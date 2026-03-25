"""Gradient-step factories and fine-tuning technique utilities.

Provides:
- ``make_ablation_grad_step``: flexible gradient-step factory supporting frozen
  backbone, LLRD, and gradient surgery.
- LoRA parameter management utilities.
- ``compute_ewc_fisher``: diagonal Fisher information estimation.
- ``gradient_projection_step``: project RL gradient onto the subspace
  orthogonal to the BC gradient to prevent catastrophic forgetting.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import optax

from src.diffusion.forward import forward_process
from src.diffusion.schedules import ScheduleFn

_EPS: float = 1e-5
_MAX_WEIGHT: float = 1000.0

LossFn = Callable[
    [Any, jax.Array, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, dict],
]
ModelApplyFn = Callable[
    [Any, jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[Any]], jnp.ndarray
]


# ---------------------------------------------------------------------------
# Gradient-step factory
# ---------------------------------------------------------------------------

def make_ablation_grad_step(
    loss_fn: LossFn,
    frozen_backbone: bool = False,
    llrd_decay: float = 1.0,
    gradient_surgery: bool = False,
    bc_loss_fn: Optional[LossFn] = None,
    max_grad_norm: float = 1.0,
) -> Callable:
    """Return a JIT-compatible gradient update function for ablation experiments.

    Supports three orthogonal modifications relative to a plain Adam step:

    1. **Frozen backbone** (``frozen_backbone=True``): zero all gradients
       except those belonging to the final output Dense layer
       (``params['params']['Dense_5']``). Only the action head is updated.

    2. **Layer-wise learning-rate decay** (``llrd_decay < 1.0``): scale the
       gradient of each named sub-pytree by ``llrd_decay^depth``, where
       depth increases for earlier layers.  Applied before the optimiser.

    3. **Gradient surgery** (``gradient_surgery=True``): if the cosine
       similarity between the RL gradient and the BC gradient is negative
       (conflict), remove the conflicting component of the RL gradient.
       Requires ``bc_loss_fn`` to be provided.

    Args:
        loss_fn:          RL / ablation loss function (returns ``(loss, info)``).
        frozen_backbone:  If True, zero all gradients except the output head.
        llrd_decay:       Exponential LR decay factor per layer depth (1.0 = off).
        gradient_surgery: If True, project out BC-conflicting gradient components.
        bc_loss_fn:       BC loss function used by gradient surgery (required when
                          ``gradient_surgery=True``).
        max_grad_norm:    Global gradient clipping threshold.

    Returns:
        ``step(state, acts, obs, valid, rng, advantages, **kwargs)
          -> (state, metrics)``
        where ``state`` is a Flax ``TrainState`` and ``**kwargs`` are forwarded
        to ``loss_fn`` (e.g. ``step_idx`` for curriculum losses).

    Raises:
        ValueError: If ``gradient_surgery=True`` but ``bc_loss_fn`` is ``None``.
    """
    if gradient_surgery and bc_loss_fn is None:
        raise ValueError(
            "gradient_surgery=True requires bc_loss_fn to be provided."
        )

    def step(
        state: Any,
        acts: jnp.ndarray,
        obs: jnp.ndarray,
        valid: jnp.ndarray,
        rng: jax.Array,
        advantages: jnp.ndarray,
        **kwargs: Any,
    ) -> tuple[Any, dict]:
        """Single gradient update step.

        Args:
            state:      Flax ``TrainState``.
            acts:       ``[B, H]`` int32 action sequences.
            obs:        ``[B, D]`` float32 observations.
            valid:      ``[B]`` bool validity mask.
            rng:        PRNG key.
            advantages: ``[B]`` float per-sample weights.
            **kwargs:   Extra keyword arguments forwarded to ``loss_fn``.

        Returns:
            Tuple of updated ``TrainState`` and metrics dict.
        """
        def _loss_fn(params: Any) -> tuple[jnp.ndarray, dict]:
            return loss_fn(params, rng, acts, obs, valid, advantages, **kwargs)

        (_, info), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
            state.params
        )

        # Gradient surgery: project out conflicting BC gradient components.
        if gradient_surgery:
            rng_bc, _ = jax.random.split(rng)

            def _bc_loss(params: Any) -> tuple[jnp.ndarray, dict]:
                return bc_loss_fn(
                    params, rng_bc, acts, obs, valid,
                    jnp.ones_like(advantages),
                )

            _, bc_grads = jax.value_and_grad(_bc_loss, has_aux=True)(
                state.params
            )
            grads = _apply_gradient_surgery(grads, bc_grads)

        # Frozen backbone: zero all gradients outside the output head.
        if frozen_backbone:
            grads = _zero_backbone_grads(grads)

        # Layer-wise LR decay: scale gradients by decay^depth.
        if llrd_decay < 1.0:
            grads = _apply_llrd(grads, llrd_decay)

        # Global gradient clipping.
        grads, _ = optax.clip_by_global_norm(max_grad_norm).update(
            grads, state.opt_state, state.params
        )

        state = state.apply_gradients(grads=grads)
        info["grad_norm"] = optax.tree.norm(grads)
        return state, info

    return step


# ---------------------------------------------------------------------------
# Internal gradient modifiers
# ---------------------------------------------------------------------------

def _flatten_grads(grads: Any) -> jnp.ndarray:
    """Flatten a gradient pytree to a 1-D vector.

    Args:
        grads: Gradient pytree.

    Returns:
        1-D float32 array of concatenated gradient values.
    """
    leaves = jax.tree.leaves(grads)
    return jnp.concatenate([leaf.ravel() for leaf in leaves])


def _apply_gradient_surgery(
    rl_grads: Any,
    bc_grads: Any,
) -> Any:
    """Project RL gradients onto the orthogonal complement of the BC gradient.

    Only the conflicting component (negative cosine similarity) is removed.
    If the two gradients are already aligned (positive cosine similarity),
    the RL gradient is returned unchanged.

    Args:
        rl_grads: RL loss gradient pytree.
        bc_grads: BC loss gradient pytree.

    Returns:
        Projected gradient pytree.
    """
    rl_flat = _flatten_grads(rl_grads)
    bc_flat = _flatten_grads(bc_grads)

    dot = jnp.dot(rl_flat, bc_flat)
    bc_norm_sq = jnp.dot(bc_flat, bc_flat) + _EPS
    proj_coef = dot / bc_norm_sq  # scalar

    # Only project when gradients conflict (negative cosine similarity).
    proj_coef = jnp.where(dot < 0.0, proj_coef, 0.0)

    projected_grads = jax.tree.map(
        lambda rl_g, bc_g: rl_g - proj_coef * bc_g,
        rl_grads, bc_grads,
    )
    return projected_grads


def _zero_backbone_grads(grads: Any) -> Any:
    """Zero all gradients except the output Dense layer (frozen backbone).

    Assumes a Flax params structure with a top-level ``'params'`` key.
    The output head is identified by the key ``'Dense_5'`` under
    ``params['params']`` (the final linear projection in DenoisingTransformer;
    with obs_encoder_layers=2: Dense_0/1=obs_enc, Dense_2=obs_tok,
    Dense_3/4=time_emb, Dense_5=action logit head).

    Args:
        grads: Full gradient pytree.

    Returns:
        Gradient pytree with backbone gradients zeroed.
    """
    def _zero_unless_head(path: tuple, leaf: jnp.ndarray) -> jnp.ndarray:
        # path is a tuple of dict keys / indices.
        path_str = "/".join(str(p.key) for p in path if hasattr(p, "key"))
        # Preserve only the output head Dense layer (Dense_5 with obs_encoder_layers=2).
        is_head = "Dense_5" in path_str
        return leaf if is_head else jnp.zeros_like(leaf)

    return jax.tree_util.tree_map_with_path(_zero_unless_head, grads)


def _apply_llrd(grads: Any, decay: float) -> Any:
    """Apply layer-wise learning-rate decay to a gradient pytree.

    Assigns a depth index to each top-level sub-tree in ``params['params']``
    and scales gradients by ``decay^depth``.  Deeper (earlier) layers receive
    smaller gradients.

    Args:
        grads: Gradient pytree (Flax ``params`` structure).
        decay: Decay factor per layer depth (0 < decay <= 1).

    Returns:
        Gradient pytree with per-layer scaling applied.
    """
    if "params" not in grads:
        return grads

    param_dict = grads["params"]
    keys = sorted(param_dict.keys())
    n = len(keys)

    scaled = {}
    for depth, key in enumerate(keys):
        scale = decay ** (n - 1 - depth)  # later layers (smaller depth idx) = higher LR
        scaled[key] = jax.tree.map(lambda g: g * scale, param_dict[key])

    return {**grads, "params": scaled}


# ---------------------------------------------------------------------------
# LoRA utilities
# ---------------------------------------------------------------------------

def make_lora_params(
    params: Any,
    rank: int = 8,
    alpha: float = 16.0,
    rng: Optional[jax.Array] = None,
) -> Any:
    """Create a LoRA delta-parameter pytree with the same structure as ``params``.

    For every Dense kernel of shape ``[in_features, out_features]``, creates
    two low-rank matrices A ``[in_features, rank]`` and B ``[rank, out_features]``
    initialised to random-normal / zeros respectively (standard LoRA init).
    Bias terms and non-Dense parameters are set to None (not trained).

    Args:
        params:  Base model parameter pytree.
        rank:    LoRA rank r.
        alpha:   LoRA scaling alpha (effective scale = alpha / rank).
        rng:     PRNG key; uses ``jax.random.PRNGKey(0)`` if ``None``.

    Returns:
        LoRA parameter pytree ``{'A': ..., 'B': ...}`` with leaves that
        are either ``(A_matrix, B_matrix)`` tuples or ``None``.
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    def _make_delta(path: tuple, leaf: jnp.ndarray) -> Optional[tuple]:
        path_str = "/".join(str(p.key) for p in path if hasattr(p, "key"))
        # Only decompose Dense kernels (2-D matrices).
        if "kernel" in path_str and leaf.ndim == 2:
            in_f, out_f = leaf.shape
            nonlocal rng
            rng, a_rng = jax.random.split(rng)
            a_mat = jax.random.normal(a_rng, (in_f, rank)) * 0.02
            b_mat = jnp.zeros((rank, out_f))
            return (a_mat, b_mat)
        return None

    lora_tree = jax.tree_util.tree_map_with_path(_make_delta, params)
    return {"lora": lora_tree, "rank": rank, "alpha": alpha}


def apply_lora_forward(
    params: Any,
    lora_params: dict,
) -> Any:
    """Compute effective parameters by adding LoRA deltas to the base params.

    For each Dense kernel where a LoRA pair ``(A, B)`` exists, computes:
        ``kernel_effective = kernel_base + (alpha / rank) * A @ B``

    Leaves with ``lora_pair == None`` are returned unchanged.

    Args:
        params:      Base model parameter pytree (frozen).
        lora_params: Output of :func:`make_lora_params`.

    Returns:
        Effective parameter pytree with LoRA updates applied.
    """
    lora_tree = lora_params["lora"]
    scale = lora_params["alpha"] / lora_params["rank"]

    def _apply(base_leaf: jnp.ndarray, lora_leaf: Optional[tuple]) -> jnp.ndarray:
        if lora_leaf is None:
            return base_leaf
        a_mat, b_mat = lora_leaf
        return base_leaf + scale * (a_mat @ b_mat)

    return jax.tree.map(_apply, params, lora_tree)


# ---------------------------------------------------------------------------
# EWC Fisher estimation
# ---------------------------------------------------------------------------

def compute_ewc_fisher(
    apply_fn: ModelApplyFn,
    params: Any,
    rng: jax.Array,
    obs: jnp.ndarray,
    acts: jnp.ndarray,
    valid: jnp.ndarray,
    num_actions: int,
    schedule_fn: ScheduleFn,
    schedule_deriv_fn: ScheduleFn,
) -> Any:
    """Compute the diagonal empirical Fisher information matrix.

    The empirical Fisher approximation is:
        F_i = E[(d/dtheta_i L(theta; x))^2]

    where the expectation is over data samples.  This is estimated by
    computing per-sample gradients (via ``jax.vmap`` over the batch) and
    squaring and averaging them.

    This function is *not* JIT-compiled; it is called once at setup time to
    pre-compute the Fisher for EWC regularisation.

    Args:
        apply_fn:          Model apply closure (eval mode, rng=None).
        params:            Pretrained model parameters (reference point).
        rng:               PRNG key for diffusion noise sampling.
        obs:               ``[B, D]`` observations from a reference batch.
        acts:              ``[B, H]`` int32 reference action sequences.
        valid:             ``[B]`` bool validity mask.
        num_actions:       Real action vocabulary size.
        schedule_fn:       alpha(t) noise schedule.
        schedule_deriv_fn: d(alpha)/dt.

    Returns:
        Diagonal Fisher pytree with the same structure as ``params``.
    """
    mask_id = num_actions
    B = acts.shape[0]

    def single_sample_loss(
        p: Any,
        sample_rng: jax.Array,
        act: jnp.ndarray,
        ob: jnp.ndarray,
    ) -> jnp.ndarray:
        """Scalar ELBO loss for a single sample.

        Args:
            p:          Model parameters.
            sample_rng: PRNG key.
            act:        ``[H]`` int32 action sequence.
            ob:         ``[D]`` float32 observation.

        Returns:
            Scalar loss.
        """
        sample_rng, t_rng, mask_rng = jax.random.split(sample_rng, 3)
        t = jax.random.uniform(t_rng, (1,), minval=_EPS, maxval=1.0)
        alpha_t = schedule_fn(t)

        neg_alpha_dot = -schedule_deriv_fn(t)
        weight = neg_alpha_dot / jnp.maximum(1.0 - alpha_t, _EPS)
        weight = jnp.minimum(weight, _MAX_WEIGHT)

        z_t = forward_process(mask_rng, act[None], alpha_t, mask_id)  # [1, H]
        logits = apply_fn(p, ob[None], z_t, t, None)  # [1, H, A]

        is_masked = (z_t == mask_id).astype(jnp.float32)
        n_masked = jnp.maximum(is_masked.sum(), 1.0)

        targets = jax.nn.one_hot(act[None], num_actions)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        ce = -jnp.sum(targets * log_probs, axis=-1)  # [1, H]
        return (weight * (ce * is_masked).sum() / n_masked).squeeze()

    grad_fn = jax.grad(single_sample_loss)

    rngs = jax.random.split(rng, B)
    # vmap: in_axes=(None, 0, 0, 0) — params shared, rest batched.
    per_sample_grads = jax.vmap(
        grad_fn, in_axes=(None, 0, 0, 0)
    )(params, rngs, acts, obs)

    # Fisher diagonal = mean of squared per-sample gradients.
    fisher = jax.tree.map(
        lambda g: jnp.mean(g ** 2, axis=0), per_sample_grads
    )
    return fisher


# ---------------------------------------------------------------------------
# Gradient projection step
# ---------------------------------------------------------------------------

def gradient_projection_step(
    state: Any,
    acts: jnp.ndarray,
    obs: jnp.ndarray,
    valid: jnp.ndarray,
    rng: jax.Array,
    advantages: jnp.ndarray,
    rl_loss_fn: LossFn,
    bc_loss_fn: LossFn,
    max_grad_norm: float = 1.0,
) -> tuple[Any, dict]:
    """Gradient-projection update: remove BC-conflicting RL gradient components.

    Computes both the RL gradient and the BC (uniform-advantage) gradient,
    then projects the RL gradient onto the orthogonal complement of the BC
    gradient subspace before applying the update.  This prevents catastrophic
    forgetting by ensuring the RL update does not directly oppose the BC
    direction.

    Args:
        state:        Flax ``TrainState``.
        acts:         ``[B, H]`` int32 action sequences.
        obs:          ``[B, D]`` float32 observations.
        valid:        ``[B]`` bool validity mask.
        rng:          PRNG key.
        advantages:   ``[B]`` GRPO advantages.
        rl_loss_fn:   RL loss function.
        bc_loss_fn:   BC loss function (uniform advantages).
        max_grad_norm: Global gradient clipping threshold.

    Returns:
        Tuple of ``(updated_state, metrics_dict)``.
    """
    rng, rng_rl, rng_bc = jax.random.split(rng, 3)

    (_, rl_info), rl_grads = jax.value_and_grad(
        lambda p: rl_loss_fn(p, rng_rl, acts, obs, valid, advantages),
        has_aux=True,
    )(state.params)

    _, bc_grads = jax.value_and_grad(
        lambda p: bc_loss_fn(
            p, rng_bc, acts, obs, valid, jnp.ones_like(advantages)
        ),
        has_aux=True,
    )(state.params)

    projected = _apply_gradient_surgery(rl_grads, bc_grads)
    clipped, _ = optax.clip_by_global_norm(max_grad_norm).update(
        projected, state.opt_state, state.params
    )

    # Cosine similarity for logging.
    rl_flat = _flatten_grads(rl_grads)
    bc_flat = _flatten_grads(bc_grads)
    cos_sim = jnp.dot(rl_flat, bc_flat) / (
        jnp.linalg.norm(rl_flat) * jnp.linalg.norm(bc_flat) + _EPS
    )

    state = state.apply_gradients(grads=clipped)
    rl_info["grad_norm"] = optax.tree.norm(clipped)
    rl_info["gradient_alignment"] = cos_sim
    return state, rl_info

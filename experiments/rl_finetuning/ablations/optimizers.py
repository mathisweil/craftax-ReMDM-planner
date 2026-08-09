"""Optimizer factory functions for all RL fine-tuning ablations.

Each factory returns an ``optax.GradientTransformation`` ready for use
in a Flax ``TrainState``.  All factories are pure functions with no
side effects.

LoRA parameter injection is also handled here: ``make_lora_params``
creates trainable LoRA matrices, and ``apply_fn_with_lora`` wraps
the base model apply function to include LoRA contributions.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax


def make_optimizer_standard(
    config: dict, params: Any = None
) -> optax.GradientTransformation:
    """AdamW with global gradient clipping — baseline optimizer.

    Args:
        config: UPPERCASE config dict with ``LR``, ``WEIGHT_DECAY``,
                and ``MAX_GRAD_NORM``.
        params: Unused; accepted for uniform interface.

    Returns:
        Optax gradient transformation.
    """
    return optax.chain(
        optax.clip_by_global_norm(config.get("MAX_GRAD_NORM", 1.0)),
        optax.adamw(
            config.get("LR", 3e-4),
            weight_decay=config.get("WEIGHT_DECAY", 1e-4),
            eps=1e-5,
        ),
    )


def _get_llrd_label(path: tuple, head_fragments: tuple[str, ...] = ()) -> str:
    """Assign a learning-rate group label to a parameter path.

    Groups (from fastest to slowest LR):
    - ``head``:       final output projection (highest LR = base_lr)
    - ``block_{i}``:  TransformerBlock at index i
    - ``obs_enc``:    observation encoder layers (lowest LR)

    Args:
        path:           Tuple of ``jax.tree_util.KeyEntry`` objects.
        head_fragments: Path substrings that identify head parameters.
                        Parameters not in any TransformerBlock and not
                        matching any head fragment are classified as
                        ``obs_enc``.

    Returns:
        Group label string.
    """
    path_str = "/".join(str(k.key) if hasattr(k, "key") else str(k) for k in path)
    if "TransformerBlock_" in path_str:
        block_str = path_str.split("TransformerBlock_")[1].split("/")[0]
        try:
            return f"block_{int(block_str)}"
        except ValueError:
            return "head"
    # Parameters outside TransformerBlocks: distinguish head vs obs encoder.
    # The head is the final Dense projection (action output); obs encoder
    # includes early Dense layers, LayerNorms, and embeddings.
    if head_fragments and any(frag in path_str for frag in head_fragments):
        return "head"
    # Heuristic: the last Dense layer in a @nn.compact module typically has
    # the highest index.  Fall back to obs_enc for everything else.
    return "obs_enc"


def make_optimizer_llrd(config: dict, params: Any) -> optax.GradientTransformation:
    """AdamW with Layer-wise Learning Rate Decay (LLRD).

    Assigns lower learning rates to earlier (more general) layers.
    LR for a layer at depth d from the top = base_lr * decay^d.

    - Head (final projection):  base_lr * decay^0 = base_lr
    - TransformerBlock_N-1:     base_lr * decay^1
    - TransformerBlock_0:       base_lr * decay^N
    - Obs encoder:              base_lr * decay^(N+1)

    Args:
        config: UPPERCASE config dict with ``LR``, ``WEIGHT_DECAY``,
                ``MAX_GRAD_NORM``, ``LLRD_DECAY``, and ``N_LAYERS``.
        params: Parameter pytree used to build the label tree.

    Returns:
        Optax multi_transform with per-group learning rates.
    """
    base_lr = config.get("LR", 3e-4)
    decay = config.get("LLRD_DECAY", 0.9)
    n_layers = config.get("N_LAYERS", 4)
    max_grad_norm = config.get("MAX_GRAD_NORM", 1.0)
    weight_decay = config.get("WEIGHT_DECAY", 1e-4)

    # Identify head parameters: the final Dense layer outside
    # TransformerBlocks is the action head (highest LR = base_lr).
    # We scan all param paths to find the highest Dense_N index
    # outside TransformerBlocks.
    head_fragments: list[str] = []
    max_dense_idx = -1
    all_paths: list[str] = jax.tree.leaves(
        jax.tree_util.tree_map_with_path(
            lambda path, _: "/".join(
                str(k.key) if hasattr(k, "key") else str(k) for k in path
            ),
            params,
        )
    )
    for p in all_paths:
        if "TransformerBlock_" not in p:
            for segment in p.split("/"):
                if segment.startswith("Dense_"):
                    try:
                        idx = int(segment.split("_")[1])
                        if idx > max_dense_idx:
                            max_dense_idx = idx
                    except (ValueError, IndexError):
                        pass
    if max_dense_idx >= 0:
        head_fragments = [f"Dense_{max_dense_idx}"]

    # Build label tree
    label_tree = jax.tree_util.tree_map_with_path(
        lambda path, _: _get_llrd_label(path, tuple(head_fragments)), params
    )

    # Build optimizer for each label
    transforms: dict[str, optax.GradientTransformation] = {
        "head": optax.adamw(base_lr, weight_decay=weight_decay, eps=1e-5),
    }
    obs_lr = base_lr * (decay ** (n_layers + 1))
    transforms["obs_enc"] = optax.adamw(obs_lr, weight_decay=weight_decay, eps=1e-5)
    for i in range(n_layers):
        depth_from_top = n_layers - i
        lr_i = base_lr * (decay**depth_from_top)
        transforms[f"block_{i}"] = optax.adamw(
            lr_i, weight_decay=weight_decay, eps=1e-5
        )

    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.multi_transform(transforms, label_tree),
    )


def make_optimizer_frozen_paths(
    config: dict, params: Any, frozen_path_fragments: list[str]
) -> optax.GradientTransformation:
    """Adam with gradient masking for specified parameter paths.

    Parameters whose full path string contains ANY fragment from
    ``frozen_path_fragments`` receive zero gradient (effectively frozen).

    Args:
        config:                 UPPERCASE config dict.
        params:                 Parameter pytree.
        frozen_path_fragments:  Path substrings identifying frozen params.

    Returns:
        Optax transformation with frozen parameters.
    """
    max_grad_norm = config.get("MAX_GRAD_NORM", 1.0)
    lr = config.get("LR", 3e-4)
    weight_decay = config.get("WEIGHT_DECAY", 1e-4)

    def should_freeze(path: tuple) -> bool:
        path_str = "/".join(str(k.key) if hasattr(k, "key") else str(k) for k in path)
        return any(frag in path_str for frag in frozen_path_fragments)

    # multi_transform, not masked: optax.masked leaves NON-selected updates
    # untouched rather than zeroing them, so masking in the trainable params
    # would hand every frozen parameter its raw clipped gradient as an update
    # (roughly SGD at lr=1.0, in the ascent direction).
    label_tree = jax.tree_util.tree_map_with_path(
        lambda path, _: "frozen" if should_freeze(path) else "trainable", params
    )

    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.multi_transform(
            {
                "trainable": optax.adamw(lr, weight_decay=weight_decay, eps=1e-5),
                "frozen": optax.set_to_zero(),
            },
            label_tree,
        ),
    )


def _num_input_axes(path_str: str, ndim: int) -> int:
    """Number of leading kernel axes that index the input of a linear map.

    Flax ``DenseGeneral`` kernels inside ``MultiHeadDotProductAttention`` are
    3-D and split differently at each end of the block::

        query/key/value  (d_model, n_heads, head_dim)  -> 1 input axis
        out              (n_heads, head_dim, d_model)  -> 2 input axes

    A plain ``Dense`` kernel is ``(d_in, d_out)`` and takes the general case.

    Args:
        path_str: Canonical ``/``-joined parameter path.
        ndim:     Rank of the kernel array.

    Returns:
        Count of leading axes forming the input dimension.
    """
    return ndim - 1 if path_str.endswith("out/kernel") else 1


def make_lora_params(
    params: Any,
    rank: int,
    rng: jax.Array,
    path_fragment: str = "MultiHeadDotProductAttention",
) -> dict[str, dict[str, jax.Array]]:
    """Create LoRA A and B matrices for all target attention kernels.

    A kernel of shape ``S`` is treated as a ``[d_in, d_out]`` matrix, where
    ``d_in`` is the product of its input axes (see :func:`_num_input_axes`)
    and ``d_out`` the product of the rest:

    - A: ``[d_in, rank]`` — Gaussian initialised
    - B: ``[rank, d_out]`` — zero initialised (so initial LoRA delta = 0)

    Kernels of any rank are supported.  Restricting this to 2-D matched no
    parameter in ``DenoisingTransformer``, whose attention kernels are all
    3-D, so the LoRA ablation previously trained nothing at all.

    Args:
        params:        Parameter pytree to inspect for target shapes.
        rank:          LoRA rank r.
        rng:           PRNG key.
        path_fragment: Substring identifying target layers.

    Returns:
        Dict mapping canonical path string -> {"A": array, "B": array}.
    """
    lora: dict[str, dict[str, jax.Array]] = {}

    def _init(path: tuple, leaf):
        path_str = "/".join(str(k.key) if hasattr(k, "key") else str(k) for k in path)
        if path_fragment in path_str and "kernel" in path_str and leaf.ndim >= 2:
            nonlocal rng
            split = _num_input_axes(path_str, leaf.ndim)
            d_in = int(np.prod(leaf.shape[:split]))
            d_out = int(np.prod(leaf.shape[split:]))
            rng, a_rng = jax.random.split(rng)
            lora[path_str] = {
                "A": jax.random.normal(a_rng, (d_in, rank)) * 0.02,
                "B": jnp.zeros((rank, d_out)),
            }

    jax.tree_util.tree_map_with_path(_init, params)
    return lora


def apply_fn_with_lora(
    base_apply_fn: Callable,
    base_params: Any,
    lora_params: dict[str, dict[str, jax.Array]],
    alpha: float,
    rank: int,
    obs: jnp.ndarray,
    z_t: jnp.ndarray,
    t: jnp.ndarray,
    rng: jax.Array | None = None,
) -> jnp.ndarray:
    """Apply base model with LoRA by injecting effective weights into the param tree.

    For each attention kernel W at path p, the effective weight is:
        W_eff = W_frozen + (alpha / rank) * lora_B @ lora_A

    The base model is called with W_eff in place of W, without any
    modification to the model definition.

    Args:
        base_apply_fn: Original model apply fn (params, obs, z_t, t, rng) -> logits.
        base_params:   Frozen base parameters (never modified in-place).
        lora_params:   LoRA parameter dict from ``make_lora_params``.
        alpha:         LoRA alpha scaling factor.
        rank:          LoRA rank (must match the rank used in ``make_lora_params``).
        obs:           ``[B, obs_dim]`` observations.
        z_t:           ``[B, H]`` noisy action tokens.
        t:             ``[B]`` diffusion times.
        rng:           Optional PRNG key for dropout.

    Returns:
        ``[B, H, num_actions]`` logits.
    """
    scale = alpha / max(rank, 1)

    def inject(path: tuple, param: jnp.ndarray) -> jnp.ndarray:
        path_str = "/".join(str(k.key) if hasattr(k, "key") else str(k) for k in path)
        if path_str in lora_params:
            ab = lora_params[path_str]
            # A: [d_in, rank], B: [rank, d_out] → delta: [d_in, d_out],
            # reshaped back to the kernel's own (possibly 3-D) shape.
            delta = (ab["A"] @ ab["B"]).reshape(param.shape)
            return param + scale * delta
        return param

    effective_params = jax.tree_util.tree_map_with_path(inject, base_params)
    return base_apply_fn(effective_params, obs, z_t, t, rng)


def make_optimizer_lora_only(
    config: dict,
    base_params: Any,
    lora_params: dict[str, dict[str, jax.Array]],
) -> optax.GradientTransformation:
    """Adam that only updates LoRA parameters; base params receive zero gradient.

    The combined parameter tree passed to the TrainState is expected to be
    ``{"base": base_params, "lora": lora_params}``.

    Args:
        config:      UPPERCASE config dict.
        base_params: Frozen base parameters.
        lora_params: Trainable LoRA parameters.

    Returns:
        Optax transformation for the combined ``{"base": ..., "lora": ...}`` tree.
    """
    lr = config.get("LR", 3e-4)
    max_grad_norm = config.get("MAX_GRAD_NORM", 1.0)
    weight_decay = config.get("WEIGHT_DECAY", 1e-4)

    # multi_transform, not masked: see make_optimizer_frozen_paths.  With
    # optax.masked the base parameters would receive their raw gradients.
    label_tree = {
        "base": jax.tree.map(lambda _: "frozen", base_params),
        "lora": jax.tree.map(lambda _: "trainable", lora_params),
    }

    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.multi_transform(
            {
                "trainable": optax.adamw(lr, weight_decay=weight_decay, eps=1e-5),
                "frozen": optax.set_to_zero(),
            },
            label_tree,
        ),
    )


def gradient_surgery(g_rl: Any, g_bc: Any) -> Any:
    """Project RL gradients onto the plane orthogonal to conflicting BC gradients.

    For each parameter leaf: if dot(g_rl_i, g_bc_i) < 0, projects the RL
    gradient to remove the component pointing against g_bc_i.
    Otherwise, keeps g_rl_i unchanged.

    This is the PCGrad operation applied per-parameter-tensor.

    Args:
        g_rl: RL gradient pytree.
        g_bc: BC gradient pytree (reference direction).

    Returns:
        Projected RL gradient pytree.
    """

    def _project(g_r: jnp.ndarray, g_b: jnp.ndarray) -> jnp.ndarray:
        dot = jnp.sum(g_r * g_b)
        norm_sq = jnp.sum(g_b * g_b) + 1e-10
        projected = g_r - (dot / norm_sq) * g_b
        return jnp.where(dot < 0, projected, g_r)

    return jax.tree.map(_project, g_rl, g_bc)

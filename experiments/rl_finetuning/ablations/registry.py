"""AblationSpec registry: all 18+ RL fine-tuning ablations.

Usage::

    from experiments.rl_finetuning.ablations.registry import REGISTRY, AblationSpec
    spec = REGISTRY["ewc"]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from experiments.rl_finetuning.ablations.losses import (
    LossFn,
    make_loss_advantage_clip,
    make_loss_baseline,
    make_loss_bc_wins,
    make_loss_entropy_bonus,
    make_loss_ewc,
    make_loss_frozen_backbone,
    make_loss_gradient_surgery,
    make_loss_kl_penalty,
    make_loss_low_t,
    make_loss_mixed_replay,
    make_loss_normalized_adv,
    make_loss_param_isolation,
    make_loss_reward_quality,
    make_loss_t_curriculum,
    make_loss_trust_region_kl,
)
from experiments.rl_finetuning.ablations.optimizers import (
    make_optimizer_frozen_paths,
    make_optimizer_llrd,
    make_optimizer_standard,
)

# Type alias for optimizer factory
# Signature: (config, params) -> optax.GradientTransformation
OptimizerFactory = Callable[[dict, Any], Any]

# Type alias for loss factory: (ctx: LossContext, **extra_kwargs) -> LossFn
LossFactory = Callable[..., LossFn]


@dataclass
class AblationSpec:
    """Specification for a single RL fine-tuning ablation.

    Args:
        name:                   Short identifier used in CLI and output paths.
        group:                  Ablation group: "Baseline", "A", "B", "C", or "D".
        description:            One-line human-readable description.
        hypothesis:             What this ablation tests (failure hypothesis).
        loss_factory:           Callable(ctx, **extra) -> LossFn.
        optimizer_factory:      Callable(config, params) -> GradientTransformation.
        frozen_path_fragments:  Parameter path substrings to freeze (zero gradient).
        wins_only:              If True, pre-filter batch to windows with return > threshold.
        gradient_surgery:       If True, apply PCGrad to RL vs. BC gradients.
        mixed_replay:           If True, mix offline buffer into each batch.
        t_curriculum:           If True, anneal t range over training.
        reward_filtering:       If True, discard windows with return < percentile.
        running_stats:          If True, normalise advantages with running mean/std.
        action_diversity_filter: If True, discard degenerate (all-same-action) plans.
        reward_model_weighting: If True, weight advantages with a learned reward model.
        extra_loss_kwargs:      Extra keyword arguments forwarded to loss_factory.
    """

    name: str
    group: str
    description: str
    hypothesis: str
    loss_factory: LossFactory
    optimizer_factory: OptimizerFactory = field(
        default_factory=lambda: make_optimizer_standard
    )
    frozen_path_fragments: list[str] = field(default_factory=list)
    wins_only: bool = False
    gradient_surgery: bool = False
    mixed_replay: bool = False
    t_curriculum: bool = False
    reward_filtering: bool = False
    running_stats: bool = False
    action_diversity_filter: bool = False
    reward_model_weighting: bool = False
    extra_loss_kwargs: dict = field(default_factory=dict)


def _std_opt(config: dict, params: Any) -> Any:
    return make_optimizer_standard(config, params)


def _llrd_opt(config: dict, params: Any) -> Any:
    return make_optimizer_llrd(config, params)


def _frozen_backbone_opt(config: dict, params: Any) -> Any:
    # Freeze everything except the final Dense (action head)
    # In Flax @nn.compact, the head is typically the last Dense with kernel shape [d_model, num_actions]
    # We freeze by keeping everything that is NOT the head
    # Since we can't reliably identify the head by name without running init, we freeze
    # TransformerBlock and obs encoder params, leaving any remaining Dense (the head) trainable.
    frozen = ["TransformerBlock_", "SinusoidalPosEmbed_"]
    # Also freeze obs encoder dense layers (all Dense_* before transformer blocks)
    frozen += ["Dense_0", "Dense_1", "LayerNorm_0", "LayerNorm_1"]
    return make_optimizer_frozen_paths(config, params, frozen)


def _head_only_opt(config: dict, params: Any) -> Any:
    # Freeze: all transformer blocks, obs encoder, positional embeddings, action embeddings
    frozen = [
        "TransformerBlock_",
        "SinusoidalPosEmbed_",
        "Dense_0",
        "Dense_1",
        "Dense_2",
        "LayerNorm_0",
        "LayerNorm_1",
        "Embed_",
    ]
    return make_optimizer_frozen_paths(config, params, frozen)


def _attention_only_opt(config: dict, params: Any) -> Any:
    # Freeze FFN layers (Dense_0, Dense_1 inside TransformerBlocks = FFN)
    # Keep MultiHeadDotProductAttention trainable
    frozen = ["TransformerBlock_/Dense_", "LayerNorm_"]
    # Also freeze obs encoder
    frozen += ["Dense_0", "Dense_1", "SinusoidalPosEmbed_"]
    return make_optimizer_frozen_paths(config, params, frozen)


def _ffn_only_opt(config: dict, params: Any) -> Any:
    # Freeze attention weights, keep FFN dense layers trainable
    frozen = [
        "MultiHeadDotProductAttention_",
        "Dense_0",
        "Dense_1",
        "SinusoidalPosEmbed_",
        "Embed_",
    ]
    return make_optimizer_frozen_paths(config, params, frozen)


def _layer_ablation_top_n_opt(n: int) -> OptimizerFactory:
    """Factory: freeze all transformer blocks except the top n + head."""

    def _opt(config: dict, params: Any) -> Any:
        n_layers = config.get("N_LAYERS", 4)
        # Top n blocks have the highest indices
        trainable_block_indices = list(range(n_layers - n, n_layers))
        # Freeze all blocks NOT in trainable set
        frozen = []
        for i in range(n_layers):
            if i not in trainable_block_indices:
                frozen.append(f"TransformerBlock_{i}")
        # Freeze obs encoder but keep head trainable
        frozen += ["Dense_0", "Dense_1", "SinusoidalPosEmbed_", "Embed_"]
        # Note: the final output Dense (head) is NOT frozen; it has a
        # higher index than Dense_0/Dense_1 and is therefore not matched.
        return make_optimizer_frozen_paths(config, params, frozen)

    return _opt


REGISTRY: dict[str, AblationSpec] = {
    "baseline_rl": AblationSpec(
        name="baseline_rl",
        group="Baseline",
        description="Return-weighted ELBO — no modifications",
        hypothesis="Diagnoses whether the RL signal alone causes collapse",
        loss_factory=make_loss_baseline,
        optimizer_factory=_std_opt,
    ),
    "kl_penalty": AblationSpec(
        name="kl_penalty",
        group="A",
        description="Return-weighted ELBO + soft KL penalty vs. pretrained",
        hypothesis="If this helps: catastrophic forgetting is the primary cause; "
        "soft regularisation suffices",
        loss_factory=make_loss_kl_penalty,
        optimizer_factory=_std_opt,
    ),
    "ewc": AblationSpec(
        name="ewc",
        group="A",
        description="ELBO + Elastic Weight Consolidation (Fisher diagonal regularisation)",
        hypothesis="If EWC helps: forgetting pretrained representations is the proximate cause",
        loss_factory=make_loss_ewc,
        optimizer_factory=_std_opt,
        # fisher is injected at runtime via extra_loss_kwargs by training.py
        extra_loss_kwargs={"fisher": None},  # placeholder; replaced at runtime
    ),
    "llrd": AblationSpec(
        name="llrd",
        group="A",
        description="Baseline ELBO with Layer-wise Learning Rate Decay",
        hypothesis="If LLRD helps: deep gradient flow into early layers corrupts representations",
        loss_factory=make_loss_baseline,
        optimizer_factory=_llrd_opt,
    ),
    "lora": AblationSpec(
        name="lora",
        group="A",
        description="Baseline ELBO with LoRA adaptation (rank-r attention projections only)",
        hypothesis="If LoRA works: too many unconstrained degrees of freedom cause collapse",
        loss_factory=make_loss_baseline,
        optimizer_factory=_std_opt,  # replaced at runtime with LoRA-specific optimizer
    ),
    "mixed_replay": AblationSpec(
        name="mixed_replay",
        group="A",
        description="Baseline ELBO with offline PPO data mixed into online batches",
        hypothesis="If mixed replay helps: online data distribution alone is too corrupted",
        loss_factory=make_loss_mixed_replay,
        optimizer_factory=_std_opt,
        mixed_replay=True,
    ),
    "trust_region_kl": AblationSpec(
        name="trust_region_kl",
        group="A",
        description="Baseline ELBO + hard KL trust region via quadratic barrier",
        hypothesis="If hard constraint helps: soft KL is insufficient — a hard boundary is needed",
        loss_factory=make_loss_trust_region_kl,
        optimizer_factory=_std_opt,
    ),
    "t_curriculum": AblationSpec(
        name="t_curriculum",
        group="B",
        description="ELBO with t range annealed from high-t to low-t over training",
        hypothesis="If curriculum helps: ordering of learning signals matters",
        loss_factory=make_loss_t_curriculum,
        optimizer_factory=_std_opt,
        t_curriculum=True,
    ),
    "entropy_bonus": AblationSpec(
        name="entropy_bonus",
        group="B",
        description="Baseline ELBO minus entropy bonus (encourages action diversity)",
        hypothesis="If entropy bonus helps: collapse is mode-collapse; not a gradient problem",
        loss_factory=make_loss_entropy_bonus,
        optimizer_factory=_std_opt,
    ),
    "gradient_surgery": AblationSpec(
        name="gradient_surgery",
        group="B",
        description="PCGrad: RL gradient projected to remove conflict with BC gradient",
        hypothesis="If PCGrad helps: gradients are conflicting and resolvable by projection",
        loss_factory=make_loss_gradient_surgery,
        optimizer_factory=_std_opt,
        gradient_surgery=True,
    ),
    "advantage_clip": AblationSpec(
        name="advantage_clip",
        group="B",
        description="Baseline ELBO with PPO-style advantage clipping to [1-eps, 1+eps]",
        hypothesis="If clipping helps: large advantage magnitudes destabilise training",
        loss_factory=make_loss_advantage_clip,
        optimizer_factory=_std_opt,
    ),
    "normalized_adv": AblationSpec(
        name="normalized_adv",
        group="B",
        description="Baseline ELBO with (A - mean) / (std + eps) advantage normalisation",
        hypothesis="If std normalisation helps: simple mean normalisation is too loose",
        loss_factory=make_loss_normalized_adv,
        optimizer_factory=_std_opt,
    ),
    "bc_wins": AblationSpec(
        name="bc_wins",
        group="B",
        description="Uniform ELBO on win windows only (no advantage weighting)",
        hypothesis="If BC on wins helps: the return weighting is the specific cause",
        loss_factory=make_loss_bc_wins,
        optimizer_factory=_std_opt,
        wins_only=True,
    ),
    "low_t": AblationSpec(
        name="low_t",
        group="B",
        description="Return-weighted ELBO restricted to low-t (fine-detail) regime",
        hypothesis="If low-t helps: high-t (coarse-structure) gradients are biased",
        loss_factory=make_loss_low_t,
        optimizer_factory=_std_opt,
    ),
    "frozen_backbone": AblationSpec(
        name="frozen_backbone",
        group="C",
        description="Baseline ELBO with all params frozen except the final output head",
        hypothesis="If frozen backbone helps: deep gradient flow into backbone causes collapse",
        loss_factory=make_loss_frozen_backbone,
        optimizer_factory=_frozen_backbone_opt,
    ),
    "head_only": AblationSpec(
        name="head_only",
        group="C",
        description="Baseline ELBO updating only the final linear projection",
        hypothesis="If head-only works: backbone representations are fine; only decision boundary needs updating",
        loss_factory=make_loss_param_isolation,
        optimizer_factory=_head_only_opt,
    ),
    "attention_only": AblationSpec(
        name="attention_only",
        group="C",
        description="Baseline ELBO updating only attention weights (Q/K/V/O); FFN frozen",
        hypothesis="If attention-only works: model needs routing updates, not feature updates",
        loss_factory=make_loss_param_isolation,
        optimizer_factory=_attention_only_opt,
    ),
    "ffn_only": AblationSpec(
        name="ffn_only",
        group="C",
        description="Baseline ELBO updating only FFN layers; attention frozen",
        hypothesis="If FFN-only works: stored knowledge (FFN as memory) needs updating; not attention",
        loss_factory=make_loss_param_isolation,
        optimizer_factory=_ffn_only_opt,
    ),
    "layer_ablation_top1": AblationSpec(
        name="layer_ablation_top1",
        group="C",
        description="Baseline ELBO updating only the top-1 transformer block",
        hypothesis="Minimal unfrozen depth needed; collapse depth correlates with gradient flow depth",
        loss_factory=make_loss_param_isolation,
        optimizer_factory=_layer_ablation_top_n_opt(1),
    ),
    "layer_ablation_top2": AblationSpec(
        name="layer_ablation_top2",
        group="C",
        description="Baseline ELBO updating only the top-2 transformer blocks",
        hypothesis="Minimal unfrozen depth needed; collapse depth correlates with gradient flow depth",
        loss_factory=make_loss_param_isolation,
        optimizer_factory=_layer_ablation_top_n_opt(2),
    ),
    "layer_ablation_top3": AblationSpec(
        name="layer_ablation_top3",
        group="C",
        description="Baseline ELBO updating only the top-3 transformer blocks",
        hypothesis="Minimal unfrozen depth needed; collapse depth correlates with gradient flow depth",
        loss_factory=make_loss_param_isolation,
        optimizer_factory=_layer_ablation_top_n_opt(3),
    ),
    "reward_filtering": AblationSpec(
        name="reward_filtering",
        group="D",
        description="Baseline ELBO trained only on top-75th-percentile return windows",
        hypothesis="If filtering helps: noisy/low-return data poisons gradients",
        loss_factory=make_loss_reward_quality,
        optimizer_factory=_std_opt,
        reward_filtering=True,
    ),
    "running_stats": AblationSpec(
        name="running_stats",
        group="D",
        description="Baseline ELBO with EMA running mean/std for advantage normalisation",
        hypothesis="If running stats help: batch normalisation is too noisy for small batches",
        loss_factory=make_loss_reward_quality,
        optimizer_factory=_std_opt,
        running_stats=True,
    ),
    "action_diversity": AblationSpec(
        name="action_diversity",
        group="D",
        description="Baseline ELBO with degenerate (all-same-action) plans discarded",
        hypothesis="If diversity filtering helps: degenerate PPO plans corrupt training",
        loss_factory=make_loss_reward_quality,
        optimizer_factory=_std_opt,
        action_diversity_filter=True,
    ),
    "reward_model": AblationSpec(
        name="reward_model",
        group="D",
        description="Baseline ELBO with advantages re-weighted by a learned MLP reward model",
        hypothesis="If reward model helps: raw returns are too sparse; learned model smooths signal",
        loss_factory=make_loss_reward_quality,
        optimizer_factory=_std_opt,
        reward_model_weighting=True,
    ),
}

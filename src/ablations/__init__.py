"""Ablation-study utilities for ReMDM RL fine-tuning experiments.

Submodules
----------
losses         : Loss-function factories for all ablation variants.
techniques     : Gradient-step factories, LoRA helpers, EWC Fisher, projection.
diagnostics    : Gradient alignment, representation drift, per-t-bin loss, etc.
visualisations : Matplotlib plotting and Polars summary tables.
runner         : End-to-end run_ablation_v2 training loop.
"""

from src.ablations.losses import (
    make_loss_baseline,
    make_loss_kl,
    make_loss_bc_wins,
    make_loss_low_t,
    make_loss_ewc,
    make_loss_mixed_replay,
    make_loss_t_curriculum,
    make_loss_entropy_reg,
    make_loss_token_advantage,
    make_loss_trust_region,
)
from src.ablations.techniques import (
    make_ablation_grad_step,
    make_lora_params,
    apply_lora_forward,
    compute_ewc_fisher,
    gradient_projection_step,
)
from src.ablations.diagnostics import (
    compute_gradient_alignment,
    compute_representation_drift,
    compute_output_kl,
    compute_per_t_loss,
    compute_token_entropy,
    compute_collapse_fraction,
    compute_per_layer_grad_norm,
)
from src.ablations.visualisations import (
    plot_training_dynamics,
    plot_summary_bars,
    plot_scatter_diagnostics,
    plot_per_method_deep_dive,
    plot_t_bin_heatmap,
    plot_failure_mode_map,
    plot_achievement_bars,
    make_summary_table,
    make_correlation_table,
)
from src.ablations.runner import run_ablation_v2

__all__ = [
    # losses
    "make_loss_baseline",
    "make_loss_kl",
    "make_loss_bc_wins",
    "make_loss_low_t",
    "make_loss_ewc",
    "make_loss_mixed_replay",
    "make_loss_t_curriculum",
    "make_loss_entropy_reg",
    "make_loss_token_advantage",
    "make_loss_trust_region",
    # techniques
    "make_ablation_grad_step",
    "make_lora_params",
    "apply_lora_forward",
    "compute_ewc_fisher",
    "gradient_projection_step",
    # diagnostics
    "compute_gradient_alignment",
    "compute_representation_drift",
    "compute_output_kl",
    "compute_per_t_loss",
    "compute_token_entropy",
    "compute_collapse_fraction",
    "compute_per_layer_grad_norm",
    # visualisations
    "plot_training_dynamics",
    "plot_summary_bars",
    "plot_scatter_diagnostics",
    "plot_per_method_deep_dive",
    "plot_t_bin_heatmap",
    "plot_failure_mode_map",
    "plot_achievement_bars",
    "make_summary_table",
    "make_correlation_table",
    # runner
    "run_ablation_v2",
]

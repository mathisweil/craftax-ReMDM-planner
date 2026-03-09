"""Batch logging for ReMDM training, mirroring Craftax_Baselines/logz/batch_logging.py.

Buffers metric dicts from ``jax.debug.callback`` across ``NUM_REPEATS`` runs
for the same update step, then aggregates and calls ``wandb.log`` once all
repeats have reported.
"""

import time

import jax.numpy as jnp
import numpy as np
import wandb

batch_logs: dict = {}
log_times: list = []

# Keys that should be averaged across repeats (scalars).
MEAN_KEYS = {
    "episode_return",
    "episode_length",
    "diffusion_loss",
    "mean_t",
    "frac_masked",
    "grad_norm",
    "action_entropy",
    "action_unique_frac",
    "mean_step_reward",
    "reward_std",
    "plan_diversity",
    "num_completed_eps",
    "achievements",
}


def create_log_dict(metric, config):
    """Build a flat logging dict from a JAX metric pytree.

    Matches the Craftax_Baselines ``create_log_dict`` interface and adds
    ReMDM-specific diffusion metrics.
    """
    to_log = {
        "episode_return": metric.get("episode_return", float("nan")),
        "episode_length": metric.get("episode_length", float("nan")),
    }

    # Diffusion-specific metrics.
    for k in (
        "diffusion_loss", "mean_t", "frac_masked", "grad_norm",
        "action_entropy", "action_unique_frac",
        "mean_step_reward", "reward_std", "plan_diversity",
        "num_completed_eps",
    ):
        if k in metric:
            to_log[k] = metric[k]

    # Craftax achievements (any key containing "achievement").
    sum_achievements = 0.0
    for k, v in metric.items():
        if "achievement" in k.lower() and k != "achievements":
            to_log[k] = v
            sum_achievements += v / 100.0
    to_log["achievements"] = sum_achievements

    return to_log


def batch_log(update_step, log, config):
    """Buffer a single repeat's log dict and aggregate when all repeats arrive.

    Mirrors ``Craftax_Baselines.logz.batch_logging.batch_log``.
    """
    update_step = int(update_step)
    if update_step not in batch_logs:
        batch_logs[update_step] = []

    batch_logs[update_step].append(log)

    if len(batch_logs[update_step]) == config["NUM_REPEATS"]:
        agg_logs: dict = {}
        for key in batch_logs[update_step][0]:
            agg = []
            for i in range(config["NUM_REPEATS"]):
                val = batch_logs[update_step][i][key]
                if not jnp.isnan(val):
                    agg.append(val)

            if len(agg) > 0:
                if key in MEAN_KEYS or "achievement" in key.lower():
                    agg_logs[key] = np.mean(agg)
                else:
                    agg_logs[key] = np.array(agg)

        log_times.append(time.time())

        if config.get("DEBUG"):
            if len(log_times) == 1:
                print("Started logging")
            elif len(log_times) > 1:
                dt = log_times[-1] - log_times[-2]
                steps_between = (
                    config["NUM_STEPS"] * config["NUM_ENVS"] * config["NUM_REPEATS"]
                )
                sps = steps_between / dt
                agg_logs["sps"] = sps

        wandb.log(agg_logs)

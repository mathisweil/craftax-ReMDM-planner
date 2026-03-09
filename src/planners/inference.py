import time
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import wandb
from craftax.craftax_env import make_craftax_env_from_name

from src.models.remdm import sample_plan
from Craftax_Baselines.wrappers import (
    AutoResetEnvWrapper,
    BatchEnvWrapper,
    LogWrapper,
)
from src.envs.wrappers import PlannerWrapper

from .common import SCHEDULE_MAP
from .utils import (
    _build_model,
    _load_checkpoint,
    _make_apply_fns,
)

def run_inference(config: Dict[str, Any]) -> None:
    env = make_craftax_env_from_name(config["ENV_NAME"], True)
    env_params = env.default_params
    num_actions: int = env.action_space(env_params).n
    obs_dim: int = env.observation_space(env_params).shape[0]
    config["NUM_ACTIONS"] = num_actions

    num_envs: int = config["NUM_ENVS"]
    plan_horizon: int = config["PLAN_HORIZON"]
    replan_every: int = config["REPLAN_EVERY"]
    diffusion_steps: int = config["DIFFUSION_STEPS"]
    schedule_fn = SCHEDULE_MAP[config["DIFFUSION_SCHEDULE"]]
    remask_strategy: str = config["REMASK_STRATEGY"]
    eta: float = config["ETA"]
    t_on: float = config.get("T_ON", 0.7)
    t_off: float = config.get("T_OFF", 0.3)
    use_loop: bool = config.get("USE_LOOP", False)
    temperature: float = config.get("TEMPERATURE", 1.0)
    top_p: Optional[float] = config.get("TOP_P", None)
    eval_steps: int = config.get("EVAL_STEPS", 1000)

    model = _build_model(config, num_actions)
    apply_inference, _ = _make_apply_fns(model)

    assert config.get("CHECKPOINT_PATH"), "--checkpoint_path required for inference"
    model_params = _load_checkpoint(config, model, obs_dim, config["CHECKPOINT_PATH"])

    def planner_apply_fn(rng, model_params, obs):
        return sample_plan(
            apply_inference, model_params, rng, obs,
            num_actions, plan_horizon, diffusion_steps, schedule_fn,
            remask_strategy, eta, use_loop, t_on, t_off, temperature, top_p,
        )

    env_w = PlannerWrapper(
        BatchEnvWrapper(AutoResetEnvWrapper(LogWrapper(env)), num_envs=num_envs),
        num_envs=num_envs,
        plan_horizon=plan_horizon,
        replan_every=replan_every,
        planner_apply_fn=planner_apply_fn,
    )

    rng = jax.random.PRNGKey(config["SEED"])

    @jax.jit
    def _eval_loop(rng):
        rng, env_rng = jax.random.split(rng)
        obs, state = env_w.reset(env_rng, env_params)

        def _step(carry, _):
            obs, state, rng = carry
            rng, step_rng = jax.random.split(rng)
            obs, state, action, reward, done, info = env_w.step(
                step_rng, state, obs, model_params, env_params,
            )
            return (obs, state, rng), (reward, done, info)

        _, (rewards, dones, infos) = jax.lax.scan(_step, (obs, state, rng), None, eval_steps)
        return rewards, dones, infos

    t0 = time.time()
    rewards, dones, infos = _eval_loop(rng)
    elapsed = time.time() - t0

    ep_returns, ep_mask = infos["returned_episode_returns"], infos["returned_episode"]
    completed = ep_mask.sum()
    mean_return = jnp.where(completed > 0, (ep_returns * ep_mask).sum() / completed, jnp.nan)

    print(f"Eval time: {elapsed:.1f}s  ({eval_steps * num_envs} steps)")
    print(f"Completed episodes: {int(completed)} | Mean return: {float(mean_return):.2f} | Mean step reward: {float(rewards.mean()):.4f}")

    if config.get("USE_WANDB"):
        eval_log = {
            "eval/mean_return": float(mean_return),
            "eval/completed_episodes": int(completed),
            "eval/mean_step_reward": float(rewards.mean()),
            "eval/sps": eval_steps * num_envs / max(elapsed, 1e-6),
        }
        sum_achievements = 0.0
        for k, v in infos.items():
            if "achievement" in k.lower():
                val = float(jnp.where(
                    completed > 0,
                    (v * ep_mask).sum() / completed,
                    jnp.nan,
                ))
                eval_log[f"eval/{k}"] = val
                sum_achievements += val / 100.0
        eval_log["eval/achievements"] = sum_achievements
        wandb.log(eval_log)
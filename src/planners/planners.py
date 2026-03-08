from __future__ import annotations

import pathlib
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from craftax.craftax_env import make_craftax_env_from_name
from flax.training.train_state import TrainState

from src.models.remdm import (
    ScheduleFn,
    compute_loss,
    cosine_schedule,
    linear_schedule,
    sample_plan,
)

from Craftax_Baselines.wrappers import (
    AutoResetEnvWrapper,
    BatchEnvWrapper,
    LogWrapper,
)
from src.envs.wrappers import PlannerWrapper
from .utils import (
    _build_model,
    _init_model_params,
    _create_train_state,
    _load_checkpoint,
    _load_ppo_checkpoint,
    _save_model,
    _make_env_stack,
    _valid_window_mask,
    _sample_windows_from_chunk,
    _make_apply_fns,
)

SCHEDULE_MAP: Dict[str, ScheduleFn] = {
    "cosine": cosine_schedule,
    "linear": linear_schedule,
}


# =============================================================================
# Data Collection  (step 2a, saves trajectories to disk)
# =============================================================================


def collect_offline_data(config: Dict[str, Any]) -> None:
    """Roll out a pre-trained PPO checkpoint and save (obs, actions, dones) to disk.

    Requires ``--ppo_checkpoint_path`` pointing to a pre-trained ActorCritic checkpoint.

    Data is saved with per-environment contiguity preserved::

        obs:     [num_envs, num_iters, obs_dim]
        actions: [num_envs, num_iters]
        dones:   [num_envs, num_iters]
    """
    assert config.get("PPO_CHECKPOINT_PATH"), (
        "--ppo_checkpoint_path is required for --mode collect.\n"
        "Train a PPO agent first with ppo_rnn.py or ppo_rnd.py."
    )

    env = make_craftax_env_from_name(config["ENV_NAME"], True)
    env_params = env.default_params
    num_actions: Sequence[int] | int = env.action_space(env_params).n
    obs_dim: int = env.observation_space(env_params).shape[0]
    num_envs: int = config["COLLECT_NUM_ENVS"]
    layer_size: int = config.get("LAYER_SIZE", 512)

    ppo_agent = _load_ppo_checkpoint(
        config["PPO_CHECKPOINT_PATH"], num_actions, obs_dim, layer_size,
        model_type=config.get("PPO_MODEL_TYPE"),
    )

    env_w, _ = _make_env_stack(config, num_envs)

    rng = jax.random.PRNGKey(config["SEED"])
    rng, env_rng, collect_rng = jax.random.split(rng, 3)

    obs, env_state = env_w.reset(env_rng, env_params)
    done = jnp.zeros(num_envs, dtype=bool)
    hidden = ppo_agent.init_hidden(num_envs)
    num_iters: int = config["COLLECT_NUM_STEPS"] // num_envs

    all_obs: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    all_dones: List[np.ndarray] = []

    if ppo_agent.model_type == "ppo_rnn":
        @jax.jit
        def _step_rnn(
            rng: jax.Array,
            env_state: Any,
            obs: jnp.ndarray,
            done: jnp.ndarray,
            hidden: jax.Array,
        ) -> Tuple[jax.Array, Any, jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.Array]:
            rng, k1, k2 = jax.random.split(rng, 3)
            pi, _, new_hidden = ppo_agent.apply(obs, hidden=hidden, done=done)
            action = pi.sample(seed=k1)
            obs_next, env_state, _, done_next, _ = env_w.step(
                k2, env_state, action, env_params
            )
            return rng, env_state, obs_next, action, done_next, new_hidden

        for i in range(num_iters):
            collect_rng, env_state, obs_next, action, done, hidden = _step_rnn(
                collect_rng, env_state, obs, done, hidden
            )
            all_obs.append(np.array(obs))
            all_actions.append(np.array(action))
            all_dones.append(np.array(done))
            obs = obs_next
            if (i + 1) % 500 == 0:
                print(f"  {(i + 1) * num_envs:,} / {config['COLLECT_NUM_STEPS']:,} steps")
    else:
        @jax.jit
        def _step(
            rng: jax.Array,
            env_state: Any,
            obs: jnp.ndarray,
            done: jnp.ndarray,
        ) -> Tuple[jax.Array, Any, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            rng, k1, k2 = jax.random.split(rng, 3)
            pi, _, _ = ppo_agent.apply(ppo_agent.params, obs)
            action = pi.sample(seed=k1)
            obs_next, env_state, _, done_next, _ = env_w.step(
                k2, env_state, action, env_params
            )
            return rng, env_state, obs_next, action, done_next

        for i in range(num_iters):
            collect_rng, env_state, obs_next, action, done = _step(
                collect_rng, env_state, obs, done
            )
            all_obs.append(np.array(obs))
            all_actions.append(np.array(action))
            all_dones.append(np.array(done))
            obs = obs_next
            if (i + 1) % 500 == 0:
                print(f"  {(i + 1) * num_envs:,} / {config['COLLECT_NUM_STEPS']:,} steps")

    # Stack into [num_iters, num_envs, ...] then transpose -> [num_envs, num_iters, ...]
    obs_arr = np.stack(all_obs, axis=0).transpose(1, 0, 2)   # [E, T, obs_dim]
    act_arr = np.stack(all_actions, axis=0).transpose(1, 0)   # [E, T]
    done_arr = np.stack(all_dones, axis=0).transpose(1, 0)    # [E, T]

    out_path = config["OFFLINE_DATA_PATH"]
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, obs=obs_arr, actions=act_arr, dones=done_arr)
    print(
        f"Saved {obs_arr.shape[0]}x{obs_arr.shape[1]} transitions "
        f"({obs_arr.shape[0] * obs_arr.shape[1]:,} total) to '{out_path}'"
    )


# =============================================================================
# Offline Training, from saved trajectories  (step 3)
# =============================================================================


def make_train_offline(
    config: Dict[str, Any],
    offline_data: Dict[str, np.ndarray],
) -> Callable[[jax.Array], Dict[str, Any]]:
    """Return ``train(rng)`` for offline MDLM training on pre-collected trajectories.

    Args:
        config:       Configuration dict (all-uppercase keys).
        offline_data: Dict with ``obs`` [E, T, obs_dim], ``actions`` [E, T],
                      and ``dones`` [E, T].

    Returns:
        train: ``Callable[[PRNGKey], dict]`` — JIT-able training function.
    """
    raw_obs = offline_data["obs"]      # [E, T, obs_dim]
    raw_act = offline_data["actions"]  # [E, T]
    raw_done = offline_data["dones"]   # [E, T]
    num_envs_data, traj_len, obs_dim = raw_obs.shape

    plan_horizon: int = config["PLAN_HORIZON"]
    batch_size: int = config["BATCH_SIZE"]
    num_actions: int = config["NUM_ACTIONS"]
    num_train_steps: int = config["NUM_TRAIN_STEPS"]
    schedule_fn: ScheduleFn = SCHEDULE_MAP[config["DIFFUSION_SCHEDULE"]]

    valid = _valid_window_mask(raw_done, plan_horizon)
    env_indices, time_indices = np.where(valid)
    num_valid = len(env_indices)
    assert num_valid > 0, (
        f"No valid {plan_horizon}-step windows in offline data "
        f"(traj_len={traj_len}). Check episode boundary density."
    )
    print(
        f"Offline data: {num_valid:,} valid {plan_horizon}-step windows "
        f"from {num_envs_data}x{traj_len} transitions"
    )

    obs_data = jnp.array(raw_obs, dtype=jnp.float32)
    act_data = jnp.array(raw_act, dtype=jnp.int32)
    env_idx_arr = jnp.array(env_indices, dtype=jnp.int32)
    time_idx_arr = jnp.array(time_indices, dtype=jnp.int32)

    model = _build_model(config, num_actions)
    _, apply_train = _make_apply_fns(model)

    def train(rng: jax.Array) -> Dict[str, Any]:
        rng, init_rng = jax.random.split(rng)
        params = _init_model_params(model, init_rng, obs_dim, plan_horizon)
        train_state = _create_train_state(
            model, params, config["LR"], config["MAX_GRAD_NORM"]
        )

        def _train_step(
            carry: Tuple[TrainState, jax.Array],
            step_idx: jnp.ndarray,
        ) -> Tuple[Tuple[TrainState, jax.Array], Dict[str, jnp.ndarray]]:
            train_state, rng = carry
            rng, sample_rng, loss_rng = jax.random.split(rng, 3)

            flat_idxs = jax.random.randint(sample_rng, (batch_size,), 0, num_valid)
            sel_env = env_idx_arr[flat_idxs]
            sel_time = time_idx_arr[flat_idxs]

            obs_batch = obs_data[sel_env, sel_time]
            act_batch = jax.vmap(
                lambda e, t: jax.lax.dynamic_slice(act_data[e], (t,), (plan_horizon,))
            )(sel_env, sel_time)

            def loss_fn(params: Any) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
                return compute_loss(
                    apply_train, params, loss_rng,
                    act_batch, obs_batch, num_actions, schedule_fn,
                )

            (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                train_state.params
            )
            train_state = train_state.apply_gradients(grads=grads)

            if config["DEBUG"] and config["USE_WANDB"]:
                def _log(info: Dict[str, Any], step: jnp.ndarray) -> None:
                    wandb.log(
                        {
                            "diffusion_loss": float(info["loss"]),
                            "mean_t": float(info["mean_t"]),
                            "frac_masked": float(info["frac_masked"]),
                        },
                        step=int(step),
                    )

                jax.debug.callback(_log, info, step_idx)

            return (train_state, rng), info

        (train_state, _), metrics = jax.lax.scan(
            _train_step, (train_state, rng), jnp.arange(num_train_steps)
        )
        return {"train_state": train_state, "metrics": metrics}

    return train


# =============================================================================
# Offline Training — directly from a PPO agent  (step 2b)
# =============================================================================


def make_train_offline_from_agent(
    config: Dict[str, Any],
    ppo_checkpoint_path: str,
) -> Callable[[jax.Array], Dict[str, Any]]:
    """Train the diffusion model using a pre-trained PPO agent for live data collection.

    Instead of loading pre-collected trajectories from disk, this function
    rolls out the fixed PPO policy at every training step and trains the
    diffusion model on the freshly collected windows.

    The training loop is a Python for-loop (not ``jax.lax.scan``) because the
    environment interaction must happen between gradient steps.

    Args:
        config:               Configuration dict (all-uppercase keys).
        ppo_checkpoint_path:  Path to a pre-trained ActorCritic checkpoint.

    Returns:
        train: ``Callable[[PRNGKey], dict]``
    """
    env = make_craftax_env_from_name(config["ENV_NAME"], True)
    env_params = env.default_params
    num_actions: Sequence[int] | int = env.action_space(env_params).n
    obs_dim: int = env.observation_space(env_params).shape[0]
    config["NUM_ACTIONS"] = num_actions

    plan_horizon: int = config["PLAN_HORIZON"]
    batch_size: int = config["BATCH_SIZE"]
    num_train_steps: int = config["NUM_TRAIN_STEPS"]
    num_envs: int = config.get("COLLECT_NUM_ENVS", config["NUM_ENVS"])
    layer_size: int = config.get("LAYER_SIZE", 512)
    schedule_fn: ScheduleFn = SCHEDULE_MAP[config["DIFFUSION_SCHEDULE"]]

    collect_steps: int = max(
        plan_horizon * 2,
        (batch_size // num_envs + 2) * plan_horizon,
    )

    ppo_agent = _load_ppo_checkpoint(
        ppo_checkpoint_path, num_actions, obs_dim, layer_size,
        model_type=config.get("PPO_MODEL_TYPE"),
    )

    env_w, _ = _make_env_stack(config, num_envs)

    model = _build_model(config, num_actions)
    _, apply_train = _make_apply_fns(model)

    if ppo_agent.model_type == "ppo_rnn":
        @jax.jit
        def _collect_chunk(
            rng: jax.Array,
            env_state: Any,
            obs: jnp.ndarray,
            done: jnp.ndarray,
            hidden: jax.Array,
        ) -> Tuple[
            jax.Array, Any, jnp.ndarray, jnp.ndarray, jax.Array,
            jnp.ndarray, jnp.ndarray, jnp.ndarray,
        ]:
            """Collect transitions via the RNN PPO policy, tracking hidden state."""
            def _step(
                carry: Tuple[jax.Array, Any, jnp.ndarray, jnp.ndarray, jax.Array],
                _: Any,
            ) -> Tuple[
                Tuple[jax.Array, Any, jnp.ndarray, jnp.ndarray, jax.Array],
                Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
            ]:
                rng, env_state, obs, done, hidden = carry
                rng, k1, k2 = jax.random.split(rng, 3)
                pi, _, new_hidden = ppo_agent.apply(obs, hidden=hidden, done=done)
                action = pi.sample(seed=k1)
                obs_next, env_state, _, done_next, _ = env_w.step(
                    k2, env_state, action, env_params
                )
                return (rng, env_state, obs_next, done_next, new_hidden), (obs, action, done)

            init_hidden = ppo_agent.init_hidden(num_envs)
            (rng, env_state, final_obs, final_done, final_hidden), (all_obs, all_acts, all_dones) = (
                jax.lax.scan(_step, (rng, env_state, obs, done, init_hidden), None, collect_steps)
            )
            return rng, env_state, final_obs, final_done, final_hidden, all_obs, all_acts, all_dones
    else:
        @jax.jit
        def _collect_chunk(
            rng: jax.Array,
            env_state: Any,
            obs: jnp.ndarray,
            done: jnp.ndarray,
            hidden: Any,
        ) -> Tuple[
            jax.Array, Any, jnp.ndarray, jnp.ndarray, Any,
            jnp.ndarray, jnp.ndarray, jnp.ndarray,
        ]:
            """Collect transitions via the MLP PPO policy (no hidden state)."""
            def _step(
                carry: Tuple[jax.Array, Any, jnp.ndarray, jnp.ndarray],
                _: Any,
            ) -> Tuple[
                Tuple[jax.Array, Any, jnp.ndarray, jnp.ndarray],
                Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
            ]:
                rng, env_state, obs, done = carry
                rng, k1, k2 = jax.random.split(rng, 3)
                pi, _, _ = ppo_agent.apply(ppo_agent.params, obs)
                action = pi.sample(seed=k1)
                obs_next, env_state, _, done_next, _ = env_w.step(
                    k2, env_state, action, env_params
                )
                return (rng, env_state, obs_next, done_next), (obs, action, done)

            (rng, env_state, final_obs, final_done), (all_obs, all_acts, all_dones) = (
                jax.lax.scan(_step, (rng, env_state, obs, done), None, collect_steps)
            )
            return rng, env_state, final_obs, final_done, None, all_obs, all_acts, all_dones

    @jax.jit
    def _grad_step(
        train_state: TrainState,
        obs_batch: jnp.ndarray,
        act_batch: jnp.ndarray,
        rng: jax.Array,
    ) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
        def loss_fn(params: Any) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
            return compute_loss(
                apply_train, params, rng, act_batch, obs_batch, num_actions, schedule_fn
            )

        (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
        return train_state.apply_gradients(grads=grads), info

    def train(rng: jax.Array) -> Dict[str, Any]:
        rng, init_rng, env_rng = jax.random.split(rng, 3)
        params = _init_model_params(model, init_rng, obs_dim, plan_horizon)
        train_state = _create_train_state(
            model, params, config["LR"], config["MAX_GRAD_NORM"]
        )

        obs, env_state = env_w.reset(env_rng, env_params)
        done = jnp.zeros(num_envs, dtype=bool)
        hidden = ppo_agent.init_hidden(num_envs)

        seed_val = int(jax.random.randint(rng, (), 0, 2**31))
        np_rng = np.random.default_rng(seed_val)

        all_metrics: List[Dict[str, Any]] = []
        t0 = time.time()

        for step in range(num_train_steps):
            rng, collect_rng, loss_rng = jax.random.split(rng, 3)

            rng, env_state, obs, done, hidden, chunk_obs, chunk_acts, chunk_dones = (
                _collect_chunk(collect_rng, env_state, obs, done, hidden)
            )

            # Transpose to [num_envs, collect_steps, ...]
            chunk_obs_np = np.array(chunk_obs).transpose(1, 0, 2)
            chunk_acts_np = np.array(chunk_acts).T
            chunk_dones_np = np.array(chunk_dones).T

            result = _sample_windows_from_chunk(
                chunk_obs_np, chunk_acts_np, chunk_dones_np,
                plan_horizon, batch_size, np_rng,
            )
            if result is None:
                continue

            obs_batch, act_batch = result
            train_state, info = _grad_step(train_state, obs_batch, act_batch, loss_rng)
            all_metrics.append({k: float(v) for k, v in info.items()})

            if config["DEBUG"] and config["USE_WANDB"]:
                wandb.log(
                    {
                        "diffusion_loss": float(info["loss"]),
                        "mean_t": float(info["mean_t"]),
                        "frac_masked": float(info["frac_masked"]),
                    },
                    step=step,
                )

            if (step + 1) % 1000 == 0:
                elapsed = time.time() - t0
                print(
                    f"  [{step + 1:>6}/{num_train_steps}] "
                    f"loss={float(info['loss']):.4f}  elapsed={elapsed:.0f}s"
                )

        return {"train_state": train_state, "metrics": all_metrics}

    return train


# =============================================================================
# Online Training  (step 4)
# =============================================================================


def make_train_online(
    config: Dict[str, Any],
    init_params: Optional[Any] = None,
) -> Callable[[jax.Array], Dict[str, Any]]:
    """Return ``train(rng)`` for online fine-tuning with the diffusion model as policy.

    At each update step:
      1. Generate plans via ``sample_plan`` and execute ``REPLAN_EVERY`` actions.
      2. Collect ``(obs, plan)`` pairs as training data (self-imitation).
      3. Fine-tune the model on those pairs for ``UPDATE_EPOCHS`` passes.

    Args:
        config:      Configuration dict (all-uppercase keys).
        init_params: Optional pre-loaded model parameters (e.g. from offline checkpoint).

    Returns:
        train: ``Callable[[PRNGKey], dict]`` — JIT-able training function.
    """
    assert config["NUM_STEPS"] % config["REPLAN_EVERY"] == 0, (
        "NUM_STEPS must be divisible by REPLAN_EVERY"
    )

    num_envs: int = config["NUM_ENVS"]
    plan_horizon: int = config["PLAN_HORIZON"]
    replan_every: int = config["REPLAN_EVERY"]
    num_updates: int = config["NUM_UPDATES"]
    update_epochs: int = config["UPDATE_EPOCHS"]
    num_minibatches: int = config["NUM_MINIBATCHES"]
    diffusion_steps: int = config["DIFFUSION_STEPS"]
    schedule_fn: ScheduleFn = SCHEDULE_MAP[config["DIFFUSION_SCHEDULE"]]
    remask_strategy: str = config["REMASK_STRATEGY"]
    eta: float = config["ETA"]
    t_on: float = config.get("T_ON", 0.7)
    t_off: float = config.get("T_OFF", 0.3)
    num_plan_cycles: int = config["NUM_STEPS"] // replan_every
    use_optimistic_resets: bool = config.get("USE_OPTIMISTIC_RESETS", False)

    num_actions: int = config["NUM_ACTIONS"]
    obs_dim: int = config["OBS_DIM"]

    env_w, env_params = _make_env_stack(
        config, num_envs,
        use_optimistic_resets=use_optimistic_resets,
        use_sequence_history=True,
    )
    model = _build_model(config, num_actions)
    apply_inference, apply_train = _make_apply_fns(model)

    def train(rng: jax.Array) -> Dict[str, Any]:
        rng, init_rng, env_rng = jax.random.split(rng, 3)
        params = _init_model_params(model, init_rng, obs_dim, plan_horizon)

        if init_params is not None:
            params = init_params

        train_state = _create_train_state(
            model, params, config["LR"], config["MAX_GRAD_NORM"]
        )
        obs, env_state = env_w.reset(env_rng, env_params)

        def _update_step(
            runner_state: Tuple[TrainState, Any, jnp.ndarray, jax.Array, int],
            _: Any,
        ) -> Tuple[Tuple[TrainState, Any, jnp.ndarray, jax.Array, int], Dict[str, jnp.ndarray]]:
            train_state, env_state, obs, rng, update_step = runner_state

            # -- Collect: generate plans and execute them --
            def _plan_and_execute(
                carry: Tuple[Any, jnp.ndarray, jax.Array],
                _: Any,
            ) -> Tuple[Tuple[Any, jnp.ndarray, jax.Array], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Any]]:
                env_state, obs, rng = carry
                rng, plan_rng = jax.random.split(rng)

                plan = sample_plan(
                    apply_inference, train_state.params, plan_rng, obs,
                    num_actions, plan_horizon, diffusion_steps, schedule_fn,
                    remask_strategy, eta, t_on, t_off,
                )

                def _exec_step(
                    carry: Tuple[Any, jnp.ndarray, jax.Array],
                    step_idx: jnp.ndarray,
                ) -> Tuple[Tuple[Any, jnp.ndarray, jax.Array], Tuple[jnp.ndarray, jnp.ndarray, Any]]:
                    env_state, _, rng = carry
                    action = plan[:, step_idx]
                    rng, step_rng = jax.random.split(rng)
                    obs_next, env_state, reward, done, info = env_w.step(
                        step_rng, env_state, action, env_params
                    )
                    return (env_state, obs_next, rng), (reward, done, info)

                (env_state, obs_next, rng), (rewards, dones, infos) = jax.lax.scan(
                    _exec_step, (env_state, obs, rng), jnp.arange(replan_every),
                )
                return (env_state, obs_next, rng), (obs, plan, rewards, dones, infos)

            (env_state, obs, rng), traj = jax.lax.scan(
                _plan_and_execute, (env_state, obs, rng), None, num_plan_cycles,
            )
            traj_obs, traj_plans, _, _, all_infos = traj

            # -- Train on collected (obs, plan) pairs --
            total_samples = num_plan_cycles * num_envs
            minibatch_size = total_samples // num_minibatches
            flat_obs = traj_obs.reshape(total_samples, obs_dim)
            flat_plans = traj_plans.reshape(total_samples, plan_horizon)

            def _update_epoch(
                carry: Tuple[TrainState, jax.Array],
                _: Any,
            ) -> Tuple[Tuple[TrainState, jax.Array], Dict[str, jnp.ndarray]]:
                train_state, rng = carry
                rng, perm_rng = jax.random.split(rng)
                perm = jax.random.permutation(perm_rng, total_samples)
                obs_mbs = flat_obs[perm].reshape(num_minibatches, minibatch_size, obs_dim)
                plan_mbs = flat_plans[perm].reshape(num_minibatches, minibatch_size, plan_horizon)

                def _update_minibatch(
                    ts_rng: Tuple[TrainState, jax.Array],
                    idx_and_mb: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
                ) -> Tuple[Tuple[TrainState, jax.Array], Dict[str, jnp.ndarray]]:
                    ts, rng = ts_rng
                    mb_idx, obs_mb, plan_mb = idx_and_mb
                    loss_rng = jax.random.fold_in(rng, mb_idx)

                    def loss_fn(params: Any) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
                        return compute_loss(
                            apply_train, params, loss_rng,
                            plan_mb, obs_mb, num_actions, schedule_fn,
                        )

                    (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(ts.params)
                    ts = ts.apply_gradients(grads=grads)
                    return (ts, rng), info

                (train_state, rng), infos = jax.lax.scan(
                    _update_minibatch,
                    (train_state, rng),
                    (jnp.arange(num_minibatches), obs_mbs, plan_mbs),
                )
                return (train_state, rng), infos

            (train_state, rng), epoch_infos = jax.lax.scan(
                _update_epoch, (train_state, rng), None, update_epochs
            )

            # -- Metrics --
            ep_returns = all_infos["returned_episode_returns"]
            ep_mask = all_infos["returned_episode"]
            mean_ep_return = jnp.where(
                ep_mask.sum() > 0,
                (ep_returns * ep_mask).sum() / ep_mask.sum(),
                jnp.nan,
            )
            metric = jax.tree.map(
                lambda x: (x * ep_mask).sum() / jnp.maximum(ep_mask.sum(), 1),
                all_infos,
            )
            metric["diffusion_loss"] = epoch_infos["loss"].mean()
            metric["mean_t"] = epoch_infos["mean_t"].mean()
            metric["frac_masked"] = epoch_infos["frac_masked"].mean()
            metric["episode_return"] = mean_ep_return

            if config["DEBUG"] and config["USE_WANDB"]:
                def _log(
                    loss: jnp.ndarray,
                    mean_t: jnp.ndarray,
                    frac_masked: jnp.ndarray,
                    ep_return: jnp.ndarray,
                    update_step: jnp.ndarray,
                ) -> None:
                    wandb.log(
                        {
                            "diffusion_loss": float(loss),
                            "mean_t": float(mean_t),
                            "frac_masked": float(frac_masked),
                            "episode_return": float(ep_return),
                        },
                        step=int(update_step),
                    )

                jax.debug.callback(
                    _log,
                    metric["diffusion_loss"], metric["mean_t"],
                    metric["frac_masked"], metric["episode_return"],
                    update_step,
                )

            runner_state = (train_state, env_state, obs, rng, update_step + 1)
            return runner_state, metric

        runner_state = (train_state, env_state, obs, rng, 0)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, num_updates
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


# =============================================================================
# Entry Points
# =============================================================================


def run_collect(config: Dict[str, Any]) -> None:
    """Collect offline data from a PPO agent and save to disk."""
    collect_offline_data(config)


def run_offline(config: Dict[str, Any]) -> None:
    """Offline training dispatcher.

    Two sub-modes, selected automatically:
      - ``--ppo_checkpoint_path`` provided -> collect data live from the agent.
      - ``--offline_data_path`` provided   -> load pre-saved .npz trajectories.
    """
    if config.get("USE_WANDB"):
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=f"remdm-offline-{config['ENV_NAME']}",
        )

    if config.get("PPO_CHECKPOINT_PATH"):
        print(
            "Offline training mode: live collection from PPO agent "
            f"({config['PPO_CHECKPOINT_PATH']})"
        )
        train_fn = make_train_offline_from_agent(config, config["PPO_CHECKPOINT_PATH"])

        rng = jax.random.PRNGKey(config["SEED"])
        t0 = time.time()
        outs = [train_fn(jax.random.fold_in(rng, i)) for i in range(config["NUM_REPEATS"])]
        t1 = time.time()
        print(f"Offline training time: {t1 - t0:.1f}s")

        if config["SAVE_POLICY"]:
            _save_model(outs[0]["train_state"], config, "diffusion_offline")

    else:
        assert config.get("OFFLINE_DATA_PATH"), (
            "Either --ppo_checkpoint_path or --offline_data_path must be provided "
            "for --mode offline."
        )
        print(
            f"Offline training mode: loading trajectories from '{config['OFFLINE_DATA_PATH']}'"
        )

        env = make_craftax_env_from_name(config["ENV_NAME"], True)
        config["NUM_ACTIONS"] = env.action_space(env.default_params).n

        offline_data = dict(np.load(config["OFFLINE_DATA_PATH"]))
        if offline_data["obs"].ndim == 2:
            print(
                "WARNING: flat data format detected. Reshaping to [1, N, obs_dim]. "
                "Episode-boundary masking will be limited."
            )
            offline_data["obs"] = offline_data["obs"][np.newaxis]
            offline_data["actions"] = offline_data["actions"][np.newaxis]
            offline_data["dones"] = (
                offline_data.get("dones", np.zeros_like(offline_data["actions"], dtype=bool))[np.newaxis]
            )

        n_envs, n_steps, obs_dim = offline_data["obs"].shape
        print(
            f"Loaded {n_envs}x{n_steps} transitions "
            f"(obs_dim={obs_dim}, num_actions={config['NUM_ACTIONS']})"
        )

        rng = jax.random.PRNGKey(config["SEED"])
        train_fn = make_train_offline(config, offline_data)
        train_jit = jax.jit(train_fn)

        t0 = time.time()
        outs = [train_jit(jax.random.fold_in(rng, i)) for i in range(config["NUM_REPEATS"])]
        t1 = time.time()
        print(f"Offline training time: {t1 - t0:.1f}s")

        if config["SAVE_POLICY"]:
            _save_model(outs[0]["train_state"], config, "diffusion_offline")


def run_online(config: Dict[str, Any]) -> None:
    """Online fine-tuning: diffusion model collects its own data."""
    env = make_craftax_env_from_name(config["ENV_NAME"], True)
    config["NUM_ACTIONS"] = env.action_space(env.default_params).n
    config["OBS_DIM"] = env.observation_space(env.default_params).shape[0]

    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=f"remdm-online-{config['ENV_NAME']}",
        )

    init_params: Optional[Any] = None
    if config.get("OFFLINE_CHECKPOINT_PATH"):
        model = _build_model(config, config["NUM_ACTIONS"])
        init_params = _load_checkpoint(
            config, model, config["OBS_DIM"], config["OFFLINE_CHECKPOINT_PATH"]
        )

    rng = jax.random.PRNGKey(config["SEED"])
    train_fn = make_train_online(config, init_params=init_params)
    train_jit = jax.jit(train_fn)

    t0 = time.time()
    outs = [
        train_jit(jax.random.fold_in(rng, i))
        for i in range(config["NUM_REPEATS"])
    ]
    t1 = time.time()
    print(f"Online training time: {t1 - t0:.1f}s")
    total_steps = config["NUM_UPDATES"] * config["NUM_STEPS"] * config["NUM_ENVS"]
    print(f"SPS: {total_steps / (t1 - t0):.0f}")

    if config["SAVE_POLICY"]:
        _save_model(outs[0]["runner_state"][0], config, "diffusion_online")


# =============================================================================
# Inference  (step 5)
# =============================================================================


def run_inference(config: Dict[str, Any]) -> None:
    """Evaluate a trained diffusion planner using ``PlannerWrapper``."""
    env = make_craftax_env_from_name(config["ENV_NAME"], True)
    env_params = env.default_params
    num_actions: int = env.action_space(env_params).n
    obs_dim: int = env.observation_space(env_params).shape[0]
    config["NUM_ACTIONS"] = num_actions

    num_envs: int = config["NUM_ENVS"]
    plan_horizon: int = config["PLAN_HORIZON"]
    replan_every: int = config["REPLAN_EVERY"]
    diffusion_steps: int = config["DIFFUSION_STEPS"]
    schedule_fn: ScheduleFn = SCHEDULE_MAP[config["DIFFUSION_SCHEDULE"]]
    remask_strategy: str = config["REMASK_STRATEGY"]
    eta: float = config["ETA"]
    t_on: float = config.get("T_ON", 0.7)
    t_off: float = config.get("T_OFF", 0.3)
    eval_steps: int = config.get("EVAL_STEPS", 1000)

    model = _build_model(config, num_actions)
    apply_inference, _ = _make_apply_fns(model)

    assert config.get("CHECKPOINT_PATH"), "--checkpoint_path required for inference"
    model_params = _load_checkpoint(config, model, obs_dim, config["CHECKPOINT_PATH"])

    def planner_apply_fn(
        rng: jax.Array,
        model_params: Any,
        obs: jnp.ndarray,
    ) -> jnp.ndarray:
        return sample_plan(
            apply_inference, model_params, rng, obs,
            num_actions, plan_horizon, diffusion_steps, schedule_fn,
            remask_strategy, eta, t_on, t_off,
        )

    env_w = LogWrapper(env)
    env_w = AutoResetEnvWrapper(env_w)
    env_w = BatchEnvWrapper(env_w, num_envs=num_envs)
    env_w = PlannerWrapper(
        env_w,
        num_envs=num_envs,
        plan_horizon=plan_horizon,
        replan_every=replan_every,
        planner_apply_fn=planner_apply_fn,
    )

    rng = jax.random.PRNGKey(config["SEED"])

    @jax.jit
    def _eval_loop(
        rng: jax.Array,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
        rng, env_rng = jax.random.split(rng)
        obs, state = env_w.reset(env_rng, env_params)

        def _step(
            carry: Tuple[jnp.ndarray, Any, jax.Array],
            _: Any,
        ) -> Tuple[Tuple[jnp.ndarray, Any, jax.Array], Tuple[jnp.ndarray, jnp.ndarray, Any]]:
            obs, state, rng = carry
            rng, step_rng = jax.random.split(rng)
            obs, state, action, reward, done, info = env_w.step(
                step_rng, state, obs, model_params, env_params,
            )
            return (obs, state, rng), (reward, done, info)

        (_, _, _), (rewards, dones, infos) = jax.lax.scan(
            _step, (obs, state, rng), None, eval_steps,
        )
        return rewards, dones, infos

    t0 = time.time()
    rewards, dones, infos = _eval_loop(rng)
    t1 = time.time()

    ep_returns = infos["returned_episode_returns"]
    ep_mask = infos["returned_episode"]
    completed = ep_mask.sum()
    mean_return = jnp.where(
        completed > 0,
        (ep_returns * ep_mask).sum() / completed,
        jnp.nan,
    )
    print(f"Eval time: {t1 - t0:.1f}s  ({eval_steps * num_envs} steps)")
    print(f"Completed episodes: {int(completed)}")
    print(f"Mean episode return: {float(mean_return):.2f}")
    print(f"Mean step reward: {float(rewards.mean()):.4f}")




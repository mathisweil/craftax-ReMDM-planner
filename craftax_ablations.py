"""
Craftax RL Fine-Tuning Ablations
=================================
Mirrors the MiniHack ablations notebook, adapted for Craftax/JAX.

Four ablations:
  1. KL Penalty       — rules out catastrophic forgetting / updates too large
  2. Frozen Backbone  — rules out deep gradient flow destabilising representations
  3. BC on Wins       — isolates ELBO t-marginalisation as the specific cause
  4. Low-t Only       — tests high-t gradient dominance hypothesis

Baseline RL result (from earlier run): ~2/226 score, collapsed from pretrained.

Structure: Python for-loop outer training (flexibility),
           JAX-jitted inner grad step and rollout.
"""

# ---------------------------------------------------------------------------
# 0. Setup
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
from typing import Any
import pathlib
import json
import numpy as np
import matplotlib.pyplot as plt
import yaml

import jax
import jax.numpy as jnp
import wandb

from src.planners.env import make_env, Transition
from src.planners.model import build_model, create_train_state, make_apply_fns
from src.planners.ppo import PPOAgent, build_ppo_network, load_ppo_params
from src.diffusion.schedules import SCHEDULE_MAP
from src.diffusion.forward import forward_process
from src.diffusion.sampling import sample_plan
from src.planners.model import load_checkpoint
from src.planners.common import make_grad_step

REMASK_STRATEGIES = ["rescale", "cap", "conf"]
DIFFUSION_SCHEDULES = ["cosine", "linear"]
PPO_TYPES = ["ppo", "ppo_rnn", "ppo_rnd"]

# ---------------------------------------------------------------------------
# 2. Environment, Model, PPO
# ---------------------------------------------------------------------------

# Environment
env, env_params = make_env(config, config["num_envs"])

num_actions = env.action_space(env_params).n
obs_shape = env.observation_space(env_params).shape
obs_dim = obs_shape[0]

# PPO collector
model_type = config["PPO_MODEL_TYPE"]
ppo_net = build_ppo_network(model_type, num_actions, config["LAYER_SIZE"], config)
ppo_params = load_ppo_params(
    config["PPO_CHECKPOINT_PATH"], ppo_net, model_type, config["num_envs"], obs_shape, config["LAYER_SIZE"],
)
ppo = PPOAgent(ppo_net, ppo_params, model_type, config["LAYER_SIZE"])

# Noise schedule
schedule_fn, schedule_deriv_fn = SCHEDULE_MAP[config["DIFFUSION_SCHEDULE"]]

# Diffusion model — pure Flax dataclass, no randomness, safe to build once.
net = build_model(config, num_actions)
apply_eval, apply_train = make_apply_fns(net)
grad_step = make_grad_step(
    apply_train, num_actions, schedule_fn, schedule_deriv_fn,
    config.get("TRAIN_SIGMA", 0.0), config.get("LABEL_SMOOTHING", 0.0),
)

print("Environment, model, and PPO ready.")


# ---------------------------------------------------------------------------
# Load pretrained checkpoint
# ---------------------------------------------------------------------------

pretrained_params = load_checkpoint(
    net,
    jax.random.PRNGKey(0),
    obs_dim,
    config["PLAN_HORIZON"],
    config["OFFLINE_CHECKPOINT_PATH"],
)


# ---------------------------------------------------------------------------
# 3. Shared Infrastructure
# ---------------------------------------------------------------------------

valid_per_rollout = config["NUM_STEPS"] - config["PLAN_HORIZON"] + 1


@jax.jit
def collect_rollout(env_state, obs, done, hstate, rng):
    """Run one PPO rollout. Returns flattened (obs, acts, valid, returns) windows."""
    def _env_step(carry, _):
        es, ob, dn, hs, rng = carry
        rng, act_rng, step_rng = jax.random.split(rng, 3)
        action, new_hs = ppo.act(ob, dn, hs, act_rng, temperature=config["COLLECT_TEMP"])
        new_obs, es, reward, new_done, info = env.step(step_rng, es, action, env_params)
        t = Transition(done=dn, action=action, reward=reward, obs=ob, info=info)
        return (es, new_obs, new_done, new_hs, rng), t

    (env_state, obs, done, hstate, rng), traj = jax.lax.scan(
        _env_step, (env_state, obs, done, hstate, rng), None, config["NUM_STEPS"],
    )

    def _window(t_idx):
        obs_t  = traj.obs[t_idx]
        acts   = jax.lax.dynamic_slice(traj.action, (t_idx, 0),     (config["PLAN_HORIZON"],     config["NUM_ENVS"]))
        dones  = jax.lax.dynamic_slice(traj.done,   (t_idx + 1, 0), (config["PLAN_HORIZON"] - 1, config["NUM_ENVS"]))
        valid  = ~jnp.any(dones, axis=0)
        rews   = jax.lax.dynamic_slice(traj.reward, (t_idx, 0),     (config["PLAN_HORIZON"],     config["NUM_ENVS"]))
        window_return = jnp.sum(rews, axis=0)
        return obs_t, jnp.swapaxes(acts, 0, 1), valid, window_return

    obs_w, act_w, valid_w, ret_w = jax.vmap(_window)(jnp.arange(valid_per_rollout))

    flat_obs     = obs_w.reshape(-1, obs_dim)
    flat_acts    = act_w.reshape(-1, config["PLAN_HORIZON"])
    flat_valid   = valid_w.reshape(-1)
    flat_returns = ret_w.reshape(-1)

    info_returned = traj.info["returned_episode"]
    env_score = jax.tree.map(
        lambda x: (x * info_returned).sum() / (info_returned.sum() + 1e-8),
        traj.info,
    )

    return env_state, obs, done, hstate, rng, flat_obs, flat_acts, flat_valid, flat_returns, env_score


@jax.jit
def eval_policy(params, rng):
    """Quick eval: run sample_plan + env for EVAL_STEPS steps, return score."""
    n_cycles = config["EVAL_STEPS"] // config["EVAL_REPLAN"]
    rng, env_rng = jax.random.split(rng)
    val_obs, val_es = env.reset(env_rng, env_params)

    def _cycle(carry, _):
        es, vo, rng = carry
        rng, p_rng = jax.random.split(rng)
        plan = sample_plan(
            apply_eval, params, p_rng, vo,
            num_actions, config["PLAN_HORIZON"],
            num_steps=config["VAL_DIFFUSION_STEPS"],
            schedule_fn=schedule_fn,
            remask_strategy=config["REMASK_STRATEGY"],
            eta=config["ETA"],
            use_loop=config["USE_LOOP"],
            t_on=config["T_ON"],
            t_off=config["T_OFF"],
            temperature=config["TEMPERATURE"],
            top_p=config["TOP_P"],
        )

        def _step(c, step_i):
            es_i, vo_i, r = c
            r, s_rng = jax.random.split(r)
            vo_next, es_next, _, _, info = env.step(s_rng, es_i, plan[:, step_i], env_params)
            return (es_next, vo_next, r), info

        (es, vo, rng), infos = jax.lax.scan(_step, (es, vo, rng), jnp.arange(config["EVAL_REPLAN"]))
        return (es, vo, rng), infos

    _, cycle_infos = jax.lax.scan(_cycle, (val_es, val_obs, rng), None, n_cycles)
    infos = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), cycle_infos)
    ret   = infos["returned_episode"]
    score = jax.tree.map(lambda x: (x * ret).sum() / (ret.sum() + 1e-8), infos)
    return score


def make_empty_history():
    return {
        "iter": [], "loss": [],
        "env_score_iter": [], "env_score": [],
        "eval_iter": [], "eval_score": [],
        "grad_align_iter": [], "grad_align": [], "rl_grad_norm": [], "bc_grad_norm": [],
        "repr_drift_iter": [], "repr_drift": [],
        "t_analysis_iter": [], "norm_low_t": [], "norm_high_t": [], "lowhigh_cos": [],
    }


print(f"Rollout: {valid_per_rollout} windows/rollout | {config["NUM_ENVS"] * valid_per_rollout} samples/rollout")


# ---------------------------------------------------------------------------
# Loss functions for each ablation
# ---------------------------------------------------------------------------

_EPS        = 1e-5
_MAX_WEIGHT = 1000.0


def _base_loss(apply_fn, params, rng, acts, obs, valid, advantages,
               t_min=_EPS, t_max=1.0):
    """Core MDLM loss with configurable t range. Used by all ablations."""
    B = acts.shape[0]
    mask_id = num_actions
    rng, t_rng, mask_rng, drop_rng = jax.random.split(rng, 4)

    t             = jax.random.uniform(t_rng, (B,), minval=t_min, maxval=t_max)
    alpha_t       = schedule_fn(t)
    neg_alpha_dot = -schedule_deriv_fn(t)
    weight        = (1.0 - config["TRAIN_SIGMA"]) * neg_alpha_dot / jnp.maximum(1.0 - alpha_t, _EPS)
    weight        = jnp.minimum(weight, _MAX_WEIGHT)

    z_t          = forward_process(mask_rng, acts, alpha_t, mask_id)
    logits       = apply_fn(params, obs, z_t, t, drop_rng)        # [B, H, V]
    is_masked    = (z_t == mask_id).astype(jnp.float32)
    valid_masked = is_masked * valid[:, None].astype(jnp.float32)

    targets   = jax.nn.one_hot(acts, num_actions)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    ce        = -jnp.sum(targets * log_probs, axis=-1)             # [B, H]

    n_masked   = jnp.maximum(valid_masked.sum(axis=-1), 1.0)
    per_sample = weight * (ce * valid_masked).sum(axis=-1) / n_masked

    if advantages is not None:
        per_sample = per_sample * jax.lax.stop_gradient(advantages)

    return per_sample.mean()


def make_loss_baseline(apply_fn):
    """Standard return-weighted ELBO — no modifications."""
    def loss(params, acts, obs, valid, rng, advantages):
        return _base_loss(apply_fn, params, rng, acts, obs, valid, advantages)
    return loss


def make_loss_kl(apply_fn, ref_params):
    """Return-weighted ELBO + KL penalty against the frozen pretrained model."""
    def loss(params, acts, obs, valid, rng, advantages):
        rl = _base_loss(apply_fn, params, rng, acts, obs, valid, advantages)

        rng2, t_rng, mask_rng, drop_rng = jax.random.split(rng, 4)
        B = acts.shape[0]
        t        = jax.random.uniform(t_rng, (B,), minval=_EPS, maxval=1.0)
        alpha_t  = schedule_fn(t)
        z_t      = forward_process(mask_rng, acts, alpha_t, num_actions)
        is_masked = (z_t == num_actions).astype(jnp.float32)
        valid_m   = is_masked * valid[:, None].astype(jnp.float32)

        cur_log = jax.nn.log_softmax(apply_fn(params, obs, z_t, t, drop_rng), axis=-1)
        ref_log = jax.nn.log_softmax(
            apply_fn(jax.lax.stop_gradient(ref_params), obs, z_t, t, drop_rng), axis=-1
        )
        cur_prob  = jnp.exp(cur_log)
        kl        = (cur_prob * (cur_log - ref_log)).sum(-1)        # [B, H]
        kl_masked = (kl * valid_m).sum(-1) / jnp.maximum(valid_m.sum(-1), 1.0)
        return rl + config["KL_COEF"] * kl_masked.mean()
    return loss


def make_loss_bc_wins(apply_fn):
    """Uniform loss on all samples — ignores advantages."""
    def loss(params, acts, obs, valid, rng, advantages):
        return _base_loss(apply_fn, params, rng, acts, obs, valid, advantages=None)
    return loss


def make_loss_low_t(apply_fn, t_max=0.2):
    """Return-weighted ELBO restricted to t ∈ [ε, t_max]."""
    def loss(params, acts, obs, valid, rng, advantages):
        return _base_loss(apply_fn, params, rng, acts, obs, valid, advantages,
                          t_min=_EPS, t_max=t_max)
    return loss


print(f"Loss functions defined. KL_COEF={config["KL_COEF"]} | T_MAX_LOW={config["T_MAX_LOW"]}")


# ---------------------------------------------------------------------------
# Gradient step factory
# ---------------------------------------------------------------------------

def make_ablation_grad_step(loss_fn, frozen_backbone=False):
    """Returns a jitted grad step for a given loss function.

    frozen_backbone: if True, zero out gradients for all non-head params.
    """
    def _step(state, acts, obs, valid, rng, advantages):
        def _loss(params):
            return loss_fn(params, acts, obs, valid, rng, advantages)

        loss_val, grads = jax.value_and_grad(_loss)(state.params)

        if frozen_backbone:
            def _mask_grad(path, grad):
                path_str = "/".join(str(p.key) for p in path)
                if "Dense_5" in path_str:
                    return grad
                return jnp.zeros_like(grad)
            grads = jax.tree_util.tree_map_with_path(_mask_grad, grads)

        state = state.apply_gradients(grads=grads)
        return state, loss_val

    return jax.jit(_step)


def compute_return_weights(flat_returns, wins_only=False):
    """Normalise returns to per-sample advantage weights."""
    if wins_only:
        return (flat_returns > config["WIN_THRESHOLD"]).astype(jnp.float32)
    clipped = jnp.clip(flat_returns, 0.0, None)
    weights = clipped / (jnp.mean(clipped) + 1e-8)
    return jnp.clip(weights, config["RETURN_WEIGHT_FLOOR"], config["RETURN_WEIGHT_CAP"])


print("Grad step factory defined.")


# ---------------------------------------------------------------------------
# Diagnostic functions
# ---------------------------------------------------------------------------

GRAD_ALIGN_EVERY = 25
T_ANALYSIS_EVERY = 25


@jax.jit
def compute_grad_alignment(params, ref_params, acts, obs, valid, rng, advantages):
    """Cosine similarity between RL gradient and oracle BC gradient.

    cos_sim < 0  → RL gradient actively points AWAY from improvement.
    cos_sim ~ 0  → RL gradient is noise.
    cos_sim > 0  → RL gradient points in a useful direction.
    """
    rng_rl, rng_bc = jax.random.split(rng)

    def rl_loss(p):
        return _base_loss(apply_train, p, rng_rl, acts, obs, valid, advantages)

    def bc_loss(p):
        return _base_loss(apply_train, p, rng_bc, acts, obs, valid, advantages=None)

    rl_grads = jax.grad(rl_loss)(params)
    bc_grads = jax.grad(bc_loss)(ref_params)

    rl_flat = jnp.concatenate([g.ravel() for g in jax.tree.leaves(rl_grads)])
    bc_flat = jnp.concatenate([g.ravel() for g in jax.tree.leaves(bc_grads)])

    cos_sim = jnp.dot(rl_flat, bc_flat) / (
        jnp.linalg.norm(rl_flat) * jnp.linalg.norm(bc_flat) + 1e-10
    )
    return cos_sim, jnp.linalg.norm(rl_flat), jnp.linalg.norm(bc_flat)


@jax.jit
def compute_repr_drift(params, ref_params, obs, acts, rng):
    """KL divergence between current and pretrained model predictions.

    Measures how much the model has drifted from its pretrained state.
    High drift + performance collapse → representations corrupted.
    """
    B = obs.shape[0]
    rng, t_rng, mask_rng = jax.random.split(rng, 3)
    t       = jax.random.uniform(t_rng, (B,), minval=0.3, maxval=0.7)
    alpha_t = schedule_fn(t)
    z_t     = forward_process(mask_rng, acts, alpha_t, num_actions)

    cur_logits = apply_eval(params,     obs, z_t, t)
    ref_logits = apply_eval(ref_params, obs, z_t, t)

    cur_log  = jax.nn.log_softmax(cur_logits, axis=-1)
    ref_log  = jax.nn.log_softmax(ref_logits, axis=-1)
    ref_prob = jnp.exp(ref_log)

    kl = (ref_prob * (ref_log - cur_log)).sum(-1).mean()
    return kl


@jax.jit
def compute_t_gradient_analysis(params, acts, obs, valid, advantages, rng):
    """Compare gradient norm contribution from high-t vs low-t samples.

    If high-t dominates and those gradients point in the wrong direction,
    this supports the bias hypothesis.
    """
    rng_lo, rng_hi = jax.random.split(rng)

    def loss_low(p):
        return _base_loss(apply_train, p, rng_lo, acts, obs, valid, advantages,
                          t_min=_EPS, t_max=0.2)

    def loss_high(p):
        return _base_loss(apply_train, p, rng_hi, acts, obs, valid, advantages,
                          t_min=0.8, t_max=1.0)

    grads_low  = jax.grad(loss_low)(params)
    grads_high = jax.grad(loss_high)(params)

    low_flat  = jnp.concatenate([g.ravel() for g in jax.tree.leaves(grads_low)])
    high_flat = jnp.concatenate([g.ravel() for g in jax.tree.leaves(grads_high)])
    norm_low  = jnp.linalg.norm(low_flat)
    norm_high = jnp.linalg.norm(high_flat)
    cos_sim   = jnp.dot(low_flat, high_flat) / (norm_low * norm_high + 1e-10)

    return norm_low, norm_high, cos_sim


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_ablation(name, params_init, loss_fn, frozen_backbone=False, wins_only=False):
    print(f"\n{'='*60}")
    print(f"ABLATION: {name}")
    print(f"{'='*60}")

    state     = create_train_state(net, params_init, config["LR"], config["MAX_GRAD_NORM"])
    grad_step = make_ablation_grad_step(loss_fn, frozen_backbone=frozen_backbone)
    history   = make_empty_history()

    rng = jax.random.PRNGKey(42)
    rng, env_rng = jax.random.split(rng)
    obs, env_state = env.reset(env_rng, env_params)
    done   = jnp.zeros(config["NUM_ENVS"], dtype=bool)
    hstate = ppo.init_hidden(config["NUM_ENVS"])

    running_loss = running_score = n_log = 0.0

    for iteration in range(1, config["MAX_ITER"] + 1):
        rng, rollout_rng = jax.random.split(rng)
        (env_state, obs, done, hstate, rng,
         flat_obs, flat_acts, flat_valid, flat_returns, env_score) = \
            collect_rollout(env_state, obs, done, hstate, rollout_rng)

        advantages = compute_return_weights(flat_returns, wins_only=wins_only)

        n_samples = flat_obs.shape[0]
        rng, perm_rng, loss_rng, align_rng, drift_rng, t_rng = jax.random.split(rng, 6)
        perm       = jax.random.permutation(perm_rng, n_samples)
        flat_obs   = flat_obs[perm]
        flat_acts  = flat_acts[perm]
        flat_valid = flat_valid[perm]
        advantages = advantages[perm]

        obs_b = flat_obs[:config["BATCH_SIZE"]]
        act_b = flat_acts[:config["BATCH_SIZE"]]
        val_b = flat_valid[:config["BATCH_SIZE"]]
        adv_b = advantages[:config["BATCH_SIZE"]]

        state, loss_val = grad_step(state, act_b, obs_b, val_b, loss_rng, adv_b)

        running_loss  += float(loss_val)
        running_score += float(env_score.get("returned_episode_returns", jnp.array(0.0)))
        n_log += 1

        # Gradient alignment
        if iteration % GRAD_ALIGN_EVERY == 0:
            cos_sim, rl_norm, bc_norm = compute_grad_alignment(
                state.params, pretrained_params, act_b, obs_b, val_b, align_rng, adv_b
            )
            history["grad_align_iter"].append(iteration)
            history["grad_align"].append(float(cos_sim))
            history["rl_grad_norm"].append(float(rl_norm))
            history["bc_grad_norm"].append(float(bc_norm))
            print(f"  [{name}] iter {iteration} | grad_align={float(cos_sim):+.4f} | "
                  f"rl_norm={float(rl_norm):.4f} | bc_norm={float(bc_norm):.4f}")

        # Representation drift
        if iteration % GRAD_ALIGN_EVERY == 0:
            drift = compute_repr_drift(state.params, pretrained_params, obs_b, act_b, drift_rng)
            history["repr_drift_iter"].append(iteration)
            history["repr_drift"].append(float(drift))
            print(f"  [{name}] iter {iteration} | repr_drift (KL)={float(drift):.6f}")

        # t-distribution analysis
        if iteration % T_ANALYSIS_EVERY == 0:
            norm_lo, norm_hi, lo_hi_cos = compute_t_gradient_analysis(
                state.params, act_b, obs_b, val_b, adv_b, t_rng
            )
            history["t_analysis_iter"].append(iteration)
            history["norm_low_t"].append(float(norm_lo))
            history["norm_high_t"].append(float(norm_hi))
            history["lowhigh_cos"].append(float(lo_hi_cos))

        if iteration % 10 == 0:
            ml = running_loss  / max(n_log, 1)
            ms = running_score / max(n_log, 1)
            history["iter"].append(iteration)
            history["loss"].append(ml)
            history["env_score_iter"].append(iteration)
            history["env_score"].append(ms)
            wins_in_buf = int(jnp.sum(flat_returns > config["WIN_THRESHOLD"]))
            print(f"  [{name}] iter {iteration}/{config["MAX_ITER"]} | loss={ml:.4f} | "
                  f"score={ms:.3f} | wins={wins_in_buf}")
            running_loss = running_score = n_log = 0

        if iteration % config["EVAL_EVERY"] == 0:
            rng, eval_rng = jax.random.split(rng)
            eval_info  = eval_policy(state.params, eval_rng)
            eval_score = float(eval_info.get("returned_episode_returns", jnp.array(0.0)))
            history["eval_iter"].append(iteration)
            history["eval_score"].append(eval_score)
            print(f"  [{name}] Eval score: {eval_score:.4f}")

    rng, eval_rng = jax.random.split(rng)
    final_eval  = eval_policy(state.params, eval_rng)
    final_score = float(final_eval.get("returned_episode_returns", jnp.array(0.0)))
    print(f"  [{name}] FINAL score: {final_score:.4f}")
    return history, final_score, state.params


# ---------------------------------------------------------------------------
# 4. Baseline Evaluation (Pretrained, No Fine-Tuning)
# ---------------------------------------------------------------------------

print("Evaluating pretrained model (no fine-tuning)...")
rng = jax.random.PRNGKey(0)
rng, eval_rng    = jax.random.split(rng)
baseline_info    = eval_policy(pretrained_params, eval_rng)
baseline_score   = float(baseline_info.get("returned_episode_returns", jnp.array(0.0)))
print(f"Pretrained baseline score: {baseline_score:.4f}")


# ---------------------------------------------------------------------------
# 5. Ablation 0: Baseline RL (Return-Weighted ELBO)
# ---------------------------------------------------------------------------

loss_baseline_rl = make_loss_baseline(apply_train)
history_baseline_rl, score_baseline_rl, _ = run_ablation(
    name="Baseline-RL",
    params_init=jax.tree.map(jnp.array, pretrained_params),
    loss_fn=loss_baseline_rl,
    frozen_backbone=False,
    wins_only=False,
)
print(f"Baseline-RL final score: {score_baseline_rl:.4f} (pretrained: {baseline_score:.4f})")
BASELINE_RL_SCORE = score_baseline_rl


# ---------------------------------------------------------------------------
# 6. Ablation 1: KL Penalty
# ---------------------------------------------------------------------------

loss_kl = make_loss_kl(apply_train, pretrained_params)
history_kl, score_kl, _ = run_ablation(
    name="KL-Penalty",
    params_init=jax.tree.map(jnp.array, pretrained_params),
    loss_fn=loss_kl,
    frozen_backbone=False,
    wins_only=False,
)
print(f"KL-Penalty final score: {score_kl:.4f} (baseline: {baseline_score:.4f})")


# ---------------------------------------------------------------------------
# 7. Ablation 2: Frozen Backbone
# ---------------------------------------------------------------------------

loss_baseline = make_loss_baseline(apply_train)
history_frozen, score_frozen, _ = run_ablation(
    name="Frozen-Backbone",
    params_init=jax.tree.map(jnp.array, pretrained_params),
    loss_fn=loss_baseline,
    frozen_backbone=True,
    wins_only=False,
)
print(f"Frozen-Backbone final score: {score_frozen:.4f} (baseline: {baseline_score:.4f})")


# ---------------------------------------------------------------------------
# 8. Ablation 3: BC on Wins
# ---------------------------------------------------------------------------

loss_bc = make_loss_bc_wins(apply_train)
history_bc, score_bc, _ = run_ablation(
    name="BC-on-Wins",
    params_init=jax.tree.map(jnp.array, pretrained_params),
    loss_fn=loss_bc,
    frozen_backbone=False,
    wins_only=True,
)
print(f"BC-on-Wins final score: {score_bc:.4f} (baseline: {baseline_score:.4f})")


# ---------------------------------------------------------------------------
# 9. Ablation 4: Low-t Only
# ---------------------------------------------------------------------------

loss_lowt = make_loss_low_t(apply_train, t_max=config["T_MAX_LOW"])
history_lowt, score_lowt, _ = run_ablation(
    name="Low-t-Only",
    params_init=jax.tree.map(jnp.array, pretrained_params),
    loss_fn=loss_lowt,
    frozen_backbone=False,
    wins_only=False,
)
print(f"Low-t-Only final score: {score_lowt:.4f} (baseline: {baseline_score:.4f})")


# ---------------------------------------------------------------------------
# 10. Synthetic Sanity Check
# ---------------------------------------------------------------------------

print("Running synthetic sanity check...")
print("Collecting oracle PPO data for synthetic task...")

rng_synth = jax.random.PRNGKey(999)
rng_synth, env_rng = jax.random.split(rng_synth)
obs_s, es_s = env.reset(env_rng, env_params)
done_s   = jnp.zeros(config["NUM_ENVS"], dtype=bool)
hstate_s = ppo.init_hidden(config["NUM_ENVS"])

rng_synth, rollout_rng = jax.random.split(rng_synth)
_, _, _, _, _, oracle_obs, oracle_acts, oracle_valid, _, _ = collect_rollout(
    es_s, obs_s, done_s, hstate_s, rollout_rng
)
print(f"Oracle data: {oracle_obs.shape[0]} windows")

# Perturb 10% of actions
rng_synth, perturb_rng = jax.random.split(rng_synth)
perturb_mask   = jax.random.bernoulli(perturb_rng, 0.1, shape=oracle_acts.shape)
random_actions = jax.random.randint(perturb_rng, oracle_acts.shape, 0, num_actions)
perturbed_acts = jnp.where(perturb_mask, random_actions, oracle_acts)

synth_state   = create_train_state(net, jax.tree.map(jnp.array, pretrained_params), config["LR"], config["MAX_GRAD_NORM"])
synth_history = {"iter": [], "loss": [], "recovery_rate": []}


@jax.jit
def synth_step(state, rng):
    """Train on oracle actions (ground truth recovery task)."""
    def _loss(params):
        return _base_loss(apply_train, params, rng,
                          oracle_acts[:config["BATCH_SIZE"]],
                          oracle_obs[:config["BATCH_SIZE"]],
                          oracle_valid[:config["BATCH_SIZE"]],
                          advantages=None)
    loss_val, grads = jax.value_and_grad(_loss)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss_val


print("Training synthetic recovery task...")
for iteration in range(1, config["MAX_ITER"] + 1):
    rng_synth, step_rng, eval_rng = jax.random.split(rng_synth, 3)
    synth_state, loss_val = synth_step(synth_state, step_rng)

    if iteration % 10 == 0:
        synth_history["iter"].append(iteration)
        synth_history["loss"].append(float(loss_val))

    if iteration % config["EVAL_EVERY"] == 0:
        eval_info  = eval_policy(synth_state.params, eval_rng)
        eval_score = float(eval_info.get("returned_episode_returns", jnp.array(0.0)))
        synth_history["recovery_rate"].append(eval_score)
        print(f"  [Synthetic] iter {iteration} | loss={float(loss_val):.4f} | eval_score={eval_score:.4f}")

score_synthetic = synth_history["recovery_rate"][-1] if synth_history["recovery_rate"] else 0.0
print(f"\nSynthetic final score: {score_synthetic:.4f}")
print("Interpretation: ", end="")
if score_synthetic > baseline_score - 0.005:
    print("WORKS — infrastructure is fine, problem is specific to RL signal")
elif score_synthetic < baseline_score * 0.5:
    print("FAILS — either infrastructure broken or H1 is very strong")
else:
    print("NEUTRAL — partial recovery")


# ---------------------------------------------------------------------------
# 11. Summary and Full Analysis
# ---------------------------------------------------------------------------

results = [
    ("Pretrained (no FT)", baseline_score),
    ("Baseline RL",        score_baseline_rl),
    ("KL Penalty",         score_kl),
    ("Frozen Backbone",    score_frozen),
    ("BC on Wins",         score_bc),
    ("Low-t Only",         score_lowt),
    ("Synthetic (BC)",     score_synthetic),
]

print("\n" + "=" * 60)
print(f'{"Method":<25} | {"Score":>10} | {"Delta":>10} | {"Verdict":>12}')
print("=" * 60)
for name, score in results:
    delta = score - baseline_score
    if name == "Pretrained (no FT)":
        verdict = "BASELINE"
    elif delta < -0.005:
        verdict = "COLLAPSE"
    elif delta > 0.005:
        verdict = "IMPROVEMENT"
    else:
        verdict = "NEUTRAL"
    flag = "  ← BASELINE" if name == "Pretrained (no FT)" else ""
    print(f"  {name:<23} | {score:>10.4f} | {delta:>+9.4f} | {verdict:>12}{flag}")
print("=" * 60)

all_histories = {
    "Baseline-RL":     history_baseline_rl,
    "KL-Penalty":      history_kl,
    "Frozen-Backbone": history_frozen,
    "BC-on-Wins":      history_bc,
    "Low-t-Only":      history_lowt,
}

print("\n--- Gradient Alignment Analysis ---")
print("(cos_sim < 0 = gradient actively wrong, ~0 = noise, > 0 = useful signal)\n")
for name, hist in all_histories.items():
    if hist["grad_align"]:
        mean_align  = np.mean(hist["grad_align"])
        final_align = hist["grad_align"][-1]
        trend = "↓" if len(hist["grad_align"]) > 1 and hist["grad_align"][-1] < hist["grad_align"][0] else "→"
        print(f"  {name:<23}: mean={mean_align:+.4f}  final={final_align:+.4f}  {trend}")

print("\n--- Representation Drift (KL from pretrained) ---")
print("(higher = more drift from pretrained representations)\n")
for name, hist in all_histories.items():
    if hist["repr_drift"]:
        mean_drift  = np.mean(hist["repr_drift"])
        final_drift = hist["repr_drift"][-1]
        print(f"  {name:<23}: mean={mean_drift:.6f}  final={final_drift:.6f}")

print("\n--- t-Distribution: High-t vs Low-t Gradient Norms ---")
print("(if norm_high >> norm_low: high-t dominates = supports bias hypothesis)\n")
for name, hist in all_histories.items():
    if hist["norm_high_t"]:
        ratio = np.mean(hist["norm_high_t"]) / (np.mean(hist["norm_low_t"]) + 1e-10)
        cos   = np.mean(hist["lowhigh_cos"])
        print(f"  {name:<23}: high/low ratio={ratio:.2f}  low-high alignment={cos:+.4f}")

print("\n--- Paper Verdict ---")
all_collapse = all(s < baseline_score - 0.005 for _, s in results[1:-1])
synth_works  = score_synthetic > baseline_score - 0.005
mean_align_baseline = (
    np.mean(history_baseline_rl["grad_align"])
    if history_baseline_rl["grad_align"] else 0.0
)

if all_collapse and not synth_works:
    print("  ALL ablations collapse AND synthetic fails.")
    print("  STRONGEST case for H1: fundamental incompatibility with RL fine-tuning.")
elif all_collapse and synth_works:
    print("  ALL ablations collapse BUT synthetic works.")
    print("  Infrastructure is fine. Problem is RL signal from environment interaction.")
    print("  H1 supported: the model cannot improve from environment-generated data.")
else:
    print("  Mixed results — some ablations work. Check individual verdicts above.")

if mean_align_baseline < -0.01:
    print(f"  Gradient alignment = {mean_align_baseline:+.4f}: RL gradient actively points WRONG direction.")
    print("  This PROVES the gradient is not a valid policy gradient surrogate.")
elif abs(mean_align_baseline) < 0.05:
    print(f"  Gradient alignment = {mean_align_baseline:+.4f}: RL gradient is noise.")
    print("  Signal too weak to improve, but not actively harmful.")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

colors = {
    "Baseline-RL":     "#9E9E9E",
    "KL-Penalty":      "#2196F3",
    "Frozen-Backbone": "#FF7043",
    "BC-on-Wins":      "#4CAF50",
    "Low-t-Only":      "#9C27B0",
}

# 1. Eval score over training
ax = axes[0, 0]
ax.axhline(baseline_score, color="black", linestyle="--", linewidth=2,
           label=f"Pretrained ({baseline_score:.4f})", alpha=0.8)
for name, hist in all_histories.items():
    if hist["eval_score"]:
        ax.plot(hist["eval_iter"], hist["eval_score"],
                color=colors[name], marker="o", linewidth=2, markersize=4, label=name)
ax.set_title("Eval Score Over Training")
ax.set_xlabel("Iteration"); ax.set_ylabel("Score")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# 2. Final score bar chart
ax2 = axes[0, 1]
all_names_bar  = ["Pretrained", "Baseline\nRL", "KL\nPenalty",
                   "Frozen\nBackbone", "BC on\nWins", "Low-t\nOnly", "Synthetic"]
all_scores_bar = [baseline_score, score_baseline_rl, score_kl,
                   score_frozen, score_bc, score_lowt, score_synthetic]
bar_colors_list = ["#607D8B", "#9E9E9E", "#2196F3", "#FF7043", "#4CAF50", "#9C27B0", "#FF9800"]
bars = ax2.bar(range(len(all_names_bar)), all_scores_bar, color=bar_colors_list, alpha=0.85)
ax2.axhline(baseline_score, color="black", linestyle="--", alpha=0.5)
ax2.set_xticks(range(len(all_names_bar)))
ax2.set_xticklabels(all_names_bar, fontsize=8)
ax2.set_title("Final Score: All Methods"); ax2.set_ylabel("Score")
ax2.grid(axis="y", alpha=0.3)
for bar, score in zip(bars, all_scores_bar):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
             f"{score:.4f}", ha="center", va="bottom", fontsize=7)

# 3. Gradient alignment
ax3 = axes[0, 2]
ax3.axhline(0, color="black", linestyle="--", alpha=0.5, label="Zero (noise)")
for name, hist in all_histories.items():
    if hist["grad_align"]:
        ax3.plot(hist["grad_align_iter"], hist["grad_align"],
                 color=colors[name], marker="s", linewidth=1.5, markersize=4, label=name)
ax3.set_title("Gradient Alignment (cos sim with BC gradient)")
ax3.set_xlabel("Iteration"); ax3.set_ylabel("Cosine Similarity")
ax3.legend(fontsize=7); ax3.grid(alpha=0.3)
ax3.set_ylim(-1.1, 1.1)

# 4. Representation drift
ax4 = axes[1, 0]
for name, hist in all_histories.items():
    if hist["repr_drift"]:
        ax4.plot(hist["repr_drift_iter"], hist["repr_drift"],
                 color=colors[name], marker="^", linewidth=1.5, markersize=4, label=name)
ax4.set_title("Representation Drift (KL from pretrained)")
ax4.set_xlabel("Iteration"); ax4.set_ylabel("KL Divergence")
ax4.legend(fontsize=7); ax4.grid(alpha=0.3)

# 5. High-t vs Low-t gradient norm ratio
ax5 = axes[1, 1]
for name, hist in all_histories.items():
    if hist["norm_high_t"]:
        ratios = [h / (l + 1e-10) for h, l in zip(hist["norm_high_t"], hist["norm_low_t"])]
        ax5.plot(hist["t_analysis_iter"], ratios,
                 color=colors[name], marker="D", linewidth=1.5, markersize=4, label=name)
ax5.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="Equal contribution")
ax5.set_title("High-t / Low-t Gradient Norm Ratio")
ax5.set_xlabel("Iteration"); ax5.set_ylabel("Ratio (>1 = high-t dominates)")
ax5.legend(fontsize=7); ax5.grid(alpha=0.3)

# 6. Synthetic sanity check
ax6 = axes[1, 2]
if synth_history["recovery_rate"]:
    ax6.plot(
        range(config["EVAL_EVERY"], config["EVAL_EVERY"] * (len(synth_history["recovery_rate"]) + 1), config["EVAL_EVERY"]),
        synth_history["recovery_rate"],
        color="#FF9800", marker="o", linewidth=2, markersize=5, label="Synthetic BC",
    )
ax6.axhline(baseline_score, color="black", linestyle="--", alpha=0.7,
            label=f"Pretrained ({baseline_score:.4f})")
ax6.set_title("Synthetic Sanity Check (BC on oracle data)")
ax6.set_xlabel("Iteration"); ax6.set_ylabel("Score")
ax6.legend(fontsize=8); ax6.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("craftax_full_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: craftax_full_analysis.png")


# ---------------------------------------------------------------------------
# Save results to JSON
# ---------------------------------------------------------------------------

results_dict = {
    "env": "Craftax-Classic-Symbolic-v1",
    "max_iter": config["MAX_ITER"],
    "baseline_score":    float(baseline_score),
    "baseline_rl_score": float(score_baseline_rl),
    "kl_penalty":        float(score_kl),
    "frozen_backbone":   float(score_frozen),
    "bc_on_wins":        float(score_bc),
    "low_t_only":        float(score_lowt),
    "synthetic":         float(score_synthetic),
    "histories": {
        k: {hk: [float(v) for v in hv] for hk, hv in hist.items()}
        for k, hist in {
            "baseline_rl":     history_baseline_rl,
            "kl_penalty":      history_kl,
            "frozen_backbone": history_frozen,
            "bc_on_wins":      history_bc,
            "low_t_only":      history_lowt,
            "synthetic":       synth_history,
        }.items()
    },
}

with open("craftax_full_results.json", "w") as f:
    json.dump(results_dict, f, indent=2)
print("Saved: craftax_full_results.json")


def _build_parser(default_cfg_path: str, default_ablations_cfg_path: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ablation study for ReMDM discrete diffusion planner for Craftax",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config files
    p.add_argument("--config", default=default_cfg_path)
    p.add_argument("--ablations_config", default=default_ablations_cfg_path)

    # Mode
    p.add_argument(
        "--mode", required=True,
        choices=["collect", "offline", "online", "inference", "train_reward"],
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)

    # Paths
    p.add_argument("--ppo_checkpoint_path", type=str, default=None)
    p.add_argument("--offline_data_path", type=str, default=None)
    p.add_argument("--offline_checkpoint_path", type=str, default=None)
    p.add_argument("--checkpoint_path", type=str, default=None)
    p.add_argument("--checkpoint_dir", type=str, default=None)
    p.add_argument("--reward_load_path", type=str, default=None)
    p.add_argument("--reward_save_path", type=str, default=None)

    # Environment
    p.add_argument("--env_name", type=str, default=None)
    p.add_argument("--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--optimistic_reset_ratio", type=int, default=None)

    # Architecture
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--n_heads", type=int, default=None)
    p.add_argument("--n_layers", type=int, default=None)
    p.add_argument("--d_ff", type=int, default=None)
    p.add_argument("--obs_encoder_layers", type=int, default=None)
    p.add_argument("--obs_encoder_width", type=int, default=None)
    p.add_argument("--dropout_rate", type=float, default=None)

    # Diffusion
    p.add_argument("--plan_horizon", type=int, default=None)
    p.add_argument("--diffusion_schedule", type=str, choices=DIFFUSION_SCHEDULES, default=None)
    p.add_argument("--diffusion_steps", type=int, default=None)
    p.add_argument("--diffusion_steps_eval", type=int, default=None)
    p.add_argument("--train_sigma", type=float, default=None)
    p.add_argument("--label_smoothing", type=float, default=None)
    p.add_argument("--remask_strategy", type=str, choices=REMASK_STRATEGIES, default=None)
    p.add_argument("--eta", type=float, default=None)
    p.add_argument("--use_loop", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--t_on", type=float, default=None)
    p.add_argument("--t_off", type=float, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top_p", type=float, default=None)

    # Optimisation
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max_grad_norm", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)

    # Offline training
    p.add_argument("--total_timesteps", type=lambda x: int(float(x)), default=None)
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--num_steps", type=int, default=None)
    p.add_argument("--num_minibatches", type=int, default=None)
    p.add_argument("--update_epochs", type=int, default=None)
    p.add_argument("--num_repeats", type=int, default=None)
    p.add_argument("--collect_temperature", type=float, default=None)
    p.add_argument("--val_interval", type=int, default=None)
    p.add_argument("--val_diffusion_steps", type=int, default=None)
    p.add_argument("--val_replan_every", type=int, default=None)
    p.add_argument("--val_steps", type=int, default=None)
    p.add_argument("--return_weight_cap", type=float, default=None)
    p.add_argument("--lr_warmup_steps", type=int, default=None)

    # Ablations
    p.add_argument("--ewc_lambda", type=float, default=None)
    p.add_argument("--llrd_decay", type=float, default=None)
    p.add_argument("--lora_rank", type=int, default=None)
    p.add_argument("--lora_alpha", type=float, default=None)
    p.add_argument("--mixed_replay_ratio", type=float, default=None)
    p.add_argument("--t_curriculum_start", type=float, default=None)
    p.add_argument("--t_curriculum_end", type=float, default=None)
    p.add_argument("--t_curriculum_steps", type=int, default=None)
    p.add_argument("--entropy_coef", type=float, default=None)
    p.add_argument("--trust_region_kl", type=float, default=None)
    p.add_argument("--trust_region_penalty_coef", type=float, default=None)
    p.add_argument("--gradient_surgery", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--kl_coef", type=float, default=None)
    p.add_argument("--win_threshold", type=float, default=None)
    p.add_argument("--t_max_low", type=float, default=None)
    p.add_argument("--max_iter", type=int, default=None)
    p.add_argument("--eval_every", type=int, default=None)
    p.add_argument("--return_weight_floor", type=float, default=None)
    p.add_argument("--n_t_bins", type=int, default=None)
    p.add_argument("--num_seeds", type=int, default=None)

    # Logging
    p.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)

    return p


if __name__ == "__main__":
    backend = jax.default_backend()
    print(f"JAX backend: {backend} | Devices: {jax.devices()}")
    if backend != "gpu":
        import warnings

        warnings.warn(f"JAX is using '{backend}', not GPU. pip install jax[cuda12]")

    default_cfg = str(pathlib.Path(__file__).parent / "configs" / "defaults.yaml")
    default_ablations_cfg = str(pathlib.Path(__file__).parent / "configs" / "ablations.yaml")

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=default_cfg)
    pre.add_argument("--ablations_config", default=default_ablations_cfg)
    pre_args, _ = pre.parse_known_args()

    with open(pre_args.config) as f:
        yaml_cfg = yaml.safe_load(f) or {}

    with open(pre_args.ablations_config) as f:
        ablations_cfg = yaml.safe_load(f) or {}

    parser = _build_parser(default_cfg, default_ablations_cfg)
    args, rest = parser.parse_known_args()
    if rest:
        raise ValueError(f"Unknown arguments: {rest}")

    config: dict[str, Any] = {k.upper(): v for k, v in yaml_cfg.items()}
    config.update({k.upper(): v for k, v in ablations_cfg.items()})

    cli = {k.upper(): v for k, v in vars(args).items() if v is not None and k not in ("config", "ablations_config")}
    config.update(cli)

    if config.get("SEED") is None:
        config["SEED"] = np.random.randint(2 ** 31)

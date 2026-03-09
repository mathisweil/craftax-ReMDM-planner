import jax
import jax.numpy as jnp

from src.models.remdm import (
    ScheduleFn,
    compute_loss,
    cosine_schedule,
    linear_schedule,
)

SCHEDULE_MAP: dict[str, ScheduleFn] = {
    "cosine": cosine_schedule,
    "linear": linear_schedule,
}

def _global_grad_norm(grads) -> jnp.ndarray:
    """L2 norm across all gradient leaves."""
    leaves = jax.tree.leaves(grads)
    sq_sum = jnp.array([jnp.sum(g ** 2) for g in leaves])
    return jnp.sqrt(jnp.sum(sq_sum))

def _action_stats(act_batch: jnp.ndarray, num_actions: int) -> dict[str, jnp.ndarray]:
    """Compute action distribution entropy and mode fraction over a batch of action sequences."""
    flat = act_batch.reshape(-1)
    counts = jnp.bincount(flat, length=num_actions).astype(jnp.float32)
    probs = counts / jnp.maximum(counts.sum(), 1.0)
    log_probs = jnp.log(jnp.where(probs > 0, probs, 1.0))
    entropy = -jnp.sum(probs * log_probs)
    unique_frac = jnp.sum(probs > 0).astype(jnp.float32) / num_actions
    return {"action_entropy": entropy, "action_unique_frac": unique_frac}

def _make_grad_step(apply_train, num_actions: int, schedule_fn, sigma_t: float):
    """Return a pure function: (train_state, act_batch, obs_batch, rng) -> (train_state, info)."""
    def grad_step(train_state, act_batch, obs_batch, rng):
        def loss_fn(params):
            return compute_loss(
                apply_train, params, rng,
                act_batch, obs_batch, num_actions, schedule_fn,
                sigma_t=sigma_t,
            )

        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
        
        info["loss"] = loss 
        
        info["grad_norm"] = _global_grad_norm(grads)
        info.update(_action_stats(act_batch, num_actions))
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, info

    return grad_step
"""Literature-anchored specification tests — step-8 gap closure.

Complements tests/test_method_spec.py with the surfaces the step-8 audit
found untested in this repo: the reverse-posterior/remasking/carry-over
chain, the final greedy cleanup, decode temperature, label smoothing and
the loss weight clip. Every expected value derives from a cited source
or a derivation written in the docstring — never from the current output
of this or the sibling repo. The minihack twin file carries the same
chain/cleanup/temperature/smoothing assertions with the same inputs and
tolerances (weight clip and empty-mask zero already exist there).

References as in tests/test_method_spec.py (MDLM arXiv:2406.07524;
ReMDM arXiv:2503.00307; Holtzman arXiv:1904.09751).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.diffusion.loss import compute_loss
from src.diffusion.sampling import _decode, sample_plan
from src.diffusion.schedules import SCHEDULE_MAP

ATOL = 1e-6


# ---------------------------------------------------------------------------
# Reverse posterior + remasking + carry-over, jointly, through sample_plan
# ---------------------------------------------------------------------------


def _time_coded_apply(num_actions: int, horizon: int):
    """Stub model whose argmax token encodes the decode time.

    token 0 for t > 0.9, token 1 for 0.5 < t <= 0.9, token 2 otherwise.
    Deterministic under temperature=0 (argmax decode).
    """

    def apply_fn(params, obs, z, t, rng):
        idx = jnp.where(t[0] > 0.9, 0, jnp.where(t[0] > 0.5, 1, 2))
        logits = jax.nn.one_hot(idx, num_actions) * 30.0
        return jnp.broadcast_to(logits, (obs.shape[0], horizon, num_actions))

    return apply_fn


@pytest.mark.parametrize("eta", [0.0, 0.5])
def test_posterior_remask_carryover_token_distribution(eta):
    """Final token distribution matches the ReMDM Alg 1 posterior chain.

    Source: ReMDM eq (6) masked branch (unmask probability
    (alpha_s - (1-sigma) alpha_t) / (1 - alpha_t)), eq (7) sigma_max,
    Sec 4.1 rescale sigma = eta * sigma_max, and MDLM Sec 3.2.3 carry-over.

    Derivation (linear schedule, K=3, grid t=1,2/3,1/3 with s=2/3,1/3,0;
    the stub decodes token 0 at t=1, token 1 at t=2/3, token 2 at
    t<=1/3):
      step 1 (t=1):   alpha_t=0, alpha_s=1/3, p_unmask=1/3 -> token 0.
      step 2 (t=2/3): alpha_t=1/3, alpha_s=2/3, sigma_max=1, sigma=eta;
                      committed remask w.p. eta; masked unmask w.p.
                      (2/3-(1-eta)/3)/(2/3) = (1+eta)/2 -> token 1.
      step 3 (t=1/3): alpha_s=1 -> sigma_max=0 (no remask), p_unmask=1
                      -> every remaining mask becomes token 2.
    Hence P(0) = (1-eta)/3, P(1) = (1+eta)/3, P(2) = 1/3.
    eta=0 additionally pins carry-over: a committed token is never
    re-decided when sigma=0.

    Statistical: N = 512*32 = 16384 independent per-position outcomes;
    max sigma = sqrt(0.25/16384) = 0.0039; bound 0.02 = 5.1 sigma.
    """
    B, H, V = 512, 32, 3
    fn, _ = SCHEDULE_MAP["linear"]
    plan = sample_plan(
        _time_coded_apply(V, H), None, jax.random.PRNGKey(0),
        jnp.zeros((B, 4)), V, H, num_steps=3, schedule_fn=fn,
        remask_strategy="rescale", eta=eta, use_loop=False,
        temperature=0.0, top_p=None,
    )
    freq = np.bincount(np.asarray(plan).ravel(), minlength=V) / (B * H)
    expected = np.array([(1 - eta) / 3, (1 + eta) / 3, 1 / 3])
    assert np.all(np.abs(freq - expected) < 0.02), (freq, expected)


def test_final_cleanup_commits_all_remaining_masks():
    """With zero denoising steps, the final greedy cleanup decodes every
    position at t=0 (argmax), leaving no MASK token.

    Source: spec-method 4.9 (final-step commit of remaining masks is the
    project safety net; craftax cleanup is unconditional). The stub
    decodes token 2 at t=0, so the output must be all-2.
    """
    B, H, V = 8, 16, 3
    fn, _ = SCHEDULE_MAP["linear"]
    plan = sample_plan(
        _time_coded_apply(V, H), None, jax.random.PRNGKey(0),
        jnp.zeros((B, 4)), V, H, num_steps=0, schedule_fn=fn,
        remask_strategy="rescale", eta=0.0, use_loop=False,
        temperature=0.0, top_p=None,
    )
    assert (np.asarray(plan) == 2).all()


# ---------------------------------------------------------------------------
# Decode temperature
# ---------------------------------------------------------------------------


def test_decode_temperature_scales_logits_before_sampling():
    """Sampling frequencies follow softmax(logits / temperature).

    Source: spec-method 5.2 (softmax temperature before filtering;
    standard technique). Derivation: logits [0, 1] at temperature 0.5
    give softmax([0, 2]) = [1/(1+e^2), e^2/(1+e^2)] = [0.1192, 0.8808].
    Statistical: 8192 draws; sigma = sqrt(0.8808*0.1192/8192) = 0.00358;
    bound 0.0143 = 4 sigma.
    """
    logits = jnp.broadcast_to(jnp.array([0.0, 1.0]), (8192, 1, 2))
    tokens = np.asarray(
        _decode(jax.random.PRNGKey(0), logits, temperature=0.5, top_p=None)
    ).ravel()
    p1 = math.exp(2) / (1 + math.exp(2))
    assert abs(tokens.mean() - p1) < 0.0143


# ---------------------------------------------------------------------------
# Label smoothing
# ---------------------------------------------------------------------------


def test_label_smoothing_matches_closed_form():
    """Smoothed CE = -[(1-eps)+eps/V] log p_true - (eps/V) sum log p_other.

    Source: spec-method 3.7 (smoothing target (1-eps)*onehot + eps/V;
    eps=0 is the exact ELBO). Derivation: model probs [0.7,0.1,0.1,0.1],
    true class 0, V=4, eps=0.3: coefficient on -log 0.7 is
    (1-0.3)+0.3/4 = 0.775; each other class gets 0.3/4 = 0.075.
    t pinned to 1 (linear, w=1, everything masked) makes the loss equal
    that CE exactly. Same inputs and expectations as the minihack twin.
    """
    V, H, B = 4, 8, 4
    fn, deriv = SCHEDULE_MAP["linear"]
    probs = jnp.array([0.7, 0.1, 0.1, 0.1])

    def apply_fn(params, obs, z, t, rng):
        return jnp.broadcast_to(jnp.log(probs), (obs.shape[0], H, V))

    x0 = jnp.zeros((B, H), dtype=jnp.int32)
    obs = jnp.zeros((B, 3))
    expected = -0.775 * math.log(0.7) - 0.075 * 3 * math.log(0.1)
    loss, _ = compute_loss(
        apply_fn, None, jax.random.PRNGKey(0), x0, obs, jnp.ones(B), V,
        fn, deriv, label_smoothing=0.3, t_min=1.0, t_max=1.0,
    )
    assert abs(float(loss) - expected) < 1e-5
    loss0, _ = compute_loss(
        apply_fn, None, jax.random.PRNGKey(0), x0, obs, jnp.ones(B), V,
        fn, deriv, label_smoothing=0.0, t_min=1.0, t_max=1.0,
    )
    assert abs(float(loss0) - (-math.log(0.7))) < 1e-5


# ---------------------------------------------------------------------------
# Loss weight clip and the empty-mask edge case (minihack twins exist)
# ---------------------------------------------------------------------------


def test_loss_weight_clip_bound():
    """w(t) is clipped at 1000 (project numerics guard, spec-method 3.4;
    the same _MAX_WEIGHT the minihack twin pins via an explicit zt).

    compute_loss samples zt internally, so this is statistical: at
    t=1e-4 (linear) the raw weight is 1/t = 10^4, clipped to 10^3.
    E[loss] = 1000 * ln V * P(mask) = 1000 * ln 5 * 1e-4 = 0.1 ln 5.
    The unclipped hypothesis gives 1.0 ln 5 (10x larger). Masked count
    over N = 131072*8 positions is ~Poisson(104.9); sigma(loss) =
    1000 * lnV * sqrt(104.9)/N = 0.0098 lnV; bound 0.04 lnV = 4.1 sigma.
    """
    V, H, B = 5, 8, 131072
    fn, deriv = SCHEDULE_MAP["linear"]

    def apply_fn(params, obs, z, t, rng):
        return jnp.zeros((obs.shape[0], H, V))

    x0 = jnp.zeros((B, H), dtype=jnp.int32)
    loss, _ = compute_loss(
        apply_fn, None, jax.random.PRNGKey(0), x0, jnp.zeros((B, 1)),
        jnp.ones(B), V, fn, deriv, t_min=1e-4, t_max=1e-4,
    )
    assert abs(float(loss) - 0.1 * math.log(V)) < 0.04 * math.log(V)


def test_loss_zero_when_nothing_masked():
    """Loss is exactly 0.0 when the batch has no masked positions.

    Source: spec-method 3.4 (zero-on-empty project convention; the
    minihack twin asserts the same through an explicit unmasked zt).
    At t=0, alpha=1 and the forward process keeps every token
    (uniform draws in [0,1) are always < 1), so nothing is masked.
    """
    V, H, B = 5, 8, 4
    fn, deriv = SCHEDULE_MAP["linear"]

    def apply_fn(params, obs, z, t, rng):
        return jnp.zeros((obs.shape[0], H, V))

    x0 = jnp.zeros((B, H), dtype=jnp.int32)
    loss, _ = compute_loss(
        apply_fn, None, jax.random.PRNGKey(0), x0, jnp.zeros((B, 1)),
        jnp.ones(B), V, fn, deriv, t_min=0.0, t_max=0.0,
    )
    assert float(loss) == 0.0

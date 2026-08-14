"""Literature-anchored specification tests.

Every expected value here is derived from a primary source (cited
per test; full references below) or from a derivation written out in
the docstring - never from the current output of this or the sibling repo. The minihack repo
carries the same assertions with the same inputs and tolerances wherever
the mathematics is parameter-free and shared.

Tolerances: closed-form checks use atol=1e-6 (an order of magnitude above
float32 round-off, which the parity probes measured at <4e-8 on these
functions). Statistical checks state their sampling distribution and use a
4-sigma bound (or the stated multiple) with the derivation in the docstring.

References:
- MDLM: Sahoo et al., "Simple and Effective Masked Diffusion Language
  Models", NeurIPS 2024. arXiv:2406.07524.
- Shi: Shi et al., "Simplified and Generalized Masked Diffusion for
  Discrete Data", NeurIPS 2024. arXiv:2406.04329.
- ReMDM: Wang et al., "Remasking Discrete Diffusion Models with
  Inference-Time Scaling", NeurIPS 2025. arXiv:2503.00307.
- Holtzman et al., "The Curious Case of Neural Text Degeneration",
  ICLR 2020. arXiv:1904.09751.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from src.diffusion.forward import forward_process
from src.diffusion.loss import compute_loss
from src.diffusion.sampling import (
    _nucleus_sample,
    _sigma_max,
    sample_plan_inpainting,
    sigma_cap,
    sigma_conf,
    sigma_rescale,
)
from src.diffusion.schedules import (
    SCHEDULE_MAP,
    cosine_schedule,
    cosine_schedule_deriv,
    linear_schedule,
    linear_schedule_deriv,
)

ATOL = 1e-6
T_GRID = jnp.array([0.0, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0])


# ---------------------------------------------------------------------------
# Noise schedules
# ---------------------------------------------------------------------------


def test_linear_schedule_closed_form():
    """alpha(t) = 1 - t; alpha'(t) = -1.

    Source: MDLM (Sahoo et al.) App E.1 eq (90) family; ReMDM Sec 3
    convention alpha(0)=1, alpha(1)=0. Same grid and tolerance as the
    minihack twin test.
    """
    expected = np.array([1.0, 2 / 3, 0.5, 1 / 3, 0.0])
    assert np.allclose(np.asarray(linear_schedule(T_GRID)), expected, atol=ATOL)
    assert np.allclose(np.asarray(linear_schedule_deriv(T_GRID)), -1.0, atol=ATOL)


def test_cosine_schedule_closed_form():
    """alpha(t) = cos(pi t / 2); alpha'(t) = -(pi/2) sin(pi t / 2).

    Source: MDLM App E.1 eq (92) ("Cosine"). Values: cos(pi/6)=sqrt(3)/2,
    cos(pi/4)=sqrt(2)/2, cos(pi/3)=1/2.
    """
    expected = np.array([1.0, math.sqrt(3) / 2, math.sqrt(2) / 2, 0.5, 0.0])
    assert np.allclose(np.asarray(cosine_schedule(T_GRID)), expected, atol=ATOL)
    expected_d = np.array(
        [0.0, -math.pi / 4, -(math.pi / 2) * math.sqrt(2) / 2,
         -(math.pi / 2) * math.sqrt(3) / 2, -math.pi / 2]
    )
    assert np.allclose(np.asarray(cosine_schedule_deriv(T_GRID)), expected_d, atol=ATOL)


def test_schedule_registry_names_follow_mdlm_e1():
    """The label "cosine" must denote MDLM eq (92) (ADJUDICATION B-6)."""
    fn, deriv = SCHEDULE_MAP["cosine"]
    assert abs(float(fn(jnp.array(0.5))) - math.sqrt(2) / 2) < ATOL
    assert abs(float(deriv(jnp.array(0.5))) + (math.pi / 2) * math.sqrt(2) / 2) < ATOL


# ---------------------------------------------------------------------------
# Forward corruption
# ---------------------------------------------------------------------------


def test_forward_marginal_endpoints():
    """q(z_t|x) = Cat(alpha_t x + (1-alpha_t) m): alpha=1 identity,
    alpha=0 all-MASK, deterministically (uniform draws lie in [0,1)).

    Source: MDLM Sec 3.2.1.
    """
    rng = jax.random.PRNGKey(0)
    x0 = jax.random.randint(rng, (4, 32), 0, 17)
    z_keep = forward_process(rng, x0, jnp.ones(4), mask_id=17)
    assert bool(jnp.all(z_keep == x0))
    z_mask = forward_process(rng, x0, jnp.zeros(4), mask_id=17)
    assert bool(jnp.all(z_mask == 17))


def test_forward_marginal_rate():
    """Empirical mask rate at alpha=0.7 is 0.3 within 4 sigma.

    Source: MDLM Sec 3.2.1. N = 200*64 = 12800 Bernoulli(0.3) draws;
    sigma = sqrt(0.3*0.7/12800) = 0.00405; bound = 4 sigma = 0.0162.
    """
    rng = jax.random.PRNGKey(1)
    x0 = jnp.zeros((200, 64), dtype=jnp.int32)
    zt = forward_process(rng, x0, jnp.full(200, 0.7), mask_id=17)
    rate = float((zt == 17).mean())
    assert abs(rate - 0.3) < 0.0162


# ---------------------------------------------------------------------------
# Loss: NELBO estimator
# ---------------------------------------------------------------------------

_V, _H = 5, 8


def _uniform_apply(params, obs, z, t, rng):
    return jnp.zeros((obs.shape[0], _H, _V))


def _loss(t_pin, schedule_name, valid, B):
    fn, deriv = SCHEDULE_MAP[schedule_name]
    rng = jax.random.PRNGKey(0)
    x0 = jax.random.randint(rng, (B, _H), 0, _V)
    obs = jnp.zeros((B, 3))
    loss, info = compute_loss(
        _uniform_apply, None, rng, x0, obs, valid, _V, fn, deriv,
        t_min=t_pin, t_max=t_pin,
    )
    return float(loss)


def test_loss_all_masked_linear_t1():
    """Loss = w(1) * log V = log V with everything masked, uniform logits.

    Source: MDLM eq (10); Shi eq (4). Derivation: t pinned to 1 makes the
    forward step deterministic (alpha=0, all masked); uniform logits give
    CE = log V per position; sum/H = log V; linear w(1) = 1.
    """
    got = _loss(1.0, "linear", jnp.ones(4), B=4)
    assert abs(got - math.log(_V)) < 1e-5


def test_loss_weight_uses_analytic_derivative_cosine_t1():
    """Cosine at t=1: w(1) = -alpha'(1)/(1-alpha(1)) = pi/2, so loss =
    (pi/2) log V.

    Source: MDLM eq (10) with the eq (92) schedule: alpha'(1) =
    -(pi/2) sin(pi/2) = -pi/2 and alpha(1)=0. Pins the analytic-derivative
    form of the weight.
    """
    got = _loss(1.0, "cosine", jnp.ones(4), B=4)
    assert abs(got - (math.pi / 2) * math.log(_V)) < 1e-5


def test_loss_invalid_samples_contribute_zero():
    """A sample with valid=0 contributes exactly zero to the batch mean.

    Source: the validity mask is the benchmark-forced analogue of PAD
    exclusion (MDLM Sec 3.2.3: loss over masked positions of real data
    only). Derivation: B=2 at t=1 (all masked), valid=[1,0]:
    mean(log V, 0) = log V / 2.
    """
    got = _loss(1.0, "linear", jnp.array([1.0, 0.0]), B=2)
    assert abs(got - math.log(_V) / 2) < 1e-5


def test_loss_denominator_is_per_token_not_per_masked():
    """E[loss] at pinned t=0.5 (cosine) = w(0.5) * log V * (1 - alpha),
    which distinguishes the per-token denominator (1/H) from the
    pre-FIX-1 per-masked-count denominator, whose expectation is
    w(0.5) * log V (3.41x larger here).

    Source: MDLM eq (8)/(10); Shi eq (4) (no division by the realised
    masked count). Derivation: per-sample loss = w * log V * n_hat/H with
    n_hat ~ Bin(H=8, 1-alpha=0.29289); mean over B=8192 samples has sigma
    = w*logV*std(n_hat)/H/sqrt(B) = w*logV*0.00178; the bound below is
    0.01*w*logV (~5.6 sigma), far smaller than the 2.41x separation
    between the two hypotheses.
    """
    alpha = math.sqrt(2) / 2
    w = (math.pi / 2) * math.sin(math.pi / 4) / (1 - alpha)
    expected = w * math.log(_V) * (1 - alpha)
    got = _loss(0.5, "cosine", jnp.ones(8192), B=8192)
    assert abs(got - expected) < 0.01 * w * math.log(_V)


# ---------------------------------------------------------------------------
# Reverse step: remasking schedules and the sigma bound
# ---------------------------------------------------------------------------


def test_sigma_strategies_closed_form_and_bound():
    """Same grid and assertions as the minihack twin (ReMDM eq (7), Sec 4.1)."""
    eta = 0.5
    for k in range(1, 10):
        a_t = jnp.array(1 - k / 10)
        a_s = jnp.array(1 - (k + 1) / 10)
        smax = min(1.0, (1 - float(a_s)) / float(a_t))
        assert abs(float(_sigma_max(a_t, a_s)) - smax) < ATOL
        assert abs(float(sigma_rescale(a_t, a_s, eta)) - eta * smax) < ATOL
        assert abs(float(sigma_cap(a_t, a_s, eta)) - min(eta, smax)) < ATOL


def test_conf_strategy_softmax_of_stored_psi():
    """sigma_conf = softmax(-psi) * eta * sigma_max over unmasked positions,
    zero at masked; lower psi => higher remask probability; sums to
    eta * sigma_max.

    Source: ReMDM Sec 4.1 (Confidence-Based Schedule).
    """
    eta = 0.5
    a_t, a_s = jnp.array(0.5), jnp.array(0.1)
    smax = float(_sigma_max(a_t, a_s))
    psi = jnp.array([[0.9, 0.2, jnp.inf, 0.5]])
    unmasked = jnp.array([[True, True, False, True]])
    sigma = np.asarray(sigma_conf(a_t, a_s, eta, psi, unmasked))
    assert sigma[0, 2] == 0.0
    assert sigma[0, 1] > sigma[0, 3] > sigma[0, 0]
    assert abs(sigma[0, [0, 1, 3]].sum() - eta * smax) < 1e-5
    assert (sigma <= smax + ATOL).all()


# ---------------------------------------------------------------------------
# Nucleus filtering
# ---------------------------------------------------------------------------


def test_nucleus_sample_support_and_frequencies():
    """Nucleus sampling draws only from the smallest prefix with cumulative
    mass >= p, with renormalised probabilities.

    Source: ReMDM Sec 5 (nucleus sampling, Holtzman et al.). For probs
    [0.5, 0.3, 0.15, 0.05] and p=0.9 the support is {0,1,2} with
    renormalised probs [0.5263, 0.3158, 0.1579]. Statistical: 4096 draws;
    per-token sigma = sqrt(p_i(1-p_i)/4096) <= 0.0078; bound = 4 sigma
    = 0.0313.
    """
    probs = jnp.array([0.5, 0.3, 0.15, 0.05])
    logits = jnp.log(jnp.broadcast_to(probs, (4096, 1, 4)))
    tokens = np.asarray(_nucleus_sample(jax.random.PRNGKey(0), logits, 0.9)).ravel()
    counts = np.bincount(tokens, minlength=4) / tokens.size
    assert counts[3] == 0.0, "out-of-nucleus token was sampled"
    renorm = np.array([0.5, 0.3, 0.15]) / 0.95
    assert np.all(np.abs(counts[:3] - renorm) < 0.0313)


# ---------------------------------------------------------------------------
# Prefix locking through the full corrected chain
# ---------------------------------------------------------------------------


def test_prefix_lock_survives_loop_mode_chain():
    """A locked prefix is bit-identical after a full ReMDM chain with loop
    mode and the conf strategy active, and the output contains no MASK.

    Source: planning-as-inpainting (Diffuser Sec 3.3: conditioned values
    are fixed throughout denoising) on top of ReMDM Alg 1 / Sec 4.2 loop
    (FIX-2). Uses a uniform-logits stub model.
    """
    B, H, V = 3, 8, 5
    fn, _ = SCHEDULE_MAP["cosine"]

    def apply_fn(params, obs, z, t, rng):
        return jnp.zeros((obs.shape[0], H, V))

    history = jnp.tile(jnp.arange(H, dtype=jnp.int32) % V, (B, 1))
    hist_len = jnp.array([0, 3, H], dtype=jnp.int32)
    plan = sample_plan_inpainting(
        apply_fn, None, jax.random.PRNGKey(0), jnp.zeros((B, 3)),
        history, hist_len, V, H, diffusion_steps=6, schedule_fn=fn,
        remask_strategy="conf", eta=0.5, use_loop=True, t_on=0.7, t_off=0.3,
        temperature=0.5, top_p=0.95,
    )
    plan = np.asarray(plan)
    assert (plan != V).all(), "output contains MASK tokens"
    assert (plan[1, :3] == np.asarray(history)[1, :3]).all()
    assert (plan[2] == np.asarray(history)[2]).all()

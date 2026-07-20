"""NWJ objective: stationarity at the true ratio, the two-expectation restoring force,
and per-stratum bound-gap calibration.

Toy (adapted from tests/test_identities.py): per event, truth theta has t0 ~ N(0, 30)
(slot 5) and E ~ U(0.5, 9.5) (slot 6); each hit carries time t ~ N(t0, s_t) in h[3] and
a proxy x ~ N(E, 1) in h[0]. Both exact marginals are analytic, so ``ExactNet``'s logit
IS the true per-hit log ratio log p(x|theta)/p(x); its normalizer Z(theta) = 1 (log Z =
0). ``TiltedNet(eps)`` adds a coherent ``eps * theta_E`` per hit — a pure theta-dependent
shift, i.e. exactly the kind of drift the NWJ normalizer a_psi(theta) is meant to absorb.

Unlike test_identities (event-grouped padded batches for the score penalty), NWJ works on
flat per-hit (obs, theta) pairs: matched = aligned rows, shuffled = a marginal
permutation, exactly as hit_batch/_batch_loss feed the trainer.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.scipy.stats import norm

from hitman.train.nwj import (CLIP_C, N_STRATA, ZNet, nwj_hit_loss)

# Time/charge resolutions are deliberately broad relative to the t0 spread and E range:
# the per-hit ratio e^f is then light-tailed, so at the calibrated optimum e^{f-a} on
# mismatched pairs stays well under the winsorization clip (100) — the regime the clip
# assumes. A very peaked ratio (tiny S_T/S_X) makes e^f lognormal-heavy and E[e^f]=1
# relies on rare mass beyond the clip, a toy pathology real detectors don't have.
S_T, SIG_T0, S_X = 10.0, 30.0, 2.0
S_MARG = float(np.sqrt(S_T**2 + SIG_T0**2))
E_LO, E_HI = 0.5, 9.5
N_HIT = 8
# Winsorization off for the exact-ratio stationarity/identity checks: the EXACT ratio's
# e^{f} has a heavy right tail (tight time resolution => peak r can exceed 100), so the
# default clip is occasionally active AT the exact ratio and would bias the stationarity
# gradient. The clip is a finite-sample tail guard, not part of the population identity.
BIG = float(np.log(1e12))


class ExactNet(eqx.Module):
    """The true per-hit log ratio (Z == 1). eqx.Module so it is a valid loss argument."""

    def __call__(self, h, th):
        lr_t = norm.logpdf(h[3], th[5], S_T) - norm.logpdf(h[3], 0.0, S_MARG)
        marg_x = (norm.cdf((h[0] - E_LO) / S_X) - norm.cdf((h[0] - E_HI) / S_X)) / (E_HI - E_LO)
        lr_x = norm.logpdf(h[0], th[6], S_X) - jnp.log(marg_x)
        return lr_t + lr_x


class ScaledNet(eqx.Module):
    """c * (ExactNet + eps * theta_E): scale knob c on a possibly-tilted critic."""

    c: jnp.ndarray
    eps: float = eqx.field(static=True, default=0.0)

    def __call__(self, h, th):
        return self.c * (ExactNet()(h, th) + self.eps * th[6])


class TiltGradNet(eqx.Module):
    """ExactNet + eps * theta_E with eps a DYNAMIC leaf, so d/d(eps) is well defined."""

    eps: jnp.ndarray

    def __call__(self, h, th):
        return ExactNet()(h, th) + self.eps * th[6]


class ConstZ(eqx.Module):
    """Normalizer a_psi(theta) = coeff * theta_E (+ base). coeff=eps calibrates a
    TiltedNet(eps) (log Z = eps*theta_E); coeff=0, base=0 is a == 0 (log Z of ExactNet).
    """

    coeff: float = eqx.field(static=True, default=0.0)
    base: float = eqx.field(static=True, default=0.0)

    def a_hit(self, th):
        return self.coeff * th[6] + self.base

    def a_charge(self, th):
        return 0.0 * th[0]


def gen_hit_batch(key, n_events, n_hit=N_HIT):
    """Flat per-hit (obs (M,4), theta (M,7)) with M = n_events*n_hit; row i's theta is its
    generating event's theta (matched); the marginal permutation inside nwj_hit_loss makes
    the shuffled pairs."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    t0 = SIG_T0 * jax.random.normal(k1, (n_events,))
    e = jax.random.uniform(k2, (n_events,), minval=E_LO, maxval=E_HI)
    theta = jnp.zeros((n_events, 7)).at[:, 5].set(t0).at[:, 6].set(e)
    t = t0[:, None] + S_T * jax.random.normal(k3, (n_events, n_hit))
    x = e[:, None] + S_X * jax.random.normal(k4, (n_events, n_hit))
    obs = jnp.stack([x, jnp.zeros_like(x), jnp.zeros_like(x), t], axis=-1).reshape(-1, 4)
    hyp = jnp.repeat(theta, n_hit, axis=0)
    return obs, hyp


# ---------------------------------------------------------------------------
# Test 1: exact-ratio toy
# ---------------------------------------------------------------------------

def test_exact_ratio_stationary_and_znet_learns_logZ():
    """At f = exact ratio and a == 0 (= log Z), dL/d(scale) vanishes; and a flexible
    a_psi trained from a wrong init converges toward log Z = 0."""
    # (a) stationarity: gradient of L wrt the critic scale c at c=1, a=0.
    zero = ConstZ()

    def L_scale(c, batch, key):
        return nwj_hit_loss(ScaledNet(c=c), zero, batch, key, BIG)[0]

    grads = []
    for i in range(12):
        kb, kp = jax.random.split(jax.random.PRNGKey(i))
        grads.append(float(jax.grad(L_scale)(1.0, gen_hit_batch(kb, 4000), kp)))
    grads = np.array(grads)
    se = grads.std(ddof=1) / np.sqrt(len(grads))
    assert abs(grads.mean()) < 4 * se, f"not stationary at exact ratio: {grads.mean():.4f}±{se:.4f}"

    # (b) a flexible a_psi (ZNet) fit against the FIXED exact critic must drive a_hit -> 0
    # (= log Z) from a deliberately wrong (+1.5) init.
    exact = ExactNet()
    znet = ZNet(width=32, depth=2, key=jax.random.PRNGKey(0))
    # wrong init: offset the hit head's output layer bias by +1.5
    znet = eqx.tree_at(lambda z: z.hit_mlp.layers[-1].bias, znet,
                       znet.hit_mlp.layers[-1].bias + 1.5)
    th_probe = gen_hit_batch(jax.random.PRNGKey(999), 2000)[1]
    a0 = float(jnp.mean(jnp.abs(jax.vmap(znet.a_hit)(th_probe))))

    opt = optax.adam(3e-3)
    opt_state = opt.init(eqx.filter(znet, eqx.is_inexact_array))

    # Train the normalizer on the unwinsorized objective: the a-direction of the
    # winsorized loss is unbounded below (the clipped exp is flat, so a can run to -inf
    # while the linear term drops) — the clip is a tail guard for a WARM-STARTED a, not a
    # from-scratch optimizer target. At the optimum the clip is inactive anyway
    # (test_bound_gap).
    @eqx.filter_jit
    def step(znet, opt_state, batch, key):
        loss, grads = eqx.filter_value_and_grad(
            lambda z: nwj_hit_loss(exact, z, batch, key, BIG)[0])(znet)
        updates, opt_state = opt.update(grads, opt_state)
        return eqx.apply_updates(znet, updates), opt_state, loss

    key = jax.random.PRNGKey(1)
    for s in range(1500):
        key, kb, kp = jax.random.split(key, 3)
        znet, opt_state, _ = step(znet, opt_state, gen_hit_batch(kb, 1500), kp)
    a1 = float(jnp.mean(jnp.abs(jax.vmap(znet.a_hit)(th_probe))))
    assert a0 > 1.0, f"wrong init not actually wrong: mean|a|={a0:.3f}"
    assert a1 < 0.15, f"a_psi did not converge to log Z=0: mean|a|={a1:.3f} (was {a0:.3f})"


# ---------------------------------------------------------------------------
# Test 2: deflation toy — the two-expectation restoring force
# ---------------------------------------------------------------------------

def test_two_expectation_restoring_force_and_deflation_not_descent():
    """The NWJ tilt-direction gradient is E_matched[s] - E_model[s] (self-normalized IW
    over the shuffled sample) — the counterweight one-sided moment penalties lack — and
    deflating the critic is NOT a descent direction, whereas it IS for a one-sided
    penalty."""
    eps = 0.05
    a_calib = ConstZ(coeff=eps)          # a = log Z of TiltedNet(eps) = eps*theta_E
    obs, hyp = gen_hit_batch(jax.random.PRNGKey(7), 40000)
    batch = (obs, hyp)
    key = jax.random.PRNGKey(11)

    # NWJ gradient in the tilt (theta_E) direction at eps, with a tracking (calibrated).
    def L_eps(e, batch, key):
        return nwj_hit_loss(TiltGradNet(eps=e), a_calib, batch, key, BIG)[0]

    grad_eps = float(jax.grad(L_eps)(eps, batch, key))

    # Manual two-expectation form: -E_matched[s] + E_shuffled[w s], w = e^{f-a}.
    perm = jax.random.permutation(key, hyp.shape[0])
    hyp_s = hyp[perm]
    net = TiltGradNet(eps=jnp.asarray(eps))
    f_s = jax.vmap(net)(obs, hyp_s)
    a_s = jax.vmap(a_calib.a_hit)(hyp_s)
    w = np.asarray(jnp.exp(f_s - a_s))
    s_s, s_m = np.asarray(hyp_s[:, 6]), np.asarray(hyp[:, 6])
    E_matched = s_m.mean()
    analytic = -E_matched + (w * s_s).mean()
    E_model_sn = (w * s_s).sum() / w.sum()          # self-normalized model expectation

    # (i) the gradient IS the two-expectation form (exact identity).
    assert abs(grad_eps - analytic) < 1e-3, f"grad {grad_eps:.4f} != two-exp {analytic:.4f}"
    # (ii) calibrated => it equals E_model[s] - E_matched[s] (self-normalized).
    assert abs(w.mean() - 1.0) < 0.02, f"a not calibrated: mean(w)={w.mean():.4f}"
    assert abs(grad_eps - (E_model_sn - E_matched)) < 0.05
    # (iii) the counterweight is MATERIAL: a one-sided penalty's force is ~-E_matched (~-5),
    # the NWJ gradient is tiny because E_model[s] ~ E_matched[s] cancels it.
    assert abs(E_matched) > 4.0
    assert abs(grad_eps) < 0.3 * abs(E_matched)

    # Deflation: scale c on the (calibrated) critic. NWJ is minimized at the true scale
    # c=1, so deflating (c<1) INCREASES the loss; the one-sided score-mean penalty
    # decreases monotonically toward the collapsed critic f==0.
    def L_c(c):
        return float(nwj_hit_loss(ScaledNet(c=c, eps=eps), a_calib, batch, key, BIG)[0])

    def score_E_mean(c):
        sc = jax.vmap(lambda h, th: jax.grad(
            lambda t: ScaledNet(c=c, eps=eps)(h, t))(th)[6])(obs, hyp)
        return float(jnp.mean(sc))

    penalty = lambda c: 0.5 * score_E_mean(c) ** 2
    assert L_c(0.9) > L_c(1.0), "deflation should NOT be a descent direction for NWJ"
    assert penalty(0.9) < penalty(1.0), "deflation should reduce the one-sided penalty"


# ---------------------------------------------------------------------------
# Test 3: bound-gap toy — E_shuffled[e^{f-a}] -> 1 per stratum, clip inactive at optimum
# ---------------------------------------------------------------------------

def test_bound_gap_converges_and_clip_inactive():
    """Fit a_psi against the exact critic to convergence: the per-E-stratum bound gap
    E_shuffled[e^{f-a}] - 1 -> 0 in every populated stratum, and the winsorization clip is
    inactive at the calibrated optimum (clip bias vanishes)."""
    exact = ExactNet()
    znet = ZNet(width=32, depth=2, key=jax.random.PRNGKey(2))
    opt = optax.adam(2e-3)
    opt_state = opt.init(eqx.filter(znet, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(znet, opt_state, batch, key):
        loss, grads = eqx.filter_value_and_grad(
            lambda z: nwj_hit_loss(exact, z, batch, key, BIG)[0])(znet)
        updates, opt_state = opt.update(grads, opt_state)
        return eqx.apply_updates(znet, updates), opt_state, loss

    key = jax.random.PRNGKey(3)
    for s in range(2500):
        key, kb, kp = jax.random.split(key, 3)
        znet, opt_state, _ = step(znet, opt_state, gen_hit_batch(kb, 1500), kp)

    # (i) the population identity E_shuffled[e^{f-a}] -> 1 per stratum: verified with the
    # UNWINSORIZED estimator (the identity is about the true mean), averaged over a few
    # large batches to tame the lognormal-mean sampling noise.
    gaps = []
    clip_fracs = []
    for i in range(4):
        kb, kp = jax.random.split(jax.random.PRNGKey(5000 + i))
        _, aux_b = nwj_hit_loss(exact, znet, gen_hit_batch(kb, 20000), kp, BIG)
        gaps.append(np.asarray(aux_b.gap))
        # (ii) clip-inactivity receipt: evaluate clip_frac with the PRODUCTION clip.
        _, aux_c = nwj_hit_loss(exact, znet, gen_hit_batch(kb, 20000), kp, CLIP_C)
        clip_fracs.append(float(aux_c.clip_frac))
    gap = np.nanmean(np.stack(gaps), axis=0)
    assert (~np.isnan(gap)).sum() >= N_STRATA - 1     # all interior strata sampled
    assert np.nanmax(np.abs(gap)) < 0.05, f"bound gap not calibrated to 1: {gap}"
    # at the calibrated optimum the winsorization clip touches < 0.1% of shuffled rows:
    # the clip bias vanishes because it is (essentially) never engaged.
    assert np.mean(clip_fracs) < 1e-3, f"clip not inactive at optimum: {np.mean(clip_fracs):.4g}"

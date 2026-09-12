"""Exact-quadrature tilted-family MLE (hitman.wc.train.nwj).

Toy (per-hit, adapted from tests/test_identities.py): truth theta has t0 ~ N(0,30) in
slot 5 and E ~ U(0.5,9.5) in slot 6; each hit carries t ~ N(t0, S_T) in h[3] and a proxy
x ~ N(E, S_X) in h[0]. Both exact marginals are analytic, so ``ExactNet``'s logit IS the
true per-hit log ratio log p(x|theta)/p_marg(x) with partition Z(theta) == 1. The toy
observation grid (x, t) with its analytic marginal pmf gives an EXACT log Z by quadrature,
mirroring the production sensor x time grid — so the MLE loss is checked against first
principles with no Monte-Carlo estimator anywhere.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm

from hitman.nn import HitNet
from hitman.wc.train.nwj import (build_hit_grid, grid_logZ, hit_logZ_targets, mle_hit_loss)

S_T, SIG_T0, S_X = 10.0, 30.0, 2.0
S_MARG = float(np.sqrt(S_T**2 + SIG_T0**2))
E_LO, E_HI = 0.5, 9.5
N_HIT = 8


class ExactNet(eqx.Module):
    """The true per-hit log ratio (Z == 1)."""

    def __call__(self, h, th):
        lr_t = norm.logpdf(h[3], th[5], S_T) - norm.logpdf(h[3], 0.0, S_MARG)
        marg_x = (norm.cdf((h[0] - E_LO) / S_X) - norm.cdf((h[0] - E_HI) / S_X)) / (E_HI - E_LO)
        return lr_t + norm.logpdf(h[0], th[6], S_X) - jnp.log(marg_x)


class ScaledNet(eqx.Module):
    """c * ExactNet: the f-contrast scale knob (c dynamic leaf, differentiable)."""

    c: jnp.ndarray

    def __call__(self, h, th):
        return self.c * ExactNet()(h, th)


class TiltXNet(eqx.Module):
    """ExactNet + delta * x: a coherent tilt along the observable x = h[0] (delta a leaf)."""

    delta: jnp.ndarray

    def __call__(self, h, th):
        return ExactNet()(h, th) + self.delta * h[0]


class ScaledTiltNet(eqx.Module):
    """c * (ExactNet + delta * x) with c, delta static — for the deflation scan."""

    c: float = eqx.field(static=True)
    delta: float = eqx.field(static=True)

    def __call__(self, h, th):
        return self.c * (ExactNet()(h, th) + self.delta * h[0])


def build_toy_grid(nx=140, nt=140):
    """(x, t) quadrature grid with the analytic marginal pmf p_marg(x)*N(t;0,S_MARG),
    empty cells dropped. Returns (grid_obs (G,4), grid_logp (G,))."""
    xs = np.linspace(E_LO - 5 * S_X, E_HI + 5 * S_X, nx)
    ts = np.linspace(-4 * S_MARG, 4 * S_MARG, nt)
    px = (norm.cdf((xs - E_LO) / S_X) - norm.cdf((xs - E_HI) / S_X)) / (E_HI - E_LO)
    pt = np.exp(-ts**2 / (2 * S_MARG**2)) / (S_MARG * np.sqrt(2 * np.pi))
    P = np.asarray(px)[:, None] * np.asarray(pt)[None, :]
    P /= P.sum()
    X, T = np.meshgrid(xs, ts, indexing="ij")
    obs = np.stack([X.ravel(), 0 * X.ravel(), 0 * X.ravel(), T.ravel()], 1).astype(np.float32)
    keep = P.ravel() > 0
    return jnp.asarray(obs[keep]), jnp.asarray(np.log(P.ravel()[keep]).astype(np.float32))


def gen_hit_batch(key, n_events, n_hit=N_HIT):
    """Flat per-hit (obs (M,4), theta (M,7)) with M = n_events*n_hit (matched pairs)."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    t0 = SIG_T0 * jax.random.normal(k1, (n_events,))
    e = jax.random.uniform(k2, (n_events,), minval=E_LO, maxval=E_HI)
    theta = jnp.zeros((n_events, 7)).at[:, 5].set(t0).at[:, 6].set(e)
    t = t0[:, None] + S_T * jax.random.normal(k3, (n_events, n_hit))
    x = e[:, None] + S_X * jax.random.normal(k4, (n_events, n_hit))
    obs = jnp.stack([x, jnp.zeros_like(x), jnp.zeros_like(x), t], axis=-1).reshape(-1, 4)
    return obs, jnp.repeat(theta, n_hit, axis=0)


GO, GP = build_toy_grid()


def _subsample(theta, n, seed):
    return theta[jax.random.randint(jax.random.PRNGKey(seed), (n,), 0, theta.shape[0])]


# ---------------------------------------------------------------------------
# Test 1: exact-ratio toy — MLE gradient vanishes at the true ratio (exact partition)
# ---------------------------------------------------------------------------

def test_exact_ratio_mle_stationary():
    """log Z of the true ratio is 0 on the exact grid, and the MLE gradient in the
    f-scale direction vanishes at the true ratio (with the EXACT toy partition)."""
    thetas = gen_hit_batch(jax.random.PRNGKey(9), 2000)[1]
    logZ = grid_logZ(ExactNet(), _subsample(thetas, 256, 0), GO, GP, chunk=32)
    assert float(jnp.max(jnp.abs(logZ))) < 5e-3, f"log Z(exact) != 0: {float(jnp.max(jnp.abs(logZ)))}"

    def L(c, obs, hyp, thz):
        net = ScaledNet(c=c)
        return -jnp.mean(jax.vmap(net)(obs, hyp)) + jnp.mean(grid_logZ(net, thz, GO, GP, 32))

    grads = []
    for i in range(10):
        obs, hyp = gen_hit_batch(jax.random.PRNGKey(i), 3000)
        grads.append(float(jax.grad(L)(1.0, obs, hyp, _subsample(hyp, 256, 100 + i))))
    grads = np.array(grads)
    se = grads.std(ddof=1) / np.sqrt(len(grads))
    assert abs(grads.mean()) < 4 * se, f"MLE not stationary at true ratio: {grads.mean():.4f}±{se:.4f}"


# ---------------------------------------------------------------------------
# Test 2: deflation toy — restoring force = E_data[g] - E_model[g], EXACT via the grid
# ---------------------------------------------------------------------------

def test_restoring_force_exact_no_iw():
    """At a tilted f, the MLE gradient in the tilt direction equals -E_data[g] + E_model[g]
    with the model expectation computed EXACTLY from the grid (softmax over quadrature
    points), not by self-normalized importance weighting. The counterweight E_model is
    material — the two-expectation structure the one-sided penalties lacked — and deflating
    the f-contrast is not a descent direction."""
    delta = 0.15
    g = lambda o: o[0]                                    # tilt observable = hit x
    obs, hyp = gen_hit_batch(jax.random.PRNGKey(1), 8000)
    thz = _subsample(hyp, 512, 2)

    def L(d):
        net = TiltXNet(delta=d)
        return -jnp.mean(jax.vmap(net)(obs, hyp)) + jnp.mean(grid_logZ(net, thz, GO, GP, 64))

    grad = float(jax.grad(L)(delta))

    # Exact model expectation via the grid (no IW): p_f(grid|theta) = softmax(logp + f).
    net = TiltXNet(delta=delta)

    def e_model(theta):
        f = jax.vmap(lambda o: net(o, theta))(GO)
        return jnp.sum(jax.nn.softmax(GP + f) * jax.vmap(g)(GO))

    E_model = float(jnp.mean(jax.vmap(e_model)(thz)))
    E_data = float(jnp.mean(jax.vmap(g)(obs)))

    assert abs(grad - (-E_data + E_model)) < 1e-3, \
        f"grad {grad:.4f} != -E_data+E_model {-E_data + E_model:.4f}"
    assert abs(E_model - E_data) > 0.1, "counterweight not material (E_model ~ E_data)"

    # Deflation is not a descent direction at the tilted point: shrinking the f-contrast
    # (c<1) does not lower the MLE below its value at the true scale.
    def L_scale(c):
        net_c = ScaledTiltNet(c=c, delta=delta)
        return (-jnp.mean(jax.vmap(net_c)(obs, hyp))
                + jnp.mean(grid_logZ(net_c, thz, GO, GP, 64)))

    assert L_scale(0.85) > L_scale(1.0) - 1e-6, "deflation should not reduce the MLE loss"


# ---------------------------------------------------------------------------
# Test 3: grid_logZ (differentiable) matches hit_logZ_targets (numpy reference)
# ---------------------------------------------------------------------------

def test_grid_logZ_matches_reference():
    """The differentiable grid quadrature agrees with the independent numpy reference that
    integrates the full grid — they share the quadrature."""
    hitnet = HitNet(key=jax.random.PRNGKey(1))
    S, T = 24, 30
    pos = jax.random.normal(jax.random.PRNGKey(0), (S, 3)) * 500.0
    t = jnp.linspace(-100.0, 200.0, T)
    rng = np.random.default_rng(0)
    p1 = np.abs(rng.normal(size=(S, T)))
    p1[p1 < 0.2] = 0.0                                   # some empty cells
    p1 = p1 / p1.sum()
    go, gp = build_hit_grid(pos, t, jnp.asarray(p1, jnp.float32))
    with np.errstate(divide="ignore"):
        full_logp = jnp.asarray(np.where(p1 > 0, np.log(p1), -np.inf), jnp.float32)
    thetas = jax.random.uniform(jax.random.PRNGKey(3), (16, 7))
    a = np.asarray(grid_logZ(hitnet, thetas, go, gp, chunk=4))
    b = hit_logZ_targets(hitnet, thetas, pos, t, full_logp, chunk=4)
    assert np.max(np.abs(a - b)) < 1e-4, f"grid_logZ vs reference max|diff|={np.max(np.abs(a - b))}"


# ---------------------------------------------------------------------------
# Test 4: MLE is bounded below — a localized f-spike is priced by log Z (no runaway)
# ---------------------------------------------------------------------------

def test_mle_bounded_below_on_spike():
    """A net that puts f = +amp on one grid cell cannot lower the loss without bound: log Z
    grows ~amp while the matched term barely moves (data rarely lands there), so the loss
    is coercive in amp (bounded below). This is exactly what the linear-tail NWJ lacked."""
    obs, hyp = gen_hit_batch(jax.random.PRNGKey(1), 6000)
    thz = _subsample(hyp, 256, 5)
    cell = int(np.argmax(np.asarray(GP)))                # a high-density (populated) cell

    class SpikeNet(eqx.Module):
        amp: jnp.ndarray

        def __call__(self, h, th):
            return self.amp * jnp.exp(-jnp.sum((h - GO[cell]) ** 2) / (2 * 5.0**2))

    def L(amp):
        net = SpikeNet(amp=amp)
        return -jnp.mean(jax.vmap(net)(obs, hyp)) + jnp.mean(grid_logZ(net, thz, GO, GP, 64))

    vals = [float(L(a)) for a in (0.0, 5.0, 15.0, 30.0)]
    assert all(np.isfinite(vals)), f"loss not finite on spike: {vals}"
    assert vals[0] < vals[1] < vals[2] < vals[3], f"loss not coercive in amp: {vals}"
    assert float(jax.grad(L)(20.0)) > 0.0, "loss should be coercive (dL/damp > 0) — no runaway"

"""Moment vectors for GMM-weighted identity training and IM-test receipts.

For a TRUE ratio, at every theta (moments computed at simulator truth):
  score:    E[g] = 0,                 g = grad_theta log r
  Bartlett: E[B] = 0,                 B = hess_theta log r + g g^T
and both hold for any theta-measurable projection (instrument), e.g. the event's
own direction d_hat(theta) — the ray-projected components below. Stratification in
true E = fixed indicator instruments.

Two stacked vectors per event (order matters — receipts and W share it):

LEAN (19, trained against; physics-selected, stable W):
  [ g_0..g_6, g.d_hat,  B_00..B_66 (diag),  B_tE,  B_rr, B_Er, B_tr ]
FULL (39, test-only; adds all Bartlett off-diagonals):
  [ g_0..g_6, g.d_hat,  vech(B) (28, upper-tri row-major),  B_rr, B_Er, B_tr ]

Component convention: theta = (x, y, z, zen, az, t, E); r-suffix = projection of the
spatial block onto d_hat = (sin zen cos az, sin zen sin az, cos zen).
"""

from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from hitman.spec import WC_HYP_SPEC

# Parameter names/dimension come from the injected hypothesis spec (WC by default) rather
# than a hardcoded 7 — swap ``WC_HYP_SPEC`` for another :class:`hitman.spec.HypSpec` to
# retarget the identity vectors.
_NDIM = WC_HYP_SPEC.dim
VECH_IDX = np.array([(i, j) for i in range(_NDIM) for j in range(i, _NDIM)])  # 28 pairs
_P = list(WC_HYP_SPEC.names)


def direction(theta):
    """Water-Cherenkov ray instrument: unit direction from (zen, az)."""
    zen, az = theta[WC_HYP_SPEC.zenith], theta[WC_HYP_SPEC.azimuth]
    s = jnp.sin(zen)
    return jnp.stack([s * jnp.cos(az), s * jnp.sin(az), jnp.cos(zen)])


class ProjectionInstrument(NamedTuple):
    """An injectable projection instrument ``v(theta)`` for the moment/identity vectors.

    ``project`` maps ``theta`` to a unit vector spanning the ``spatial`` parameter block; the
    Bartlett rows are then projected onto it, and the two scalar ``partners`` are the
    parameters whose cross-coupling to the projection is reported (WC: t and E). Injecting a
    different instrument (e.g. a shower-axis direction with partners ``(dt, X_max)``) retargets
    "project onto the ray" to "project onto any instrument" without touching the receipt
    numerics of the default. ``partner_names`` label the receipt components.

    DESIGN proposal 4. The WC default :data:`RAY_INSTRUMENT` reproduces the pre-refactor
    ``lean_vector``/``full_vector`` bit-for-bit (locked by ``tests/test_moments_instrument.py``).
    """

    project: Callable
    spatial: Tuple[int, ...] = (0, 1, 2)
    partners: Tuple[int, int] = (5, 6)
    partner_names: Tuple[str, str] = ("t", "E")


# Water-Cherenkov default: the (zen, az) ray over the (x, y, z) spatial block, with the
# t (5) and E (6) parameters as the reported cross-coupling partners.
RAY_INSTRUMENT = ProjectionInstrument(
    project=direction,
    spatial=(0, 1, 2),
    partners=(WC_HYP_SPEC.index("t"), WC_HYP_SPEC.index("E")),
    partner_names=("t", "E"),
)


def moment_names(names=None, instrument: ProjectionInstrument = RAY_INSTRUMENT):
    """(LEAN_NAMES, FULL_NAMES) for the given parameter names + projection instrument.

    Order matches ``lean_vector``/``full_vector`` exactly (receipts and W share it)."""
    names = _P if names is None else list(names)
    p_lo, p_hi = instrument.partner_names
    head = ["g_" + n for n in names] + ["g_ray"]
    lean = (head + ["B_" + n + n for n in names]
            + [f"B_{p_lo}{p_hi}", "B_rayray", f"B_{p_hi}ray", f"B_{p_lo}ray"])
    idx = np.array([(i, j) for i in range(len(names)) for j in range(i, len(names))])
    full = (head + [f"B_{names[i]}{names[j]}" for i, j in idx]
            + ["B_rayray", f"B_{p_hi}ray", f"B_{p_lo}ray"])
    return lean, full


LEAN_NAMES, FULL_NAMES = moment_names(_P, RAY_INSTRUMENT)
N_LEAN, N_FULL = len(LEAN_NAMES), len(FULL_NAMES)


def grad_hess(loglik, theta, ev):
    """(g, H) of a scalar loglik(theta, ev) at theta — one event."""
    g = jax.grad(loglik)(theta, ev)
    H = jax.hessian(loglik)(theta, ev)
    return g, H


def _ray_entries(B, d, instrument: ProjectionInstrument = RAY_INSTRUMENT):
    sp = jnp.asarray(instrument.spatial)
    p_lo, p_hi = instrument.partners
    Bss = B[sp][:, sp]
    return jnp.stack([d @ Bss @ d, B[p_hi][sp] @ d, B[p_lo][sp] @ d])


def lean_vector(g, H, theta, instrument: ProjectionInstrument = RAY_INSTRUMENT):
    B = H + jnp.outer(g, g)
    d = instrument.project(theta)
    sp = jnp.asarray(instrument.spatial)
    p_lo, p_hi = instrument.partners
    return jnp.concatenate([
        g, (g[sp] @ d)[None], jnp.diag(B), B[p_lo, p_hi][None],
        _ray_entries(B, d, instrument)])


def full_vector(g, H, theta, instrument: ProjectionInstrument = RAY_INSTRUMENT):
    B = H + jnp.outer(g, g)
    d = instrument.project(theta)
    sp = jnp.asarray(instrument.spatial)
    vech = B[VECH_IDX[:, 0], VECH_IDX[:, 1]]
    return jnp.concatenate([g, (g[sp] @ d)[None], vech,
                            _ray_entries(B, d, instrument)])


def shrunk_inverse(S, n, ridge_frac=0.05):
    """Ledoit–Wolf-flavored shrinkage inverse: S_shrunk = (1-a)S + a*diag(S),
    a from a simple n-scaled rule plus a diagonal ridge floor. Deterministic,
    conservative — W stability beats W optimality (an unstable W^-1 direction is
    worse than a mild efficiency loss)."""
    d = np.sqrt(np.clip(np.diag(S), 1e-30, None))
    C = S / np.outer(d, d)                      # correlation
    p = S.shape[0]
    a = min(0.9, p / max(n, 1) + ridge_frac)
    C = (1 - a) * C + a * np.eye(p)
    Ci = np.linalg.inv(C)
    return Ci / np.outer(d, d)


def estimate_blocks(m, strata, n_strata, ridge_frac=0.05):
    """Per-stratum mean, covariance, shrunk inverse W, and P99.9 bounds.

    m: (N, M) per-event moment vectors; strata: (N,) int. Returns dict of arrays
    keyed mu (K,M), S (K,M,M), W (K,M,M), bounds (K,M), n (K,)."""
    M = m.shape[1]
    out = {k: np.zeros(s) for k, s in [("mu", (n_strata, M)), ("S", (n_strata, M, M)),
                                       ("W", (n_strata, M, M)), ("bounds", (n_strata, M))]}
    out["n"] = np.zeros(n_strata)
    for k in range(n_strata):
        mk = m[strata == k]
        out["n"][k] = len(mk)
        out["mu"][k] = mk.mean(0)
        out["S"][k] = np.cov(mk.T)
        out["W"][k] = shrunk_inverse(out["S"][k], len(mk), ridge_frac)
        out["bounds"][k] = np.quantile(np.abs(mk), 0.999, axis=0)
    return out

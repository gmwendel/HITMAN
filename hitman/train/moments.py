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

import jax
import jax.numpy as jnp
import numpy as np

VECH_IDX = np.array([(i, j) for i in range(7) for j in range(i, 7)])  # 28 pairs
_P = ["x", "y", "z", "zen", "az", "t", "E"]
LEAN_NAMES = (
    ["g_" + n for n in _P]
    + ["g_ray"]
    + ["B_" + n + n for n in _P]
    + ["B_tE", "B_rayray", "B_Eray", "B_tray"]
)
FULL_NAMES = (
    LEAN_NAMES[:8]
    + [f"B_{_P[i]}{_P[j]}" for i, j in VECH_IDX]
    + ["B_rayray", "B_Eray", "B_tray"]
)
N_LEAN, N_FULL = len(LEAN_NAMES), len(FULL_NAMES)


def direction(theta):
    zen, az = theta[3], theta[4]
    s = jnp.sin(zen)
    return jnp.stack([s * jnp.cos(az), s * jnp.sin(az), jnp.cos(zen)])


def grad_hess(loglik, theta, ev):
    """(g, H) of a scalar loglik(theta, ev) at theta — one event."""
    g = jax.grad(loglik)(theta, ev)
    H = jax.hessian(loglik)(theta, ev)
    return g, H


def _ray_entries(B, d):
    return jnp.stack([d @ B[:3, :3] @ d, B[6, :3] @ d, B[5, :3] @ d])


def lean_vector(g, H, theta):
    B = H + jnp.outer(g, g)
    d = direction(theta)
    return jnp.concatenate([
        g, (g[:3] @ d)[None], jnp.diag(B), B[5, 6][None], _ray_entries(B, d)])


def full_vector(g, H, theta):
    B = H + jnp.outer(g, g)
    d = direction(theta)
    vech = B[VECH_IDX[:, 0], VECH_IDX[:, 1]]
    return jnp.concatenate([g, (g[:3] @ d)[None], vech, _ray_entries(B, d)])


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

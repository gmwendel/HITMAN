"""Per-hypothesis normalization Z(theta) from a histogrammed hit marginal.

p(x) over (sensor, time) is 2-D and histogrammable to sub-percent precision from the
training set. Z(theta) = sum_grid r_hat(x, theta) p_hat(x) must equal 1 for a valid
ratio; dividing by it converts the NRE output into a proper conditional density and
— because E_{x~p_hat(.|theta)}[grad log r_hat] = grad log Z exactly — subtracting
N_hits * log Z(theta) from the event NLL is a derived first-order bias correction
(design report, Addendum 7). Per-term caveat: the full event correction is
N * log Z_hit + log Z_charge; this module implements the hit term.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class MarginalGrid(NamedTuple):
    p_grid: jnp.ndarray    # (n_sensors, n_tbins) joint pmf over (sensor, t-bin)
    t_centers: jnp.ndarray  # (n_tbins,)
    pmt_pos: jnp.ndarray   # (n_sensors, 3)
    coverage: float        # fraction of the sample inside the t-grid


def build_marginal_grid(store, *, n_sample: int = 30_000_000, time_sigma: float = 50.0,
                        t_lo: float = -160.0, t_hi: float = 260.0, dt: float = 0.5,
                        seed: int = 0) -> MarginalGrid:
    """Histogram the (augmented) training hit marginal on a (sensor, time) grid."""
    rng = np.random.default_rng(seed)
    n = min(n_sample, store.n_hits)
    rows = np.sort(rng.choice(store.n_hits, n, replace=False))
    pmt = np.asarray(store.pmt_id[rows])
    ev = np.asarray(store.event_id[rows])
    shift = rng.normal(0.0, time_sigma, store.n_events).astype(np.float32)
    t = np.asarray(store.hits[rows, 3]) + shift[ev]
    edges = np.arange(t_lo, t_hi + dt / 2, dt)
    n_pmts = len(np.asarray(store.pmt_pos))
    h = np.zeros((n_pmts, len(edges) - 1))
    for s in range(n_pmts):
        h[s], _ = np.histogram(t[pmt == s], edges)
    coverage = h.sum() / n
    return MarginalGrid(
        p_grid=jnp.asarray(h / h.sum(), jnp.float32),
        t_centers=jnp.asarray(0.5 * (edges[1:] + edges[:-1]), jnp.float32),
        pmt_pos=jnp.asarray(store.pmt_pos, jnp.float32),
        coverage=float(coverage),
    )


def z_of_theta(hitnet, grid: MarginalGrid, theta: jnp.ndarray,
               chunk_size: int = 100_000) -> jnp.ndarray:
    """Z(theta) = E_{p_hat(x)}[r_hat(x, theta)] evaluated on the grid (chunked)."""
    n_s, n_t = grid.p_grid.shape
    obs_style = getattr(hitnet, "obs_style", "xyz")
    if obs_style == "id_t":
        ids = jnp.repeat(jnp.arange(n_s, dtype=jnp.int32), n_t)
        ts = jnp.tile(grid.t_centers, n_s)
        f = jax.jit(jax.vmap(lambda i, t_: hitnet((i, t_), theta)))
        parts = [f(ids[i:i + chunk_size], ts[i:i + chunk_size])
                 for i in range(0, n_s * n_t, chunk_size)]
    else:
        g = jnp.concatenate([jnp.repeat(grid.pmt_pos, n_t, axis=0),
                             jnp.tile(grid.t_centers, n_s)[:, None]], axis=1)
        f = jax.jit(jax.vmap(lambda h: hitnet(h, theta)))
        parts = [f(g[i:i + chunk_size]) for i in range(0, n_s * n_t, chunk_size)]
    r = jnp.exp(jnp.concatenate(parts)).reshape(n_s, n_t)
    return jnp.sum(r * grid.p_grid)


def znll(nll, hitnet, grid: MarginalGrid, n_hits) -> callable:
    """Wrap an event NLL with the derived normalization correction.

    corrected(theta) = nll(theta) + n_hits * log Z_hit(theta). Differentiable, so it
    corrects MLE/NUTS surfaces (unlike the theta-constant log p_hat(x) term). NOTE:
    a Z evaluation costs one grid pass (~1 training batch) per call — use at polish/
    posterior stages, or amortize Z with a small fitted surrogate for inner loops.
    """
    def corrected(theta):
        return nll(theta) + n_hits * jnp.log(z_of_theta(hitnet, grid, theta))

    return corrected

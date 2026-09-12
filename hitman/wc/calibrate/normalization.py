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
    pmt = np.asarray(store.pmt_id[rows]).astype(np.int64)
    ev = np.asarray(store.event_id[rows])
    shift = rng.normal(0.0, time_sigma, store.n_events).astype(np.float32)
    t = np.asarray(store.hits[rows, 3]) + shift[ev]
    edges = np.arange(t_lo, t_hi + dt / 2, dt)
    n_pmts = len(np.asarray(store.pmt_pos))
    n_tb = len(edges) - 1
    # single flat bincount: ~7x faster than a per-PMT histogram loop, identical output
    inside = (t >= t_lo) & (t < edges[-1])
    tbin = ((t[inside] - t_lo) / dt).astype(np.int64)
    h = np.bincount(pmt[inside] * n_tb + np.minimum(tbin, n_tb - 1),
                    minlength=n_pmts * n_tb).reshape(n_pmts, n_tb).astype(np.float64)
    coverage = h.sum() / n
    return MarginalGrid(
        p_grid=jnp.asarray(h / h.sum(), jnp.float32),
        t_centers=jnp.asarray(0.5 * (edges[1:] + edges[:-1]), jnp.float32),
        pmt_pos=jnp.asarray(store.pmt_pos, jnp.float32),
        coverage=float(coverage),
    )


def make_z_fn(hitnet, grid: MarginalGrid, chunk_size: int = 2**16):
    """Z(theta) as a single jitted function with the grid observations HOISTED.

    Building the grid points and a fresh ``jax.jit`` wrapper inside every call costs
    ~110 ms/call of pure retrace/dispatch overhead (audit finding 3) — fatal if Z sits
    in an optimizer/sampler inner loop. Here the grid is materialized once, the chunk
    loop is a ``lax.map`` inside ONE jitted function, and repeated calls are sub-ms.
    """
    n_s, n_t = grid.p_grid.shape
    n_pts = n_s * n_t
    n_chunks = -(-n_pts // chunk_size)
    pad = n_chunks * chunk_size - n_pts
    obs_style = getattr(hitnet, "obs_style", "xyz")
    if obs_style == "id_t":
        ids = jnp.repeat(jnp.arange(n_s, dtype=jnp.int32), n_t)
        ts = jnp.tile(grid.t_centers, n_s)
        chunks = (jnp.pad(ids, (0, pad)).reshape(n_chunks, chunk_size),
                  jnp.pad(ts, (0, pad)).reshape(n_chunks, chunk_size))

        def eval_chunk(c, theta):
            i, t_ = c
            return jax.vmap(lambda ii, tt: hitnet((ii, tt), theta))(i, t_)
    else:
        g = jnp.concatenate([jnp.repeat(grid.pmt_pos, n_t, axis=0),
                             jnp.tile(grid.t_centers, n_s)[:, None]], axis=1)
        chunks = jnp.pad(g, ((0, pad), (0, 0))).reshape(n_chunks, chunk_size, -1)

        def eval_chunk(c, theta):
            return jax.vmap(lambda h: hitnet(h, theta))(c)

    p_flat = jnp.pad(grid.p_grid.reshape(-1), (0, pad)).reshape(n_chunks, chunk_size)

    @jax.jit
    def z_fn(theta):
        contrib = jax.lax.map(
            lambda cp: jnp.sum(jnp.exp(eval_chunk(cp[0], theta)) * cp[1]),
            (chunks, p_flat))
        return jnp.sum(contrib)

    return z_fn


def z_of_theta(hitnet, grid: MarginalGrid, theta: jnp.ndarray,
               chunk_size: int = 2**16) -> jnp.ndarray:
    """Z(theta) = E_{p_hat(x)}[r_hat(x, theta)] on the grid (one-shot convenience).

    For repeated evaluation (optimizer loops, grad/hessian at many points) use
    ``make_z_fn`` — this convenience wrapper rebuilds the grid every call.
    """
    return make_z_fn(hitnet, grid, chunk_size)(theta)


class ChargeGrid(NamedTuple):
    p: jnp.ndarray        # (n_bins,) pmf over charge-observable bins
    centers: jnp.ndarray  # (n_bins, obs_dim) bin-center charge observations
    coverage: float


def build_charge_grid(store, *, n_q_bins: int = 200, q_hi_quantile: float = 0.999,
                      n_sample: int = 2_000_000, seed: int = 0) -> ChargeGrid:
    """Histogram the per-event charge observable (q_tot, n_hits) marginal.

    n_hits is discrete (exact integer bins up to its sampled max); q_tot gets
    ``n_q_bins`` linear bins to its ``q_hi_quantile``. Completes the per-term Z:
    event correction = N*log Z_hit(theta) + log Z_charge(theta) (TODO item 18).
    """
    rng = np.random.default_rng(seed)
    n = min(n_sample, store.n_events)
    ev = np.sort(rng.choice(store.n_events, n, replace=False))
    c = np.asarray(store.charge[ev])          # (n, 2) = (q_tot, n_hits)
    q, k = c[:, 0], c[:, 1].astype(np.int64)
    q_hi = np.quantile(q, q_hi_quantile)
    k_max = int(k.max())
    qbin = np.clip((q / q_hi * n_q_bins).astype(np.int64), 0, n_q_bins - 1)
    inside = q <= q_hi
    h = np.bincount(qbin[inside] * (k_max + 1) + k[inside],
                    minlength=n_q_bins * (k_max + 1)).astype(np.float64)
    occ = h > 0                               # keep only occupied cells (sparse grid)
    q_centers = (np.arange(n_q_bins) + 0.5) * (q_hi / n_q_bins)
    kk, qq = np.meshgrid(np.arange(k_max + 1), q_centers)
    centers = np.stack([qq.reshape(-1)[occ], kk.reshape(-1)[occ]], axis=1)
    return ChargeGrid(p=jnp.asarray(h[occ] / h[occ].sum(), jnp.float32),
                      centers=jnp.asarray(centers, jnp.float32),
                      coverage=float(inside.mean()))


def make_z_charge_fn(chargenet, cgrid: ChargeGrid):
    """Z_charge(theta) = E_{p_hat(c)}[r_hat_c(c, theta)], jitted, grid hoisted."""

    @jax.jit
    def z_fn(theta):
        logits = jax.vmap(lambda c: chargenet(c, theta))(cgrid.centers)
        return jnp.sum(jnp.exp(logits) * cgrid.p)

    return z_fn


def znll(nll, hitnet, grid: MarginalGrid, n_hits, chargenet=None,
         charge_grid: ChargeGrid = None) -> callable:
    """Wrap an event NLL with the derived normalization correction.

    corrected(theta) = nll(theta) + n_hits * log Z_hit(theta)
                       [+ log Z_charge(theta) when chargenet+charge_grid are given].
    Differentiable, so it corrects MLE/NUTS surfaces (unlike the theta-constant
    log p_hat(x) term). Grid evaluations are hoisted+jitted (make_z_fn), so the
    wrapped NLL is safe in optimizer/sampler inner loops (one grid pass per call).
    """
    z_hit = make_z_fn(hitnet, grid)
    z_charge = make_z_charge_fn(chargenet, charge_grid) if chargenet is not None else None

    def corrected(theta):
        out = nll(theta) + n_hits * jnp.log(z_hit(theta))
        if z_charge is not None:
            out = out + jnp.log(z_charge(theta))
        return out

    return corrected

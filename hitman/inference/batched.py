"""Batched event inference: padded packing, multistart MLE, lockstep NUTS.

This is the production inference path (audit finding 1): `multistart_mle` on ragged
per-event hits compiles once per DISTINCT hit count (seconds each, hundreds of counts
per survey), so every survey script re-implemented the same padded/masked pattern by
hand. This module is that validated pattern, promoted:

- ``pad_events``: ragged hits -> fixed (N, n_pad) padded arrays + mask (one compile
  total). Events with more than ``n_pad`` hits CANNOT be represented and are dropped
  — LOUDLY (the scripts dropped them silently, a low-K selection bias).
- ``make_padded_nll``: obs_style-dispatched masked event NLL builder.
- ``batched_multistart_mle``: chart-space multistart (no zenith pole / azimuth wrap),
  bounded Adam descent, top-k seed screening; returns per-event MLE plus the ranked
  minima needed to seed multi-chain NUTS.
- ``batched_nuts``: window-adapt ONCE on a representative event (events at a test
  point are iid — step size/mass matrix transfer), then run all events x chains in
  lockstep with jit(vmap(scan)); per-event split R-hat on a monitored coordinate.
"""

import warnings
from typing import NamedTuple

import blackjax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from hitman.inference.mle import CHART_SCALE, chart_from_theta, cylinder_seeds, theta_from_chart
from hitman.inference.nuts import box_log_prior


class PaddedEvents(NamedTuple):
    hits: jnp.ndarray      # (N, n_pad, 4) xyzt (zero-padded)
    pmt_id: jnp.ndarray    # (N, n_pad) int32
    t: jnp.ndarray         # (N, n_pad) float32
    mask: jnp.ndarray      # (N, n_pad) float32 1/0
    charge: jnp.ndarray    # (N, 2)
    event_indices: np.ndarray  # indices (into the source batch) actually packed
    dropped_fraction: float    # fraction of requested events with > n_pad hits


def pad_events(batch, indices=None, n_pad: int = 160) -> PaddedEvents:
    """Pack ragged events into padded arrays (vectorized; no per-event Python loop).

    ``batch`` is an EventBatch-like object with .hits (n_hits, 4), .pmt_id, .charge
    (n_events, 2 = q_tot, n_hits). Events exceeding ``n_pad`` hits are dropped with a
    warning — account for ``dropped_fraction`` in any efficiency/coverage statement.
    """
    counts = np.asarray(batch.charge[:, 1]).astype(np.int64)
    if indices is None:
        indices = np.arange(len(counts))
    indices = np.asarray(indices)
    fits = counts[indices] <= n_pad
    dropped = 1.0 - float(fits.mean()) if len(indices) else 0.0
    if dropped > 0:
        warnings.warn(f"pad_events: dropping {(~fits).sum()} of {len(indices)} events "
                      f"({100*dropped:.2f}%) with > {n_pad} hits — low-K selection bias "
                      f"if ignored", stacklevel=2)
    keep = indices[fits]
    offsets = np.concatenate([[0], np.cumsum(counts)])
    n = counts[keep]
    # (N, n_pad) flat row index per slot, clipped into range; mask kills padding slots
    slot = np.arange(n_pad)[None, :]
    rows = np.minimum(offsets[keep][:, None] + slot, len(np.asarray(batch.hits)) - 1)
    mask = (slot < n[:, None]).astype(np.float32)
    hits = np.asarray(batch.hits)[rows] * mask[:, :, None]
    ids = (np.asarray(batch.pmt_id)[rows] * mask).astype(np.int32)
    return PaddedEvents(
        hits=jnp.asarray(hits, jnp.float32),
        pmt_id=jnp.asarray(ids),
        t=jnp.asarray(hits[:, :, 3], jnp.float32),
        mask=jnp.asarray(mask),
        charge=jnp.asarray(np.asarray(batch.charge)[keep], jnp.float32),
        event_indices=keep,
        dropped_fraction=dropped,
    )


def make_padded_nll(hitnet, chargenet):
    """Masked single-event NLL over padded slots; obs_style dispatched once."""
    obs_style = getattr(hitnet, "obs_style", "xyz")

    def nll(theta, ev: tuple):
        hits, ids, t, mask, charge = ev
        if obs_style == "id_t":
            logits = jax.vmap(lambda i, t_: hitnet((i, t_), theta))(ids, t)
        else:
            logits = jax.vmap(lambda h: hitnet(h, theta))(hits)
        return -(jnp.sum(logits * mask) + chargenet(charge, theta))

    return nll


class MLEConfig(NamedTuple):
    n_seeds: int = 256
    top_k: int = 16
    descent_steps: int = 250
    learning_rate: float = 3e-2
    radius: float = 800.0
    half_height: float = 800.0
    t_range: tuple = (-10.0, 10.0)
    e_range: tuple = (0.5, 8.0)


def _bounds(cfg: MLEConfig):
    big = 1e9
    lo = jnp.array([-cfg.radius, -cfg.radius, -cfg.half_height, -big, -big, -big,
                    cfg.t_range[0], cfg.e_range[0]]) / CHART_SCALE
    hi = jnp.array([cfg.radius, cfg.radius, cfg.half_height, big, big, big,
                    cfg.t_range[1], cfg.e_range[1]]) / CHART_SCALE
    return lo, hi


class MLEResult(NamedTuple):
    theta: jnp.ndarray      # (N, 7) best-fit hypothesis
    chart_minima: jnp.ndarray  # (N, top_k_kept, 8) ranked chart-space minima (NUTS seeds)
    sigma: jnp.ndarray      # (N, 7) Fisher (pinv-Hessian) uncertainties


def batched_multistart_mle(hitnet, chargenet, padded: PaddedEvents, *, key,
                           cfg: MLEConfig = MLEConfig(), keep_minima: int = 8,
                           chunk: int = 100) -> MLEResult:
    """Chart-space multistart MLE for every packed event (one compile, chunked vmap)."""
    nll = make_padded_nll(hitnet, chargenet)
    seeds_u = jax.vmap(chart_from_theta)(cylinder_seeds(
        key, cfg.n_seeds, radius=cfg.radius, half_height=cfg.half_height,
        t_range=cfg.t_range, e_range=(cfg.e_range[0] + 0.5, cfg.e_range[1] - 1.0)))
    u_lo, u_hi = _bounds(cfg)
    opt = optax.adam(cfg.learning_rate)

    @jax.jit
    def mle_event(ev):
        nll_u = lambda u: nll(theta_from_chart(u * CHART_SCALE), ev)
        seed_nlls = jax.vmap(lambda u: nll_u(u / CHART_SCALE))(seeds_u)
        _, top = jax.lax.top_k(-seed_nlls, cfg.top_k)

        def descend(u0):
            def step(carry, _):
                u, s = carry
                g = jax.grad(nll_u)(u)
                du, s = opt.update(g, s)
                return (jnp.clip(optax.apply_updates(u, du), u_lo, u_hi), s), None

            (u, _), _ = jax.lax.scan(step, (u0, opt.init(u0)), None,
                                     length=cfg.descent_steps)
            return u, nll_u(u)

        us, nlls = jax.vmap(descend)(seeds_u[top] / CHART_SCALE)
        order = jnp.argsort(nlls)
        u_best = us[order[0]]
        theta = theta_from_chart(u_best * CHART_SCALE)
        H = jax.hessian(lambda th: nll(th, ev))(theta)
        cov = jnp.linalg.pinv(0.5 * (H + H.T))
        sigma = jnp.sqrt(jnp.clip(jnp.diag(cov), 1e-12))
        return theta, us[order[:keep_minima]], sigma

    mle_b = jax.jit(jax.vmap(mle_event))
    outs = []
    n = padded.hits.shape[0]
    for i in range(0, n, chunk):
        ev = (padded.hits[i:i+chunk], padded.pmt_id[i:i+chunk], padded.t[i:i+chunk],
              padded.mask[i:i+chunk], padded.charge[i:i+chunk])
        outs.append(mle_b(ev))
    return MLEResult(theta=jnp.concatenate([o[0] for o in outs]),
                     chart_minima=jnp.concatenate([o[1] for o in outs]),
                     sigma=jnp.concatenate([o[2] for o in outs]))


class BatchedNUTSResult(NamedTuple):
    samples: np.ndarray     # (N, n_chains*(n_steps-n_burn), 7) theta draws
    post_mean: np.ndarray   # (N, 7)
    post_std: np.ndarray    # (N, 7)
    divergent_fraction: np.ndarray  # (N,)
    rhat: np.ndarray        # (N,) split-style R-hat on the monitored coordinate


def batched_nuts(hitnet, chargenet, padded: PaddedEvents, mle: MLEResult, *, key,
                 cfg: MLEConfig = MLEConfig(), n_steps: int = 700, n_burn: int = 200,
                 n_chains: int = 8, gauge_stiffness: float = 50.0,
                 prior_stiffness: float = 1e2, rhat_coord: int = 2,
                 chunk: int = 100, max_doublings: int = 10) -> BatchedNUTSResult:
    """Lockstep multi-chain NUTS for every packed event, seeded from the MLE minima.

    Chains sample in chart space with the |d| gauge term (the NLL is invariant to the
    direction norm — an improper flat mode that breaks HMC without it) and a smooth
    box prior matching the MLE bounds.

    ``max_doublings`` caps the NUTS tree depth. In LOCKSTEP vmapped chains every
    lane pays the deepest tree in the batch each draw (measured: mean 8.4
    leapfrogs/draw, batch max 55 — a 4-6x padding waste at the blackjax default
    of 10 doublings). A cap of ~5 (32 leapfrogs) covers the p99 tree at our
    geometry; verify with R-hat/divergence receipts whenever changing it.
    """
    nll = make_padded_nll(hitnet, chargenet)
    u_lo, u_hi = _bounds(cfg)
    prior = box_log_prior(u_lo * CHART_SCALE, u_hi * CHART_SCALE, stiffness=prior_stiffness)

    def make_logd(ev):
        def logd(u):
            gauge = -gauge_stiffness * (jnp.linalg.norm(u[3:6]) - 1.0) ** 2
            return (-nll(theta_from_chart(u * CHART_SCALE), ev)
                    + prior(u * CHART_SCALE) + gauge)
        return logd

    ev0 = (padded.hits[0], padded.pmt_id[0], padded.t[0], padded.mask[0], padded.charge[0])
    key, akey = jax.random.split(key)
    adapt = blackjax.window_adaptation(blackjax.nuts, make_logd(ev0),
                                       target_acceptance_rate=0.8,
                                       max_num_doublings=max_doublings)
    (_, params), _ = adapt.run(akey, mle.chart_minima[0, 0], num_steps=400)

    def sample_event(ev, u0s, k):
        def one_chain(u0, ck):
            # params already carries max_num_doublings (window_adaptation forwards
            # extra kwargs into its returned parameters dict)
            kernel = blackjax.nuts(make_logd(ev), **params)
            state = kernel.init(u0)

            def step(st, kk):
                st, info = kernel.step(kk, st)
                return st, (st.position, info.is_divergent)

            _, (pos, div) = jax.lax.scan(step, state, jax.random.split(ck, n_steps))
            return jax.vmap(theta_from_chart)(pos[n_burn:] * CHART_SCALE), div.mean()

        thetas, divs = jax.vmap(one_chain)(u0s, jax.random.split(k, n_chains))
        cm = thetas[:, :, rhat_coord].mean(1)
        W = thetas[:, :, rhat_coord].var(1).mean()
        m = thetas.shape[1]
        rhat = jnp.sqrt((W * (m - 1) / m + cm.var()) / jnp.clip(W, 1e-30))
        flat = thetas.reshape(-1, 7)
        return flat, flat.mean(0), flat.std(0), divs.mean(), rhat

    se_b = jax.jit(jax.vmap(sample_event))
    n = padded.hits.shape[0]
    u0s = mle.chart_minima[:, :n_chains]
    keys = jax.random.split(key, n)
    acc = []
    for i in range(0, n, chunk):
        ev = (padded.hits[i:i+chunk], padded.pmt_id[i:i+chunk], padded.t[i:i+chunk],
              padded.mask[i:i+chunk], padded.charge[i:i+chunk])
        acc.append(se_b(ev, u0s[i:i+chunk], keys[i:i+chunk]))
    return BatchedNUTSResult(
        samples=np.concatenate([np.asarray(a[0]) for a in acc]).astype(np.float32),
        post_mean=np.concatenate([np.asarray(a[1]) for a in acc]),
        post_std=np.concatenate([np.asarray(a[2]) for a in acc]),
        divergent_fraction=np.concatenate([np.asarray(a[3]) for a in acc]),
        rhat=np.concatenate([np.asarray(a[4]) for a in acc]),
    )

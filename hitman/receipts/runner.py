"""L2 receipt runner: one command -> receipts.json for a model dir + test points.

    python -m hitman.receipts <model_dir> <testpoint.root>[:tag:x,y,z,zen,az,t,E] ...

The heavy, model/JAX-facing loaders live here; the numerical receipt logic lives in the
sibling modules (``reweight``, ``chi2``, ``forward``, ``score``, ``mle``) and is unit-
tested with synthetic stand-ins. The seam is deliberate: :func:`compute_forward_block`
takes precomputed log-weights and MC arrays, so the whole forward battery is exercised
in tests with tiny random nets and no ROOT file.
"""

import json
import os
import subprocess
import time
from typing import Optional

import numpy as np

from hitman.receipts import forward as fwd
from hitman.receipts import schema
from hitman.receipts.reweight import self_normalized_weights

TIME_SIGMA = 50.0  # +/-50 ns event-time augmentation the classifier was calibrated with


# ---- provenance --------------------------------------------------------------


def git_sha(cwd: Optional[str] = None) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


# ---- forward receipt core (numpy-only; the tested seam) ----------------------


def z_rings(pmt_pos, n_rings: int):
    """Assign each PMT to a z-ring with roughly equal geometry coverage."""
    z = np.asarray(pmt_pos)[:, 2]
    edges = np.unique(np.percentile(z, np.linspace(0, 100, n_rings + 1)))
    n = len(edges) - 1
    ring = np.clip(np.digitize(z, edges) - 1, 0, n - 1)
    return ring, n


def compute_forward_block(
    logw,
    pool_pmt,
    pool_t,
    pmt_pos,
    mc_pmt,
    mc_t,
    mc_ntot,
    n_mc_events: int,
    chargenet_logit_c,
    n_grid,
    train_n_hist,
    n_rings: int = 8,
    time_bins=None,
    ring_time_bins=None,
):
    """Assemble the serialized ``forward`` block + (self_norm, ess) from arrays.

    Everything model-specific has already been reduced to ``logw`` (pool log-ratios)
    and ``chargenet_logit_c`` (chargenet logits over ``n_grid``); this function is pure
    numpy and is the unit-tested entry point.
    """
    pool = self_normalized_weights(logw)
    pmt_pos = np.asarray(pmt_pos)
    n_pmts = len(pmt_pos)

    pmf, mean_ntot = fwd.implied_ntot_pmf(chargenet_logit_c, train_n_hist, n_grid)

    per_pmt = fwd.per_pmt_charge_receipt(
        pool, pool_pmt, n_pmts, mc_pmt, n_mc_events, mean_ntot
    )
    toa = fwd.toa_receipt(pool, pool_t, mc_t, bins=time_bins)
    ntot = fwd.ntot_receipt(pmf, n_grid, train_n_hist, mc_ntot, n_mc_events)

    ring, nr = z_rings(pmt_pos, n_rings)
    mc_ring = ring[np.asarray(mc_pmt)]
    pool_ring = ring[np.asarray(pool_pmt)]
    if ring_time_bins is None:
        ring_time_bins = np.linspace(
            np.floor(np.min(mc_t)), np.percentile(mc_t, 99.5), 41
        )
    ring_time = fwd.per_ring_time_receipt(
        pool, pool_t, pool_ring, mc_t, mc_ring, nr, ring_time_bins
    )

    forward_block = {
        "per_pmt_charge": schema.chi2_block(per_pmt),
        "toa": schema.chi2_block(toa),
        "n_tot": schema.ntot_block(ntot),
        "per_ring_time": schema.ring_time_block(ring_time),
    }
    return forward_block, pool.self_norm, pool.ess


# ---- model / data loaders (JAX; used by the CLI) -----------------------------


def load_models(model_dir: str, store):
    """Load hitnet (class from models.json) + chargenet from a model dir."""
    import equinox as eqx
    import jax

    from hitman.nn import ChargeNet, HitNet

    cls = "HitNet"
    mj = os.path.join(model_dir, "models.json")
    if os.path.exists(mj):
        with open(mj) as f:
            cls = json.load(f).get("hitnet_class", "HitNet")
    if cls == "FrameHitNet":
        from hitman.nn import FrameHitNet

        skel = FrameHitNet(
            np.asarray(store.pmt_pos), np.asarray(store.pmt_dir), key=jax.random.PRNGKey(0)
        )
    else:
        skel = HitNet(key=jax.random.PRNGKey(0))
    hitnet = eqx.tree_deserialise_leaves(os.path.join(model_dir, "hitnet.eqx"), skel)
    chargenet = eqx.tree_deserialise_leaves(
        os.path.join(model_dir, "chargenet.eqx"), ChargeNet(key=jax.random.PRNGKey(1))
    )
    obs_style = getattr(hitnet, "obs_style", "xyz")
    return hitnet, chargenet, obs_style


def build_marginal_pool(store, n_pool: int, seed: int = 0):
    """Sample an augmented marginal pool (matching the classifier's calibration)."""
    rng = np.random.default_rng(seed)
    n_pool = min(n_pool, store.n_hits)
    rows = np.sort(rng.choice(store.n_hits, n_pool, replace=False))
    pool_pmt = np.asarray(store.pmt_id[rows])
    pool_ev = np.asarray(store.event_id[rows])
    shift = rng.normal(0.0, TIME_SIGMA, store.n_events).astype(np.float32)
    pool_t = np.asarray(store.hits[rows, 3]) + shift[pool_ev]
    return pool_pmt, pool_t, np.asarray(store.pmt_pos)


def hit_logits(hitnet, obs_style, pool_pmt, pool_t, pmt_pos, theta, chunk: int = 500_000):
    """log r_hit over the pool at fixed theta (chunked; obs-style aware)."""
    import jax
    import jax.numpy as jnp

    M = len(pool_pmt)
    if obs_style == "id_t":
        fn = jax.jit(jax.vmap(lambda i, t: hitnet((i, t), theta)))
        return np.concatenate(
            [
                np.asarray(fn(jnp.asarray(pool_pmt[i : i + chunk]), jnp.asarray(pool_t[i : i + chunk])))
                for i in range(0, M, chunk)
            ]
        )
    obs = np.concatenate([pmt_pos[pool_pmt], pool_t[:, None]], axis=1).astype(np.float32)
    fn = jax.jit(jax.vmap(lambda h: hitnet(h, theta)))
    return np.concatenate(
        [np.asarray(fn(jnp.asarray(obs[i : i + chunk]))) for i in range(0, M, chunk)]
    )


def chargenet_logit_grid(chargenet, n_grid, theta):
    """log r_charge over an integer N grid at fixed theta (uses (N, N) as (q, nhit))."""
    import jax
    import jax.numpy as jnp

    c_obs = jnp.asarray(np.stack([n_grid, n_grid], axis=1).astype(np.float32))
    return np.asarray(jax.jit(jax.vmap(lambda c: chargenet(c, theta)))(c_obs))


# ---- top-level orchestration -------------------------------------------------


def run_model_dir(
    model_dir: str,
    testpoints,
    store_path: str,
    n_pool: int = 4_000_000,
    out_name: str = "receipts.json",
    seed: int = 0,
    config: Optional[dict] = None,
):
    """Full L2 pass: build pool, compute forward receipts per test point, write receipts.json.

    ``testpoints`` is a list of ``(tag, root_path, truth7)``. Score-identity and MLE
    receipts are attached when the MLE/score helpers are supplied by the caller
    (kept out of the default path so the forward battery — the tested seam — has no
    hard dependency on the reconstruction stack).
    """
    import jax.numpy as jnp

    from hitman.data import HitStore, RatDSExtractor

    store = HitStore(store_path)
    hitnet, chargenet, obs_style = load_models(model_dir, store)
    pool_pmt, pool_t, pmt_pos = build_marginal_pool(store, n_pool, seed=seed)

    train_counts = np.asarray(store.charge[:, 1]).astype(int)

    testpoint_blocks = {}
    for tag, path, truth in testpoints:
        theta = jnp.asarray([float(v) for v in truth])
        mc = RatDSExtractor([path]).load()
        mc_pmt = np.asarray(mc.pmt_id)
        mc_t = np.asarray(mc.hits[:, 3])
        mc_ntot = np.asarray(mc.charge[:, 1])
        n_mc_events = mc.n_events

        logw = hit_logits(hitnet, obs_style, pool_pmt, pool_t, pmt_pos, theta)
        n_grid = np.arange(0, int(mc_ntot.max() * 2 + 50))
        train_n_hist = np.bincount(train_counts, minlength=len(n_grid))[: len(n_grid)]
        logit_c = chargenet_logit_grid(chargenet, n_grid, theta)

        forward_block, self_norm, ess = compute_forward_block(
            logw, pool_pmt, pool_t, pmt_pos, mc_pmt, mc_t, mc_ntot,
            n_mc_events, logit_c, n_grid, train_n_hist,
        )
        testpoint_blocks[tag] = schema.testpoint_receipt(
            truth=truth, self_norm=self_norm, ess=ess, forward=forward_block
        )
        print(f"[{tag}] E[r]={self_norm:.3f} ESS={ess/1e3:.0f}k "
              f"chi2/dof pmt={forward_block['per_pmt_charge']['chi2_dof']:.2f} "
              f"toa={forward_block['toa']['chi2_dof']:.2f} "
              f"ntot={forward_block['n_tot']['chi2_dof']:.2f} "
              f"ring={forward_block['per_ring_time']['global']['chi2_dof']:.2f}", flush=True)

    meta = {
        "model_dir": os.path.abspath(model_dir),
        "git_sha": git_sha(os.path.dirname(os.path.abspath(__file__))),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hitman_version": __import__("hitman").__version__,
        "store": store_path,
        "n_pool": int(min(n_pool, store.n_hits)),
        "testpoint_tags": [t[0] for t in testpoints],
        "config": config or {},
    }
    receipts = schema.assemble(meta, testpoint_blocks)
    out_path = os.path.join(model_dir, out_name)
    schema.write_receipts(out_path, receipts)
    print(f"wrote {out_path}")
    return receipts, out_path

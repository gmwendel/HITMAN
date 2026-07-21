"""L2 receipt runner: one command -> receipts.json for a model dir + test points.

    python -m hitman.receipts <model_dir> <testpoint.root>[:tag:x,y,z,zen,az,t,E] ...

The detector-coupled I/O — WC model-class loading, the HitStore marginal pool, the ROOT MC
loader — lives here; the reusable, ``(model, data)``-agnostic battery lives in
:mod:`hitman.receipts.harness` (DESIGN proposal 3) and the numeric receipt logic in the
sibling modules (``reweight``, ``chi2``, ``forward``, ``score``, ``mle``). ``run_model_dir``
is the WC instantiation: it wires an :class:`~hitman.receipts.harness.NREReceiptModel` and
the default ``z_rings`` strata through
:func:`~hitman.receipts.harness.forward_block_from_model`. The generic seam
(:func:`~hitman.receipts.harness.compute_forward_block`) is re-exported here so the
historical ``from hitman.receipts.runner import compute_forward_block`` path is unchanged.
"""

import json
import os
import subprocess
import time
from typing import Optional

import numpy as np

from hitman.receipts import schema
# Detector-agnostic battery pieces live in harness; re-exported for backward compatibility.
from hitman.receipts.harness import (  # noqa: F401
    NREReceiptModel,
    ReceiptModel,
    chargenet_logit_grid,
    compute_forward_block,
    forward_block_from_model,
    hit_logits,
    percentile_strata,
    z_rings,
)

TIME_SIGMA = 50.0  # +/-50 ns event-time augmentation the classifier was calibrated with


# ---- provenance --------------------------------------------------------------


def git_sha(cwd: Optional[str] = None) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


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
    model = NREReceiptModel(hitnet, chargenet, obs_style)
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

        n_grid = np.arange(0, int(mc_ntot.max() * 2 + 50))
        train_n_hist = np.bincount(train_counts, minlength=len(n_grid))[: len(n_grid)]

        forward_block, self_norm, ess = forward_block_from_model(
            model, pool_pmt, pool_t, pmt_pos, mc_pmt, mc_t, mc_ntot,
            n_mc_events, n_grid, train_n_hist, theta,
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

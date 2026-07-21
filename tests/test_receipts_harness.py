"""Proposal 3: reusable (model, data) receipt harness — strata + model protocol.

LOCK: the WC path (NREReceiptModel + z_rings, through forward_block_from_model) reproduces
the direct compute_forward_block numbers bit-for-bit. Genericity: a custom strata_fn and a
custom ReceiptModel both run the same battery, so a downstream detector (shower-frame E x r
strata, a density model) reuses it with no core change.
"""

import jax
import jax.numpy as jnp
import numpy as np

from hitman.nn import ChargeNet, HitNet
from hitman.receipts import schema
from hitman.receipts.harness import (NREReceiptModel, ReceiptModel, compute_forward_block,
                                     forward_block_from_model, percentile_strata, z_rings)


def _synthetic(seed=0):
    rng = np.random.default_rng(seed)
    n_pmts = 12
    pmt_pos = np.column_stack(
        [rng.uniform(-500, 500, n_pmts), rng.uniform(-500, 500, n_pmts),
         np.linspace(-600, 600, n_pmts)]).astype(np.float32)
    theta = jnp.array([0.0, 0.0, 0.0, np.pi / 2, 0.0, 0.0, 3.0])
    hitnet = HitNet(width=16, depth=2, key=jax.random.PRNGKey(0))
    chargenet = ChargeNet(width=16, depth=2, key=jax.random.PRNGKey(1))
    n_pool = 20_000
    pool_pmt = rng.integers(0, n_pmts, n_pool)
    pool_t = rng.normal(0.0, 3.0, n_pool).astype(np.float32)
    n_mc_events = 500
    mc_ntot = rng.integers(5, 30, n_mc_events)
    total = int(mc_ntot.sum())
    mc_pmt = rng.integers(0, n_pmts, total)
    mc_t = rng.normal(0.0, 3.0, total).astype(np.float32)
    n_grid = np.arange(0, int(mc_ntot.max() * 2 + 50))
    train_counts = rng.integers(3, 40, 5000)
    train_n_hist = np.bincount(train_counts, minlength=len(n_grid))[: len(n_grid)]
    return dict(hitnet=hitnet, chargenet=chargenet, theta=theta, pmt_pos=pmt_pos,
                pool_pmt=pool_pmt, pool_t=pool_t, mc_pmt=mc_pmt, mc_t=mc_t,
                mc_ntot=mc_ntot, n_mc_events=n_mc_events, n_grid=n_grid,
                train_n_hist=train_n_hist)


def _direct_block(d):
    obs = np.concatenate([d["pmt_pos"][d["pool_pmt"]], d["pool_t"][:, None]], axis=1
                         ).astype(np.float32)
    logw = np.asarray(jax.jit(jax.vmap(lambda h: d["hitnet"](h, d["theta"])))(jnp.asarray(obs)))
    c_obs = jnp.asarray(np.stack([d["n_grid"], d["n_grid"]], axis=1).astype(np.float32))
    logit_c = np.asarray(jax.jit(jax.vmap(lambda c: d["chargenet"](c, d["theta"])))(c_obs))
    return compute_forward_block(
        logw, d["pool_pmt"], d["pool_t"], d["pmt_pos"], d["mc_pmt"], d["mc_t"],
        d["mc_ntot"], d["n_mc_events"], logit_c, d["n_grid"], d["train_n_hist"], n_rings=6)


# ---- (a) LOCK: model-protocol path == direct compute_forward_block ----------
def test_nre_model_reproduces_direct_block():
    d = _synthetic()
    ref_block, ref_sn, ref_ess = _direct_block(d)
    model = NREReceiptModel(d["hitnet"], d["chargenet"], obs_style="xyz")
    got_block, got_sn, got_ess = forward_block_from_model(
        model, d["pool_pmt"], d["pool_t"], d["pmt_pos"], d["mc_pmt"], d["mc_t"],
        d["mc_ntot"], d["n_mc_events"], d["n_grid"], d["train_n_hist"], d["theta"],
        n_rings=6)
    assert got_block == ref_block          # exact dict equality (bit-for-bit floats)
    assert got_sn == ref_sn and got_ess == ref_ess


# ---- (b) default strata_fn == explicit z_rings ------------------------------
def test_default_strata_is_z_rings():
    d = _synthetic()
    obs = np.concatenate([d["pmt_pos"][d["pool_pmt"]], d["pool_t"][:, None]], axis=1
                         ).astype(np.float32)
    logw = np.asarray(jax.jit(jax.vmap(lambda h: d["hitnet"](h, d["theta"])))(jnp.asarray(obs)))
    c_obs = jnp.asarray(np.stack([d["n_grid"], d["n_grid"]], axis=1).astype(np.float32))
    logit_c = np.asarray(jax.jit(jax.vmap(lambda c: d["chargenet"](c, d["theta"])))(c_obs))
    args = (logw, d["pool_pmt"], d["pool_t"], d["pmt_pos"], d["mc_pmt"], d["mc_t"],
            d["mc_ntot"], d["n_mc_events"], logit_c, d["n_grid"], d["train_n_hist"])
    a, _, _ = compute_forward_block(*args, n_rings=6)
    b, _, _ = compute_forward_block(*args, n_rings=6, strata_fn=z_rings)
    assert a == b


# ---- (c) a custom strata_fn runs the battery --------------------------------
def test_custom_strata_fn_runs():
    d = _synthetic()
    def radial_rings(pmt_pos, n):            # core-distance analog of z-rings
        r = np.hypot(pmt_pos[:, 0], pmt_pos[:, 1])
        return percentile_strata(r, n)
    obs = np.concatenate([d["pmt_pos"][d["pool_pmt"]], d["pool_t"][:, None]], axis=1
                         ).astype(np.float32)
    logw = np.asarray(jax.jit(jax.vmap(lambda h: d["hitnet"](h, d["theta"])))(jnp.asarray(obs)))
    c_obs = jnp.asarray(np.stack([d["n_grid"], d["n_grid"]], axis=1).astype(np.float32))
    logit_c = np.asarray(jax.jit(jax.vmap(lambda c: d["chargenet"](c, d["theta"])))(c_obs))
    block, sn, ess = compute_forward_block(
        logw, d["pool_pmt"], d["pool_t"], d["pmt_pos"], d["mc_pmt"], d["mc_t"],
        d["mc_ntot"], d["n_mc_events"], logit_c, d["n_grid"], d["train_n_hist"],
        n_rings=4, strata_fn=radial_rings)
    assert np.isfinite(block["per_ring_time"]["global"]["chi2_dof"])
    tp = schema.testpoint_receipt((0.,) * 7, sn, ess, forward=block)
    schema.validate_schema(schema.assemble({"model_dir": "m"}, {"pt": tp}))


# ---- (d) strata + protocol ergonomics ---------------------------------------
def test_percentile_strata_equal_occupancy():
    vals = np.arange(100.0)
    labels, n = percentile_strata(vals, 5)
    assert n == 5
    counts = np.bincount(labels, minlength=5)
    assert counts.min() >= 18 and counts.max() <= 22   # ~equal occupancy


def test_nre_model_conforms_to_protocol():
    d = _synthetic()
    model = NREReceiptModel(d["hitnet"], d["chargenet"])
    assert isinstance(model, ReceiptModel)

    class DensityModel:                      # a non-NRE conforming model
        def pool_log_ratios(self, pool_pmt, pool_t, pmt_pos, theta):
            return np.zeros(len(pool_pmt))
        def count_grid_logits(self, n_grid, theta):
            return np.zeros(len(n_grid))
    assert isinstance(DensityModel(), ReceiptModel)

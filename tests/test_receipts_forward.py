"""End-to-end wiring of the forward receipt block with tiny random nets — no ROOT,
no trained model, no HitStore. Exercises compute_forward_block + schema serialization,
the seam the real CLI runs on."""

import jax
import jax.numpy as jnp
import numpy as np

from hitman.nn import ChargeNet, HitNet
from hitman.receipts import schema
from hitman.receipts.runner import compute_forward_block


def _synthetic_inputs(seed=0):
    rng = np.random.default_rng(seed)
    n_pmts = 12
    pmt_pos = np.column_stack(
        [rng.uniform(-500, 500, n_pmts), rng.uniform(-500, 500, n_pmts),
         np.linspace(-600, 600, n_pmts)]
    ).astype(np.float32)

    theta = jnp.array([0.0, 0.0, 0.0, np.pi / 2, 0.0, 0.0, 3.0])
    hitnet = HitNet(width=16, depth=2, key=jax.random.PRNGKey(0))
    chargenet = ChargeNet(width=16, depth=2, key=jax.random.PRNGKey(1))

    # marginal pool
    n_pool = 20_000
    pool_pmt = rng.integers(0, n_pmts, n_pool)
    pool_t = rng.normal(0.0, 3.0, n_pool).astype(np.float32)
    obs = np.concatenate([pmt_pos[pool_pmt], pool_t[:, None]], axis=1).astype(np.float32)
    logw = np.asarray(jax.jit(jax.vmap(lambda h: hitnet(h, theta)))(jnp.asarray(obs)))

    # MC test ensemble
    n_mc_events = 500
    mc_ntot = rng.integers(5, 30, n_mc_events)
    total = int(mc_ntot.sum())
    mc_pmt = rng.integers(0, n_pmts, total)
    mc_t = rng.normal(0.0, 3.0, total).astype(np.float32)

    # chargenet grid + a training-N histogram
    n_grid = np.arange(0, int(mc_ntot.max() * 2 + 50))
    train_counts = rng.integers(3, 40, 5000)
    train_n_hist = np.bincount(train_counts, minlength=len(n_grid))[: len(n_grid)]
    c_obs = jnp.asarray(np.stack([n_grid, n_grid], axis=1).astype(np.float32))
    logit_c = np.asarray(jax.jit(jax.vmap(lambda c: chargenet(c, theta)))(c_obs))

    return dict(
        logw=logw, pool_pmt=pool_pmt, pool_t=pool_t, pmt_pos=pmt_pos,
        mc_pmt=mc_pmt, mc_t=mc_t, mc_ntot=mc_ntot, n_mc_events=n_mc_events,
        logit_c=logit_c, n_grid=n_grid, train_n_hist=train_n_hist,
        truth=(0.0, 0.0, 0.0, np.pi / 2, 0.0, 0.0, 3.0),
    )


def test_forward_block_shape_and_finiteness():
    d = _synthetic_inputs()
    block, self_norm, ess = compute_forward_block(
        d["logw"], d["pool_pmt"], d["pool_t"], d["pmt_pos"],
        d["mc_pmt"], d["mc_t"], d["mc_ntot"], d["n_mc_events"],
        d["logit_c"], d["n_grid"], d["train_n_hist"], n_rings=6,
    )
    assert set(block) == {"per_pmt_charge", "toa", "n_tot", "per_ring_time"}
    for key in ("per_pmt_charge", "toa"):
        assert np.isfinite(block[key]["chi2_dof"])
        assert 0.0 <= block[key]["mc_var_share"] <= 1.0
    assert np.isfinite(block["n_tot"]["chi2_dof"])
    assert np.isfinite(block["n_tot"]["mean_sur"])
    assert np.isfinite(block["per_ring_time"]["global"]["chi2_dof"])
    assert len(block["per_ring_time"]["rings"]) >= 1
    assert self_norm > 0 and ess > 0


def test_forward_block_assembles_into_valid_receipt():
    d = _synthetic_inputs()
    block, self_norm, ess = compute_forward_block(
        d["logw"], d["pool_pmt"], d["pool_t"], d["pmt_pos"],
        d["mc_pmt"], d["mc_t"], d["mc_ntot"], d["n_mc_events"],
        d["logit_c"], d["n_grid"], d["train_n_hist"], n_rings=6,
    )
    tp = schema.testpoint_receipt(d["truth"], self_norm, ess, forward=block)
    receipts = schema.assemble({"model_dir": "synthetic"}, {"e3_zen090": tp})
    schema.validate_schema(receipts)  # raises on malformed

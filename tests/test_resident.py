import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hitman.data.structures import EventBatch
from hitman.nn import ChargeNet, HitNet
from hitman.train import DeviceData, charge_batch, fit_resident, hit_batch
from hitman.train.loop import _batch_loss


def _toy_data(n_events=200, n_pmts=16, seed=0):
    rng = np.random.default_rng(seed)
    pmt_pos = rng.normal(0, 500, (n_pmts, 3)).astype(np.float32)
    counts = rng.poisson(8, n_events).astype(np.int64)
    n_hits = int(counts.sum())
    pmt_id = rng.integers(0, n_pmts, n_hits).astype(np.int32)
    hyp = rng.normal(0, 1, (n_events, 7)).astype(np.float32)
    hyp[:, 6] = rng.uniform(1, 3, n_events)
    hits = np.concatenate(
        [pmt_pos[pmt_id], rng.normal(10, 5, (n_hits, 1)).astype(np.float32)], axis=1
    )
    batch = EventBatch(
        hyp=hyp,
        charge=np.stack([counts, counts], axis=1).astype(np.float32),
        hits=hits,
        event_id=np.repeat(np.arange(n_events, dtype=np.int32), counts),
        pmt_id=pmt_id,
    )
    return batch, pmt_pos


def test_hit_batch_reconstructs_positions():
    batch, pmt_pos = _toy_data()
    data = DeviceData.from_batch(batch, pmt_pos)
    rows = jnp.array([0, 7, data.n_hits - 1])
    obs, hyp = hit_batch(data, rows)
    np.testing.assert_allclose(np.asarray(obs), np.asarray(batch.hits)[np.asarray(rows)], rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(hyp), batch.hyp[np.asarray(batch.event_id)[np.asarray(rows)]], rtol=1e-6
    )


def test_charge_batch_identity():
    batch, pmt_pos = _toy_data()
    data = DeviceData.from_batch(batch, pmt_pos)
    rows = jnp.array([3, 5])
    obs, hyp = charge_batch(data, rows)
    np.testing.assert_array_equal(np.asarray(obs), batch.charge[[3, 5]])
    np.testing.assert_array_equal(np.asarray(hyp), batch.hyp[[3, 5]])


def test_resident_loss_matches_materialized():
    batch, pmt_pos = _toy_data()
    data = DeviceData.from_batch(batch, pmt_pos)
    model = HitNet(width=32, depth=2, key=jax.random.PRNGKey(0))
    rows = jnp.arange(64)
    key = jax.random.PRNGKey(42)
    obs, hyp = hit_batch(data, rows)
    resident = _batch_loss(model, obs, hyp, key, 0.0)
    materialized = _batch_loss(
        model,
        jnp.asarray(np.asarray(batch.hits)[:64]),
        jnp.asarray(batch.hyp[np.asarray(batch.event_id)[:64]]),
        key,
        0.0,
    )
    np.testing.assert_allclose(float(resident), float(materialized), rtol=1e-6)


def test_fit_resident_trains_both_networks():
    batch, pmt_pos = _toy_data(n_events=400)
    data = DeviceData.from_batch(batch, pmt_pos)
    res_h = fit_resident(
        HitNet(width=32, depth=2, key=jax.random.PRNGKey(0)),
        data, hit_batch, data.n_hits,
        key=jax.random.PRNGKey(1), batch_size=512, max_epochs=3, patience=3, verbose=False,
    )
    res_c = fit_resident(
        ChargeNet(width=32, depth=2, key=jax.random.PRNGKey(2)),
        data, charge_batch, data.n_events,
        key=jax.random.PRNGKey(3), batch_size=128, max_epochs=3, patience=3, verbose=False,
    )
    for res in (res_h, res_c):
        assert len(res.val_loss) == 3
        assert np.isfinite(res.val_loss).all()
    # parameters actually moved
    w0 = HitNet(width=32, depth=2, key=jax.random.PRNGKey(0)).mlp.layers[0].weight
    assert not np.allclose(np.asarray(res_h.model.mlp.layers[0].weight), np.asarray(w0))


def test_from_batch_requires_pmt_id():
    batch, pmt_pos = _toy_data()
    with pytest.raises(ValueError, match="pmt_id"):
        DeviceData.from_batch(batch._replace(pmt_id=None), pmt_pos)


@pytest.mark.skipif(
    not os.path.exists("/tank/playground/hitman-sbi-modernization/datagen/water_1M/store/meta.json"),
    reason="1M store not available",
)
def test_from_store_matches_streaming_layout():
    from hitman.data import HitStore

    store = HitStore("/tank/playground/hitman-sbi-modernization/datagen/water_1M/store")
    # cheap check on a slice: DeviceData formed from a small event subset
    sub = store.event_batch(np.arange(100))
    data = DeviceData.from_batch(sub, np.asarray(store.pmt_pos))
    rows = jnp.arange(min(500, data.n_hits))
    obs, _ = hit_batch(data, rows)
    np.testing.assert_allclose(np.asarray(obs), np.asarray(sub.hits)[: len(rows)], rtol=1e-6)

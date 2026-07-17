import jax
import jax.numpy as jnp
import numpy as np

from hitman import event_log_ratio
from hitman.nn import ChargeNet, HitNet
from hitman.train import DeviceData, event_val_bce, fit_event_model
from tests.test_resident import _toy_data


def _nets():
    return (
        HitNet(width=32, depth=2, key=jax.random.PRNGKey(0)),
        ChargeNet(width=32, depth=2, key=jax.random.PRNGKey(1)),
    )


def test_event_val_bce_matches_event_log_ratio():
    batch, pmt_pos = _toy_data(n_events=50)
    data = DeviceData.from_batch(batch, pmt_pos)
    hitnet, chargenet = _nets()

    event_start = 30
    hit_start = int(np.searchsorted(np.asarray(batch.event_id), event_start))
    key = jax.random.PRNGKey(7)
    got = float(event_val_bce(hitnet, chargenet, data, hit_start, event_start, key))

    # reference: compose via the public batch-level API on the same holdout
    val = batch.select(np.arange(event_start, batch.n_events))
    dev = val.to_device()
    perm = jax.random.permutation(key, dev.n_events)
    l_joint = event_log_ratio(hitnet, chargenet, dev, dev.hyp)
    l_marg = event_log_ratio(hitnet, chargenet, dev, dev.hyp[perm])
    want = float(
        0.5 * (jnp.mean(jax.nn.softplus(-l_joint)) + jnp.mean(jax.nn.softplus(l_marg)))
    )
    np.testing.assert_allclose(got, want, rtol=1e-5)


def test_fit_event_model_trains_and_reports_three_metrics():
    batch, pmt_pos = _toy_data(n_events=600, seed=3)
    data = DeviceData.from_batch(batch, pmt_pos)
    hitnet, chargenet = _nets()
    res = fit_event_model(
        hitnet, chargenet, data,
        key=jax.random.PRNGKey(2),
        hit_batch_size=1024, charge_batch_size=128, charge_passes_per_round=2,
        max_rounds=4, patience=4, verbose=False,
    )
    assert len(res.hit_val) == len(res.charge_val) == len(res.event_val) == 4
    assert np.isfinite(res.event_val).all()
    # independent selection: each net snapshots at its own optimum
    assert res.best_hit_round == int(np.argmin(res.hit_val))
    assert res.best_charge_round == int(np.argmin(res.charge_val))
    assert np.isfinite(res.final_event_bce)
    # both nets moved
    h0, c0 = _nets()
    assert not np.allclose(
        np.asarray(res.hitnet.mlp.layers[0].weight), np.asarray(h0.mlp.layers[0].weight)
    )
    assert not np.allclose(
        np.asarray(res.chargenet.mlp.layers[0].weight), np.asarray(c0.mlp.layers[0].weight)
    )


def test_nets_stop_independently():
    # tiny patience: chargenet (few steps/round on toy data) plateaus quickly while
    # the loop keeps running until BOTH nets exhaust their own patience
    batch, pmt_pos = _toy_data(n_events=600, seed=9)
    data = DeviceData.from_batch(batch, pmt_pos)
    hitnet, chargenet = _nets()
    res = fit_event_model(
        hitnet, chargenet, data,
        key=jax.random.PRNGKey(4),
        hit_batch_size=1024, charge_batch_size=128, charge_passes_per_round=1,
        max_rounds=30, patience=2, verbose=False,
    )
    n_rounds = len(res.event_val)
    # loop ended only after both nets were >= patience past their own best
    assert n_rounds - 1 - res.best_hit_round >= 2
    assert n_rounds - 1 - res.best_charge_round >= 2
    # and the two optima are tracked separately (fields exist and are valid rounds)
    assert 0 <= res.best_hit_round < n_rounds
    assert 0 <= res.best_charge_round < n_rounds


def test_composed_selection_mode():
    batch, pmt_pos = _toy_data(n_events=400, seed=11)
    data = DeviceData.from_batch(batch, pmt_pos)
    hitnet, chargenet = _nets()
    res = fit_event_model(
        hitnet, chargenet, data,
        key=jax.random.PRNGKey(5),
        hit_batch_size=1024, charge_batch_size=128,
        max_rounds=3, patience=3, selection="composed", verbose=False,
    )
    assert res.best_hit_round == res.best_charge_round == int(np.argmin(res.event_val))


def test_holdout_is_event_aligned():
    batch, pmt_pos = _toy_data(n_events=100, seed=5)
    data = DeviceData.from_batch(batch, pmt_pos)
    event_start = 90
    hit_start = int(np.searchsorted(np.asarray(batch.event_id), event_start))
    # no training hit belongs to a validation event and vice versa
    assert (np.asarray(batch.event_id)[:hit_start] < event_start).all()
    assert (np.asarray(batch.event_id)[hit_start:] >= event_start).all()

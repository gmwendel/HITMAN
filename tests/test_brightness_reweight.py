"""Brightness-weighted hit batch maker (the E-wall fix): normalization + augmentation
contract + ratio-preservation-at-alpha0 receipts. Weight axis is PARENT-EVENT brightness
(nhit), so each hit inherits its event's w(nhit) — this is the axis that targets the
high-E tail, unlike run11's per-(sensor,t) weight."""
import jax
import jax.numpy as jnp
import numpy as np

from hitman.nn import HitNet
from hitman.train import (BrightnessWeightTable, DeviceData, build_brightness_weights,
                          hit_batch, make_brightness_weighted_hit_batch, train_recipe)
from hitman.train.loop import _batch_loss
from tests.test_resident import _toy_data


def _fake_store(batch, pmt_pos):
    class S: pass
    st = S()
    st.n_hits = batch.n_hits
    st.n_events = batch.n_events
    st.pmt_id = np.asarray(batch.pmt_id)
    st.event_id = np.asarray(batch.event_id)
    st.hits = np.asarray(batch.hits)
    st.charge = np.asarray(batch.charge)
    st.pmt_pos = pmt_pos
    return st


def test_brightness_weight_normalization_mean1_over_hits():
    # E_hit[w] == 1: the mean per-hit weight is 1 (per-hit gradient scale preserved).
    batch, pmt_pos = _toy_data(n_events=800, seed=21)
    st = _fake_store(batch, pmt_pos)
    tab = build_brightness_weights(st, alpha=0.25, n_max=100)
    nhit = np.clip(st.charge[:, 1].astype(int), 0, 100)          # per-event brightness
    per_hit_w = np.asarray(tab.w)[nhit[np.asarray(st.event_id)]]  # weight of every hit
    np.testing.assert_allclose(per_hit_w.mean(), 1.0, rtol=1e-4)


def test_brighter_events_get_larger_weight():
    batch, pmt_pos = _toy_data(n_events=800, seed=22)
    st = _fake_store(batch, pmt_pos)
    tab = build_brightness_weights(st, alpha=0.5, n_max=100)
    h = np.bincount(np.clip(st.charge[:, 1].astype(int), 0, 100), minlength=101)
    common = int(np.argmax(h))                          # modal (typical) multiplicity
    rare = int(np.max(np.where(h > 0)[0]))              # brightest occupied bin
    assert tab.w[rare] >= tab.w[common]


def test_alpha0_recovers_unweighted_loss():
    # alpha=0 -> all weights exactly 1 -> weighted loss == unweighted loss (same rows/keys)
    batch, pmt_pos = _toy_data(n_events=400, seed=23)
    data = DeviceData.from_batch(batch, pmt_pos)
    st = _fake_store(batch, pmt_pos)
    tab = build_brightness_weights(st, alpha=0.0, n_max=100)
    np.testing.assert_allclose(np.asarray(tab.w), 1.0, rtol=1e-6)
    make_w = make_brightness_weighted_hit_batch(tab, obs_style="xyz", time_sigma=50.0)
    model = HitNet(width=32, depth=2, key=jax.random.PRNGKey(0))
    rows = jnp.arange(512, dtype=jnp.int32)
    k = jax.random.PRNGKey(3)
    obs_u, hyp_u = hit_batch(data, rows, k)
    obs_w, hyp_w, w = make_w(data, rows, k)
    np.testing.assert_allclose(np.asarray(obs_u), np.asarray(obs_w), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(hyp_u), np.asarray(hyp_w), rtol=1e-6)
    lu = _batch_loss(model, obs_u, hyp_u, jax.random.PRNGKey(4), 0.0)
    lw = _batch_loss(model, obs_w, hyp_w, jax.random.PRNGKey(4), 0.0, w)
    np.testing.assert_allclose(float(lu), float(lw), rtol=1e-5)


def test_augmentation_contract_matches_hit_batch():
    # Same time-shuffle augmentation as hit_batch: identical obs & hyp under the same key.
    batch, pmt_pos = _toy_data(n_events=400, seed=24)
    data = DeviceData.from_batch(batch, pmt_pos)
    st = _fake_store(batch, pmt_pos)
    tab = build_brightness_weights(st, alpha=0.3, n_max=100)
    make_w = make_brightness_weighted_hit_batch(tab, obs_style="xyz", time_sigma=50.0)
    rows = jnp.arange(300, dtype=jnp.int32)
    k = jax.random.PRNGKey(11)
    obs_u, hyp_u = hit_batch(data, rows, k, time_sigma=50.0)
    obs_w, hyp_w, w = make_w(data, rows, k)
    # augmentation applied identically (hit times shifted, theta_t shifted)
    np.testing.assert_allclose(np.asarray(obs_u), np.asarray(obs_w), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(hyp_u), np.asarray(hyp_w), rtol=1e-6)
    # unaugmented obs (no key) differs in the time column -> augmentation really ran
    obs_noaug, _, _ = make_w(data, rows, None)
    assert not np.allclose(np.asarray(obs_noaug)[:, 3], np.asarray(obs_w)[:, 3])


def test_weight_is_parent_brightness_and_time_invariant():
    # The weight of a row equals table.w[parent nhit] and does NOT depend on the aug key
    # (brightness is time-invariant); the augmentation only touches obs/hyp times.
    batch, pmt_pos = _toy_data(n_events=400, seed=25)
    data = DeviceData.from_batch(batch, pmt_pos)
    st = _fake_store(batch, pmt_pos)
    tab = build_brightness_weights(st, alpha=0.4, n_max=100)
    make_w = make_brightness_weighted_hit_batch(tab)
    rows = jnp.arange(256, dtype=jnp.int32)
    ev = np.asarray(batch.event_id)[np.asarray(rows)]
    nhit = np.clip(np.asarray(batch.charge)[ev, 1].astype(int), 0, 100)
    expected = np.asarray(tab.w)[nhit]
    _, _, w1 = make_w(data, rows, jax.random.PRNGKey(1))
    _, _, w2 = make_w(data, rows, jax.random.PRNGKey(999))
    np.testing.assert_allclose(np.asarray(w1), expected, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(w1), np.asarray(w2), rtol=1e-6)  # key-invariant


def test_id_t_obs_style():
    batch, pmt_pos = _toy_data(n_events=200, seed=26)
    data = DeviceData.from_batch(batch, pmt_pos)
    st = _fake_store(batch, pmt_pos)
    tab = build_brightness_weights(st, alpha=0.25, n_max=100)
    make_w = make_brightness_weighted_hit_batch(tab, obs_style="id_t")
    (pmt, t), hyp, w = make_w(data, jnp.arange(64, dtype=jnp.int32), jax.random.PRNGKey(0))
    assert pmt.shape == (64,) and t.shape == (64,) and w.shape == (64,)
    assert np.isfinite(np.asarray(w)).all()


def test_weighted_recipe_smoke():
    batch, pmt_pos = _toy_data(n_events=300, seed=27)
    data = DeviceData.from_batch(batch, pmt_pos)
    st = _fake_store(batch, pmt_pos)
    tab = build_brightness_weights(st, alpha=0.25, n_max=100)
    make_w = make_brightness_weighted_hit_batch(tab, obs_style="xyz")
    res = train_recipe(HitNet(width=32, depth=2, key=jax.random.PRNGKey(0)),
                       data, make_w, data.n_hits, key=jax.random.PRNGKey(1),
                       sgd_batch=512, sgd_max_steps=200, cosine_steps=100,
                       cosine_batch=512, val_every=50, patience_steps=100,
                       verbose=False)
    assert np.isfinite(res.best_val)

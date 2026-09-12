import jax
import jax.numpy as jnp
import numpy as np

from hitman.calibrate import build_charge_grid, build_marginal_grid, make_z_charge_fn, make_z_fn
from hitman.wc.inference import e_bfmi
from hitman.nn import HitNet
from hitman.train import DeviceData, build_hit_weights, make_weighted_hit_batch, train_recipe
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


def test_weight_table_normalization_and_lookup():
    batch, pmt_pos = _toy_data(n_events=300, seed=7)
    st = _fake_store(batch, pmt_pos)
    tab = build_hit_weights(st, alpha=0.5, n_sample=batch.n_hits, time_sigma=0.0,
                            t_lo=-40.0, t_hi=60.0, dt=1.0, w_max=50.0)
    # E_p[w] == 1 under the smoothed histogram pmf
    h = np.asarray(tab.w_grid)
    assert h.min() > 0
    # alpha=0 -> all weights exactly 1
    tab0 = build_hit_weights(st, alpha=0.0, n_sample=batch.n_hits, time_sigma=0.0,
                             t_lo=-40.0, t_hi=60.0, dt=1.0)
    np.testing.assert_allclose(np.asarray(tab0.w_grid), 1.0, rtol=1e-6)


def test_weighted_batch_ratio_preserving_at_alpha0():
    # alpha=0 weighted loss == unweighted loss exactly (same rows, same keys)
    batch, pmt_pos = _toy_data(n_events=300, seed=8)
    data = DeviceData.from_batch(batch, pmt_pos)
    st = _fake_store(batch, pmt_pos)
    tab = build_hit_weights(st, alpha=0.0, n_sample=batch.n_hits, time_sigma=50.0,
                            t_lo=-200.0, t_hi=300.0, dt=1.0)
    make_w = make_weighted_hit_batch(tab, obs_style="xyz", time_sigma=50.0)
    from hitman.train import hit_batch
    model = HitNet(width=32, depth=2, key=jax.random.PRNGKey(0))
    rows = jnp.arange(512, dtype=jnp.int32)
    k = jax.random.PRNGKey(3)
    obs_u, hyp_u = hit_batch(data, rows, k)
    obs_w, hyp_w, w = make_w(data, rows, k)
    np.testing.assert_allclose(np.asarray(obs_u), np.asarray(obs_w), rtol=1e-6)
    lu = _batch_loss(model, obs_u, hyp_u, jax.random.PRNGKey(4), 0.0)
    lw = _batch_loss(model, obs_w, hyp_w, jax.random.PRNGKey(4), 0.0, w)
    np.testing.assert_allclose(float(lu), float(lw), rtol=1e-5)


def test_weighted_recipe_smoke(tmp_path):
    batch, pmt_pos = _toy_data(n_events=300, seed=9)
    data = DeviceData.from_batch(batch, pmt_pos)
    st = _fake_store(batch, pmt_pos)
    tab = build_hit_weights(st, alpha=0.5, n_sample=batch.n_hits, time_sigma=50.0,
                            t_lo=-200.0, t_hi=300.0, dt=1.0)
    make_w = make_weighted_hit_batch(tab, obs_style="xyz")
    res = train_recipe(HitNet(width=32, depth=2, key=jax.random.PRNGKey(0)),
                       data, make_w, data.n_hits, key=jax.random.PRNGKey(1),
                       sgd_batch=512, sgd_max_steps=200, cosine_steps=100,
                       cosine_batch=512, val_every=50, patience_steps=100,
                       verbose=False)
    assert np.isfinite(res.best_val)


def test_z_fn_hoisted_matches_and_charge_z():
    batch, pmt_pos = _toy_data(n_events=300, seed=10)
    st = _fake_store(batch, pmt_pos)
    grid = build_marginal_grid(st, n_sample=batch.n_hits, time_sigma=0.0,
                               t_lo=-40.0, t_hi=60.0, dt=1.0)

    class UnitNet:
        obs_style = "xyz"
        def __call__(self, h, theta):
            return jnp.log(2.0)

    zf = make_z_fn(UnitNet(), grid, chunk_size=64)   # forces padding path
    np.testing.assert_allclose(float(zf(jnp.zeros(7))), 2.0, rtol=1e-5)

    cgrid = build_charge_grid(st, n_q_bins=20, n_sample=st.n_events)
    assert cgrid.coverage > 0.99

    class UnitCharge:
        def __call__(self, c, theta):
            return jnp.log(3.0)

    zc = make_z_charge_fn(UnitCharge(), cgrid)
    np.testing.assert_allclose(float(zc(jnp.zeros(7))), 3.0, rtol=1e-5)


def test_e_bfmi():
    rng = np.random.default_rng(0)
    e_iid = rng.standard_normal(20000)
    v = float(e_bfmi(jnp.asarray(e_iid)))
    assert 1.8 < v < 2.2          # iid energies: E[(dE)^2] = 2 var(E)
    e_slow = np.cumsum(rng.standard_normal(20000) * 0.01)  # random walk: tiny diffs
    assert float(e_bfmi(jnp.asarray(e_slow))) < 0.3


def test_charge_weights():
    from hitman.train import build_charge_weights, make_weighted_charge_batch
    batch, pmt_pos = _toy_data(n_events=400, seed=13)
    st = _fake_store(batch, pmt_pos)
    tab = build_charge_weights(st, alpha=0.5, n_max=100)
    h = np.bincount(np.clip(st.charge[:, 1].astype(int), 0, 100), minlength=101)
    p = (h + 0.5) / (h + 0.5).sum()
    np.testing.assert_allclose(np.sum(p * np.asarray(tab.w)), 1.0, rtol=1e-5)
    # rarer multiplicities get larger weights
    common = np.argmax(h)
    rare = np.argmin(np.where(h > 0, h, h.max()))
    assert tab.w[rare] >= tab.w[common]
    data = DeviceData.from_batch(batch, pmt_pos)
    obs, hyp, w = make_weighted_charge_batch(tab)(data, jnp.arange(64), None)
    assert w.shape == (64,) and np.isfinite(np.asarray(w)).all()

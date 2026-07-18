import jax
import jax.numpy as jnp
import numpy as np

from hitman.calibrate import (adjust_samples, build_marginal_grid, estimate_sandwich,
                              z_of_theta, znll)
from hitman.nn import HitNet
from hitman.train import DeviceData, hit_batch, train_recipe
from tests.test_resident import _toy_data


def test_train_recipe_stages_and_global_best(tmp_path):
    batch, pmt_pos = _toy_data(n_events=400, seed=41)
    data = DeviceData.from_batch(batch, pmt_pos)
    res = train_recipe(
        HitNet(width=32, depth=2, key=jax.random.PRNGKey(0)),
        data, hit_batch, data.n_hits,
        key=jax.random.PRNGKey(1),
        sgd_batch=512, sgd_max_steps=400, cosine_steps=200, cosine_batch=512,
        val_every=50, patience_steps=200, min_delta=1e-4,
        checkpoint_dir=str(tmp_path), verbose=False,
    )
    stages = {h[0] for h in res.history}
    assert stages == {"sgd", "cosine"}
    # global best is the min over the WHOLE history (both stages, one yardstick)
    assert np.isclose(res.best_val, min(h[2] for h in res.history))
    assert (tmp_path / "best.eqx").exists()
    assert res.best_stage in ("sgd", "cosine")


def test_min_delta_gates_patience():
    batch, pmt_pos = _toy_data(n_events=300, seed=43)
    data = DeviceData.from_batch(batch, pmt_pos)
    # huge min_delta => plateau clock never resets => stage stops at patience_steps
    res = train_recipe(
        HitNet(width=32, depth=2, key=jax.random.PRNGKey(0)),
        data, hit_batch, data.n_hits,
        key=jax.random.PRNGKey(2),
        sgd_batch=512, sgd_max_steps=5000, cosine_steps=100, cosine_batch=512,
        val_every=50, patience_steps=150, min_delta=10.0, verbose=False,
    )
    sgd_steps = [h[1] for h in res.history if h[0] == "sgd"]
    assert max(sgd_steps) <= 200  # stopped by the gated plateau, not max_steps


def test_sandwich_recovers_composite_inflation():
    # d=2 composite likelihood: K correlated Gaussian observations per event treated
    # as independent. Known result: H = K*I, J = K*(1+(K-1)*rho)*I => pull = sqrt(J)/sqrt(H).
    rng = np.random.default_rng(0)
    K, rho, n = 8, 0.5, 40000
    cov = rho * np.ones((K, K)) + (1 - rho) * np.eye(K)
    L = np.linalg.cholesky(cov)
    x = rng.standard_normal((n, K)) @ L.T  # events x K correlated obs, mean 0
    # score of composite loglik for mean param mu at mu=0: sum_k x_k ; hessian = -K
    scores = x.sum(axis=1, keepdims=True)
    hessians = -K * np.ones((n, 1, 1))
    sw = estimate_sandwich(scores, hessians)
    expected_pull = np.sqrt(1 + (K - 1) * rho)
    np.testing.assert_allclose(sw.pull_prediction[0], expected_pull, rtol=0.05)
    # adjusted samples attain the godambe covariance
    fake_samples = rng.standard_normal((5000, 1)) * np.sqrt(sw.fisher_cov[0, 0])
    adj = adjust_samples(fake_samples, np.zeros(1), sw)
    np.testing.assert_allclose(adj.var(), sw.godambe_cov[0, 0], rtol=0.1)


def test_z_of_theta_and_znll():
    batch, pmt_pos = _toy_data(n_events=300, seed=45)

    class FakeStore:
        pass

    st = FakeStore()
    st.n_hits = batch.n_hits
    st.n_events = batch.n_events
    st.pmt_id = np.asarray(batch.pmt_id)
    st.event_id = np.asarray(batch.event_id)
    st.hits = np.asarray(batch.hits)
    st.pmt_pos = pmt_pos
    grid = build_marginal_grid(st, n_sample=batch.n_hits, time_sigma=0.0,
                               t_lo=-40.0, t_hi=60.0, dt=1.0)
    assert grid.coverage > 0.99

    class UnitNet:
        obs_style = "xyz"
        def __call__(self, h, theta):
            return jnp.asarray(0.0)  # r == 1 everywhere

    z = float(z_of_theta(UnitNet(), grid, jnp.zeros(7)))
    np.testing.assert_allclose(z, 1.0, rtol=1e-5)

    class ScaledNet(UnitNet):
        def __call__(self, h, theta):
            return jnp.log(3.0)  # r == 3 everywhere -> Z = 3

    z3 = float(z_of_theta(ScaledNet(), grid, jnp.zeros(7)))
    np.testing.assert_allclose(z3, 3.0, rtol=1e-5)
    # znll: corrected NLL differs by n_hits*log Z; a global scale cancels exactly
    base = lambda th: jnp.asarray(5.0)
    corr = znll(base, ScaledNet(), grid, n_hits=10)
    np.testing.assert_allclose(float(corr(jnp.zeros(7))), 5.0 + 10 * np.log(3.0), rtol=1e-5)


def test_sandwich_null_direction_passthrough():
    # param 0 identified (H=5, J=20 -> pull 2), param 1 unidentified (H ~ 0)
    rng = np.random.default_rng(1)
    n = 20000
    scores = np.stack([2.0 * rng.standard_normal(n), rng.standard_normal(n)], axis=1)
    scores[:, 0] *= np.sqrt(20) / 2.0
    H_ev = np.zeros((n, 2, 2)); H_ev[:, 0, 0] = -5.0; H_ev[:, 1, 1] = -1e-12
    sw = estimate_sandwich(scores, H_ev)
    np.testing.assert_allclose(sw.pull_prediction[0], np.sqrt(20) / np.sqrt(5), rtol=0.05)
    assert np.isnan(sw.pull_prediction[1])
    samples = rng.standard_normal((3000, 2))
    adj = adjust_samples(samples, np.zeros(2), sw)
    # unidentified coordinate passes through untouched; identified one is stretched
    np.testing.assert_allclose(adj[:, 1], samples[:, 1], atol=1e-9)
    np.testing.assert_allclose(adj[:, 0].std() / samples[:, 0].std(),
                               sw.pull_prediction[0], rtol=0.05)

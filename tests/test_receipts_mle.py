import numpy as np

from hitman.wc.receipts.mle import mle_receipt, opening_angles_deg


def test_mle_receipt_bias_resolution_pull_closure():
    rng = np.random.default_rng(0)
    truth = np.array([100.0, -50.0, 200.0, np.pi / 2, 0.5, 1.0, 3.0])
    n = 4000
    fits = np.tile(truth, (n, 1))
    # inject known Gaussian scatter on x, z, t, E with known sigmas
    sig = {0: 80.0, 2: 90.0, 5: 0.7, 6: 0.4}
    sigmas = np.ones((n, 7)) * 1e-6
    for i, s in sig.items():
        fits[:, i] = truth[i] + rng.normal(0, s, n)
        sigmas[:, i] = s

    res = mle_receipt(fits, truth, sigmas=sigmas, mean_nhit=42.0)
    by = {p.name: p for p in res.params}
    for i, s in sig.items():
        name = ["x", "y", "z", "zenith", "azimuth", "t", "E"][i]
        assert abs(by[name].bias) < 0.1 * s          # unbiased
        assert abs(by[name].resolution - s) < 0.1 * s  # IQR/1.35 ~ sigma
        assert abs(by[name].pull_sigma - 1.0) < 0.1    # curvature honesty
    assert res.psi_median_deg < 1e-6  # directions exactly at truth
    assert res.n_events == n


def test_opening_angle_zero_for_identical_direction():
    truth = np.array([0, 0, 0, 1.0, 2.0, 0, 3.0])
    fits = np.tile(truth, (5, 1))
    psi = opening_angles_deg(fits, truth)
    np.testing.assert_allclose(psi, 0.0, atol=1e-6)

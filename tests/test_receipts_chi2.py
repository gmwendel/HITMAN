import numpy as np

from hitman.receipts.chi2 import chi2_comparison


def test_perfect_match_is_zero_chi2():
    obs = np.array([1.0, 2.0, 3.0])
    res = chi2_comparison(obs, np.ones(3), obs.copy(), np.ones(3), ddof=0)
    assert res.chi2 == 0.0
    assert res.chi2_dof == 0.0
    np.testing.assert_allclose(res.pulls, 0.0)


def test_known_offset_chi2():
    # obs=0, pred=1, each err=1 -> var=2, per-bin chi2 = 1/2, over 4 bins.
    obs = np.zeros(4)
    pred = np.ones(4)
    res = chi2_comparison(obs, np.ones(4), pred, np.ones(4), ddof=0)
    assert abs(res.chi2 - 4 * 0.5) < 1e-12
    assert res.dof == 4
    np.testing.assert_allclose(res.pulls, 1 / np.sqrt(2.0))  # (pred-obs)/sqrt(var) > 0


def test_mc_var_share_is_one_when_surrogate_error_zero():
    obs = np.zeros(5)
    res = chi2_comparison(obs, np.ones(5), np.ones(5), np.zeros(5), ddof=0)
    assert abs(res.mc_var_share - 1.0) < 1e-12


def test_mask_controls_dof():
    obs = np.zeros(10)
    mask = np.arange(10) < 6
    res = chi2_comparison(obs, np.ones(10), np.ones(10), np.ones(10), mask=mask, ddof=1)
    assert res.n_selected == 6
    assert res.dof == 5

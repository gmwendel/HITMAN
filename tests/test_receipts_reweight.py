import numpy as np

from hitman.receipts.reweight import (
    group_fractions,
    reweighted_density,
    self_normalized_weights,
    weighted_fraction,
)


def test_uniform_logits_give_unit_norm_and_full_ess():
    pw = self_normalized_weights(np.zeros(1000))
    assert pw.self_norm == 1.0
    np.testing.assert_allclose(pw.weights, np.full(1000, 1e-3))
    assert abs(pw.ess - 1000) < 1e-6


def test_self_norm_reads_mean_ratio():
    # logw = log(2) everywhere -> E[r] = 2 (a mis-normalized/overconfident estimator).
    pw = self_normalized_weights(np.full(500, np.log(2.0)))
    assert abs(pw.self_norm - 2.0) < 1e-9


def test_peaked_weights_have_small_ess():
    logw = np.full(1000, -50.0)
    logw[0] = 0.0  # one dominating member
    pw = self_normalized_weights(logw)
    assert pw.ess < 1.5


def test_weighted_fraction_matches_plain_mean_for_uniform_weights():
    w = np.full(100, 0.01)
    member = np.arange(100) < 40
    p, sigma = weighted_fraction(w, member)
    assert abs(p - 0.4) < 1e-9
    assert sigma > 0


def test_reweighted_density_is_flat_and_normalized_for_uniform_pool():
    rng = np.random.default_rng(0)
    pool = rng.uniform(0.0, 10.0, 200_000)
    pw = self_normalized_weights(np.zeros(len(pool)))
    bins = np.linspace(0.0, 10.0, 21)
    dens, err = reweighted_density(pool, pw.weights, bins)
    # integrates to 1 over the window and is flat at 0.1 /unit
    assert abs(np.sum(dens * np.diff(bins)) - 1.0) < 1e-9
    np.testing.assert_allclose(dens, 0.1, atol=0.01)
    assert np.all(err >= 0)


def test_group_fractions_sum_to_one():
    rng = np.random.default_rng(1)
    ids = rng.integers(0, 5, 10_000)
    pw = self_normalized_weights(rng.normal(size=10_000))
    frac, err = group_fractions(pw.weights, ids, 5)
    assert abs(frac.sum() - 1.0) < 1e-9
    assert frac.shape == (5,) and err.shape == (5,)

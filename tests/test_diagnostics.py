import numpy as np

from hitman.diagnostics import (
    expected_coverage,
    reliability_curve,
    sbc_ranks,
    self_normalization,
)


def test_self_normalization_is_one_for_zero_logits():
    assert self_normalization(np.zeros(1000)) == 1.0


def test_reliability_curve_calibrated_classifier():
    rng = np.random.default_rng(0)
    # A perfectly calibrated classifier: P(joint) = sigmoid(logit) matches label frequency.
    logits = rng.normal(0, 2, 200_000)
    prob = 1 / (1 + np.exp(-logits))
    labels = rng.uniform(size=len(prob)) < prob
    _, _, ece = reliability_curve(logits[labels], logits[~labels])
    assert ece < 0.02


def test_sbc_ranks_uniform_for_calibrated_posterior():
    rng = np.random.default_rng(1)
    n_sims, n_draws, dim = 500, 200, 3
    truths = rng.normal(size=(n_sims, dim))
    # Posterior draws from the same distribution as the truth -> calibrated by construction.
    samples = rng.normal(size=(n_sims, n_draws, dim))
    ranks = sbc_ranks(samples, truths)
    assert ranks.shape == (n_sims, dim)
    levels, coverage = expected_coverage(ranks, n_draws)
    np.testing.assert_allclose(
        coverage, np.broadcast_to(levels[:, None], coverage.shape), atol=0.06
    )

import jax.numpy as jnp
import numpy as np

from hitman.train.losses import classifier_accuracy, nre_loss


def test_bce_at_zero_logits_is_log2():
    logits = jnp.zeros(100)
    np.testing.assert_allclose(float(nre_loss(logits, logits)), np.log(2.0), rtol=1e-6)


def test_balance_penalty_vanishes_when_balanced():
    logits = jnp.zeros(100)  # sigmoid = 0.5 both classes -> sum = 1
    base = float(nre_loss(logits, logits, balance_weight=0.0))
    regularized = float(nre_loss(logits, logits, balance_weight=100.0))
    np.testing.assert_allclose(base, regularized, rtol=1e-6)


def test_balance_penalty_active_for_overconfident_estimator():
    joint = jnp.full(100, 5.0)
    marginal = jnp.full(100, 4.0)  # sigmoid sums to ~1.97, far from 1
    assert float(nre_loss(joint, marginal, balance_weight=1.0)) > float(
        nre_loss(joint, marginal, balance_weight=0.0)
    )


def test_perfect_classifier_accuracy():
    assert float(classifier_accuracy(jnp.full(10, 3.0), jnp.full(10, -3.0))) == 1.0

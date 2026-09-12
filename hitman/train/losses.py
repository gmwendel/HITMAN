"""Ratio-estimation losses.

Binary cross-entropy on logits (numerically stable; equivalent to 1.x's sigmoid head +
BCE), with an optional balancing regularizer (BNRE, arXiv:2208.13624) that pushes the
learned ratio conservative — recommended for inference-grade estimators.
"""

import jax
import jax.numpy as jnp


def nre_loss(
    joint_logits: jnp.ndarray, marginal_logits: jnp.ndarray, balance_weight: float = 0.0,
    weights: jnp.ndarray = None,
) -> jnp.ndarray:
    """Mean BCE for label-1 joint pairs and label-0 marginal pairs (+ BNRE penalty).

    ``weights`` (per-row, depending on the OBSERVATION only — never on theta or the
    pairing) reweights both class terms at the same x, so the Bayes-optimal logit is
    unchanged (the weight cancels in the pointwise minimizer): this redirects
    ACCURACY toward low-p(x) regions without biasing the learned ratio. Weights are
    expected mean-1 under the training marginal (see hitman.wc.train.reweight).
    """
    if weights is None:
        bce = 0.5 * (
            jnp.mean(jax.nn.softplus(-joint_logits)) + jnp.mean(jax.nn.softplus(marginal_logits))
        )
    else:
        bce = 0.5 * (
            jnp.mean(weights * jax.nn.softplus(-joint_logits))
            + jnp.mean(weights * jax.nn.softplus(marginal_logits))
        )
    # The BNRE penalty is skipped entirely when it is switched off. ``balance_weight`` is a
    # static Python float, so this branch resolves at trace time and costs nothing -- while
    # computing it anyway adds two full-batch sigmoids and their reductions to BOTH the
    # forward and the backward pass. At batch 2^18 against a small network that is a
    # measurable fraction of the step, and ``bce + 0.0 * balance**2`` is exactly ``bce`` for
    # any finite balance, so nothing is lost.
    if balance_weight == 0.0:
        return bce
    balance = (
        jnp.mean(jax.nn.sigmoid(joint_logits)) + jnp.mean(jax.nn.sigmoid(marginal_logits)) - 1.0
    )
    return bce + balance_weight * balance**2


def classifier_accuracy(joint_logits: jnp.ndarray, marginal_logits: jnp.ndarray) -> jnp.ndarray:
    """Fraction of correctly classified pairs (threshold at logit 0)."""
    return 0.5 * (jnp.mean(joint_logits > 0) + jnp.mean(marginal_logits <= 0))

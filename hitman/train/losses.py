"""Ratio-estimation losses.

Binary cross-entropy on logits (numerically stable; equivalent to 1.x's sigmoid head +
BCE), with an optional balancing regularizer (BNRE, arXiv:2208.13624) that pushes the
learned ratio conservative — recommended for inference-grade estimators.
"""

import jax
import jax.numpy as jnp


def nre_loss(
    joint_logits: jnp.ndarray, marginal_logits: jnp.ndarray, balance_weight: float = 0.0
) -> jnp.ndarray:
    """Mean BCE for label-1 joint pairs and label-0 marginal pairs (+ BNRE penalty)."""
    bce = 0.5 * (
        jnp.mean(jax.nn.softplus(-joint_logits)) + jnp.mean(jax.nn.softplus(marginal_logits))
    )
    balance = (
        jnp.mean(jax.nn.sigmoid(joint_logits)) + jnp.mean(jax.nn.sigmoid(marginal_logits)) - 1.0
    )
    return bce + balance_weight * balance**2


def classifier_accuracy(joint_logits: jnp.ndarray, marginal_logits: jnp.ndarray) -> jnp.ndarray:
    """Fraction of correctly classified pairs (threshold at logit 0)."""
    return 0.5 * (jnp.mean(joint_logits > 0) + jnp.mean(marginal_logits <= 0))

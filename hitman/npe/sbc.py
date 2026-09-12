"""Calibration diagnostics for the trained flow: SBC rank statistics + TARP coverage.

Both operate on precomputed contexts ``c`` (M, context_dim) paired with the true theta
(M, 7) that generated each x. Use ``hitman.npe.loss.encode`` to build contexts from an
event batch first. Plotting lives elsewhere; these return arrays.

* **SBC** (Talts et al. 2018, arXiv:1804.06788): under a calibrated posterior the rank
  of theta_true among K posterior draws is uniform on {0..K} per coordinate. For the
  circular coordinate the rank uses the raw [0, 2pi) value -- a fixed measurable
  statistic, so uniformity still holds under calibration.
* **TARP** (Lemos et al. 2023, arXiv:2302.03026): distance-to-random-reference test of
  JOINT coverage. Returns the credibility grid and the empirical coverage curve; a
  calibrated posterior lies on the diagonal.
"""

import jax
import jax.numpy as jnp
import numpy as np


def sbc_ranks(flow, context, theta_true, key, n_samples=100):
    """Per-coordinate SBC ranks. Returns (M, n_dim) ints in {0..n_samples}."""
    M = theta_true.shape[0]
    keys = jax.random.split(key, M)

    def one(c, th, k):
        s = flow.sample_n(k, c, n_samples)          # (n_samples, n_dim)
        return jnp.sum(s < th[None, :], axis=0)     # (n_dim,)

    return np.asarray(jax.vmap(one)(context, theta_true, keys))


def _theta_dist(a, b, scales, circular_mask, period):
    """Standardized Euclidean distance with angular difference on circular dims.

    a, b: (..., n_dim). Broadcasts. Returns (...,)."""
    d = a - b
    if circular_mask is not None:
        ang = jnp.abs(jnp.mod(d + period / 2.0, period) - period / 2.0)
        d = jnp.where(circular_mask, ang, d)
    return jnp.sqrt(jnp.sum((d / scales) ** 2, axis=-1))


def tarp_coverage(flow, context, theta_true, key, n_samples=100, n_alpha=21,
                  references=None, scales=None):
    """TARP expected-coverage curve.

    Returns (alpha_grid (n_alpha,), coverage (n_alpha,), atarp (M,)) where ``atarp`` are
    the per-event credibility values whose ECDF is the coverage curve.
    """
    theta_true = jnp.asarray(theta_true)
    M, n_dim = theta_true.shape
    circular_mask = getattr(flow, "circular_mask", None)
    period = getattr(flow, "period", 2.0 * np.pi)

    if scales is None:
        scales = jnp.std(theta_true, axis=0) + 1e-6
    if references is None:
        k_ref, key = jax.random.split(key)
        mean = jnp.mean(theta_true, axis=0)
        references = mean[None, :] + 1.5 * scales[None, :] * jax.random.normal(
            k_ref, (M, n_dim))
    references = jnp.asarray(references)

    keys = jax.random.split(key, M)

    def one(c, th, ref, k):
        s = flow.sample_n(k, c, n_samples)                       # (n_samples, n_dim)
        d_samp = _theta_dist(ref[None, :], s, scales, circular_mask, period)
        d_true = _theta_dist(ref, th, scales, circular_mask, period)
        return jnp.mean((d_samp < d_true).astype(jnp.float32))   # credibility of true

    atarp = np.asarray(jax.vmap(one)(context, theta_true, references, keys))
    alpha = np.linspace(0.0, 1.0, n_alpha)
    coverage = np.array([np.mean(atarp <= a) for a in alpha])
    return alpha, coverage, atarp

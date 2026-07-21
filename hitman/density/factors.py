"""The :class:`DensityFactor` protocol, a discrete-mark softmax factor, and composition.

A per-event likelihood is a product of exactly-normalized factors — one count factor and,
per object, one or more mark factors::

    log L(event | context) = count.log_prob(N | context)
                           + sum_objects [ sum_marks mark_k.log_prob(v_k | context) ]

Each factor is responsible for its OWN closed-form normalization (that is the whole design
invariant); composition is then just addition of log-densities, guaranteed to stay a proper
density. :func:`marked_poisson_loglik` assembles the per-event number, masking padded slots.
"""

from typing import Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
from jax.scipy.special import logsumexp


@runtime_checkable
class DensityFactor(Protocol):
    """A single exactly-normalized likelihood factor.

    ``log_prob(obs, *context)`` returns the log density/pmf of one observation ``obs`` given
    conditioning ``context`` (whatever the factor needs: spline node values, a softmax logit
    vector, a mark intensity + yield variable, ...). Normalization is closed-form and exact
    at every ``(obs, context)`` — no Monte Carlo, no grid in the data dimension.

    Conforming factors here: :class:`SoftmaxMarkFactor` (discrete mark),
    :class:`hitman.density.logspline.LogSplineFactor` (continuous 1-D mark),
    :class:`hitman.density.count.CountFactor` (total count). RQ-spline angle factors
    (``hitman.npe.flow.unconstrained_rqs`` / ``circular_rqs``) also conform.
    """

    def log_prob(self, obs, *context): ...


class SoftmaxMarkFactor(eqx.Module):
    """Exact discrete-mark factor: log p(s | eta) = eta[s] - logsumexp(eta).

    Stateless — the per-object logits ``eta`` (one per sensor/category) are the context,
    produced by the model's conditioner. This is the water-Cherenkov sensor factor; a
    continuous-mark detector simply omits it.
    """

    def log_prob(self, s, eta):
        """log p(s | eta) for a discrete mark index ``s``."""
        return eta[s] - logsumexp(eta)

    def log_partition(self, eta):
        """log sum_s exp(eta_s) — the (yield-free) log mark intensity."""
        return logsumexp(eta)


def marked_poisson_loglik(count_ll, mark_lls, mask=None):
    """Compose one event's log-likelihood: ``count_ll + sum(mark_lls)``.

    Parameters
    ----------
    count_ll : scalar
        The count factor's ``log_prob(N | context)``.
    mark_lls : (n_obj,) array
        Per-object summed mark log-densities (already summed over each object's mark
        factors). For padded batches, pad to a fixed ``n_obj`` and pass ``mask``.
    mask : (n_obj,) array or None
        1.0 for real objects, 0.0 for padded slots.
    """
    ll = mark_lls if mask is None else mark_lls * mask
    return count_ll + jnp.sum(ll)

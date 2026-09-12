"""Closed-form 1-D log-spline density factor (piecewise-exponential, exact partition).

The reusable, coordinate-agnostic core of the splinemle time density. ``ell`` is continuous
piecewise-linear on FIXED knots ``u_0 < ... < u_J``; the node values ``ell_j`` are the
per-object *context* (produced by whatever conditioner the model owns). The partition
``Z = integral e^{ell}`` is the exact analytic integral of the piecewise-exponential, so the
factor is normalized by construction and differentiable, with hard support ``[u_0, u_J]``.

These functions moved verbatim out of ``hitman.wc.splinemle.model`` (which re-exports them for
backward compatibility); the numerics are unchanged. :class:`LogSplineFactor` wraps them as
a :class:`hitman.density.factors.DensityFactor` over one continuous 1-D mark ``u`` given
``node_vals``. It is the ``f(dt | Theta)`` / bounded-angle building block for a
chain-of-conditionals likelihood.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

# Loss-side finite floor for the log density at out-of-support u. The TRUE model has zero
# density there (log = -inf); on finite data a finite floor keeps a stray/padded slot a
# large-but-finite penalty with clean gradients. Receipts/equivalence use floor=-inf (the
# exact hard-support model).
SUPPORT_FLOOR = -40.0


def log_expm1_over_x(z):
    """log(expm1(z) / z): C-infinity, -> 0 as z -> 0, overflow-safe for large |z|.

    expm1(z)/z > 0 for all real z, so its log is everywhere defined. Near 0 we use the
    Taylor series log((e^z-1)/z) = z/2 + z^2/24 - z^4/2880 + O(z^6). Away from 0 we use the
    overflow-safe identity |e^z - 1| = e^{max(z,0)} * (-expm1(-|z|)), i.e.
    log(expm1(z)/z) = max(z,0) + log(-expm1(-|z|)) - log|z|. The double-``where`` keeps the
    unused branch NaN-free so reverse-mode gradients are finite at every z (incl. exactly 0).
    """
    small = jnp.abs(z) < 1e-2
    z_big = jnp.where(small, 1.0, z)  # dummy in the small region; result discarded there
    az = jnp.abs(z_big)
    log_abs_expm1 = jnp.maximum(z_big, 0.0) + jnp.log(-jnp.expm1(-az))
    big = log_abs_expm1 - jnp.log(az)
    series = z * (0.5 + z * (1.0 / 24.0 - z * z / 2880.0))
    return jnp.where(small, series, big)


def interval_log_integrals(node_vals, knots):
    """Per-interval log integral of e^{ell} where ell is piecewise-linear.

    On [u_j, u_{j+1}]: ell(u) = a_j + b_j (u - u_j) with a_j = node_vals[j],
    b_j = (node_vals[j+1]-node_vals[j])/dk_j. The integral is
    e^{a_j} dk_j * expm1(b_j dk_j)/(b_j dk_j); note b_j dk_j = node_vals[j+1]-node_vals[j].
    Returns (J,) log integrals. Exact at b_j = 0 (uniform interval) via the series switch.
    """
    dk = knots[1:] - knots[:-1]
    a = node_vals[:-1]
    z = node_vals[1:] - node_vals[:-1]  # == b_j * dk_j
    return a + jnp.log(dk) + log_expm1_over_x(z)


def logZ_time(node_vals, knots):
    """Exact log partition of the log-spline density: logsumexp of interval integrals."""
    return logsumexp(interval_log_integrals(node_vals, knots))


def ell_at(u, node_vals, knots):
    """Piecewise-linear ell(u). ``jnp.interp`` clamps to the edge values outside the knot
    span; callers mask the out-of-support region explicitly, so the clamp is never used."""
    return jnp.interp(u, knots, node_vals)


def log_prob_u(u, node_vals, knots, floor=-jnp.inf):
    """log p_hat(u) = ell(u) - log Z inside [u_0, u_J], else ``floor`` (-inf = true model).
    Because u is an affine reparametrization of the mark with unit Jacobian, this is also
    the mark log-density."""
    inside = (u >= knots[0]) & (u <= knots[-1])
    lp = ell_at(u, node_vals, knots) - logZ_time(node_vals, knots)
    return jnp.where(inside, lp, floor)


def density_u(u, node_vals, knots):
    """p_hat(u) = exp(log_prob_u); zero outside support. Vectorizes over ``u`` arrays."""
    lp = jax.vmap(lambda uu: log_prob_u(uu, node_vals, knots))(jnp.atleast_1d(u))
    return jnp.exp(lp)


class LogSplineFactor(eqx.Module):
    """Exactly-normalized 1-D log-spline density factor over a continuous mark ``u``.

    Stateless apart from its fixed ``knots`` (static) and out-of-support ``floor`` (static);
    the shape parameters (``node_vals``, one per knot) are the per-object *context* supplied
    by the model's conditioner at call time. Conforms to
    :class:`hitman.density.factors.DensityFactor`.

    Examples
    --------
    >>> f = LogSplineFactor(knots=(-1.0, 0.0, 1.0))
    >>> lp = f.log_prob(0.2, node_vals)          # exactly normalized in u
    >>> Z = f.log_partition(node_vals)           # closed-form log partition
    """

    knots: tuple = eqx.field(static=True)
    floor: float = eqx.field(static=True, default=-jnp.inf)

    def __init__(self, knots, floor=-jnp.inf):
        self.knots = tuple(float(k) for k in knots)
        self.floor = float(floor)

    @property
    def knots_arr(self) -> jnp.ndarray:
        return jnp.asarray(self.knots, jnp.float32)

    @property
    def n_nodes(self) -> int:
        return len(self.knots)

    def log_prob(self, u, node_vals, floor=None):
        """log p(u | node_vals); zero (``floor``) outside the hard support."""
        fl = self.floor if floor is None else floor
        return log_prob_u(u, node_vals, self.knots_arr, fl)

    def log_partition(self, node_vals):
        """Exact log partition log Z(node_vals)."""
        return logZ_time(node_vals, self.knots_arr)

    def density(self, u, node_vals):
        """p(u | node_vals) on an arbitrary continuous ``u`` array (zero out of support)."""
        return density_u(u, node_vals, self.knots_arr)

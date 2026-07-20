"""run20 model: a per-hit conditional density with CLOSED-FORM normalization.

WHY (the theorem four estimators paid for)
------------------------------------------
Normalization cannot be enforced by SAMPLING in any variable: shuffled-x NWJ ran away
(twice), and exact-in-x quadrature with theta-subsampling memorized matched pairs
(Var(d_E f) exploded 5e4x). The only cure is a model whose every factor normalizes
ANALYTICALLY, identically, in the forward pass, at every (x, theta) it ever evaluates —
no Monte Carlo and no grid in the data dimension. This module is that model.

THE MODEL
---------
An observation is a hit = (sensor s in {0..240}, arrival time t in R). A hypothesis is
theta = (x, y, z, zenith, azimuth, t0, E). Every hit factorizes as

    p_hat(s, t | theta) = p_hat(s | theta) * p_hat(t | s, theta)

and the event is a marked Poisson process over sensors:

    log L(theta) = sum_hits [ log p_hat(s_i | theta) + log p_hat(t_i | s_i, theta) ]
                 + log Pois(N | Lambda(theta))

* TOF-residual coordinate.  u = t - t_geo(s, theta),  t_geo = t0 + n_eff |x_pmt - x_vtx|/c,
  with n_eff a LEARNABLE scalar (init 1.40; prior campaign measured ~1.38-1.42). In u the
  density is quasi-stationary (prompt peak near u ~ 0 for every sensor/theta), so ~22 knots
  suffice.

* Time factor -- log-spline density with a closed-form partition.  log p_hat(t|s,theta) =
  ell(u) - log Z_t, where ell is continuous piecewise-linear on FIXED knots u_0<...<u_J and
  the node values ell_j = c_j(s,theta) come from the conditioner net. Z_t is the exact
  analytic integral of the piecewise-exponential e^{ell}, so the density is normalized by
  construction and differentiable. Outside [u_0, u_J] the density is ZERO (hard support;
  <1e-4 of hits fall outside on this store -- verified, see the driver header).

* Sensor factor -- exact softmax.  log p_hat(s|theta) = eta_s(theta) - logsumexp_s' eta_s'.
  241 terms, exact.

* Count factor -- marked Poisson process.  mu_s = exp(eta_s), Lambda = sum_s mu_s =
  exp(logsumexp eta); N ~ Poisson(Lambda). This ties the count and sensor factors to the
  SAME eta (no separate ChargeNet, no charge grid) and is fully closed form. Note the
  N*log Lambda terms of the softmax and the Poisson cancel: sensor+count reduces to the
  canonical sum_i eta_{s_i} - Lambda - log N!. (Over-dispersion is a one-parameter NB2
  extension -- add a trainable log_r -- deferred; Poisson is the simplest closed form.)

* Conditioner net.  Per (s, theta) a small MLP maps five hypothesis-conditional O(2)
  invariants -- distance d, incidence cos(h.n), direction cos(e.n), Cherenkov cos(e.h), and
  E-1 -- to (eta_s, c_0..c_J). It is evaluated for ALL 241 sensors per theta (the softmax
  needs them anyway): batch (B_theta, 241, 5) -> (B_theta, 241, J+2).

Everything here is closed form and differentiable. If you ever find yourself adding a
sampled or gridded normalizer, STOP -- that is the forbidden move this whole design exists
to avoid.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, logsumexp

from hitman.nn import features as ft
from hitman.nn.mlp import get_activation

C_MM_PER_NS = 299.792458

# Fixed knots spanning the TOF-residual support, dense near the prompt peak (u ~ 0) and
# sparse across the long scattered/reflected-light tail. Chosen from the measured u
# distribution on water_5M (median ~0.1 ns, 99.999 pct ~85 ns, min ~-4 ns): support
# [-20, 110] leaves ~2e-6 of hits out-of-support (< 1e-4; MLE self-suppresses the empty
# far-left/right nodes because they only add to Z_t with no data to reward them).
DEFAULT_KNOTS = (
    -20.0, -8.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.4, 0.8, 1.2, 1.8, 2.5,
    3.5, 5.0, 7.0, 10.0, 14.0, 20.0, 28.0, 40.0, 55.0, 75.0, 110.0,
)

N_COND_FEATURES = 5

# Loss-side finite floor for the log time-density at out-of-support u. The TRUE model has
# zero density there (log = -inf); on finite data we replace -inf by this finite value so a
# stray hit (or a masked pad slot) contributes a large-but-finite penalty and gradients stay
# clean. The receipt tests use floor=-inf to check the exact hard-support model.
SUPPORT_FLOOR = -40.0


# ---------------------------------------------------------------------------
# Closed-form pieces (pure functions, all differentiable)
# ---------------------------------------------------------------------------

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
    """Exact log partition of the log-spline time density: logsumexp of interval integrals."""
    return logsumexp(interval_log_integrals(node_vals, knots))


def ell_at(u, node_vals, knots):
    """Piecewise-linear ell(u). ``jnp.interp`` clamps to the edge values outside the knot
    span; callers mask the out-of-support region explicitly, so the clamp is never used."""
    return jnp.interp(u, knots, node_vals)


def log_prob_u(u, node_vals, knots, floor=-jnp.inf):
    """log p_hat(u) = ell(u) - log Z_t inside [u_0, u_J], else ``floor`` (-inf = true model).
    Because u = t - t_geo is an affine reparametrization of t with unit Jacobian, this is
    also log p_hat(t | s, theta)."""
    inside = (u >= knots[0]) & (u <= knots[-1])
    lp = ell_at(u, node_vals, knots) - logZ_time(node_vals, knots)
    return jnp.where(inside, lp, floor)


def density_u(u, node_vals, knots):
    """p_hat(u) = exp(log_prob_u); zero outside support. Vectorizes over ``u`` arrays."""
    lp = jax.vmap(lambda uu: log_prob_u(uu, node_vals, knots))(jnp.atleast_1d(u))
    return jnp.exp(lp)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class _CondMLP(eqx.Module):
    """Plain MLP with a linear VECTOR head and a STATIC activation name.

    Mirrors ``hitman.nn.mlp.MLP`` (static ``act`` string, not a callable leaf) so the model
    survives the ``jax.make_jaxpr`` pre-flight walk and ``tree_deserialise_leaves`` into a
    default template -- ``eqx.nn.MLP`` stores the activation as a non-static callable leaf,
    which both break.
    """

    layers: tuple
    act: str = eqx.field(static=True, default="mish")

    def __init__(self, in_size: int, out_size: int, width: int, depth: int, *, key,
                 activation: str = "mish"):
        keys = jax.random.split(key, depth + 1)
        sizes = (in_size, *([width] * depth), out_size)
        self.layers = tuple(
            eqx.nn.Linear(sizes[i], sizes[i + 1], key=keys[i]) for i in range(depth + 1))
        get_activation(activation)  # validate early
        self.act = activation

    def __call__(self, x):
        f = get_activation(self.act)
        for layer in self.layers[:-1]:
            x = f(layer(x))
        return self.layers[-1](x)


class SplineMLE(eqx.Module):
    """Marked-Poisson event density with a closed-form log-spline time factor.

    Owns the learnable log(n_eff), the conditioner MLP, and the (stop-gradient) per-sensor
    geometry tables (position + normal). Knots are a static field, so a serialized model
    round-trips through the default template like ``MLP.act``.
    """

    log_n_eff: jnp.ndarray
    mlp: _CondMLP
    pmt_pos: jnp.ndarray
    pmt_normal: jnp.ndarray
    knots: tuple = eqx.field(static=True)
    count_model: str = eqx.field(static=True)

    def __init__(self, pmt_pos, pmt_normal, *, key, knots=DEFAULT_KNOTS, width: int = 192,
                 depth: int = 3, n_eff_init: float = 1.40, activation: str = "mish",
                 count_model: str = "poisson"):
        self.knots = tuple(float(k) for k in knots)
        self.log_n_eff = jnp.log(jnp.asarray(n_eff_init, jnp.float32))
        n_nodes = len(self.knots)
        self.mlp = _CondMLP(N_COND_FEATURES, n_nodes + 1, width, depth,
                            activation=activation, key=key)
        self.pmt_pos = jnp.asarray(pmt_pos, jnp.float32)
        self.pmt_normal = jnp.asarray(pmt_normal, jnp.float32)
        if count_model != "poisson":
            raise ValueError(f"unsupported count_model {count_model!r} (only 'poisson')")
        self.count_model = count_model

    @property
    def n_eff(self) -> jnp.ndarray:
        return jnp.exp(self.log_n_eff)

    @property
    def knots_arr(self) -> jnp.ndarray:
        return jnp.asarray(self.knots, jnp.float32)

    # -- geometry / conditioner ------------------------------------------------
    def _sensor_features(self, pos, nrm, theta):
        """(pmt pos (3,), normal (3,), theta (7,)) -> (5,) invariants, distance d."""
        rvec = pos - theta[:3]
        d = jnp.linalg.norm(rvec) + 1e-6
        h = rvec / d                       # unit vertex -> PMT (direct-light travel dir)
        e = ft.direction(theta)
        feats = jnp.stack([
            d / ft.POSITION_SCALE,
            jnp.dot(h, nrm),               # incidence  cos(h, n)
            jnp.dot(e, nrm),               # direction  cos(e, n)
            jnp.dot(e, h),                 # Cherenkov  cos(e, h)
            theta[ft.ENERGY] - 1.0,
        ])
        return feats, d

    def event_tables(self, theta):
        """All-sensor conditioner for one theta: (eta (241,), nodes (241, J+1),
        t_geo (241,), logZt (241,)). Evaluated for every sensor (the softmax needs them)."""
        pos = jax.lax.stop_gradient(self.pmt_pos)
        nrm = jax.lax.stop_gradient(self.pmt_normal)
        knots = self.knots_arr
        n_eff = self.n_eff
        t0 = theta[ft.TIME]

        def one(p, n):
            feats, d = self._sensor_features(p, n, theta)
            out = self.mlp(feats)
            eta = out[0]
            nodes = out[1:]
            t_geo = t0 + n_eff * d / C_MM_PER_NS
            return eta, nodes, t_geo, logZ_time(nodes, knots)

        return jax.vmap(one)(pos, nrm)

    # -- per-hit / per-event log-densities (convenience; used by tests/receipts) --
    def log_prob_time(self, t, pmt_id, theta, floor=-jnp.inf):
        """log p_hat(t | s, theta) for one hit. Closed-form normalized, zero out of support."""
        pos = self.pmt_pos[pmt_id]
        nrm = self.pmt_normal[pmt_id]
        feats, d = self._sensor_features(pos, nrm, theta)
        nodes = self.mlp(feats)[1:]
        u = t - (theta[ft.TIME] + self.n_eff * d / C_MM_PER_NS)
        return log_prob_u(u, nodes, self.knots_arr, floor)

    def density_time(self, t_array, pmt_id, theta):
        """Vectorized p_hat(t | s, theta) on an arbitrary continuous ``t`` array."""
        return jax.vmap(lambda tt: jnp.exp(self.log_prob_time(tt, pmt_id, theta)))(
            jnp.atleast_1d(t_array))

    def log_prob_sensor(self, pmt_id, theta):
        """log p_hat(s | theta) = eta_s - logsumexp eta. Exact softmax over 241 sensors."""
        eta, _, _, _ = self.event_tables(theta)
        return eta[pmt_id] - logsumexp(eta)

    def log_prob_hit(self, t, pmt_id, theta, floor=-jnp.inf):
        """log p_hat(s, t | theta) for one hit (sensor + time factors)."""
        eta, nodes, t_geo, logZt = self.event_tables(theta)
        ls = eta[pmt_id] - logsumexp(eta)
        lt = log_prob_u(t - t_geo[pmt_id], nodes[pmt_id], self.knots_arr, floor)
        return ls + lt

    def log_count(self, N, log_Lambda):
        """log Pois(N | Lambda) with log_Lambda = logsumexp(eta) (marked-Poisson tie)."""
        return N * log_Lambda - jnp.exp(log_Lambda) - gammaln(N + 1.0)

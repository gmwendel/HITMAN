"""Count factor: NB2/Poisson total-count pmf + a monotone yield head.

The reusable count model extracted from ``hitman.splinemle.model``. The N objects of an
event are iid draws from the per-object mark density, so the multinomial-given-N factorizes
away and the TOTAL count is a SEPARATE factor: ``N ~ NB2(mean=Lambda, dispersion r)`` (or
Poisson). The mean carries a MONOTONE yield head ``phi(x)`` (cumulative-softplus increments)
that lives ONLY in ``Lambda`` — an additive intensity scale cancels in a softmax mark
factor, so ``phi`` touches neither the mark factors nor the mark densities. ``x`` is the
monotone conditioning variable (water-Cherenkov: energy E).

Pure functions (all differentiable, all closed-form) plus :class:`CountFactor`, an
``eqx.Module`` bundling the trainable yield/dispersion parameters as a standalone,
composable :class:`hitman.density.factors.DensityFactor`. ``hitman.splinemle.model`` keeps
its own copies of these parameters and calls the same pure functions, so the monolithic and
the composed count log-likelihoods are identical.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln

# Default yield-head knots on [0, 10] MeV (WC) and the dispersion linearization anchor.
DEFAULT_PHI_KNOTS = (0.0, 0.75, 1.5, 2.5, 3.5, 5.0, 7.0, 10.0)
E_REF_DISP = 5.0


# ---------------------------------------------------------------------------
# Closed-form pieces (pure functions)
# ---------------------------------------------------------------------------
def monotone_phi_values(phi_e0, phi_raw):
    """Non-decreasing knot values phi_e0 + cumulative softplus(phi_raw)."""
    inc = jax.nn.softplus(phi_raw)
    return phi_e0 + jnp.concatenate([jnp.zeros(1, inc.dtype), jnp.cumsum(inc)])


def phi_at(x, phi_knots, phi_values):
    """Monotone piecewise-linear log-yield scale phi(x); lives ONLY in the count mean."""
    return jnp.interp(x, phi_knots, phi_values)


def affine_log_dispersion(x, disp, x_ref=E_REF_DISP):
    """Affine log-dispersion log r(x) = disp[0] + disp[1] (x - x_ref); r -> inf is Poisson."""
    return disp[0] + disp[1] * (x - x_ref)


def poisson_log_count(N, log_Lambda):
    """log Pois(N | Lambda) with log mean ``log_Lambda``."""
    return N * log_Lambda - jnp.exp(log_Lambda) - gammaln(N + 1.0)


def nb2_log_count(N, log_Lambda, log_r):
    """log NB2(N | mean Lambda, size r); Var = Lambda + Lambda^2/r, Fano = 1 + Lambda/r."""
    r = jnp.exp(log_r)
    log_r_plus_mu = jnp.logaddexp(log_r, log_Lambda)  # log(r + Lambda), stable
    return (gammaln(N + r) - gammaln(r) - gammaln(N + 1.0)
            + r * (log_r - log_r_plus_mu) + N * (log_Lambda - log_r_plus_mu))


def _softplus_inverse(y):
    """Inverse of softplus for y > 0: log(expm1(y)). Used to seed monotone increments."""
    return jnp.log(jnp.expm1(jnp.asarray(y, jnp.float32)))


def init_phi_params(phi_knots, phi_init_values, phi_offset, phi_anchor):
    """Seed (phi_e0, phi_raw) for the monotone phi(x) spline.

    ``phi_init_values`` (K,) are target log-yields at ``phi_knots`` (from a harvested
    curve); if None, phi initializes near-flat (phi == 0). The pre-softplus increments
    encode the SHAPE/concavity; the absolute level (phi_e0) is degenerate with the mark
    intensity level and is only a starting point:
      * ``phi_anchor=(x_a, log_meanN_a)`` centers init logLambda(x_a) ~ log_meanN_a assuming
        the mark-intensity logsumexp ~ log(phi_offset). Preferred.
      * else phi_e0 = logY(knot0) - log(phi_offset).
    """
    K = len(phi_knots)
    if phi_init_values is None:
        inc = np.full(K - 1, 1e-4, np.float32)          # near-flat
        v0 = 0.0
    else:
        v = np.asarray(phi_init_values, np.float64)
        inc = np.maximum(np.diff(v), 1e-3).astype(np.float32)  # positive => monotone init
        v0 = float(v[0])
    phi_raw = _softplus_inverse(jnp.asarray(inc, jnp.float32))
    if phi_anchor is not None:
        x_a, log_meanN_a = float(phi_anchor[0]), float(phi_anchor[1])
        cum = np.concatenate([[0.0], np.cumsum(inc)])   # phi shape relative to phi_e0
        cum_at_a = float(np.interp(x_a, np.asarray(phi_knots, float), cum))
        phi_e0 = log_meanN_a - np.log(phi_offset) - cum_at_a
    else:
        phi_e0 = v0 - np.log(phi_offset)
    return jnp.asarray(phi_e0, jnp.float32), phi_raw


# ---------------------------------------------------------------------------
# The composable count factor
# ---------------------------------------------------------------------------
class CountFactor(eqx.Module):
    """NB2/Poisson total-count factor with a monotone yield head.

    Bundles the trainable parameters (``phi_e0``, ``phi_raw``, ``disp``) so the count model
    is a standalone :class:`hitman.density.factors.DensityFactor` a downstream likelihood can
    compose. ``log_prob(N, log_intensity, x)`` returns the exactly-normalized log pmf, where
    ``log_intensity`` is the (yield-free) log mark intensity (e.g. logsumexp(eta) for a
    softmax mark, or 0 for a purely continuous-mark detector) and ``x`` is the monotone yield
    conditioning variable. The mean is ``Lambda = exp(phi(x) + log_intensity)``.
    """

    phi_e0: jnp.ndarray            # base of the monotone phi(x) yield spline
    phi_raw: jnp.ndarray           # (K_phi-1,) pre-softplus increments (monotone param)
    disp: jnp.ndarray              # (2,) affine log-dispersion in x: [disp0, disp1]
    phi_knots: tuple = eqx.field(static=True)
    count_model: str = eqx.field(static=True)
    x_ref: float = eqx.field(static=True)

    def __init__(self, *, phi_knots=DEFAULT_PHI_KNOTS, phi_init_values=None,
                 phi_offset: float = 241.0, phi_anchor=None, disp_init=(3.5, 0.15),
                 count_model: str = "nbinom", x_ref: float = E_REF_DISP):
        if count_model not in ("poisson", "nbinom"):
            raise ValueError(f"unsupported count_model {count_model!r}")
        self.phi_knots = tuple(float(k) for k in phi_knots)
        self.count_model = count_model
        self.x_ref = float(x_ref)
        phi_e0, phi_raw = init_phi_params(self.phi_knots, phi_init_values, phi_offset,
                                          phi_anchor)
        self.phi_e0 = phi_e0
        self.phi_raw = phi_raw
        self.disp = jnp.asarray(disp_init, jnp.float32)

    @property
    def phi_knots_arr(self) -> jnp.ndarray:
        return jnp.asarray(self.phi_knots, jnp.float32)

    @property
    def phi_values(self) -> jnp.ndarray:
        """Monotone (non-decreasing) knot values: phi_e0 + cumulative softplus increments."""
        return monotone_phi_values(self.phi_e0, self.phi_raw)

    def phi(self, x):
        """Monotone piecewise-linear log-yield scale phi(x). Lives ONLY in the count mean."""
        return phi_at(x, self.phi_knots_arr, self.phi_values)

    def log_dispersion(self, x):
        """Affine log-dispersion log r(x) = disp0 + disp1 (x - x_ref)."""
        return affine_log_dispersion(x, self.disp, self.x_ref)

    def log_mean(self, log_intensity, x):
        """log Lambda = phi(x) + log_intensity (the yield head modulates only the mean)."""
        return self.phi(x) + log_intensity

    def log_count(self, N, log_Lambda, x):
        """log p(N) given the log mean ``log_Lambda`` and yield variable ``x``."""
        if self.count_model == "poisson":
            return poisson_log_count(N, log_Lambda)
        return nb2_log_count(N, log_Lambda, self.log_dispersion(x))

    def log_prob(self, N, log_intensity, x):
        """DensityFactor entry point: log p(N | mark intensity, yield variable)."""
        return self.log_count(N, self.log_mean(log_intensity, x), x)

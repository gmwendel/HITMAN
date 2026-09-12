"""Conditional normalizing flow over 7-D theta, native equinox (no external flow dep).

Architecture (Durkan et al. 2019, "Neural Spline Flows", arXiv:1906.04032):

* **Rational-quadratic spline coupling layers** with alternating binary masks. Each
  layer keeps the *identity* dims fixed and transforms the *transform* dims with a
  monotonic RQ spline whose knot parameters are produced by a conditioner MLP fed the
  identity dims and the context vector ``c(x)``.
* **Domain handling** maps the bounded 7-D hypothesis into the flow's working space
  BEFORE any coupling layer, with the exact per-coordinate Jacobian folded into
  ``log_prob``:
    - x, y, z, t, E  : affine to (0, 1) over a prior box, then logit -> R.
      Base density on these is standard normal; the coupling splines use linear tails
      outside [-tail_bound, tail_bound] so the map is a bijection of all of R.
    - zenith         : cos(zen) in [-1, 1] (monotone on [0, pi]) -> affine (0,1) ->
      logit -> R. The sin(zen) factor from d cos/d zen is in the domain Jacobian.
    - azimuth        : PERIODIC. Kept on the circle [0, 2pi); the coupling layers use a
      *circular* RQ spline (periodic boundary derivative, Rezende & Racaniere 2020,
      arXiv:2002.02428) and the base density on this coordinate is uniform on [0, 2pi).
      This is the principled alternative to (sin,cos) embedding, which is NOT a valid
      7-D density (it lifts a 1-D circle into a 2-D plane, so has no 7-D Jacobian).

Both directions are exact and tested: ``log_prob(theta | c)`` (all Jacobians) and
``sample(key, c)``.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp

MIN_BIN = 1e-3      # floor on normalized spline bin width/height
MIN_DERIV = 1e-3    # floor on knot derivatives
_EPS = 1e-7
_LOG2PI = math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Rational-quadratic spline core (single scalar; batch/vmap externally)
# ---------------------------------------------------------------------------
def _rqs_scalar(x, widths, heights, derivatives, inverse, left, right, bottom, top):
    """Monotone RQ spline on [left,right] x [bottom,top] for one scalar.

    ``widths`` (K,) sum to (right-left); ``heights`` (K,) sum to (top-bottom);
    ``derivatives`` (K+1,) are the positive knot slopes. Returns (out, logabsdet).
    Formulae: Durkan et al. (2019) App. A; inverse from the nflows reference.
    """
    left = jnp.asarray(left, x.dtype)
    right = jnp.asarray(right, x.dtype)
    bottom = jnp.asarray(bottom, x.dtype)
    top = jnp.asarray(top, x.dtype)

    cumw = jnp.concatenate([left[None], left + jnp.cumsum(widths)])
    cumw = cumw.at[-1].set(right)
    cumh = jnp.concatenate([bottom[None], bottom + jnp.cumsum(heights)])
    cumh = cumh.at[-1].set(top)

    edges = cumh if inverse else cumw
    idx = jnp.clip(jnp.sum(x >= edges[1:]), 0, widths.shape[0] - 1)

    xk, xk1 = cumw[idx], cumw[idx + 1]
    yk, yk1 = cumh[idx], cumh[idx + 1]
    bw = xk1 - xk
    bh = yk1 - yk
    s = bh / bw
    dk, dk1 = derivatives[idx], derivatives[idx + 1]

    if inverse:
        dy = x - yk
        a = dy * (dk1 + dk - 2.0 * s) + bh * (s - dk)
        b = bh * dk - dy * (dk1 + dk - 2.0 * s)
        c = -s * dy
        disc = jnp.maximum(b * b - 4.0 * a * c, 0.0)
        xi = 2.0 * c / (-b - jnp.sqrt(disc))
        out = xi * bw + xk
        xi1 = 1.0 - xi
        den = s + (dk1 + dk - 2.0 * s) * xi * xi1
        dnum = s * s * (dk1 * xi * xi + 2.0 * s * xi * xi1 + dk * xi1 * xi1)
        logabsdet = -(jnp.log(dnum) - 2.0 * jnp.log(den))
        return out, logabsdet

    xi = (x - xk) / bw
    xi1 = 1.0 - xi
    num = bh * (s * xi * xi + dk * xi * xi1)
    den = s + (dk1 + dk - 2.0 * s) * xi * xi1
    out = yk + num / den
    dnum = s * s * (dk1 * xi * xi + 2.0 * s * xi * xi1 + dk * xi1 * xi1)
    logabsdet = jnp.log(dnum) - 2.0 * jnp.log(den)
    return out, logabsdet


def _normalize_bins(unnorm, span, k):
    """Softmax -> floored -> scaled so bins sum to ``span``."""
    p = jax.nn.softmax(unnorm)
    p = MIN_BIN + (1.0 - MIN_BIN * k) * p
    return p * span


def unconstrained_rqs(x, uw, uh, ud, inverse, tail_bound):
    """Real-line RQ spline with linear (slope-1) tails outside [-B, B]."""
    k = uw.shape[0]
    b = tail_bound
    inside = (x > -b) & (x < b)
    widths = _normalize_bins(uw, 2.0 * b, k)
    heights = _normalize_bins(uh, 2.0 * b, k)
    one = jnp.ones((1,), x.dtype)
    d_int = MIN_DERIV + jax.nn.softplus(ud)          # (K-1,)
    derivs = jnp.concatenate([one, d_int, one])      # boundary slope 1 -> C1 linear tail
    x_safe = jnp.where(inside, x, 0.0)
    out_in, ld_in = _rqs_scalar(x_safe, widths, heights, derivs, inverse, -b, b, -b, b)
    out = jnp.where(inside, out_in, x)
    ld = jnp.where(inside, ld_in, 0.0)
    return out, ld


def circular_rqs(x, uw, uh, ud, inverse, period):
    """Circular RQ spline: diffeomorphism of [0, period) with periodic derivative."""
    k = uw.shape[0]
    widths = _normalize_bins(uw, period, k)
    heights = _normalize_bins(uh, period, k)
    d = MIN_DERIV + jax.nn.softplus(ud)              # (K,)
    derivs = jnp.concatenate([d, d[:1]])             # d_K = d_0 -> periodic, matched slope
    x = jnp.mod(x, period)
    out, ld = _rqs_scalar(x, widths, heights, derivs, inverse, 0.0, period, 0.0, period)
    return jnp.mod(out, period), ld


# ---------------------------------------------------------------------------
# Domain transform: bounded/periodic theta <-> flow working space
# ---------------------------------------------------------------------------
class DomainTransform(eqx.Module):
    """Fixed (non-trainable) per-coordinate map into the flow's working space.

    ``kind`` is one of 'box' (affine+logit over [lo,hi]), 'cos' (cos then affine+logit;
    for a polar angle in [0, pi]) or 'circ' (identity on the circle). All Jacobians are
    exact; only interior points are guaranteed invertible (boundary values logit to
    +-inf and are clipped).
    """

    lo: tuple = eqx.field(static=True)
    hi: tuple = eqx.field(static=True)
    kind: tuple = eqx.field(static=True)
    period: float = eqx.field(static=True)
    n_dim: int = eqx.field(static=True)

    def __init__(self, lo, hi, kind, period):
        self.lo = tuple(float(v) for v in lo)
        self.hi = tuple(float(v) for v in hi)
        self.kind = tuple(kind)
        self.period = float(period)
        self.n_dim = len(kind)

    def forward(self, theta):
        """theta -> (u, log|det du/dtheta|)."""
        us, ld = [], 0.0
        for i in range(self.n_dim):
            v = theta[i]
            k = self.kind[i]
            if k == "box":
                span = self.hi[i] - self.lo[i]
                p = jnp.clip((v - self.lo[i]) / span, _EPS, 1.0 - _EPS)
                us.append(jnp.log(p) - jnp.log1p(-p))
                ld = ld - math.log(span) - jnp.log(p) - jnp.log1p(-p)
            elif k == "cos":
                w = jnp.cos(v)
                p = jnp.clip((w + 1.0) / 2.0, _EPS, 1.0 - _EPS)
                us.append(jnp.log(p) - jnp.log1p(-p))
                ld = ld - jnp.log(p) - jnp.log1p(-p) - math.log(2.0) + jnp.log(
                    jnp.abs(jnp.sin(v)) + _EPS)
            else:  # circ
                us.append(jnp.mod(v, self.period))
        return jnp.stack(us), ld

    def inverse(self, u):
        """u -> theta (no logdet; used only for sampling)."""
        ths = []
        for i in range(self.n_dim):
            z = u[i]
            k = self.kind[i]
            if k == "box":
                p = jax.nn.sigmoid(z)
                ths.append(self.lo[i] + p * (self.hi[i] - self.lo[i]))
            elif k == "cos":
                p = jax.nn.sigmoid(z)
                ths.append(jnp.arccos(jnp.clip(2.0 * p - 1.0, -1.0, 1.0)))
            else:
                ths.append(jnp.mod(z, self.period))
        return jnp.stack(ths)


# ---------------------------------------------------------------------------
# Coupling layer
# ---------------------------------------------------------------------------
class CouplingLayer(eqx.Module):
    """RQ-spline coupling: identity dims condition a spline on the transform dims."""

    conditioner: eqx.nn.MLP
    circular: tuple = eqx.field(static=True)
    identity_dims: tuple = eqx.field(static=True)
    transform_dims: tuple = eqx.field(static=True)
    n_bins: int = eqx.field(static=True)
    tail_bound: float = eqx.field(static=True)
    period: float = eqx.field(static=True)
    n_dim: int = eqx.field(static=True)

    def __init__(self, n_dim, circular, mask, context_dim, n_bins, tail_bound,
                 hidden, cond_depth, period, *, key):
        self.n_dim = n_dim
        self.circular = tuple(bool(c) for c in circular)
        self.n_bins = n_bins
        self.tail_bound = float(tail_bound)
        self.period = float(period)
        self.identity_dims = tuple(i for i in range(n_dim) if mask[i] == 1)
        self.transform_dims = tuple(i for i in range(n_dim) if mask[i] == 0)
        in_size = sum(2 if self.circular[d] else 1 for d in self.identity_dims) + context_dim
        out_size = len(self.transform_dims) * 3 * n_bins
        mlp = eqx.nn.MLP(in_size, out_size, hidden, cond_depth, key=key,
                         activation=jax.nn.gelu)
        # zero the final layer: the flow starts as a smooth, stable near-canonical map
        last = mlp.layers[-1]
        last = eqx.tree_at(lambda l: (l.weight, l.bias), last,
                           (jnp.zeros_like(last.weight), jnp.zeros_like(last.bias)))
        self.conditioner = eqx.tree_at(lambda m: m.layers[-1], mlp, last)

    def _params(self, z, context):
        feats = []
        for d in self.identity_dims:
            if self.circular[d]:
                feats.append(jnp.cos(z[d]))
                feats.append(jnp.sin(z[d]))
            else:
                feats.append(z[d])
        inp = jnp.concatenate([jnp.stack(feats), context]) if feats else context
        raw = self.conditioner(inp)
        return raw.reshape(len(self.transform_dims), 3 * self.n_bins)

    def _apply(self, z, context, inverse):
        params = self._params(z, context)
        out = z
        total_ld = 0.0
        k = self.n_bins
        for j, d in enumerate(self.transform_dims):
            uw = params[j, :k]
            uh = params[j, k:2 * k]
            ud = params[j, 2 * k:]
            if self.circular[d]:
                o, ld = circular_rqs(z[d], uw, uh, ud, inverse, self.period)
            else:
                o, ld = unconstrained_rqs(z[d], uw, uh, ud[:k - 1], inverse, self.tail_bound)
            out = out.at[d].set(o)
            total_ld = total_ld + ld
        return out, total_ld

    def forward(self, z, context):
        return self._apply(z, context, inverse=False)

    def inverse(self, z, context):
        out, _ = self._apply(z, context, inverse=True)
        return out


# ---------------------------------------------------------------------------
# The conditional flow
# ---------------------------------------------------------------------------
# Default HITMAN prior box (theta = x, y, z, zen, az, t, E). Boxes strictly contain the
# generation support (rho<=889, |z|<=924, E in [0,10]) so boundary values stay interior
# to the logit; t is padded for the +-50 ns coherent time augmentation (a few sigma).
DEFAULT_BOX = {
    0: (-950.0, 950.0),   # x [mm]
    1: (-950.0, 950.0),   # y [mm]
    2: (-980.0, 980.0),   # z [mm]
    5: (-260.0, 260.0),   # t [ns]  (base +-10 ns, widened for +-50 ns aug)
    6: (-0.5, 10.5),      # E [MeV]
}
COS_DIMS = (3,)           # zenith
CIRCULAR_DIMS = (4,)      # azimuth
TWO_PI = 2.0 * math.pi


class ConditionalFlow(eqx.Module):
    """Conditional RQ-spline flow q(theta | c). Single-example API; vmap externally."""

    domain: DomainTransform
    layers: tuple
    circular: tuple = eqx.field(static=True)
    circular_mask: jnp.ndarray
    period: float = eqx.field(static=True)
    n_dim: int = eqx.field(static=True)
    context_dim: int = eqx.field(static=True)

    def __init__(self, context_dim, *, key, n_dim=7, box=None, cos_dims=COS_DIMS,
                 circular_dims=CIRCULAR_DIMS, period=TWO_PI, n_layers=8, n_bins=8,
                 tail_bound=6.0, hidden=128, cond_depth=2):
        box = DEFAULT_BOX if box is None else box
        cos_dims = tuple(cos_dims)
        circular_dims = tuple(circular_dims)
        self.n_dim = n_dim
        self.context_dim = context_dim
        self.period = float(period)
        self.circular = tuple(i in circular_dims for i in range(n_dim))
        self.circular_mask = jnp.asarray(self.circular, jnp.bool_)

        lo, hi, kind = [], [], []
        for i in range(n_dim):
            if i in circular_dims:
                lo.append(0.0); hi.append(0.0); kind.append("circ")
            elif i in cos_dims:
                lo.append(0.0); hi.append(0.0); kind.append("cos")
            else:
                b = box[i]
                lo.append(b[0]); hi.append(b[1]); kind.append("box")
        self.domain = DomainTransform(lo, hi, kind, period)

        keys = jax.random.split(key, n_layers)
        layers = []
        for l in range(n_layers):
            mask = tuple(1 if (i + l) % 2 == 0 else 0 for i in range(n_dim))
            layers.append(CouplingLayer(
                n_dim, self.circular, mask, context_dim, n_bins, tail_bound,
                hidden, cond_depth, period, key=keys[l]))
        self.layers = tuple(layers)

    @classmethod
    def from_spec(cls, spec, context_dim, *, key, **kwargs):
        """Build a flow whose domain (dim, box, cos/circular dims, period) comes from a
        :class:`hitman.spec.HypSpec`. The WC default is exactly
        ``ConditionalFlow.from_spec(WC_HYP_SPEC, ...)``. Extra ``kwargs`` (``n_layers``,
        ``n_bins``, ...) pass through unchanged; ``box``/``cos_dims``/``circular_dims``/
        ``n_dim``/``period`` are taken from the spec and must not be overridden here."""
        clash = {"n_dim", "box", "cos_dims", "circular_dims", "period"} & set(kwargs)
        if clash:
            raise TypeError(f"from_spec derives {sorted(clash)} from the spec; drop them")
        return cls(context_dim, key=key, n_dim=spec.dim, box=spec.box,
                   cos_dims=spec.cos_dims, circular_dims=spec.circular_dims,
                   period=spec.period, **kwargs)

    # -- density / transforms ---------------------------------------------------
    def forward_to_base(self, theta, context):
        """theta -> (z in base space, log|det dz/dtheta|)."""
        z, total = self.domain.forward(theta)
        for layer in self.layers:
            z, ld = layer.forward(z, context)
            total = total + ld
        return z, total

    def inverse_from_base(self, z, context):
        for layer in reversed(self.layers):
            z = layer.inverse(z, context)
        return self.domain.inverse(z)

    def base_log_prob(self, z):
        real = -0.5 * z * z - 0.5 * _LOG2PI
        circ = jnp.full_like(z, -math.log(self.period))
        return jnp.sum(jnp.where(self.circular_mask, circ, real))

    def base_sample(self, key):
        kn, ku = jax.random.split(key)
        zr = jax.random.normal(kn, (self.n_dim,))
        zc = jax.random.uniform(ku, (self.n_dim,), minval=0.0, maxval=self.period)
        return jnp.where(self.circular_mask, zc, zr)

    def log_prob(self, theta, context):
        z, ld = self.forward_to_base(theta, context)
        return self.base_log_prob(z) + ld

    def sample(self, key, context):
        return self.inverse_from_base(self.base_sample(key), context)

    def sample_n(self, key, context, n):
        keys = jax.random.split(key, n)
        return jax.vmap(lambda k: self.sample(k, context))(keys)

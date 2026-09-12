"""SensorFrame: the universal PMT-frame invariance layer (design report Addenda 5/5b).

Zeroth-order claim: all sensors share one response law, azimuthally symmetric about
their own axis. The complete O(2)-invariant content of (PMT pose, vertex, event
direction) is four scalars plus the time residual — no local transverse frame is ever
constructed, which is what dissolves the roll/chirality convention problem:

    d       = |x_pmt - x_vtx|                    vertex-PMT distance
    cos_inc = h . n                              photocathode incidence (h = vertex->PMT line)
    cos_dir = e . n                              event direction vs PMT axis
    cos_che = e . h                              the Cherenkov variable (direct light: ~1/n)
    t_res   = (t - t0) - n_eff d / c             causal alignment; n_eff TRAINABLE

The only frame-dependent quantity is sign(n . (h x e)) — the chirality bit, whose
magnitude is fixed by the Gram determinant. It is omitted by default (reflection-
symmetric sensor response) and available as an opt-in flag.

``FrameHitNet`` composes these with a global-frame context block (raw PMT position,
vertex, direction — the symmetry-breaking information: walls, reflections; see
Addendum 5b) and owns the geometry tables, so its per-hit observation is just
``(pmt_id, t)`` — the Lookup pattern. Note the SensorFrame features are genuinely
nonseparable in (obs, hyp) (d couples them), so FrameHitNet has no split-embedding
fast path; that is the documented trade for physics-aligned coordinates.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from hitman.wc.nn import features as ft
from hitman.nn.mlp import MLP

C_MM_PER_NS = 299.792458


def compute_e_ref(normals: jnp.ndarray) -> jnp.ndarray:
    """Per-PMT transverse reference axis, anchored to the detector frame.

    z-hat projected onto the sensor's transverse plane (x-hat fallback for axial
    sensors). Not used by the invariant features — this is the stored convention for
    the OPT-IN local-azimuth (symmetry-breaking) features of Addendum 5b, decided once
    at table-build time.
    """
    z = jnp.array([0.0, 0.0, 1.0])
    x = jnp.array([1.0, 0.0, 0.0])

    def one(n):
        p = z - jnp.dot(z, n) * n
        p = jnp.where(jnp.linalg.norm(p) > 1e-3, p, x - jnp.dot(x, n) * n)
        return p / (jnp.linalg.norm(p) + 1e-9)

    return jax.vmap(one)(normals)


class SensorFrame(eqx.Module):
    """The invariant feature computation, with a trainable effective refraction index.

    ``n_eff`` is stored in log space (positivity) and initialized to the group
    velocity index of water; it is a physics constant learned jointly with the
    network (FiberPhysicsLayer precedent) and exported explicitly with the model.
    """

    log_n_eff: jnp.ndarray
    include_sign: bool = eqx.field(static=True)

    def __init__(self, n_eff_init: float = 1.38, include_sign: bool = False):
        self.log_n_eff = jnp.log(jnp.asarray(n_eff_init, jnp.float32))
        self.include_sign = include_sign

    @property
    def n_eff(self) -> jnp.ndarray:
        return jnp.exp(self.log_n_eff)

    @property
    def n_features(self) -> int:
        return 6 + (1 if self.include_sign else 0)

    def __call__(self, pmt_pos, pmt_normal, t, hyp) -> jnp.ndarray:
        """(pmt (3,), normal (3,), t scalar, hyp (7,)) -> invariant features."""
        rvec = pmt_pos - hyp[:3]
        d = jnp.linalg.norm(rvec) + 1e-6
        h = rvec / d  # vertex -> PMT line (photon travel direction for direct light)
        e = ft.direction(hyp)
        t_res = (t - hyp[ft.TIME]) - self.n_eff * d / C_MM_PER_NS
        feats = jnp.stack([
            d / ft.POSITION_SCALE,
            jnp.dot(h, pmt_normal),
            jnp.dot(e, pmt_normal),
            jnp.dot(e, h),
            t_res / ft.TIME_SCALE,
            hyp[ft.ENERGY] - 1.0,
        ])
        if self.include_sign:
            chirality = jnp.sign(jnp.dot(pmt_normal, jnp.cross(h, e)))
            feats = jnp.concatenate([feats, chirality[None]])
        return feats


class FrameHitNet(eqx.Module):
    """Per-hit log-ratio net over SensorFrame invariants + global-frame context.

    Observation is ``(pmt_id, t)``; geometry (position, normal) is resolved from the
    module's own tables (stop-gradient — they are calibration data, not weights).
    Drop-in for HitNet in the resident training/inference paths that honor
    ``obs_style`` ("id_t" here vs HitNet's implicit "xyz").
    """

    frame: SensorFrame
    mlp: MLP
    pmt_pos: jnp.ndarray
    pmt_normal: jnp.ndarray
    include_context: bool = eqx.field(static=True)

    obs_style = "id_t"

    def __init__(self, pmt_pos, pmt_normal, width: int = 256, depth: int = 3,
                 include_sign: bool = False, include_context: bool = True, *, key):
        self.frame = SensorFrame(include_sign=include_sign)
        self.include_context = include_context
        n_in = self.frame.n_features + (9 if include_context else 0)
        self.mlp = MLP(n_in, width, depth, key=key)
        self.pmt_pos = jnp.asarray(pmt_pos, jnp.float32)
        self.pmt_normal = jnp.asarray(pmt_normal, jnp.float32)

    def __call__(self, hit, hyp) -> jnp.ndarray:
        """(hit = (pmt_id, t), hyp (7,)) -> scalar logit = log r_hit."""
        pmt_id, t = hit
        pos = jax.lax.stop_gradient(self.pmt_pos)[pmt_id]
        nrm = jax.lax.stop_gradient(self.pmt_normal)[pmt_id]
        f = self.frame(pos, nrm, t, hyp)
        if self.include_context:
            ctx = jnp.concatenate([
                pos / ft.POSITION_SCALE,
                hyp[:3] / ft.POSITION_SCALE,
                ft.direction(hyp),
            ])
            f = jnp.concatenate([f, ctx])
        return self.mlp(f)

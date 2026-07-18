"""Compiled single-event MLE solver: one jitted graph, deployment-target CPU.

This is the C++/cppflow deployment artifact (design decision #1 of the JAX
modernization): a SINGLE compiled function that fuses the trained surrogate
networks, a deterministic data-driven seeder, and the whole optimization loop, so
the ratpac/Eos event loop makes ONE call per event instead of an NLopt x NLLH loop.

It is NOT an amortized/learned predictor — the optimizer itself runs inside the
graph; the only networks in the graph are the frozen trained HitNet/ChargeNet
surrogates. The reference optimum is :func:`batched_multistart_mle` (256 cylinder
seeds -> top-16 -> 250 Adam steps); this solver targets the same NLL minimum for a
fraction of the single-core work by

1. **Deterministic seeding from the hit list** (never a learned model): a
   closed-form Bancroft multilateration vertex/time seed, a charge/earliest-hit
   centroid fallback, a PCA direction axis, and an nhit->E lookup. Seeds are
   screened by NLL (a tiny top-k, not 256).
2. **Damped-Newton (Levenberg-Marquardt) descent** using the exact chart-space
   Hessian (``jax.hessian``) — quadratic local convergence in <~10 iterations from
   a decent seed, vs 250 fixed Adam steps. The LM damping regularizes the chart's
   gauge null space (the NLL is invariant to the direction-vector norm) and makes
   the step a trust-region descent that never accepts an objective increase.
3. **Early stopping** via ``lax.while_loop`` on the chart-space gradient norm /
   step size, instead of a fixed-length scan.
4. **Best-so-far tracking + NaN guard** so a diverging Newton step can only ever
   fall back to the best screened seed (bounded output, in-detector).

The HitNet **separable first layer** (obs-only ``embed_hit`` cached once per event,
hyp-only ``embed_hyp`` per proposal) is used throughout so the per-hit feature
transform and the hit side of the first GEMM are never recomputed across the
hundreds of NLL/grad/Hessian evaluations of a single event's descent.

The Bancroft photon speed is ``v = c / n_water`` (``cfg.n_water``); a FrameHitNet
deployment can pass its trained ``SensorFrame.n_eff`` here (the effective
time-of-flight index the network learned) instead of the 1.38 default.

References
----------
Bancroft, S. (1985). "An Algebraic Solution of the GPS Equations."
    IEEE Trans. Aerospace and Electronic Systems, AES-21(1), 56-59.
Chan, Y.T. and Ho, K.C. (1994). "A Simple and Efficient Estimator for
    Hyperbolic Location." IEEE Trans. Signal Processing, 42(8), 1905-1915.
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax

from hitman.inference.mle import CHART_SCALE, chart_from_theta, theta_from_chart
from hitman.nn.features import wrap_direction

C_MM_PER_NS = 299.792458  # speed of light, mm/ns


class CompiledMLEConfig(NamedTuple):
    """Configuration for :func:`make_compiled_mle`.

    Parameters
    ----------
    radius, half_height : float
        Cylindrical box bounds on the vertex (mm), matching the training support.
    t_range, e_range : tuple of float
        Box bounds on emission time (ns) and energy (MeV).
    n_water : float
        Effective (group) refractive index for the Bancroft photon speed v = c / n.
        Pass a FrameHitNet's trained ``SensorFrame.n_eff`` when available.
    pe_per_mev : float
        Photoelectron yield for the nhit -> E seed (E0 = nhit / pe_per_mev).
    use_bancroft : bool
        Include the two Bancroft multilateration roots as vertex/time seeds.
    use_centroid : bool
        Include the earliest-hit charge centroid as a fallback vertex seed.
    both_dir_signs : bool
        Seed both +/- of the PCA direction axis (the cone axis sign is ambiguous).
    screen_keep : int
        Number of NLL-screened seeds handed to the descent (multi-start).
    method : str
        Descent engine: ``"newton"`` (damped-Newton / Levenberg-Marquardt with the
        exact Hessian) or ``"lbfgs"`` (optax limited-memory BFGS with zoom line
        search; gradient-only, cheaper per step).
    min_iter : int
        Warm-up iterations run before any early-stopping test (the damped descent
        is non-monotone at the start — early rejected steps must not be mistaken
        for convergence).
    max_iter : int
        Maximum descent iterations in the while-loop (cap; early stopping usually
        ends sooner).
    ftol : float
        Objective early-stopping tolerance: after ``min_iter``, the loop ends once
        ``patience`` consecutive iterations improve the best NLL by less than
        ``ftol`` nats.
    patience : int
        Consecutive below-``ftol`` iterations required to declare convergence.
    grad_tol : float
        Chart-space gradient-norm early-stopping tolerance (secondary criterion).
    lm_init, lm_up, lm_down, lm_max : float
        Levenberg-Marquardt damping schedule (initial, reject-multiplier,
        accept-multiplier, cap); Newton method only.
    """

    radius: float = 800.0
    half_height: float = 800.0
    t_range: tuple = (-10.0, 10.0)
    e_range: tuple = (0.5, 8.0)
    n_water: float = 1.38
    pe_per_mev: float = 19.5
    use_bancroft: bool = True
    use_centroid: bool = True
    both_dir_signs: bool = True
    screen_keep: int = 2
    method: str = "newton"
    min_iter: int = 12
    max_iter: int = 30
    ftol: float = 1e-3
    patience: int = 3
    grad_tol: float = 1e-4
    lm_init: float = 1e-1
    lm_up: float = 4.0
    lm_down: float = 0.5
    lm_max: float = 1e8


def _bounds(cfg: CompiledMLEConfig):
    """Chart-space (u = chart / CHART_SCALE) box bounds, (8,) lo and hi."""
    big = 1e9
    lo = jnp.array([-cfg.radius, -cfg.radius, -cfg.half_height, -big, -big, -big,
                    cfg.t_range[0], cfg.e_range[0]]) / CHART_SCALE
    hi = jnp.array([cfg.radius, cfg.radius, cfg.half_height, big, big, big,
                    cfg.t_range[1], cfg.e_range[1]]) / CHART_SCALE
    return lo, hi


def _earliest_weights(pmt_id: jnp.ndarray, t: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Per-hit weight selecting the earliest hit on each PMT (robust to the late tail).

    Fixed-shape O(n_pad^2) segment-min over padded slots: a hit is kept (weight 1)
    iff its time equals the minimum time among all real hits sharing its PMT id.
    """
    same = (pmt_id[:, None] == pmt_id[None, :]) & (mask[None, :] > 0)
    tmin = jnp.min(jnp.where(same, t[None, :], jnp.inf), axis=1)
    return mask * (t <= tmin + 1e-6)


def _bancroft(P: jnp.ndarray, t: jnp.ndarray, w: jnp.ndarray, v: float):
    """Closed-form Bancroft (1985) multilateration: both algebraic roots.

    Squaring the range equations |P_i - x0| = v (t_i - t0) linearizes them in the
    4-vector r = [x0, b] (b = v t0) plus one scalar Lorentz-norm auxiliary Lambda;
    the weighted normal equations give r = p + Lambda q, and <r, r>_M = Lambda
    (Minkowski metric M = diag(1,1,1,-1)) is a scalar quadratic in Lambda. ``w`` are
    the per-hit weights (earliest-hit selection + padding mask).

    Returns
    -------
    x0 : jnp.ndarray, shape (2, 3)
        The two candidate vertices.
    t0 : jnp.ndarray, shape (2,)
        The two candidate emission times.
    """
    rho = v * t
    A = jnp.concatenate([P, -rho[:, None]], axis=1)          # (n_pad, 4)
    AW = A * w[:, None]
    ata_inv = jnp.linalg.pinv(AW.T @ A)                       # weighted (A^T W A)^+
    cvec = jnp.sum(P ** 2, axis=1) - rho ** 2
    p = 0.5 * ata_inv @ (AW.T @ cvec)
    q = 0.5 * ata_inv @ (AW.T @ jnp.ones_like(rho))

    def lorentz(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] - a[3] * b[3]

    aa = lorentz(q, q)
    bb = 2.0 * lorentz(p, q) - 1.0
    cc = lorentz(p, p)
    disc = jnp.sqrt(jnp.clip(bb * bb - 4.0 * aa * cc, 0.0))
    aa = jnp.where(jnp.abs(aa) < 1e-12, 1e-12, aa)
    roots = jnp.array([(-bb + disc) / (2.0 * aa), (-bb - disc) / (2.0 * aa)])
    r = p[None, :] + roots[:, None] * q[None, :]             # (2, 4)
    return r[:, :3], r[:, 3] / v


def _pca_direction(P: jnp.ndarray, w: jnp.ndarray, vertex: jnp.ndarray) -> jnp.ndarray:
    """First principal axis of hit positions about ``vertex``, signed toward the centroid.

    The cone/track axis is the leading eigenvector of the weighted scatter matrix;
    its sign is fixed by pointing it from the vertex toward the hit centroid.
    """
    wsum = jnp.sum(w) + 1e-9
    rel = (P - vertex[None, :]) * w[:, None]
    cov = rel.T @ (P - vertex[None, :]) / wsum
    evals, evecs = jnp.linalg.eigh(0.5 * (cov + cov.T))
    axis = evecs[:, -1]
    centroid = jnp.sum(P * w[:, None], axis=0) / wsum - vertex
    return jnp.where(jnp.dot(axis, centroid) < 0, -axis, axis)


def _dir_chart(unit_dir: jnp.ndarray) -> jnp.ndarray:
    """Unit 3-vector -> its (dx, dy, dz) chart slot (renormalized)."""
    return unit_dir / (jnp.linalg.norm(unit_dir) + 1e-9)


def make_compiled_mle(hitnet, chargenet, cfg: CompiledMLEConfig = CompiledMLEConfig(),
                      *, n_pad: int = 160, return_nll: bool = False) -> Callable:
    """Build the compiled single-event MLE solver.

    Parameters
    ----------
    hitnet, chargenet : eqx.Module
        Trained HitNet / ChargeNet surrogates (obs_style ``"xyz"``; the separable
        ``embed_hit`` / ``embed_hyp`` fast path is used).
    cfg : CompiledMLEConfig
        Seeding and descent configuration.
    n_pad : int
        Fixed padded-hit length (compile once per pad bucket, e.g. 96 / 160).
    return_nll : bool
        If True the returned callable also returns the achieved NLL scalar.

    Returns
    -------
    callable
        ``compiled_mle(hits_pad, pmt_id, t, mask, charge) -> theta_hat`` with
        ``hits_pad`` (n_pad, 4), ``pmt_id`` (n_pad,), ``t`` (n_pad,),
        ``mask`` (n_pad,) float 1/0, ``charge`` (2,) = (q_tot, nhit); returns the
        converged 7-parameter hypothesis ``theta`` (direction wrapped to physical
        range). Already jitted; vmap for batches. One compile per ``n_pad``.
    """
    obs_style = getattr(hitnet, "obs_style", "xyz")
    if obs_style != "xyz":
        raise NotImplementedError(
            "compiled_mle currently supports the separable xyz HitNet; got "
            f"obs_style={obs_style!r}")
    u_lo, u_hi = _bounds(cfg)
    v_phot = C_MM_PER_NS / cfg.n_water

    def solve(hits_pad, pmt_id, t, mask, charge):
        P = hits_pad[:, :3]
        w = _earliest_weights(pmt_id, t, mask)

        # ---- separable NLL closed over the (fixed) per-event hit embeddings ----
        e_hit = jax.vmap(hitnet.embed_hit)(hits_pad)          # (n_pad, width)

        def nll_u(u):
            theta = theta_from_chart(u * CHART_SCALE)
            pre = e_hit + hitnet.embed_hyp(theta)[None, :]
            logits = jax.vmap(hitnet.mlp.head)(pre)
            return -(jnp.sum(logits * mask) + chargenet(charge, theta))

        # ---- deterministic data-driven seeds (chart space) ----
        wsum = jnp.sum(w) + 1e-9
        centroid = jnp.sum(P * w[:, None], axis=0) / wsum
        tmin = jnp.min(jnp.where(mask > 0, t, jnp.inf))
        e0 = jnp.clip(charge[1] / cfg.pe_per_mev, cfg.e_range[0] + 1e-3, cfg.e_range[1] - 1e-3)

        verts, times = [], []
        if cfg.use_bancroft:
            x0b, t0b = _bancroft(P, t, w, v_phot)
            verts += [x0b[0], x0b[1]]
            times += [t0b[0], t0b[1]]
        if cfg.use_centroid:
            verts.append(centroid)
            times.append(tmin - 3.0)
        if not verts:  # always keep at least the origin
            verts.append(jnp.zeros(3))
            times.append(tmin - 3.0)

        vclip_lo = jnp.array([-cfg.radius, -cfg.radius, -cfg.half_height])
        vclip_hi = jnp.array([cfg.radius, cfg.radius, cfg.half_height])
        seeds = []
        for vtx, t0 in zip(verts, times):
            vtx = jnp.clip(vtx, vclip_lo, vclip_hi)
            t0 = jnp.clip(t0, cfg.t_range[0], cfg.t_range[1])
            axis = _pca_direction(P, w, vtx)
            dir_signs = [axis, -axis] if cfg.both_dir_signs else [axis]
            for d in dir_signs:
                seeds.append(jnp.concatenate([vtx, _dir_chart(d), jnp.array([t0, e0])]))
        seeds = jnp.stack(seeds)                              # (S, 8) chart space
        u_seeds = jnp.clip(seeds / CHART_SCALE, u_lo, u_hi)

        # ---- NLL screen: keep the best `screen_keep` seeds ----
        seed_nll = jax.vmap(nll_u)(u_seeds)
        keep = min(cfg.screen_keep, u_seeds.shape[0])
        _, top = jax.lax.top_k(-seed_nll, keep)
        u0s = u_seeds[top]

        # ---- bounded descent from each screened seed (early-stopped) ----
        # State layout: (u, u_best, f_best, aux, lam, it, stall, gnorm) where `aux`
        # is method-specific (Newton: unused; L-BFGS: optimizer state). Convergence
        # ends the while-loop after `patience` iterations with best-NLL improvement
        # below `ftol`, or on a small gradient norm, or at `max_iter`.
        eye = jnp.eye(8)

        def _converged(st):
            _, _, _, _, _, it, stall, gnorm = st
            warming = it < cfg.min_iter
            keep_going = (stall < cfg.patience) & (gnorm > cfg.grad_tol)
            return (it < cfg.max_iter) & (warming | keep_going)

        def newton_body(st):
            u, u_b, f_b, aux, lam, it, stall, _ = st
            f_here, g = jax.value_and_grad(nll_u)(u)
            H = jax.hessian(nll_u)(u)
            H = 0.5 * (H + H.T)
            delta = jnp.linalg.solve(H + lam * eye, g)
            u_new = jnp.clip(u - delta, u_lo, u_hi)
            f_new = nll_u(u_new)
            ok = jnp.isfinite(f_new) & (f_new < f_here)
            u_next = jnp.where(ok, u_new, u)
            lam_next = jnp.where(ok, jnp.maximum(lam * cfg.lm_down, 1e-9), lam * cfg.lm_up)
            improve = f_b - jnp.minimum(f_b, f_new)
            better = f_new < f_b
            u_b = jnp.where(better, u_new, u_b)
            f_b = jnp.where(better, f_new, f_b)
            stall = jnp.where(improve < cfg.ftol, stall + 1, 0)
            stall = jnp.where(it < cfg.min_iter, 0, stall)  # ignore warm-up transients
            return (u_next, u_b, f_b, aux, lam_next, it + 1, stall, jnp.linalg.norm(g))

        lbfgs = optax.lbfgs()

        def lbfgs_body(st):
            u, u_b, f_b, aux, lam, it, stall, _ = st
            f_here, g = jax.value_and_grad(nll_u)(u)
            updates, aux = lbfgs.update(g, aux, u, value=f_here, grad=g, value_fn=nll_u)
            u_new = jnp.clip(optax.apply_updates(u, updates), u_lo, u_hi)
            f_new = nll_u(u_new)
            improve = f_b - jnp.minimum(f_b, f_new)
            better = jnp.isfinite(f_new) & (f_new < f_b)
            u_next = jnp.where(jnp.isfinite(f_new), u_new, u)
            u_b = jnp.where(better, u_new, u_b)
            f_b = jnp.where(better, f_new, f_b)
            stall = jnp.where(improve < cfg.ftol, stall + 1, 0)
            stall = jnp.where(it < cfg.min_iter, 0, stall)  # ignore warm-up transients
            return (u_next, u_b, f_b, aux, lam, it + 1, stall, jnp.linalg.norm(g))

        body = lbfgs_body if cfg.method == "lbfgs" else newton_body

        def descend(u0):
            f0 = nll_u(u0)
            aux0 = lbfgs.init(u0) if cfg.method == "lbfgs" else 0.0
            init = (u0, u0, f0, aux0, jnp.asarray(cfg.lm_init), 0, 0, jnp.inf)
            _, u_b, f_b, *_ = jax.lax.while_loop(_converged, body, init)
            return u_b, f_b

        us, fs = jax.vmap(descend)(u0s)
        best = jnp.argmin(jnp.where(jnp.isfinite(fs), fs, jnp.inf))
        u_best = us[best]
        theta = wrap_direction(theta_from_chart(u_best * CHART_SCALE))
        return (theta, fs[best]) if return_nll else theta

    return jax.jit(solve)

"""Sequential single-event MLE driver: small jitted primitives, a Python-level loop.

Companion to :func:`hitman.inference.compiled.make_compiled_mle`. That solver fuses the
whole optimizer into ONE jitted graph (fixed-length loops, vmapped seeds) because it was
shaped for a GPU / for a single exported cppflow graph. On the actual **single-core CPU
deployment target** those GPU-shaped constraints are pure overhead: fixed-length loops
run their worst-case iteration count on every event, and a vmapped 2-seed descent pays
for the second seed even when the first already converged.

:func:`make_sequential_mle` instead keeps only the *primitives* jitted -- the per-event
hit embedding, the (gauge-penalized) NLL value+grad, the exact Hessian, and a
Hessian-vector product -- and drives them from an ordinary Python loop. That unlocks
everything a sequential CPU can do for free:

* a real, adaptive optimizer with a genuine line search (SciPy ``L-BFGS-B``, native box
  bounds -- no clipping) instead of a compiled fixed-length descent;
* **branching**: run the second (opposite cone-axis-sign) seed only when it can help;
* an **exact-Newton / Levenberg-Marquardt polish** that converts the L-BFGS approximate
  optimum into the exact NLL minimum in a handful of exact-Hessian steps and, as a
  by-product, yields the **Fisher matrix** (Hessian of the true NLL) for error bars.

The primitives take the per-event embed cache ``e_hit`` and ``mask``/``charge`` as
ARGUMENTS, so a single jit-compile per pad bucket serves every event.

Design provenance (round-3 optimization study, ``opt_eval/``)
-------------------------------------------------------------
On this surface, gradient-only curvature (L-BFGS, empirical-Fisher/OPG) is enough for
the bulk but NOT for the high-occupancy / near-wall tail -- the exact Hessian is
required there (same finding as the compiled solver's gauge-fixed Newton). The measured
sweet spot is **tight L-BFGS-B from both cone-axis-sign seeds -> short exact-LM polish**:
it matched or beat the compiled solver's tail on all four campaign anchors (3 MeV
zen000/zen180 at n_pad=96, two 8 MeV bias-field anchors at n_pad=256) while cutting
single-core latency well below it, because the easy bulk stops early and only the
primitives -- not a whole graph -- are re-entered per iteration. Pure derivative-free
(NLopt SBPLX/BOBYQA, the original HitmanProc engine) and HVP trust-region (Newton-CG /
trust-ncg) were benchmarked too; the former is slow-and-loose, the latter accurate but
spends hundreds of HVPs per event. See ``opt_eval/README`` for the full table.

Round-4 study (``opt_eval/`` r4 receipts) -- speed vs the knife-edge tail
------------------------------------------------------------------------
A ~4x latency reduction was targeted and found NOT achievable without regressing the
near-wall tail. Every trajectory-altering speedup moves the two 8 MeV bias-field anchors'
``f(dNLL>1)`` across the 1-nat line: diagonal preconditioning (bf 0.027 -> 0.083),
predictive seed-2 branching (flip-probe fails -- a seed-2 win is a different vertex/time
basin, not a local cone-sign flip; partial-descent and single-seed both regress zen000),
and any loosening of the descent (loose L-BFGS gtol + stronger polish is a wash at 1.00x;
a loose or skipped full-hit endgame after a subset search collapses bf to 0.18). The
binding constraint is that the tail needs the FULL tight-gtol descent on ALL hits from
BOTH cone-sign seeds; per-iteration cheapening (hit subsampling) does not pay at
n_pad <= 256 because the surrogate value+grad is already cheap, and iteration-count
cheapening (curvature/looser stopping) is what the tail cannot survive. A hull-membership
analysis shows a real slice of the tail is surrogate extrapolation at the training-support
boundary (zen000 seed-2-win events are 5.2x enriched as convex-hull vertices; the near-wall
8 MeV anchors sit wholesale at the support corner), i.e. genuinely multi-basin landscape,
not optimiser weakness. The one positive, physics-grounded result is causal emission-time
grid seeding (``robust_t``, opt-in, off by default): it removes wrong-causal-side local
minima, and together with the coarse subset search (``subsample``, whose fixed
spatially-spread subset diversifies the seed and acts as a mild regulariser on the
near-wall geometry) it lowers the tail on all four anchors below round-3 (bf 8 MeV 90-deg
f(dNLL>1) 0.027 -> 0.003, both 3 MeV anchors unchanged at 0.000/0.001) -- at a latency
cost, not a saving. Enable BOTH for the tail benefit.
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from hitman.inference.compiled import (C_MM_PER_NS, CompiledMLEConfig, _bounds,
                                       _dir_chart, _earliest_weights, _pca_direction,
                                       _quad_cloud)
from hitman.inference.mle import CHART_SCALE, theta_from_chart
from hitman.nn.features import TIME_SCALE, wrap_direction
from hitman.nn.mlp import mish

_QUAD_KEY = jax.random.PRNGKey(0)


class SequentialMLEResult(NamedTuple):
    """Return type when ``return_extras=True``.

    Attributes
    ----------
    theta : (7,) ndarray
        Best-fit hypothesis (direction wrapped to physical range).
    nll : float
        Achieved (un-penalized) negative log-likelihood at the optimum.
    fisher : (7, 7) ndarray
        Exact Hessian of the true NLL in theta-space at the optimum -- the observed
        Fisher information. ``inv(fisher)`` is the asymptotic covariance for error bars.
        All zeros if ``compute_fisher=False``.
    n_grad : int
        Number of value+grad primitive evaluations spent on this event.
    n_hess : int
        Number of exact-Hessian primitive evaluations spent on this event.
    n_seeds : int
        Number of seeds actually descended (1 or 2 under the default policy).
    """

    theta: np.ndarray
    nll: float
    fisher: np.ndarray
    n_grad: int
    n_hess: int
    n_seeds: int


def _build_primitives(hitnet, chargenet, cfg: CompiledMLEConfig, n_pad: int,
                      *, n_sub: int = 0, t_grid_n: int = 16):
    """Jit, once per pad bucket, the primitives the Python driver calls per event.

    Besides the round-3 value/grad/Hessian/HVP set this also builds two round-4
    primitives:

    * ``t_scan`` -- a robust causal emission-time line search. The per-hit
      log-likelihood has a **causal cliff** in t (a photon cannot arrive before
      emission + time-of-flight, so the surrogate learned a sharp wall for late-t
      hypotheses); gradient descent from a wrong-side seed converges to a clean but
      globally-wrong local minimum whose gradient is small -- invisible to any
      gradient/flip acceptance probe. Because t enters *every* hit's first-layer
      pre-activation as one shared rank-1 shift ``-w6 * (t / TIME_SCALE)`` (the dt
      feature column; the charge term is t-independent), the whole 1-D profile is a
      cheap ``t_grid_n``-point sweep. ``t_scan`` returns the chart point with t set to
      the grid minimiser, placing the descent on the correct causal side before it
      starts.
    * ``vg_pen_sub`` (only if ``0 < n_sub < n_pad``) -- the penalised value+grad over a
      fixed ``n_sub``-hit subset (a smaller GEMM) for the coarse far-from-optimum search
      phase; the full-hit primitives finish the endgame at full precision.
    """
    u_lo, u_hi = _bounds(cfg)
    v_phot = C_MM_PER_NS / cfg.n_water
    vclip_lo = jnp.array([-cfg.radius, -cfg.radius, -cfg.half_height])
    vclip_hi = jnp.array([cfg.radius, cfg.radius, cfg.half_height])
    head = hitnet.mlp.head
    L = hitnet.mlp.layers
    gs = cfg.gauge_stiffness
    w6 = L[0].weight[:, 6]                      # (width,) dt-column first-layer weights
    t_grid = jnp.linspace(cfg.t_range[0], cfg.t_range[1], t_grid_n)

    def _logits_fused(pre):
        # explicit (n_pad, width) GEMM head -- ~1.5x faster than vmap(head) for the
        # exact Hessian (fewer, larger XLA-CPU matmuls); numerically identical to
        # MLP.head (mish(pre) -> hidden layers with mish -> linear readout).
        x = mish(pre)
        for layer in L[1:-1]:
            x = mish(x @ layer.weight.T + layer.bias)
        return (x @ L[-1].weight.T + L[-1].bias)[:, 0]

    def _nll(u, e_hit, mask, charge, fused):
        theta = theta_from_chart(u * CHART_SCALE)
        pre = e_hit + hitnet.embed_hyp(theta)[None, :]
        logits = _logits_fused(pre) if fused else jax.vmap(head)(pre)
        return -(jnp.sum(logits * mask) + chargenet(charge, theta))

    def nll_true(u, e_hit, mask, charge):
        return _nll(u, e_hit, mask, charge, False)

    def nll_pen(u, e_hit, mask, charge):
        gauge = gs * (jnp.linalg.norm(u[3:6]) - 1.0) ** 2
        return _nll(u, e_hit, mask, charge, True) + gauge

    def fisher_theta(theta, e_hit, mask, charge):
        def f(th):
            pre = e_hit + hitnet.embed_hyp(th)[None, :]
            return -(jnp.sum(_logits_fused(pre) * mask) + chargenet(charge, th))
        return jax.hessian(f)(theta)

    def prep(hits, pmt_id, t, mask, charge):
        """Embed hits once + quadfitter-cloud seeds (both cone signs) + NLL screen."""
        P = hits[:, :3]
        w = _earliest_weights(pmt_id, t, mask)
        e_hit = jax.vmap(hitnet.embed_hit)(hits)

        def nll_u(u):
            return _nll(u, e_hit, mask, charge, False)

        wsum = jnp.sum(w) + 1e-9
        centroid = jnp.sum(P * w[:, None], axis=0) / wsum
        tmin = jnp.min(jnp.where(mask > 0, t, jnp.inf))
        e0 = jnp.clip(charge[1] / cfg.pe_per_mev, cfg.e_range[0] + 1e-3,
                      cfg.e_range[1] - 1e-3)

        V, TAU, OK = _quad_cloud(P, v_phot * t, w, cfg.quad_points,
                                 cfg.quad_rlimit_scale * cfg.radius, _QUAD_KEY)
        med_v = jnp.nanmedian(jnp.where(OK[:, None], V, jnp.nan), axis=0)
        med_t0 = jnp.nanmedian(jnp.where(OK, TAU, jnp.nan)) / v_phot
        med_v = jnp.where(jnp.all(jnp.isfinite(med_v)), med_v, centroid)
        med_t0 = jnp.where(jnp.isfinite(med_t0), med_t0, tmin - 3.0)

        def seeds_for(vtx, t0):
            vtx = jnp.clip(vtx, vclip_lo, vclip_hi)
            t0 = jnp.clip(t0, cfg.t_range[0], cfg.t_range[1])
            axis = _pca_direction(P, w, vtx)
            return [jnp.concatenate([vtx, _dir_chart(d), jnp.array([t0, e0])])
                    for d in (axis, -axis)]

        seeds = seeds_for(med_v, med_t0) + seeds_for(centroid, tmin - 3.0)
        if cfg.quad_screen > 0:
            Vc = jnp.where(OK[:, None], V, centroid[None, :])
            Tc = jnp.where(OK, TAU / v_phot, tmin - 3.0)

            def cand_nll(vtx, t0):
                vtx = jnp.clip(vtx, vclip_lo, vclip_hi)
                t0 = jnp.clip(t0, cfg.t_range[0], cfg.t_range[1])
                axis = _pca_direction(P, w, vtx)
                u = jnp.concatenate([vtx, _dir_chart(axis),
                                     jnp.array([t0, e0])]) / CHART_SCALE
                return nll_u(u)
            cn = jnp.where(OK, jax.vmap(cand_nll)(Vc, Tc), jnp.inf)
            _, best = jax.lax.top_k(-cn, cfg.quad_screen)
            for bi in range(cfg.quad_screen):
                seeds += seeds_for(Vc[best[bi]], Tc[best[bi]])

        seeds = jnp.stack(seeds)
        u_seeds = jnp.clip(seeds / CHART_SCALE, u_lo, u_hi)
        order = jnp.argsort(jax.vmap(nll_u)(u_seeds))
        return e_hit, u_seeds[order]

    def t_scan(u, e_hit, mask, charge):
        """Return u with its t slot (chart idx 6) set to the causal-grid NLL minimiser."""
        theta = theta_from_chart(u * CHART_SCALE)
        base = hitnet.embed_hyp(theta.at[5].set(0.0))          # t-term removed (t=0)
        a = e_hit + base[None, :]                              # (n_pad, width)

        # the charge term is t-independent (charge_features ignores t), so it is a
        # constant over the grid and never enters the argmin -- only the hit term is swept
        def hit_nll_at_t(tt):
            pre = a - (w6 * (tt / TIME_SCALE))[None, :]
            return -jnp.sum(_logits_fused(pre) * mask)

        vals = jax.vmap(hit_nll_at_t)(t_grid)
        return u.at[6].set(t_grid[jnp.argmin(vals)] / TIME_SCALE)

    prims = dict(
        prep=jax.jit(prep),
        nll_true=jax.jit(nll_true),
        vg_pen=jax.jit(jax.value_and_grad(nll_pen)),
        hess_pen=jax.jit(jax.hessian(nll_pen)),
        hvp_pen=jax.jit(lambda u, v, e, m, c: jax.jvp(
            lambda uu: jax.grad(nll_pen)(uu, e, m, c), (u,), (v,))[1]),
        t_scan=jax.jit(t_scan),
        fisher_theta=jax.jit(fisher_theta),
        u_lo=np.asarray(u_lo, np.float64), u_hi=np.asarray(u_hi, np.float64),
        n_sub=0,
    )
    if 0 < n_sub < n_pad:
        # coarse-search value+grad over a fixed n_sub-hit subset (smaller GEMM); the
        # full-hit primitives above finish the endgame at full precision.
        prims["vg_pen_sub"] = _build_primitives(hitnet, chargenet, cfg, n_sub)["vg_pen"]
        prims["n_sub"] = n_sub
    return prims


def make_sequential_mle(hitnet, chargenet, cfg: CompiledMLEConfig = CompiledMLEConfig(),
                        *, n_pad: int = 160, method: str = "lbfgsb", polish: int = 8,
                        max_seeds: int = 2, always_all_seeds: bool = True,
                        accept_margin: float = 0.5, lbfgs_maxiter: int = 60,
                        lbfgs_gtol: float = 1e-6, robust_t: bool = False,
                        t_grid_n: int = 24, subsample: bool = False,
                        n_sub: int = 0, sub_maxiter: int = 40, sub_gtol: float = 1e-5,
                        endgame_gtol: float = 1e-6, endgame_maxiter: int = 60,
                        compute_fisher: bool = False,
                        return_extras: bool = False) -> Callable:
    """Build the Python-level sequential single-event MLE solver.

    Parameters
    ----------
    hitnet, chargenet : eqx.Module
        Trained separable-``xyz`` HitNet / ChargeNet surrogates.
    cfg : CompiledMLEConfig
        Shared seeding / bounds / gauge / LM-damping configuration (the same dataclass
        the compiled solver uses; ``gauge_stiffness``, ``lm_*`` and the box ranges apply).
    n_pad : int
        Fixed padded-hit length (one compile per bucket, e.g. 96 / 160 / 256).
    method : {"lbfgsb", "lm", "trust-exact", "newton-cg", "trust-ncg", "sbplx", "bobyqa"}
        Sequential inner optimizer. ``"lbfgsb"`` (SciPy L-BFGS-B, native bounds) is the
        recommended default -- fastest on the bulk with the exact-LM polish supplying the
        tail precision. ``"lm"`` is a from-seed exact Levenberg-Marquardt; the ``trust*``
        / ``newton*`` methods are accurate but spend many curvature products; ``sbplx`` /
        ``bobyqa`` reproduce the derivative-free NLopt engine of the original C++ proc.
    polish : int
        Maximum exact-LM polish steps after the inner optimizer (0 disables). The polish
        is best-of monotone on the TRUE NLL -- it can only improve the result.
    max_seeds : int
        Cap on descended seeds. Seeds are the NLL-screened quadfitter-median /
        centroid vertices, each with BOTH cone-axis signs (the wrong-basin defense).
    always_all_seeds : bool
        If True (default) always descend ``max_seeds`` seeds and keep the best -- the
        robust choice (guarantees the cone-sign pair is tried). If False, descend the
        next seed only when the current best fails the ``accept_margin`` test.
    accept_margin : float
        Branch-on-acceptance threshold (nats), used only when ``always_all_seeds=False``:
        stop once the best result beats the next seed's screen NLL by this margin.
    lbfgs_maxiter, lbfgs_gtol : int, float
        L-BFGS-B stopping controls. The tight default gtol matters -- L-BFGS does the
        basin-finding; loosening it and leaning on the polish regresses the tail.
    robust_t : bool
        Round-4 causal-time seeding (opt-in ROBUSTNESS, off by default). Before
        descending each seed, run a ``t_grid_n``-point line scan over the whole feasible
        emission-time range and set the seed's t to the grid minimiser (see
        :func:`_build_primitives`); the descent optimum and its t-rescan are both polished
        (strictly best-of). This places the descent on the correct side of the surrogate's
        causal cliff, removing wrong-causal-side local minima that grad/flip acceptance
        probes cannot detect (they have a vanishing gradient). It lowers the near-wall
        tail (e.g. bf 8 MeV 90-deg f(dNLL>1) 0.027 -> 0.003 in the round-4 study) but is
        NOT a speedup -- the grid scan adds ~1 nll-sweep per seed. Enable it when tail
        robustness matters more than the ~10-40% latency it costs.
    t_grid_n : int
        Number of grid points in the causal t line scan.
    subsample : bool
        Round-4 coarse-to-fine search (opt-in, off by default). Run the far-from-optimum
        L-BFGS phase over a fixed ``n_sub``-hit subset (a cheaper GEMM, hit term rescaled
        to full weight); the endgame + LM polish run on all hits. LANDMINE: the full-hit
        endgame MUST stay tight (``endgame_gtol`` ~ ``lbfgs_gtol``) -- loosening it or
        skipping it to chase speed collapses the near-wall tail (bf 8 MeV f(dNLL>1) 0.027
        -> 0.18 in the round-4 study). With a tight endgame the subset phase is pure added
        work at n_pad <= 256 (the surrogate value+grad is cheap enough that the extra phase
        does not pay), so this is provided for future very-large-n_pad detectors, not as a
        default speedup.
    n_sub : int
        Subset size for the coarse search (0 -> auto: 48 for ``n_pad <= 96``, 64 for
        ``n_pad <= 160``, else 96). Ignored unless ``subsample`` and ``n_sub < n_pad``.
    sub_maxiter, sub_gtol : int, float
        L-BFGS-B controls for the coarse subset search phase.
    endgame_gtol, endgame_maxiter : float, int
        L-BFGS-B controls for the full-hit endgame after a subset search. Keep tight
        (the defaults) -- see the ``subsample`` landmine.
    compute_fisher : bool
        Also evaluate the exact theta-space Hessian (observed Fisher) at the optimum.
    return_extras : bool
        If True the callable returns a :class:`SequentialMLEResult`; else just ``theta``.

    Returns
    -------
    callable
        ``solve(hits_pad, pmt_id, t, mask, charge) -> theta`` (or ``SequentialMLEResult``).
        ``hits_pad`` (n_pad, 4), ``pmt_id`` / ``t`` / ``mask`` (n_pad,), ``charge`` (2,).
        NOT jitted -- it IS a Python loop by design; call it per event (no vmap).
    """
    obs_style = getattr(hitnet, "obs_style", "xyz")
    if obs_style != "xyz":
        raise NotImplementedError(
            f"make_sequential_mle supports the separable xyz HitNet; got {obs_style!r}")
    if subsample and n_sub <= 0:
        # more search hits for the larger buckets: the near-wall high-occupancy tail
        # needs a richer coarse subset to land in the right causal/cone basin
        n_sub = 48 if n_pad <= 96 else (64 if n_pad <= 160 else 96)
    build_sub = subsample and 0 < n_sub < n_pad
    p = _build_primitives(hitnet, chargenet, cfg, n_pad, t_grid_n=t_grid_n,
                          n_sub=(n_sub if build_sub else 0))
    u_lo, u_hi = p["u_lo"], p["u_hi"]
    bounds = list(zip(u_lo, u_hi))
    prep, nll_true = p["prep"], p["nll_true"]
    vg_pen, hess_pen, hvp_pen = p["vg_pen"], p["hess_pen"], p["hvp_pen"]
    fisher_theta = p["fisher_theta"]
    t_scan = p["t_scan"]
    vg_pen_sub = p.get("vg_pen_sub")
    n_sub = p.get("n_sub", 0)
    eye8 = np.eye(8)

    def solve(hits_pad, pmt_id, t, mask, charge):
        e_hit, u_seeds = prep(hits_pad, pmt_id, t, mask, charge)
        e_hit = np.asarray(e_hit)
        mask = np.asarray(mask, np.float32)
        charge = np.asarray(charge, np.float32)
        u_seeds = np.asarray(u_seeds, np.float64)
        n_avail = u_seeds.shape[0]
        cnt = dict(g=0, h=0, s=0)

        # deterministic, spatially spread hit subset for the coarse search phase
        sub_ready = False
        if vg_pen_sub is not None:
            ridx = np.where(mask > 0)[0]
            if ridx.size > n_sub:
                pick = ridx[np.linspace(0, ridx.size - 1, n_sub).astype(int)]
                e_sub = e_hit[pick]
                m_sub = np.full(n_sub, ridx.size / float(n_sub), np.float32)
                sub_ready = True

        def _vg(u):
            cnt["g"] += 1
            v, g = vg_pen(jnp.asarray(u), e_hit, mask, charge)
            return float(v), np.asarray(g, np.float64)

        def _seed(u):
            if not robust_t:
                return u
            cnt["g"] += 1
            return np.asarray(t_scan(jnp.asarray(u), e_hit, mask, charge), np.float64)

        def _f_true(u):
            return float(nll_true(jnp.asarray(u), e_hit, mask, charge))

        def _hessp(u, v):
            cnt["h"] += 1
            return np.asarray(hvp_pen(jnp.asarray(u), jnp.asarray(v), e_hit, mask, charge),
                              np.float64)

        def _lm(u, n_min, n_max, ftol):
            lam = cfg.lm_init
            best_u, best_f = u.copy(), _f_true(u)
            stall = 0
            for it in range(n_max):
                v_pen, g = vg_pen(jnp.asarray(u), e_hit, mask, charge)
                cnt["g"] += 1; cnt["h"] += 1
                H = np.asarray(hess_pen(jnp.asarray(u), e_hit, mask, charge), np.float64)
                v_pen = float(v_pen); g = np.asarray(g, np.float64)
                try:
                    delta = np.linalg.solve(H + lam * eye8, g)
                except np.linalg.LinAlgError:
                    delta = g
                u_new = np.clip(u - delta, u_lo, u_hi)
                f_pen_new = float(vg_pen(jnp.asarray(u_new), e_hit, mask, charge)[0])
                if np.isfinite(f_pen_new) and f_pen_new < v_pen:
                    u = u_new
                    lam = max(lam * cfg.lm_down, 1e-9)
                    f_true_new = _f_true(u_new)
                    improve = best_f - min(best_f, f_true_new)
                    if f_true_new < best_f:
                        best_u, best_f = u_new.copy(), f_true_new
                    stall = stall + 1 if (improve < ftol and it >= n_min) else 0
                else:
                    lam = min(lam * cfg.lm_up, cfg.lm_max)
                    stall = stall + 1 if it >= n_min else 0
                if it >= n_min and stall >= cfg.patience:
                    break
            return best_u, best_f

        def run_one(u0):
            if method == "lm":
                return _lm(u0.copy(), cfg.min_iter, cfg.max_iter, cfg.ftol)[0]
            if method == "lbfgsb":
                gtol, mit = lbfgs_gtol, lbfgs_maxiter
                if sub_ready:                          # coarse subset search first
                    def _vgs(u):
                        cnt["g"] += 1
                        v, g = vg_pen_sub(jnp.asarray(u), e_sub, m_sub, charge)
                        return float(v), np.asarray(g, np.float64)
                    rs = minimize(_vgs, u0, jac=True, method="L-BFGS-B", bounds=bounds,
                                  options=dict(maxiter=sub_maxiter, ftol=1e-9, gtol=sub_gtol))
                    u0 = np.clip(rs.x, u_lo, u_hi)
                    # the subset already found the basin: a SHORT loose full-hit endgame
                    # refines it and the exact-Hessian LM polish supplies final precision,
                    # so we need not pay a full tight-gtol descent on every hit here.
                    gtol, mit = endgame_gtol, endgame_maxiter
                r = minimize(_vg, u0, jac=True, method="L-BFGS-B", bounds=bounds,
                             options=dict(maxiter=mit, ftol=1e-9, gtol=gtol))
                return np.clip(r.x, u_lo, u_hi)
            if method == "trust-exact":
                r = minimize(_vg, u0, jac=True,
                             hess=lambda u: (cnt.__setitem__("h", cnt["h"] + 1) or
                                             np.asarray(hess_pen(jnp.asarray(u), e_hit,
                                                        mask, charge), np.float64)),
                             method="trust-exact", options=dict(maxiter=lbfgs_maxiter))
                return np.clip(r.x, u_lo, u_hi)
            if method in ("newton-cg", "trust-ncg", "trust-krylov"):
                m = "Newton-CG" if method == "newton-cg" else method
                r = minimize(_vg, u0, jac=True, hessp=_hessp, method=m,
                             options=dict(maxiter=lbfgs_maxiter))
                return np.clip(r.x, u_lo, u_hi)
            if method in ("sbplx", "bobyqa"):
                import nlopt
                opt = nlopt.opt(nlopt.LN_SBPLX if method == "sbplx" else nlopt.LN_BOBYQA, 8)
                opt.set_lower_bounds(u_lo.tolist()); opt.set_upper_bounds(u_hi.tolist())
                opt.set_min_objective(lambda x, grad: _f_true(x))
                opt.set_xtol_rel(1e-4); opt.set_maxeval(400)
                try:
                    return np.clip(np.asarray(opt.optimize(u0.tolist())), u_lo, u_hi)
                except Exception:
                    return u0
            raise ValueError(f"unknown method {method!r}")

        best_u, best_f = None, np.inf
        n_try = min(max_seeds, n_avail)
        for s in range(n_try):
            cnt["s"] += 1
            u = run_one(_seed(u_seeds[s]))
            f = _f_true(u)
            if f < best_f:
                best_f, best_u = f, u
            if (not always_all_seeds) and s + 1 < n_avail:
                # cheap acceptance: stop if the current optimum clearly beats the next
                # seed's screening NLL (unlikely the remaining seeds improve the basin)
                nxt = float(nll_true(jnp.asarray(u_seeds[s + 1]), e_hit, mask, charge))
                if best_f < nxt - accept_margin:
                    break

        # Endpoints to polish: the descent optimum, and (robust_t) its causal-t rescan --
        # the descent may have drifted the vertex (and with it the causal cliff) off the
        # seed's t. Polishing BOTH and keeping the best makes robust_t strictly best-of,
        # so it can only lower the tail; committing to the grid-snapped t before the
        # polish (a cheaper alternative) can mislead the polish on knife-edge events.
        cands = [best_u]
        if robust_t:
            u_ts = _seed(best_u)
            # only a genuinely different t is worth a second polish; if the descent
            # already sits at the causal-grid minimiser, u_ts == best_u (t_scan only
            # rewrites the t slot) and the extra polish would be pure redundancy.
            if abs(float(u_ts[6] - best_u[6])) > 1e-3:
                cands.append(u_ts)
        if polish:
            for uc in cands:
                u_p, f_p = _lm(uc.copy(), 0, int(polish), 1e-3)
                if np.isfinite(f_p) and f_p < best_f:
                    best_u, best_f = u_p, f_p
        else:
            for uc in cands[1:]:
                f = _f_true(uc)
                if f < best_f:
                    best_u, best_f = uc, f

        theta = np.asarray(wrap_direction(theta_from_chart(jnp.asarray(best_u)
                                                           * CHART_SCALE)), np.float64)
        fisher = np.zeros((7, 7))
        if compute_fisher:
            fisher = np.asarray(fisher_theta(jnp.asarray(theta, np.float32), e_hit,
                                             mask, charge), np.float64)
        if return_extras:
            return SequentialMLEResult(theta, best_f, fisher, cnt["g"], cnt["h"], cnt["s"])
        return theta

    return solve

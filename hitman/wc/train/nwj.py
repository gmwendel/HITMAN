"""Exact-quadrature tilted-family MLE for the HITMAN per-hit / per-event surrogates.

WHY (three failed estimators -> exact quadrature)
--------------------------------------------------
The surrogate is a tilted exponential family
    p_f(x|theta) = p_marg(x) * exp(f(x,theta)) / Z(theta),  Z(theta)=E_{x~p_marg}[e^{f}].
Training needs the log-partition log Z(theta). Estimating it from MC-shuffled
(marginal-permutation) pairs is DEAD for this detector:

* BCE leaves Z uncontrolled (the classifier is invariant to a theta-dependent shift).
* NWJ with a HARD clamp on the shuffled exp -> a-runaway: the clipped tail is flat, its
  restoring gradient on the normalizer a vanishes, the clipped fraction avalanches and
  a -> -inf (A3 attempt 2).
* NWJ with a LINEARLY-EXTENDED exp -> f-runaway: the normalization is carried by shuffled
  rows that land near the matched manifold, which occur at rate ~exp(-D2) with the
  data-model squared distance D2 ~ 200-800 here. The TRUE exponential prices those rare
  rows at e^{f}, so rate*cost = O(1); ANY sub-exponential tail collapses that product to
  ~0, so inflating f on matched configurations becomes nearly free and the objective is
  effectively unbounded in the matched directions (A3 attempt 3: clipf~0.009 but maxgap
  1e8, nwj -8e7).

No tail shape fixes an estimator whose informative samples arrive at rate exp(-D2). But
the per-hit observation domain is SMALL and GRIDDED (241 sensors x an 840-bin t-grid;
empirical marginal pmf cached in probe_grids.npz), so log Z is computed EXACTLY by
quadrature over that grid. No shuffled rows, no exp tail, no znet in the loss. The loss is
then the exact tilted-family MLE:  L(f) = -E_matched[f] + E_theta[log Z(theta)].

ChargeNet is treated identically with a 2-D (q, nhit) empirical-pmf grid
(``build_charge_grid``): its overlap problem does not bite (one aggregate observable, D2
tiny; only ~300 populated (q,N) cells), so the exact partition is cheap and removes the
last MC estimator. This is the route-(a) the earlier NWJ-charge fallback deferred to.

The znet is retained ONLY as a post-training INFERENCE CACHE: fit a(theta) ~= log Z_hit of
the FINAL f so downstream MLE/NUTS can subtract a fast normalizer (``fit_znet_cache``); it
plays no role in training now.
"""

import os
import time
from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from hitman.wc.nn import features as ft
from hitman.nn.mlp import MLP
from hitman.wc.train.recipe import _window_stream
from hitman.wc.train.resident import charge_batch, hit_batch

# ZNet feature transform maps theta(7) -> 8 whitened features (inference-cache only now).
N_THETA_FEATURES = 8


def theta_features(theta: jnp.ndarray) -> jnp.ndarray:
    """(theta (7,)) -> (8,) whitened normalizer-net input features (positions/scale,
    unit direction, t/scale, E-1). Used only by the ZNet inference cache."""
    return jnp.concatenate([
        theta[:3] / ft.POSITION_SCALE,
        ft.direction(theta),
        (theta[ft.TIME])[None] / ft.TIME_SCALE,
        (theta[ft.ENERGY] - 1.0)[None],
    ])


# ---------------------------------------------------------------------------
# Exact log-partition by grid quadrature
# ---------------------------------------------------------------------------

def grid_logZ(net, thetas, grid_obs, grid_logp, chunk: int = 64):
    """Exact log Z(theta) = logsumexp_g [ grid_logp[g] + f(grid_obs[g], theta) ] per theta.

    Differentiable in ``net``; vmapped over ``thetas`` in chunks of ``chunk`` (via
    ``jax.lax.map``) so the (chunk, n_grid) intermediate stays bounded. Works for any net
    with a ``(obs, theta) -> scalar`` call: hitnet over the (sensor,time) grid, chargenet
    over the (q,nhit) grid.

    Parameters
    ----------
    net : callable ``(obs, theta) -> scalar logit``.
    thetas : (M, 7) thetas to normalize.
    grid_obs : (G, d_obs) quadrature points (empty-pmf cells should be dropped upstream).
    grid_logp : (G,) log empirical marginal pmf at the grid points.
    chunk : thetas per vmap call.

    Returns (M,) exact log Z.
    """
    M = thetas.shape[0]

    def one(theta):
        f = jax.vmap(lambda o: net(o, theta))(grid_obs)          # (G,)
        return jax.scipy.special.logsumexp(grid_logp + f)

    if chunk >= M:
        return jax.vmap(one)(thetas)
    nc = -(-M // chunk)                                          # ceil
    pad = nc * chunk - M
    th = thetas if pad == 0 else jnp.concatenate(
        [thetas, jnp.broadcast_to(thetas[:1], (pad, thetas.shape[1]))], axis=0)
    # jax.checkpoint on the CHUNK BODY is load-bearing: lax.map lowers to a scan whose
    # reverse-mode pass otherwise STACKS every chunk's forward activations
    # (M x grid x width x mish-temps), so chunking the forward does NOT chunk the backward
    # — the unchunked cross-product then materializes (637 GiB at 256 thetas x 170k cells x
    # width-256 killed the A3 launch). Remat makes the scan store only each chunk's inputs
    # and recompute its grid-forward in backward, bounding peak to ONE z_chunk.
    body = jax.checkpoint(lambda tc: jax.vmap(one)(tc))
    out = jax.lax.map(body, th.reshape(nc, chunk, thetas.shape[1])).reshape(-1)
    return out[:M]


def mle_hit_loss(net, batch_matched, thetas_z, grid_obs, grid_logp, chunk: int = 64):
    """Exact tilted-family MLE loss for a ``(obs, theta) -> logit`` net.

    ``-E_matched[f] + mean_{theta in thetas_z} log Z(theta)`` with the EXACT grid
    partition. ``batch_matched = (obs (B,d), hyp (B,7))`` are true (x, theta) pairs;
    ``thetas_z`` (n_z, 7) is a per-step subsample of the matched thetas (the estimator of
    E_theta[log Z] is unbiased for the subsample average and log Z is exact per theta — no
    heavy tail anywhere). Named ``mle_hit_loss`` but reused verbatim for the charge net
    (same call signature, its own grid). Bounded below: inflating f on any configuration
    raises log Z through the quadrature by at least as much (Jensen), so f cannot run away.
    """
    obs, hyp = batch_matched
    f_m = jax.vmap(net)(obs, hyp)
    logZ = grid_logZ(net, thetas_z, grid_obs, grid_logp, chunk)
    return -jnp.mean(f_m) + jnp.mean(logZ)


def build_hit_grid(grid_pos, grid_t, p1):
    """(sensor pos (S,3), t-centers (T,), marginal pmf p1 (S,T)) -> (grid_obs (G,4),
    grid_logp (G,)) with empty-pmf cells DROPPED (they contribute 0 to logsumexp; dropping
    them just saves f-evals). G = #populated (sensor,time) cells."""
    S, T = p1.shape
    pos_rep = np.repeat(np.asarray(grid_pos), T, axis=0)               # (S*T, 3)
    t_rep = np.tile(np.asarray(grid_t), S)[:, None]                    # (S*T, 1)
    obs = np.concatenate([pos_rep, t_rep], axis=1).astype(np.float32)  # (S*T, 4)
    p = np.asarray(p1, np.float64).reshape(-1)
    keep = p > 0
    with np.errstate(divide="ignore"):
        logp = np.log(p[keep]).astype(np.float32)
    return jnp.asarray(obs[keep]), jnp.asarray(logp)


def build_charge_grid(charge, n_q_bins: int = 200, n_max: int = 320,
                      q_hi_quantile: float = 0.9995):
    """Empirical 2-D (q, nhit) marginal pmf as a quadrature grid, empties dropped.

    ChargeNet consumes charge = (q, nhit) as continuous features; the exact partition
    Z_c(theta) = E_{q,N}[e^{f_c}] is a 2-D integral. The empirical (q-bin x N) histogram IS
    that marginal, so summing p_marg * e^{f_c} over the populated cells is the exact
    quadrature (charge D2 is tiny — only ~300 cells are populated on the 5M store).
    """
    charge = np.asarray(charge)
    q = charge[:, 0]
    nh = np.clip(charge[:, 1].astype(np.int64), 0, n_max)
    q_edges = np.linspace(0.0, float(np.quantile(q, q_hi_quantile)), n_q_bins + 1)
    q_centers = 0.5 * (q_edges[:-1] + q_edges[1:])
    qi = np.clip(np.digitize(q, q_edges) - 1, 0, n_q_bins - 1)
    counts = np.zeros((n_q_bins, n_max + 1), np.float64)
    np.add.at(counts, (qi, nh), 1.0)
    pmf = (counts / counts.sum()).reshape(-1)
    QC, NN = np.meshgrid(q_centers, np.arange(n_max + 1), indexing="ij")
    obs = np.stack([QC.ravel(), NN.ravel()], axis=1).astype(np.float32)
    keep = pmf > 0
    with np.errstate(divide="ignore"):
        logp = np.log(pmf[keep]).astype(np.float32)
    return jnp.asarray(obs[keep]), jnp.asarray(logp)


# ---------------------------------------------------------------------------
# ZNet inference cache (NOT used in training)
# ---------------------------------------------------------------------------

class ZNet(eqx.Module):
    """Cache network a_psi(theta) ~= log Z_hit(theta) for a FIXED trained hitnet.

    Post-training only: downstream MLE/NUTS evaluate f(hit,theta) per hit and subtract a
    single a(theta) instead of re-integrating the grid. Fit with ``fit_znet_cache``.
    """

    hit_mlp: MLP

    def __init__(self, width: int = 128, depth: int = 3, *, key, activation: str = "mish"):
        self.hit_mlp = MLP(N_THETA_FEATURES, width, depth, key=key, activation=activation)

    def a_hit(self, theta: jnp.ndarray) -> jnp.ndarray:
        return self.hit_mlp(theta_features(theta))

    def __call__(self, theta: jnp.ndarray) -> jnp.ndarray:
        return self.hit_mlp(theta_features(theta))


def hit_logZ_targets(hitnet, thetas, grid_pos, grid_t, grid_logp_full, chunk: int = 64):
    """Numpy log Z_hit(theta) on the FULL (sensor,time) grid — the independent reference
    the differentiable ``grid_logZ`` is checked against. ``grid_logp_full`` is log p1 over
    the full S*T grid (may contain -inf); dropped-cell and full-grid logsumexp agree.
    """
    S = grid_pos.shape[0]
    T = grid_t.shape[0]
    pos_rep = jnp.repeat(grid_pos, T, axis=0)
    t_rep = jnp.tile(grid_t, S)[:, None]
    hits = jnp.concatenate([pos_rep, t_rep], axis=1)
    logw = grid_logp_full.reshape(-1)

    @jax.jit
    def one(theta):
        f = jax.vmap(lambda h: hitnet(h, theta))(hits)
        return jax.scipy.special.logsumexp(logw + f)

    thetas = jnp.asarray(thetas, jnp.float32)
    out = [np.asarray(jax.vmap(one)(thetas[i:i + chunk]))
           for i in range(0, thetas.shape[0], chunk)]
    return np.concatenate(out).astype(np.float32)


def regress_znet_hit_head(znet, thetas, targets, *, key, steps: int = 3000,
                          lr: float = 1e-3, batch: int = 1024, verbose: bool = False):
    """Fit ``znet.a_hit`` to ``targets`` (log Z) by MSE Adam. Returns (znet, final_mse)."""
    thetas = jnp.asarray(thetas, jnp.float32)
    targets = jnp.asarray(targets, jnp.float32)
    M = thetas.shape[0]
    batch = min(batch, M)
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(znet, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(znet, opt_state, th, tg):
        loss, grads = eqx.filter_value_and_grad(
            lambda z: jnp.mean((jax.vmap(z.a_hit)(th) - tg) ** 2))(znet)
        updates, opt_state = opt.update(grads, opt_state)
        return eqx.apply_updates(znet, updates), opt_state, loss

    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
    last = np.nan
    for s in range(steps):
        idx = rng.integers(0, M, size=batch)
        znet, opt_state, loss = step(znet, opt_state, thetas[idx], targets[idx])
        last = float(loss)
        if verbose and (s % max(1, steps // 10) == 0 or s == steps - 1):
            print(f"  znet-cache step {s:5d}  mse {last:.5f}", flush=True)
    return znet, last


def fit_znet_cache(hitnet, thetas, grid_obs, grid_logp, *, key, width: int = 128,
                   depth: int = 3, chunk: int = 64, steps: int = 3000, lr: float = 1e-3,
                   verbose: bool = False):
    """Post-training: exact grid log Z of the FINAL hitnet at ``thetas``, then regress a
    ZNet onto it. Returns (znet, mse, targets)."""
    targets = np.asarray(grid_logZ(hitnet, jnp.asarray(thetas, jnp.float32),
                                   grid_obs, grid_logp, chunk))
    znet = ZNet(width=width, depth=depth, key=key)
    znet, mse = regress_znet_hit_head(znet, thetas, targets, key=key, steps=steps, lr=lr,
                                      verbose=verbose)
    return znet, mse, targets


# ---------------------------------------------------------------------------
# Joint model + exact-MLE recipe
# ---------------------------------------------------------------------------

class MleModel(eqx.Module):
    """The trained pair: per-hit critic hitnet + per-event chargenet (no znet)."""

    hitnet: eqx.Module
    chargenet: eqx.Module


@dataclass
class MleRecipeResult:
    model: MleModel
    best_val: float
    best_stage: str
    best_step: int
    history: list = field(default_factory=list)


def joint_mle_loss(model, data, hrows, crows, key, hit_grid, charge_grid,
                   n_thetas_z, z_chunk, w_charge=1.0, time_sigma=50.0):
    """Exact-MLE loss for one step: hit + w_charge * charge, on matched windows ``hrows``
    (hits) / ``crows`` (events) with ``n_thetas_z`` random matched thetas each. Shared by
    the training step and the ``step_max_intermediate_gib`` audit so both trace the SAME
    graph. ``data`` and the grids ride the tracer as arguments (never closed over)."""
    hgo, hgp = hit_grid
    cgo, cgp = charge_grid
    k_aug, k_zh, k_zc = jax.random.split(key, 3)
    hobs, hhyp = hit_batch(data, hrows, k_aug, time_sigma)
    cobs, chyp = charge_batch(data, crows)
    zh = hhyp[jax.random.randint(k_zh, (n_thetas_z,), 0, hrows.shape[0])]
    zc = chyp[jax.random.randint(k_zc, (n_thetas_z,), 0, crows.shape[0])]
    lh = mle_hit_loss(model.hitnet, (hobs, hhyp), zh, hgo, hgp, z_chunk)
    lc = mle_hit_loss(model.chargenet, (cobs, chyp), zc, cgo, cgp, z_chunk)
    return lh + w_charge * lc


def _iter_subjaxprs(x):
    """Yield nested jaxprs held in an eqn param (scan/while/pjit/cond bodies). Duck-typed
    so it survives jax version churn (ClosedJaxpr has ``.jaxpr``; Jaxpr has ``.eqns``)."""
    if hasattr(x, "eqns"):                       # a Jaxpr
        yield x
    elif hasattr(x, "jaxpr") and hasattr(getattr(x, "jaxpr"), "eqns"):  # a ClosedJaxpr
        yield x.jaxpr
    elif isinstance(x, (list, tuple)):
        for y in x:
            yield from _iter_subjaxprs(y)


def _jaxpr_max_bytes(jaxpr):
    """Largest single intermediate tensor (bytes) produced anywhere in ``jaxpr``, including
    inside scan/map/cond bodies. Pure static analysis — allocates nothing."""
    import math
    m = 0
    for eqn in jaxpr.eqns:
        for ov in eqn.outvars:
            av = getattr(ov, "aval", None)
            if av is not None and hasattr(av, "shape") and hasattr(av, "dtype"):
                try:
                    m = max(m, math.prod(av.shape) * av.dtype.itemsize)
                except (TypeError, AttributeError):
                    pass
        for p in eqn.params.values():
            for sub in _iter_subjaxprs(p):
                m = max(m, _jaxpr_max_bytes(sub))
    return m


def step_max_intermediate_gib(model, data, hit_grid, charge_grid, *, hit_bs, chg_bs,
                              n_thetas_z, z_chunk, w_charge=1.0, time_sigma=50.0,
                              key=None):
    """Static audit: largest intermediate tensor (GiB) in the value_and_grad training step
    at the EXACT given config, via ``jax.make_jaxpr`` (traces abstractly — no allocation,
    CPU-safe). Turns the OOM class of failure into a 5-second pre-flight check."""
    if key is None:
        key = jax.random.PRNGKey(0)
    hrows = jnp.zeros(hit_bs, jnp.int32)
    crows = jnp.zeros(chg_bs, jnp.int32)

    def loss(m):
        return joint_mle_loss(m, data, hrows, crows, key, hit_grid, charge_grid,
                              n_thetas_z, z_chunk, w_charge, time_sigma)

    jaxpr = jax.make_jaxpr(eqx.filter_grad(loss))(model)
    return _jaxpr_max_bytes(jaxpr.jaxpr) / 2**30


def _val_rows(n_train, n_rows, n_val, max_val_rows):
    idx = np.arange(n_train, n_rows, dtype=np.int64)
    if n_val > max_val_rows:
        idx = idx[:: n_val // max_val_rows + 1][:max_val_rows]
    return jnp.asarray(idx, jnp.int32)


def train_recipe_mle(
    model: MleModel,
    data,
    n_hit_rows: int,
    n_event_rows: int,
    *,
    key,
    hit_grid,                # (grid_obs (Gh,4), grid_logp (Gh,))
    charge_grid,             # (grid_obs (Gc,2), grid_logp (Gc,))
    w_charge: float = 1.0,
    n_thetas_z: int = 256,
    z_chunk: int = 64,
    lr: float = 1e-3,
    sgd_hit_batch: int = 2**17,
    sgd_charge_batch: int = 2**14,
    sgd_max_steps: int = 400_000,
    cosine_peak: float = 3e-4,
    cosine_steps: int = 30_000,
    val_every: int = 1000,
    patience_steps: int = 10_000,
    min_delta: float = 2e-5,
    val_fraction: float = 0.1,
    max_val_rows: int = 2**16,
    time_sigma: float = 50.0,
    checkpoint_dir: str = None,
    snapshot_every: int = 1,
    extra_val=None,
    verbose: bool = True,
) -> MleRecipeResult:
    """Two-stage (sgd -> cosine) exact-MLE training of {hitnet, chargenet}.

    Per step: a matched hit window + ``n_thetas_z`` random matched thetas -> exact hit
    ``mle_hit_loss``; a matched charge window + ``n_thetas_z`` random thetas -> exact charge
    MLE; loss = hit + ``w_charge`` * charge. The grid tensors ride the jit tracer as
    ARGUMENTS (never closed over — the captured-constants guard stays armed). Validation
    is the exact-MLE NLL on a FIXED held-out batch (the grid bound gap is identically 0 now
    — E_grid[e^{f-logZ}] == 1 by construction — so it is dropped); ``extra_val`` still logs
    the deflation Jhat receipt.
    """
    h_obs_grid, h_logp_grid = hit_grid
    c_obs_grid, c_logp_grid = charge_grid

    n_val_h = max(int(n_hit_rows * val_fraction), 1)
    n_train_h = n_hit_rows - n_val_h
    n_val_e = max(int(n_event_rows * val_fraction), 1)
    n_train_e = n_event_rows - n_val_e
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    key, k_haug, k_zh, k_zc = jax.random.split(key, 4)
    val_hit_rows = _val_rows(n_train_h, n_hit_rows, n_val_h, max_val_rows)
    val_evt_rows = _val_rows(n_train_e, n_event_rows, n_val_e, max_val_rows)
    val_hobs, val_hhyp = hit_batch(data, val_hit_rows, k_haug, time_sigma)
    val_cobs, val_chyp = charge_batch(data, val_evt_rows)
    val_zh = val_hhyp[jax.random.randint(k_zh, (n_thetas_z,), 0, val_hhyp.shape[0])]
    val_zc = val_chyp[jax.random.randint(k_zc, (n_thetas_z,), 0, val_chyp.shape[0])]

    @eqx.filter_jit
    def val_fn(model, hgo, hgp, cgo, cgp):
        lh = mle_hit_loss(model.hitnet, (val_hobs, val_hhyp), val_zh, hgo, hgp, z_chunk)
        lc = mle_hit_loss(model.chargenet, (val_cobs, val_chyp), val_zc, cgo, cgp, z_chunk)
        return lh + w_charge * lc

    def make_step(opt, hit_bs, chg_bs):
        iota_h = jnp.arange(hit_bs, dtype=jnp.int32)
        iota_c = jnp.arange(chg_bs, dtype=jnp.int32)

        @eqx.filter_jit
        def step(model, opt_state, data, hstart, cstart, key, hgo, hgp, cgo, cgp):
            hrows = jnp.asarray(hstart, jnp.int32) + iota_h
            crows = jnp.asarray(cstart, jnp.int32) + iota_c

            def loss_fn(m):
                return joint_mle_loss(m, data, hrows, crows, key, (hgo, hgp), (cgo, cgp),
                                      n_thetas_z, z_chunk, w_charge, time_sigma)

            loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
            updates, opt_state = opt.update(grads, opt_state)
            return eqx.apply_updates(model, updates), opt_state, loss

        return step

    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
    best = (np.inf, model, "init", 0)
    history = []
    gstep = 0

    def run_stage(name, model, opt, hit_bs, chg_bs, max_steps, key):
        nonlocal best, gstep
        step_fn = make_step(opt, hit_bs, chg_bs)
        opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
        hit_stream = _window_stream(n_train_h, hit_bs, rng)
        chg_stream = _window_stream(n_train_e, chg_bs, rng)
        gate, gate_step = np.inf, 0
        t0 = time.time()
        for s in range(1, max_steps + 1):
            key, sk = jax.random.split(key)
            gstep += 1
            model, opt_state, _ = step_fn(
                model, opt_state, data,
                jnp.asarray(next(hit_stream), jnp.int32),
                jnp.asarray(next(chg_stream), jnp.int32), sk,
                h_obs_grid, h_logp_grid, c_obs_grid, c_logp_grid)
            if s % val_every == 0:
                v = float(val_fn(model, h_obs_grid, h_logp_grid, c_obs_grid, c_logp_grid))
                entry = {"stage": name, "step": s, "nll": v}
                if extra_val is not None:
                    entry.update(extra_val(model, s))
                history.append(entry)
                if v < best[0]:
                    best = (v, model, name, s)
                    if checkpoint_dir is not None:
                        eqx.tree_serialise_leaves(
                            os.path.join(checkpoint_dir, "best.eqx"), model)
                if checkpoint_dir is not None and (s // val_every) % snapshot_every == 0:
                    eqx.tree_serialise_leaves(
                        os.path.join(checkpoint_dir, f"{name}_step{s:07d}.eqx"), model)
                if v < gate - min_delta:
                    gate, gate_step = v, s
                if verbose:
                    print(f"[{name}] step {s:7d}  nll {v:.5f}  "
                          f"(best {best[0]:.5f}, {time.time()-t0:.0f}s)", flush=True)
                if s - gate_step >= patience_steps:
                    if verbose:
                        print(f"[{name}] value-plateau stop at step {s}", flush=True)
                    break
        return model, key

    model, key = run_stage("sgd", model, optax.adam(lr), sgd_hit_batch,
                           sgd_charge_batch, sgd_max_steps, key)
    if cosine_steps and cosine_steps > 0:
        sched = optax.cosine_decay_schedule(cosine_peak, cosine_steps, alpha=0.01)
        model, key = run_stage("cosine", best[1], optax.adam(sched), 2 * sgd_hit_batch,
                               2 * sgd_charge_batch, cosine_steps, key)

    return MleRecipeResult(model=best[1], best_val=best[0], best_stage=best[2],
                           best_step=best[3], history=history)

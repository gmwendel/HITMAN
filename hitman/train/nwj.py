"""NWJ variational objective: a self-normalizing replacement for BCE ratio training.

Why this module exists
----------------------
BCE-trained ratios are only defined up to a theta-dependent additive constant: the
learned logit approaches ``log p(x|theta)/p(x) + c(theta)`` for ANY c, because the
classifier loss is invariant to a shift that is constant in x at fixed theta. The
normalization ``Z(theta) = E_{x~p(x)}[e^{f(x,theta)}]`` is therefore uncontrolled, and
the downstream score identity / moment constraints are violated by exactly that
uncontrolled drift. Enforcing a one-sided moment identity as a penalty FAILS: the model
minimizes the penalty by deflating its own statistics (shrinking the f-contrast) rather
than matching the target, because the *model-measure counterweight* — the second
expectation that says "if you shrink f you pay in the exp term" — is absent from a
one-sided penalty.

The Nguyen-Wainwright-Jordan (NWJ / f-GAN KL) variational objective restores BOTH halves
in a single loss. For a critic ``T(x,theta) = f_phi(x,theta) - a_psi(theta)``,

    L(phi, psi) = - E_matched[ f_phi(x,theta) - a_psi(theta) ]
                  + E_shuffled[ exp( f_phi(x,theta) - a_psi(theta) ) ]  - 1 ,

where *matched* pairs are drawn from the joint p(x,theta) and *shuffled* pairs from the
product p(x)p(theta) (exactly the marginal-permutation construction the BCE path uses in
``hitman.train.loop._batch_loss``). At the joint minimum:

  * ``dL/df`` gives ``f_phi(x,theta) - a_psi(theta) = log p(x,theta)/(p(x)p(theta))``,
    i.e. f recovers the true per-hit log-ratio up to the additive a;
  * ``dL/da`` gives, per theta, ``E_{x~p(x)}[e^{f - a}] = 1`` — i.e.
    ``a_psi(theta) = log Z(theta)``. The normalizer is learned, not left free.

The exp term is the missing counterweight: the gradient of L in the direction that
deflates the f-contrast is ``E_matched[s] - E_model[s]`` (model expectation under the
self-normalized ``e^{f-a}`` reweighting of the shuffled sample), NOT the one-sided
``E_matched[s]`` a moment penalty sees. Deflation is no longer a free descent direction.

ChargeNet route (design decision, see director report)
------------------------------------------------------
The brief's first option was an EXACTLY-normalized MLE for chargenet: sum
``p_marg(N) * e^{f_c((q(N),N),theta))`` over an nhit grid. That is ill-posed here:
``ChargeNet`` consumes ``charge = (q, nhit)`` as two *continuous* features
(``charge/40 - 1``), so the exact partition ``Z_c(theta) = E_{p(q,nhit)}[e^{f_c}]`` is a
2-D integral over the joint charge marginal with q continuous. Reducing it to a 1-D sum
over the nhit grid would need a well-defined ``q(N)`` — i.e. the conditional p(q|N),
itself a continuous integral we do not have. So we take the documented fallback: chargenet
gets its OWN NWJ term with its own scalar normalizer head inside ``ZNet`` (``a_psi``
returns two outputs, hit and charge). This is exactly consistent with the hit path and
needs no per-N grid.

Optimizer split
---------------
``a_psi = log Z(theta)`` is a deterministic functional of the current phi and must TRACK
phi as it moves; a lagging normalizer re-introduces the very deflation bias NWJ removes.
We therefore give psi (znet) a larger learning rate than phi via ``optax.multi_transform``
(default 10x). This is simpler and more jit-friendly than an EMA target and keeps the
inner normalization calibrated throughout training.
"""

import os
import time
from dataclasses import dataclass, field
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from hitman.nn import features as ft
from hitman.nn.mlp import MLP
from hitman.train.recipe import _window_stream
from hitman.train.resident import charge_batch, hit_batch

# Winsorization constant: clamp (f - a) at +CLIP_C inside the shuffled exp. log(100)
# caps a single mis-normalized shuffled row's contribution to the exp mean at 100x the
# calibrated value, taming the heavy right tail of e^{f-a} early in training (when a is
# far from log Z) without biasing the optimum: at the optimum f - a = log r on shuffled
# (independent) pairs, whose exp has mean 1 and rarely exceeds 100, so the clip is
# inactive there (verified by the bound-gap toy). Static, never a trained quantity.
CLIP_C = float(np.log(100.0))

# Pre-calibration clip. The WINSORIZED loss (CLIP_C) is unbounded below in a: once f-a
# exceeds the clip on the calibrating mass, the exp is flat there and its restoring
# gradient vanishes, so training a alone runs it to -inf (observed as maxgap -> clip,
# loss -> -inf). The normalizer must therefore be pre-calibrated on an essentially
# unwinsorized objective — clip high enough to be inactive for a near log Z, yet finite
# to bound e^{f-a} against float overflow. Once a is calibrated, the production CLIP_C is
# inactive and safe in the joint phase.
PRECAL_CLIP = float(np.log(1e6))

# Fixed 1-MeV strata on true E for the per-stratum bound-gap receipt E[e^{f-a}]-1.
STRATA_EDGES = np.arange(0.0, 10.5, 1.0)          # 11 edges -> 10 strata
_EDGES = jnp.asarray(STRATA_EDGES, jnp.float32)
N_STRATA = len(STRATA_EDGES) - 1

# ZNet feature transform maps theta(7) -> 8 whitened features, mirroring the hyp-only
# columns of ``ft.hit_features``: positions/scale, unit direction, t/scale, (E-1).
N_THETA_FEATURES = 8


def theta_features(theta: jnp.ndarray) -> jnp.ndarray:
    """(theta (7,)) -> (8,) whitened normalizer-net input features.

    Z(theta) genuinely depends on theta_t: f depends on dt = t_hit - t_hyp, and the
    coherent time augmentation shifts theta_t at train time, so t is carried here.
    """
    return jnp.concatenate([
        theta[:3] / ft.POSITION_SCALE,
        ft.direction(theta),
        (theta[ft.TIME])[None] / ft.TIME_SCALE,
        (theta[ft.ENERGY] - 1.0)[None],
    ])


class ZNet(eqx.Module):
    """Normalizer network a_psi(theta) = (log Z_hit(theta), log Z_charge(theta)).

    Two separate MLP heads (theta(7) -> scalar each): the hit head is warm-started by
    regressing onto a grid estimate of log Z_hit (see ``hit_logZ_targets`` /
    ``regress_znet_hit_head``); the charge head has no cheap grid estimate and is left to
    the joint NWJ phase to calibrate (its init is ~0, a sensible start since ChargeNet is
    warm-started to a near-normalized ratio).
    """

    hit_mlp: MLP
    charge_mlp: MLP

    def __init__(self, width: int = 128, depth: int = 3, *, key, activation: str = "mish"):
        kh, kc = jax.random.split(key)
        self.hit_mlp = MLP(N_THETA_FEATURES, width, depth, key=kh, activation=activation)
        self.charge_mlp = MLP(N_THETA_FEATURES, width, depth, key=kc, activation=activation)

    def a_hit(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Scalar hit normalizer log Z_hit(theta)."""
        return self.hit_mlp(theta_features(theta))

    def a_charge(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Scalar charge normalizer log Z_charge(theta)."""
        return self.charge_mlp(theta_features(theta))

    def __call__(self, theta: jnp.ndarray):
        """(a_hit, a_charge) for one theta."""
        f = theta_features(theta)
        return self.hit_mlp(f), self.charge_mlp(f)


class NWJAux(NamedTuple):
    """Diagnostics for a NWJ loss evaluation (not differentiated).

    ``gap`` (N_STRATA,) is the per-1-MeV-E-stratum bound gap E_shuffled[e^{f-a}] - 1
    (winsorized exactly as the loss uses it), with nan for empty strata; it -> 0 in every
    stratum at the calibrated optimum. ``clip_frac`` is the fraction of shuffled rows on
    which the winsorization clip is active — a receipt that the clip bias vanishes once a
    is calibrated (clip_frac -> ~0).
    """

    gap: jnp.ndarray        # (N_STRATA,)
    clip_frac: jnp.ndarray  # scalar


def _stratum_gap(ex: jnp.ndarray, e: jnp.ndarray) -> jnp.ndarray:
    """Per-stratum mean of ``ex`` minus 1, keyed on true E ``e``; nan for empty strata."""
    k = jnp.clip(jnp.searchsorted(_EDGES, e) - 1, 0, N_STRATA - 1)
    sums = jax.ops.segment_sum(ex, k, num_segments=N_STRATA)
    cnts = jax.ops.segment_sum(jnp.ones_like(ex), k, num_segments=N_STRATA)
    return jnp.where(cnts > 0, sums / jnp.maximum(cnts, 1.0) - 1.0, jnp.nan)


def _nwj_loss(f_m, a_m, f_s, a_s, e_s, clip_c):
    """Core NWJ loss from matched/shuffled critic values; shared by hit and charge.

    ``f_m, a_m`` matched critic parts; ``f_s, a_s`` shuffled; ``e_s`` shuffled true E for
    the stratified bound-gap aux. Returns (loss, NWJAux).
    """
    d_s = f_s - a_s
    d_s_clip = jnp.minimum(d_s, clip_c)
    ex = jnp.exp(d_s_clip)
    matched = jnp.mean(f_m - a_m)
    shuffled = jnp.mean(ex)
    loss = -matched + shuffled - 1.0
    aux = NWJAux(gap=_stratum_gap(ex, e_s),
                 clip_frac=jnp.mean((d_s > clip_c).astype(jnp.float32)))
    return loss, aux


def nwj_hit_loss(hitnet, znet, batch, key, clip_c: float = CLIP_C):
    """Per-hit NWJ loss for (hitnet phi, znet hit head).

    ``batch = (obs (B,4), hyp (B,7))`` exactly as ``hit_batch`` returns. Matched pairs are
    the aligned rows (obs[i], hyp[i]); shuffled pairs are (obs[i], hyp[perm[i]]) with a
    fresh permutation from ``key`` — the same marginal construction as
    ``hitman.train.loop._batch_loss``. Returns (loss, NWJAux).

    (The brief's signature is ``(hitnet, znet, batch, clip_c)``; ``key`` is added because
    the marginal permutation needs randomness, mirroring ``_batch_loss(model, obs, hyp,
    key, ...)`` which takes a key for the identical reason.)
    """
    obs, hyp = batch
    perm = jax.random.permutation(key, hyp.shape[0])
    hyp_s = hyp[perm]
    f_m = jax.vmap(hitnet)(obs, hyp)
    a_m = jax.vmap(znet.a_hit)(hyp)
    f_s = jax.vmap(hitnet)(obs, hyp_s)
    a_s = jax.vmap(znet.a_hit)(hyp_s)
    return _nwj_loss(f_m, a_m, f_s, a_s, hyp_s[:, ft.ENERGY], clip_c)


def nwj_charge_loss(chargenet, znet, batch, key, clip_c: float = CLIP_C):
    """Per-event NWJ loss for (chargenet, znet charge head).

    ``batch = (charge (B,2), hyp (B,7))`` as ``charge_batch`` returns. Same matched /
    shuffled convention as ``nwj_hit_loss``. Returns (loss, NWJAux).
    """
    charge, hyp = batch
    perm = jax.random.permutation(key, hyp.shape[0])
    hyp_s = hyp[perm]
    f_m = jax.vmap(chargenet)(charge, hyp)
    a_m = jax.vmap(znet.a_charge)(hyp)
    f_s = jax.vmap(chargenet)(charge, hyp_s)
    a_s = jax.vmap(znet.a_charge)(hyp_s)
    return _nwj_loss(f_m, a_m, f_s, a_s, hyp_s[:, ft.ENERGY], clip_c)


# ---------------------------------------------------------------------------
# ZNet hit-head warm start: regress a_hit(theta) onto a grid estimate of log Z_hit
# ---------------------------------------------------------------------------

def hit_logZ_targets(hitnet, thetas, grid_pos, grid_t, grid_logp, chunk: int = 64):
    """Grid estimate of log Z_hit(theta) = log E_{x~p(x)}[e^{f(x,theta)}].

    p(x) over hits is approximated by the (sensor, time) marginal pmf ``grid_logp``
    (log of p1 in ``probe_grids.npz``). For each theta,
    ``log Z = logsumexp_{s,t} ( grid_logp[s,t] + f((grid_pos[s], grid_t[t]), theta) )``.

    Parameters
    ----------
    hitnet : callable ``(hit (4,), theta (7,)) -> scalar``
    thetas : (M, 7) array of thetas to evaluate.
    grid_pos : (S, 3) sensor positions.
    grid_t : (T,) time centers.
    grid_logp : (S, T) log marginal pmf (log of p1); need not be exactly normalized.
    chunk : thetas per device call (bounds the (M, S*T) intermediate).

    Returns (M,) float32 log Z estimates.
    """
    S = grid_pos.shape[0]
    T = grid_t.shape[0]
    # (S*T, 4) grid of candidate hits; (S*T,) log weights.
    pos_rep = jnp.repeat(grid_pos, T, axis=0)                       # (S*T, 3)
    t_rep = jnp.tile(grid_t, S)[:, None]                           # (S*T, 1)
    hits = jnp.concatenate([pos_rep, t_rep], axis=1)              # (S*T, 4)
    logw = grid_logp.reshape(-1)                                   # (S*T,)

    @jax.jit
    def one(theta):
        f = jax.vmap(lambda h: hitnet(h, theta))(hits)            # (S*T,)
        return jax.scipy.special.logsumexp(logw + f)

    thetas = jnp.asarray(thetas, jnp.float32)
    out = []
    for i in range(0, thetas.shape[0], chunk):
        out.append(np.asarray(jax.vmap(one)(thetas[i:i + chunk])))
    return np.concatenate(out).astype(np.float32)


def regress_znet_hit_head(znet, thetas, targets, *, key, steps: int = 3000,
                          lr: float = 1e-3, batch: int = 1024, verbose: bool = False):
    """Fit ``znet.a_hit`` to ``targets`` (log Z) by MSE Adam. Only the hit head moves.

    ``thetas`` (M,7), ``targets`` (M,). Returns the updated ZNet. The charge head gets
    zero gradient (loss touches only a_hit) so it is left at its init.
    """
    thetas = jnp.asarray(thetas, jnp.float32)
    targets = jnp.asarray(targets, jnp.float32)
    M = thetas.shape[0]
    batch = min(batch, M)
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(znet, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(znet, opt_state, th, tg):
        def loss_fn(z):
            pred = jax.vmap(z.a_hit)(th)
            return jnp.mean((pred - tg) ** 2)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(znet)
        updates, opt_state = opt.update(grads, opt_state)
        return eqx.apply_updates(znet, updates), opt_state, loss

    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
    last = np.nan
    for s in range(steps):
        idx = rng.integers(0, M, size=batch)
        znet, opt_state, loss = step(znet, opt_state, thetas[idx], targets[idx])
        last = float(loss)
        if verbose and (s % max(1, steps // 10) == 0 or s == steps - 1):
            print(f"  znet-init step {s:5d}  mse {last:.5f}", flush=True)
    return znet, last


# ---------------------------------------------------------------------------
# Joint pytree + training recipe
# ---------------------------------------------------------------------------

class JointModel(eqx.Module):
    """The trained pytree: critic hitnet phi, normalizer znet psi, chargenet."""

    hitnet: eqx.Module
    znet: ZNet
    chargenet: eqx.Module


@dataclass
class NwjRecipeResult:
    model: JointModel
    best_val: float
    best_stage: str
    best_step: int
    history: list = field(default_factory=list)


def _labels(model: JointModel) -> JointModel:
    """multi_transform label tree: 'psi' on znet leaves, 'phi' on hitnet+chargenet."""
    filt = eqx.filter(model, eqx.is_inexact_array)
    return JointModel(
        hitnet=jax.tree_util.tree_map(lambda _: "phi", filt.hitnet),
        znet=jax.tree_util.tree_map(lambda _: "psi", filt.znet),
        chargenet=jax.tree_util.tree_map(lambda _: "phi", filt.chargenet),
    )


def _val_rows(n_train, n_rows, n_val, max_val_rows):
    """Held-out row indices (last val_fraction), subsampled to max_val_rows."""
    idx = np.arange(n_train, n_rows, dtype=np.int64)
    if n_val > max_val_rows:
        idx = idx[:: n_val // max_val_rows + 1][:max_val_rows]
    return jnp.asarray(idx, jnp.int32)


def precalibrate_znet(
    model: JointModel,
    data,
    n_hit_rows: int,
    n_event_rows: int,
    *,
    key,
    clip_c: float = PRECAL_CLIP,
    w_charge: float = 1.0,
    psi_lr: float = 1e-2,
    hit_batch_size: int = 2**16,
    charge_batch_size: int = 2**14,
    max_steps: int = 2000,
    gap_tol: float = 0.1,
    log_every: int = 100,
    val_fraction: float = 0.1,
    max_val_rows: int = 2**16,
    time_sigma: float = 50.0,
    verbose: bool = True,
):
    """psi-only pre-calibration: train ONLY znet (both heads) on the NWJ objective with
    hitnet (phi) and chargenet FROZEN, until the max per-stratum |bound gap| over the hit
    AND charge heads drops below ``gap_tol`` on a fixed val batch, or ``max_steps``.

    The hit head is grid-warm-started but a_charge is not (no charge grid), and the
    winsorized NWJ loss loses its restoring gradient in a once the clip region dominates:
    calibrating a BEFORE the joint phase makes the clip's validity a guarantee, not a
    hope. Freezing is done with ``optax.set_to_zero`` on the 'phi' label — the same
    multi_transform label tree the joint recipe uses. Returns (model, steps_used).

    The CHARGE head is the primary target (it has no grid warm start and starts at a large
    gap; precal drives it under gap_tol in ~100 steps). The hit head is already
    grid-warm-started, so under the production ±time augmentation its bound gap sits at a
    small floor (mean ~0.2, max noisier on sparse high-E strata); the max-over-both gate
    may therefore run to ``max_steps`` even after the charge head is calibrated — that is
    the specified cap fallback, and the joint phase (psi at higher LR) continues refining
    a_hit.
    """
    n_val_h = max(int(n_hit_rows * val_fraction), 1)
    n_val_e = max(int(n_event_rows * val_fraction), 1)
    key, k_haug, k_hperm, k_cperm = jax.random.split(key, 4)
    val_hit_rows = _val_rows(n_hit_rows - n_val_h, n_hit_rows, n_val_h, max_val_rows)
    val_evt_rows = _val_rows(n_event_rows - n_val_e, n_event_rows, n_val_e, max_val_rows)
    val_hobs, val_hhyp = hit_batch(data, val_hit_rows, k_haug, time_sigma)
    val_cobs, val_chyp = charge_batch(data, val_evt_rows)

    @eqx.filter_jit
    def val_fn(model):
        lh, auxh = nwj_hit_loss(model.hitnet, model.znet, (val_hobs, val_hhyp),
                                k_hperm, clip_c)
        lc, auxc = nwj_charge_loss(model.chargenet, model.znet, (val_cobs, val_chyp),
                                   k_cperm, clip_c)
        return (lh + w_charge * lc, auxh.gap, auxc.gap, auxh.clip_frac, auxc.clip_frac)

    labels = _labels(model)
    opt = optax.multi_transform(
        {"phi": optax.set_to_zero(), "psi": optax.adam(psi_lr)}, labels)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    iota_h = jnp.arange(hit_batch_size, dtype=jnp.int32)
    iota_c = jnp.arange(charge_batch_size, dtype=jnp.int32)

    @eqx.filter_jit
    def step(model, opt_state, data, hstart, cstart, key):
        hrows = jnp.asarray(hstart, jnp.int32) + iota_h
        crows = jnp.asarray(cstart, jnp.int32) + iota_c
        k_aug, k_hp, k_cp = jax.random.split(key, 3)

        def loss_fn(m):
            hobs, hhyp = hit_batch(data, hrows, k_aug, time_sigma)
            cobs, chyp = charge_batch(data, crows)
            lh, _ = nwj_hit_loss(m.hitnet, m.znet, (hobs, hhyp), k_hp, clip_c)
            lc, _ = nwj_charge_loss(m.chargenet, m.znet, (cobs, chyp), k_cp, clip_c)
            return lh + w_charge * lc

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = opt.update(grads, opt_state)
        return eqx.apply_updates(model, updates), opt_state, loss

    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
    hit_stream = _window_stream(n_hit_rows - n_val_h, hit_batch_size, rng)
    chg_stream = _window_stream(n_event_rows - n_val_e, charge_batch_size, rng)
    t0 = time.time()
    steps_used = 0
    for s in range(1, max_steps + 1):
        key, sk = jax.random.split(key)
        model, opt_state, _ = step(model, opt_state, data,
                                   jnp.asarray(next(hit_stream), jnp.int32),
                                   jnp.asarray(next(chg_stream), jnp.int32), sk)
        steps_used = s
        if s % log_every == 0 or s == 1:
            v, hgap, cgap, hcf, ccf = val_fn(model)
            hmax = float(np.nanmax(np.abs(np.asarray(hgap))))
            cmax = float(np.nanmax(np.abs(np.asarray(cgap))))
            maxgap = max(hmax, cmax)
            clipf = float(max(float(hcf), float(ccf)))
            if verbose:
                print(f"[precal] step {s:5d}  val {float(v):.5f}  maxgap {maxgap:.4g}  "
                      f"(hit {hmax:.4g} chg {cmax:.4g})  clipf {clipf:.3g}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if maxgap < gap_tol:
                if verbose:
                    print(f"[precal] calibrated: maxgap {maxgap:.4g} < {gap_tol} "
                          f"at step {s}", flush=True)
                break
    return model, steps_used


def train_recipe_nwj(
    model: JointModel,
    data,
    n_hit_rows: int,
    n_event_rows: int,
    *,
    key,
    clip_c: float = CLIP_C,
    w_charge: float = 1.0,
    phi_lr: float = 1e-3,
    psi_lr_mult: float = 10.0,
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
) -> NwjRecipeResult:
    """Two-stage (sgd -> cosine) joint NWJ training of {hitnet, znet, chargenet}.

    Structure mirrors ``hitman.train.recipe.train_recipe`` (step-based validation with
    min_delta patience gating, global-best checkpointing, contiguous shuffled windows via
    ``_window_stream``) but the loss is NWJ, not BCE, and psi (znet) is trained at
    ``psi_lr_mult`` x the phi learning rate via ``optax.multi_transform``.

    Hits and events are streamed as two independent window streams (hits index
    ``data.n_hits``, events index ``data.n_event``); the last ``val_fraction`` of each is
    held out. Validation reports the NWJ loss on a FIXED held-out batch plus the per-E
    stratum bound gap E[e^{f-a}]-1 (max|gap| printed; full hit/charge vectors into
    history). ``extra_val(model, step) -> dict`` (optional) is merged into each history
    entry — the driver uses it for the deflation-alarm receipt.

    ``data`` is a CALL-TIME argument that rides the jit tracer; it is never closed over
    (a closure-captured DeviceData baked ~4 GB of constants into the compiled step and
    hard-locked the box on 2026-07-19).
    """
    n_val_h = max(int(n_hit_rows * val_fraction), 1)
    n_train_h = n_hit_rows - n_val_h
    n_val_e = max(int(n_event_rows * val_fraction), 1)
    n_train_e = n_event_rows - n_val_e

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Fixed held-out validation batch (subsampled to max_val_rows) — built once so the
    # NWJ yardstick is comparable step to step.
    def _val_rows(n_train, n_rows, n_val):
        idx = np.arange(n_train, n_rows, dtype=np.int64)
        if n_val > max_val_rows:
            idx = idx[:: n_val // max_val_rows + 1][:max_val_rows]
        return jnp.asarray(idx, jnp.int32)

    key, k_haug, k_hperm, k_cperm = jax.random.split(key, 4)
    val_hit_rows = _val_rows(n_train_h, n_hit_rows, n_val_h)
    val_evt_rows = _val_rows(n_train_e, n_event_rows, n_val_e)
    val_hobs, val_hhyp = hit_batch(data, val_hit_rows, k_haug, time_sigma)
    val_cobs, val_chyp = charge_batch(data, val_evt_rows)

    @eqx.filter_jit
    def val_fn(model):
        lh, auxh = nwj_hit_loss(model.hitnet, model.znet, (val_hobs, val_hhyp),
                                k_hperm, clip_c)
        lc, auxc = nwj_charge_loss(model.chargenet, model.znet, (val_cobs, val_chyp),
                                   k_cperm, clip_c)
        return lh + w_charge * lc, auxh.gap, auxc.gap, auxh.clip_frac, auxc.clip_frac

    def make_step(opt, hit_bs, chg_bs):
        iota_h = jnp.arange(hit_bs, dtype=jnp.int32)
        iota_c = jnp.arange(chg_bs, dtype=jnp.int32)

        @eqx.filter_jit
        def step(model, opt_state, data, hstart, cstart, key):
            hrows = jnp.asarray(hstart, jnp.int32) + iota_h
            crows = jnp.asarray(cstart, jnp.int32) + iota_c
            k_aug, k_hp, k_cp = jax.random.split(key, 3)

            def loss_fn(m):
                hobs, hhyp = hit_batch(data, hrows, k_aug, time_sigma)
                cobs, chyp = charge_batch(data, crows)
                lh, _ = nwj_hit_loss(m.hitnet, m.znet, (hobs, hhyp), k_hp, clip_c)
                lc, _ = nwj_charge_loss(m.chargenet, m.znet, (cobs, chyp), k_cp, clip_c)
                return lh + w_charge * lc

            loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
            updates, opt_state = opt.update(grads, opt_state)
            return eqx.apply_updates(model, updates), opt_state, loss

        return step

    labels = _labels(model)
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
            hstart = jnp.asarray(next(hit_stream), jnp.int32)
            cstart = jnp.asarray(next(chg_stream), jnp.int32)
            gstep += 1
            model, opt_state, _ = step_fn(model, opt_state, data, hstart, cstart, sk)
            if s % val_every == 0:
                v, hgap, cgap, hcf, ccf = val_fn(model)
                v = float(v)
                hgap = np.asarray(hgap)
                cgap = np.asarray(cgap)
                maxgap = float(np.nanmax(np.abs(hgap)))
                entry = {"stage": name, "step": s, "nwj": v, "max_hit_gap": maxgap,
                         "hit_gap": hgap.tolist(), "charge_gap": cgap.tolist(),
                         "hit_clip_frac": float(hcf), "charge_clip_frac": float(ccf)}
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
                    print(f"[{name}] step {s:7d}  nwj {v:.5f}  maxgap {maxgap:.4g}  "
                          f"clipf {float(hcf):.3g}  (best {best[0]:.5f}, "
                          f"{time.time()-t0:.0f}s)", flush=True)
                if s - gate_step >= patience_steps:
                    if verbose:
                        print(f"[{name}] value-plateau stop at step {s}", flush=True)
                    break
        return model, key

    sgd_opt = optax.multi_transform(
        {"phi": optax.adam(phi_lr), "psi": optax.adam(phi_lr * psi_lr_mult)}, labels)
    model, key = run_stage("sgd", model, sgd_opt, sgd_hit_batch, sgd_charge_batch,
                           sgd_max_steps, key)
    if cosine_steps and cosine_steps > 0:
        sched = optax.cosine_decay_schedule(cosine_peak, cosine_steps, alpha=0.01)
        cos_opt = optax.multi_transform(
            {"phi": optax.adam(sched),
             "psi": optax.adam(lambda c: psi_lr_mult * sched(c))}, labels)
        model, key = run_stage("cosine", best[1], cos_opt, 2 * sgd_hit_batch,
                               2 * sgd_charge_batch, cosine_steps, key)

    return NwjRecipeResult(model=best[1], best_val=best[0], best_stage=best[2],
                           best_step=best[3], history=history)

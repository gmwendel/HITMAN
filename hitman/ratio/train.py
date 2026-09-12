"""The ratio lane's training loop -- one loop, group-aware, with injectable pairing.

Why this exists rather than reusing ``hitman.wc.train.resident.fit_resident``
(evaluated first; the verdict is recorded here because it is the whole reason for the
duplication this replaces):

* ``fit_resident`` has the two seams that matter -- a ``make_batch`` callable and per-batch
  weight threading -- and its checkpointing, early stopping and device residency are all
  sound. But its loss goes through ``hitman.train.loop._batch_loss``, which hardcodes the
  marginal as ``hyp[permutation]``. There is no seam for the negative pairing, and that is
  precisely the construction a grouped dataset must override.
* Its validation split is the LAST ``val_fraction`` ROWS. For water-Cherenkov that leaks
  one event; for grouped rows it leaks catastrophically.
* Its batches are contiguous row WINDOWS. Coalesced reads are the right call for a
  row-independent dataset, but when rows are sorted by group a window spans only a handful
  of groups -- so the group-aware negative would be drawn from a pool of a few distinct
  hypotheses, which is a materially worse marginal.
* ``DeviceData`` is a water-Cherenkov container (pmt_id / t / charge).

Three of those four are contract changes, not parameters. Changing ``fit_resident`` in
place would alter the marginal construction and the val split under the calibrated WC
receipts, which is a physics change for that program. So: ``fit_resident`` stays exactly
as it is for water-Cherenkov, and this loop takes its good ideas with a group-aware
contract. The shared numeric core (``hitman.train.losses.nre_loss``) is genuinely shared.

Residency. The feature tables live on-device when they fit the budget, and the whole step
-- negative draw included -- runs there, so only a batch-length index array crosses the
bus per step. Otherwise they stay on host and each step gathers its own batch. The device
arrays are passed as ARGUMENTS to the jitted step, never closed over: a JAX array captured
by a ``filter_jit`` closure is baked into the compiled executable as a constant, which
both doubles residency and produces an uncacheable multi-GB executable.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from hitman.diagnostics.calibration import (
    expected_calibration_error, roc_auc, self_normalization,
)
from hitman.ratio.pairing import build_group_index, get_pairing
from hitman.train.losses import nre_loss


@dataclass
class FitResult:
    model: object
    history: list = field(default_factory=list)
    best_step: int = 0
    best_val_loss: float = float("inf")
    final_val_loss: float = float("nan")
    final_val_auc: float = float("nan")
    n_params: int = 0
    n_train: int = 0
    n_val: int = 0
    wall_sec: float = 0.0
    residency: str = "host"
    pairing: str = ""
    weighted: bool = False
    steps: int = 0
    batch_size: int = 0

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d.pop("model", None)
        return d


def ratio_bce(model, x, theta_pos, theta_neg, weights=None, balance_weight: float = 0.0):
    """NRE BCE for pre-paired positives/negatives, plus the aux logits.

    Delegates the objective itself to :func:`hitman.train.losses.nre_loss` -- one
    definition of the loss for the whole library, including the optional BNRE balancing
    term (arXiv:2208.13624) that the forked arm silently dropped.
    """
    logit_pos = model.logit_batch(x, theta_pos)
    logit_neg = model.logit_batch(x, theta_neg)
    loss = nre_loss(logit_pos, logit_neg, balance_weight, weights)
    return loss, (logit_pos, logit_neg)


def _estimate_device_gb(*arrays) -> float:
    return sum(np.asarray(a).nbytes for a in arrays if a is not None) / 1e9


def fit_ratio(
    model,
    dataset,
    *,
    key,
    pairing="group_shift",
    steps: Optional[int] = None,
    epochs: int = 30,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    clip_norm: float = 1.0,
    balance_weight: float = 0.0,
    val_every: int = 200,
    val_max_rows: int = 2_000_000,
    val_chunk_rows: int = 1_000_000,
    auc_max_rows: int = 200_000,
    residency: str = "auto",
    device_budget_gb: float = 12.0,
    use_weights: Optional[bool] = None,
    checkpoint_path: Optional[str] = None,
    on_best=None,
    verbose: bool = True,
) -> FitResult:
    """Train ``model`` on ``dataset``. Splits come from ``dataset.split`` (group-level).

    ``pairing`` is a name from :data:`hitman.ratio.pairing.PAIRINGS` or an instance. A
    non-jittable pairing (``group_derangement``) forces the host path, since it needs
    ``np.unique`` on the batch.

    ``use_weights`` defaults to "weighted iff the dataset says it is weighted". When it
    resolves to False the loss takes ``nre_loss``'s unweighted branch verbatim, so an
    unweighted dataset trains on exactly the path it would have without any weight
    plumbing at all.

    ``on_best`` is called after every improving checkpoint, so a run killed partway
    through still leaves an honest record of its best model rather than nothing.
    """
    tr = dataset.split_view("train")
    va = dataset.split_view("val")
    if tr.n_rows == 0 or va.n_rows == 0:
        raise ValueError(
            f"empty train ({tr.n_rows}) or val ({va.n_rows}) split. Splits are GROUP-level; "
            f"a small dataset can land every group on one side -- widen it or adjust the "
            f"split fractions.")

    pair = get_pairing(pairing)
    pair_name = getattr(pair, "name", type(pair).__name__)
    idx_tr = build_group_index(tr.group_id)
    idx_va = build_group_index(va.group_id)
    if getattr(pair, "group_aware", False) and min(idx_tr.n_groups, idx_va.n_groups) < 2:
        raise ValueError(
            f"pairing {pair_name!r} is group-aware but a split has < 2 groups "
            f"(train={idx_tr.n_groups}, val={idx_va.n_groups}): every 'marginal' pair "
            f"would be a joint pair.")

    weighted = (not tr.weights_are_unit) if use_weights is None else bool(use_weights)
    if steps is None:
        steps = max(1, epochs * max(1, tr.n_rows // max(1, batch_size)))
    bs = int(min(batch_size, tr.n_rows))

    optimizer = optax.chain(optax.clip_by_global_norm(clip_norm),
                            optax.adam(learning_rate))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # -- residency ----------------------------------------------------------
    est_gb = _estimate_device_gb(tr.x, tr.theta, va.x, va.theta,
                                 idx_tr.order, idx_va.order) + 2.0
    if residency == "device":
        use_device = True
    elif residency == "host":
        use_device = False
    else:
        use_device = est_gb <= float(device_budget_gb)
    if use_device and not pair.jittable:
        use_device = False   # host-only pairing (np.unique) cannot trace
    mode = "device" if use_device else "host"
    if verbose:
        print(f"[ratio] {mode}-resident (est {est_gb:.2f} GB, budget "
              f"{device_budget_gb:.1f} GB)  pairing={pair_name}  weighted={weighted}  "
              f"steps={steps}  batch={bs}", flush=True)

    def _put(a):
        return jnp.asarray(a) if use_device else np.asarray(a)

    x_tr, th_tr = _put(tr.x), _put(tr.theta)
    x_va, th_va = _put(va.x), _put(va.theta)
    w_tr = _put(tr.weight) if weighted else None
    w_va = _put(va.weight) if weighted else None
    gi_tr = idx_tr.to_device() if use_device else idx_tr
    gi_va = idx_va.to_device() if use_device else idx_va

    # Two step flavours, and which one runs is the single most important performance
    # decision in this loop.
    #
    # DEVICE-resident: the tables are already on the device, so they are passed as
    # ARGUMENTS (never closed over -- a captured jax array is baked into the executable as
    # a constant) and the gather happens inside the jit. Only a batch-length index array
    # crosses the bus per step.
    #
    # HOST-resident: the gather MUST happen on the host. Passing a host array to a jitted
    # function device-puts THE WHOLE ARRAY, so gathering inside the jit would transfer the
    # entire table every single step -- at 10^8 rows that is gigabytes per step and it
    # exhausts the allocator outright. The host path therefore does `np.take` itself and
    # hands the step a batch-sized array, which is the only host->device crossing.
    @eqx.filter_jit
    def step_from_batch(model, opt_state, xb, tp, tn, wb):
        (loss, _aux), grads = eqx.filter_value_and_grad(
            lambda m: ratio_bce(m, xb, tp, tn, wb, balance_weight), has_aux=True)(model)
        updates, opt_state = optimizer.update(grads, opt_state)
        return eqx.apply_updates(model, updates), opt_state, loss

    @eqx.filter_jit
    def step_gather(model, opt_state, x, th, w, gi, rows, key):
        neg = pair(key, rows, gi)
        xb = jnp.take(x, rows, axis=0)
        tp = jnp.take(th, rows, axis=0)
        tn = jnp.take(th, neg, axis=0)
        wb = None if w is None else jnp.take(w, rows, axis=0)
        (loss, _aux), grads = eqx.filter_value_and_grad(
            lambda m: ratio_bce(m, xb, tp, tn, wb, balance_weight), has_aux=True)(model)
        updates, opt_state = optimizer.update(grads, opt_state)
        return eqx.apply_updates(model, updates), opt_state, loss

    @eqx.filter_jit
    def eval_from_batch(model, xb, tp, tn, wb):
        loss, (lp, ln) = ratio_bce(model, xb, tp, tn, wb, balance_weight)
        return loss, lp, ln

    @eqx.filter_jit
    def eval_gather(model, x, th, w, rows, neg):
        xb = jnp.take(x, rows, axis=0)
        tp = jnp.take(th, rows, axis=0)
        tn = jnp.take(th, neg, axis=0)
        wb = None if w is None else jnp.take(w, rows, axis=0)
        loss, (lp, ln) = ratio_bce(model, xb, tp, tn, wb, balance_weight)
        return loss, lp, ln

    def draw_negatives(rows, key, gi):
        """Negatives for `rows`, drawn where the data lives.

        On host this uses the pairing's numpy twin: calling the jax version would
        device-put the 10^8-row label/order tables just to index them.
        """
        if use_device and pair.jittable:
            return pair(key, rows, gi)
        return pair.numpy(rng, np.asarray(rows), gi)

    # -- validation subset: fixed across epochs so the curve is comparable ---
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2 ** 31 - 1)))
    if 0 < val_max_rows < va.n_rows:
        val_rows_np = np.sort(rng.choice(va.n_rows, val_max_rows, replace=False))
    else:
        val_rows_np = np.arange(va.n_rows)
    val_rows_np = val_rows_np.astype(np.int32)

    vcr = max(1, int(val_chunk_rows))

    def run_val(m, key):
        """Validation in chunks, so no full-val-size activation is ever materialized.

        Peak device residency is O(val_chunk_rows), not O(val_max_rows): a single jitted
        forward over 2M rows would allocate a multi-GB ``[rows, width]`` activation per
        layer PER CLASS (joint and marginal), which OOMs under a fractional-VRAM ceiling
        long before the dataset itself does.

        The negatives are drawn ONCE for the whole subsample rather than per chunk, so the
        group-aware pairing sees the full group structure instead of whatever handful of
        groups a chunk happens to contain. Only the forward pass is chunked.

        ``val_loss`` is the chunk-size-weighted mean of the per-chunk means -- exactly the
        overall mean for the unweighted loss, and a close approximation when per-row
        weights are active (``nre_loss`` normalizes within a batch). Logits are pulled to
        host only up to ``auc_max_rows``: the host rank-sort, not the device forward, is
        what makes validation expensive.
        """
        neg_all = np.asarray(draw_negatives(
            jnp.asarray(val_rows_np) if (use_device and pair.jittable) else val_rows_np,
            key, gi_va))
        n = int(val_rows_np.shape[0])
        cap = auc_max_rows or n
        total_loss, total_n, collected = 0.0, 0, 0
        pos_chunks, neg_chunks = [], []
        for s in range(0, n, vcr):
            e = min(s + vcr, n)
            r, g = val_rows_np[s:e], neg_all[s:e]
            if use_device:
                loss_c, lp_c, ln_c = eval_gather(m, x_va, th_va, w_va,
                                                 jnp.asarray(r), jnp.asarray(g))
            else:
                wb = np.take(w_va, r, axis=0) if w_va is not None else None
                loss_c, lp_c, ln_c = eval_from_batch(
                    m, np.take(x_va, r, axis=0), np.take(th_va, r, axis=0),
                    np.take(th_va, g, axis=0), wb)
            n_c = e - s
            total_loss += float(loss_c) * n_c
            total_n += n_c
            if collected < cap:      # only pull enough logits to host to hit the cap
                take = min(cap - collected, n_c)
                pos_chunks.append(np.asarray(lp_c[:take]))
                neg_chunks.append(np.asarray(ln_c[:take]))
                collected += take
        lp = np.concatenate(pos_chunks)
        ln = np.concatenate(neg_chunks)
        return {
            "val_loss": total_loss / max(1, total_n),
            "val_auc": roc_auc(lp, ln),
            "val_ece": expected_calibration_error(lp, ln),
            "val_self_norm": self_normalization(ln),
        }

    # -- loop ---------------------------------------------------------------
    # separate streams for the epoch permutation and the validation negatives, so
    # changing the validation cadence cannot perturb the training batch order
    perm_key = jax.random.PRNGKey(int(jax.random.randint(key, (), 0, 2 ** 31 - 1)))
    val_key = jax.random.PRNGKey(int(jax.random.randint(key, (), 0, 2 ** 31 - 1)) + 1)
    epoch, pos = 0, 0

    def epoch_perm(ep):
        """Fresh permutation of the training rows.

        On device this runs on the GPU (a host permutation of 10^8 rows costs seconds per
        epoch with the device idle). On host it stays in numpy -- asking jax for it would
        allocate the whole index array on the device purely to copy it straight back.
        """
        if use_device:
            return jax.random.permutation(jax.random.fold_in(perm_key, ep), tr.n_rows)
        return rng.permutation(tr.n_rows).astype(np.int64)

    perm = epoch_perm(epoch)
    best = (float("inf"), model, 0)
    history, t0 = [], time.time()

    for step in range(1, int(steps) + 1):
        if pos + bs > tr.n_rows:
            epoch += 1
            perm = epoch_perm(epoch)
            pos = 0
        rows = perm[pos:pos + bs]
        pos += bs
        step_key = jax.random.fold_in(key, step)
        if use_device and pair.jittable:
            model, opt_state, loss = step_gather(
                model, opt_state, x_tr, th_tr, w_tr, gi_tr, rows, step_key)
        else:
            r = np.asarray(rows)
            neg = draw_negatives(r, step_key, gi_tr)
            wb = np.take(w_tr, r, axis=0) if w_tr is not None else None
            # np.take, not fancy indexing: same result, ~2x faster on a scattered gather
            # from a large host array (it calls straight into the specialized C loop)
            model, opt_state, loss = step_from_batch(
                model, opt_state, np.take(x_tr, r, axis=0), np.take(th_tr, r, axis=0),
                np.take(th_tr, neg, axis=0), wb)

        if step % val_every == 0 or step == steps:
            metrics = run_val(model, jax.random.fold_in(val_key, step))
            history.append({"step": step, "train_loss": float(loss), **metrics})
            if metrics["val_loss"] < best[0]:
                best = (metrics["val_loss"], model, step)
                if checkpoint_path is not None:
                    eqx.tree_serialise_leaves(checkpoint_path, model)
                if on_best is not None:
                    on_best({"step": step, "wall_sec": time.time() - t0,
                             "history": list(history), **metrics})
            if verbose:
                print(f"[ratio] step {step:6d}  train {float(loss):.4f}  "
                      f"val {metrics['val_loss']:.4f}  auc {metrics['val_auc']:.4f}  "
                      f"ece {metrics['val_ece']:.4f}  "
                      f"(best {best[0]:.4f} @ {best[2]}, {time.time() - t0:.0f}s)",
                      flush=True)

    best_val, best_model, best_step = best
    final = run_val(best_model, jax.random.fold_in(val_key, 0))
    from hitman.ratio.model import n_parameters
    return FitResult(
        model=best_model, history=history, best_step=int(best_step),
        best_val_loss=float(best_val), final_val_loss=final["val_loss"],
        final_val_auc=final["val_auc"], n_params=n_parameters(best_model),
        n_train=tr.n_rows, n_val=va.n_rows, wall_sec=time.time() - t0,
        residency=mode, pairing=pair_name, weighted=weighted,
        steps=int(steps), batch_size=bs,
    )

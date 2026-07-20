"""The training recipe as one loop: step-based SGD to value-plateau, then cosine polish.

Regime lessons baked in (design report, Addenda 4/8 + the 1M/5M campaign):
- Validation and early stopping are STEP-based, checked sub-epoch: at 5M events an
  "epoch" is 3350 steps and all decisive movement happens in the first ~2 epochs.
- ``min_delta`` is the aggressiveness knob: an improvement must exceed it to reset
  the patience clock, which kills the ~1e-6/epoch tail-grinding that BCE receipts
  showed to be physics-irrelevant (best models still checkpoint on ANY improvement).
- One validation yardstick across both stages; the returned model is the GLOBAL best
  (fixes the stage-carryover flaw of chained fit_resident calls).
- No exact L-BFGS stage: receipt-gated only (harmful at 5M — Addendum 8).
- Batches are contiguous shuffled windows (gather coalescing, Addendum 6).
"""

import os
import time
from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from hitman.train.loop import _batch_loss, _unpack_batch


@dataclass
class RecipeResult:
    model: object
    best_val: float
    best_stage: str
    best_step: int
    history: list = field(default_factory=list)  # (stage, step, val)


def _window_stream(n_train, batch_size, rng):
    """Endless stream of contiguous window starts in shuffled order with random phase."""
    while True:
        phase = int(rng.integers(0, batch_size)) if n_train > 2 * batch_size else 0
        n_windows = (n_train - phase) // batch_size
        for s in rng.permutation(n_windows):
            yield int(s) * batch_size + phase


def train_recipe(
    model,
    data,
    make_batch,
    n_rows: int,
    *,
    key,
    sgd_lr: float = 1e-3,
    sgd_batch: int = 2**17,
    sgd_max_steps: int = 400_000,
    cosine_peak: float = 3e-4,
    cosine_steps: int = 30_000,
    cosine_batch: int = None,
    val_every: int = 1000,
    patience_steps: int = 10_000,
    min_delta: float = 2e-5,
    val_fraction: float = 0.1,
    max_val_rows: int = 2**16,
    balance_weight: float = 0.0,
    extra_loss=None,
    checkpoint_dir: str = None,
    snapshot_every: int = 1,
    verbose: bool = True,
) -> RecipeResult:
    """Two-stage staged training in one loop with a shared validation yardstick.

    ``extra_loss(model, data, key, step) -> scalar``, when given, is added to the
    training BCE every step (e.g. the score-identity / GMM moment penalties from
    hitman.train.identities — pass them pre-scaled). It receives ``data`` as an
    argument so large arrays ride the jit tracer instead of being closure-captured
    into the compiled step as constants (a 4 GB capture hard-locked the box on
    2026-07-19), and ``step`` as a traced scalar (warm-up ramps; no recompiles).
    It is deliberately NOT added to the validation yardstick: validation stays the
    pure comparable BCE, and the extra term's effect is judged by its own receipts.
    """
    cosine_batch = cosine_batch or 2 * sgd_batch
    n_val = max(int(n_rows * val_fraction), 1)
    n_train = n_rows - n_val
    val_idx = np.arange(n_train, n_rows, dtype=np.int64)
    if n_val > max_val_rows:
        val_idx = val_idx[:: n_val // max_val_rows + 1][:max_val_rows]
    val_rows = jnp.asarray(val_idx, jnp.int32)
    key, val_key = jax.random.split(key)
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    @eqx.filter_jit
    def val_bce(model, data, rows, key):
        k_aug, k_loss = jax.random.split(key)
        obs, hyp, w = _unpack_batch(make_batch(data, rows, k_aug))
        return _batch_loss(model, obs, hyp, k_loss, balance_weight, w)

    if extra_loss is not None:
        # fixed key -> fixed identity batch + splits: a deterministic penalty
        # yardstick per val point (reported alongside BCE, still not part of it)
        @eqx.filter_jit
        def val_pen(model, data, key, step):
            return extra_loss(model, data, key, step)

    def make_step(optimizer, batch_size):
        iota = jnp.arange(batch_size, dtype=jnp.int32)

        @eqx.filter_jit
        def step(model, opt_state, data, start, key, step_no):
            rows = jnp.asarray(start, jnp.int32) + iota
            k_aug, k_loss, k_extra = jax.random.split(key, 3)

            def loss_fn(model):
                obs, hyp, w = _unpack_batch(make_batch(data, rows, k_aug))
                loss = _batch_loss(model, obs, hyp, k_loss, balance_weight, w)
                if extra_loss is not None:
                    loss = loss + extra_loss(model, data, k_extra, step_no)
                return loss

            loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
            updates, opt_state = optimizer.update(grads, opt_state)
            return eqx.apply_updates(model, updates), opt_state, loss

        return step

    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
    best = (np.inf, model, "init", 0)   # global across stages
    history = []
    gstep = 0   # global step across stages: warm-up ramps must not restart at cosine

    def run_stage(name, model, optimizer, batch_size, max_steps, key):
        nonlocal best, gstep
        step_fn = make_step(optimizer, batch_size)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        stream = _window_stream(n_train, batch_size, rng)
        gate = np.inf   # last val that reset the patience clock (min_delta gating)
        gate_step = 0
        t0 = time.time()
        for s in range(1, max_steps + 1):
            key, sk = jax.random.split(key)
            # start must be a device array: filter_jit treats a Python int as a
            # STATIC argument and recompiles the step for every window start
            start = jnp.asarray(next(stream), jnp.int32)
            gstep += 1
            model, opt_state, _ = step_fn(model, opt_state, data, start, sk,
                                          jnp.asarray(gstep, jnp.float32))
            if s % val_every == 0:
                v = float(val_bce(model, data, val_rows, val_key))
                pen = (float(val_pen(model, data, val_key,
                                     jnp.asarray(gstep, jnp.float32)))
                       if extra_loss is not None else None)
                history.append((name, s, v) if pen is None else (name, s, v, pen))
                if v < best[0]:
                    best = (v, model, name, s)
                    if checkpoint_dir is not None:
                        eqx.tree_serialise_leaves(
                            os.path.join(checkpoint_dir, "best.eqx"), model)
                if checkpoint_dir is not None and (s // val_every) % snapshot_every == 0:
                    # trajectory snapshots feed receipt-based selection; gate with
                    # snapshot_every if the file writes ever matter (~1 MB each)
                    eqx.tree_serialise_leaves(
                        os.path.join(checkpoint_dir, f"{name}_step{s:07d}.eqx"), model)
                if v < gate - min_delta:
                    gate, gate_step = v, s
                if verbose:
                    extra_txt = "" if pen is None else f"  pen {pen:.4g}"
                    print(f"[{name}] step {s:7d}  val {v:.5f}{extra_txt}  "
                          f"(best {best[0]:.5f}, {time.time()-t0:.0f}s)", flush=True)
                if s - gate_step >= patience_steps:
                    if verbose:
                        print(f"[{name}] value-plateau stop at step {s} "
                              f"(no {min_delta:g} improvement in {patience_steps} steps)",
                              flush=True)
                    break
        return model, key

    model, key = run_stage("sgd", model, optax.adam(sgd_lr), sgd_batch,
                           sgd_max_steps, key)
    sched = optax.cosine_decay_schedule(cosine_peak, cosine_steps, alpha=0.01)
    model, key = run_stage("cosine", best[1], optax.adam(sched), cosine_batch,
                           cosine_steps, key)

    return RecipeResult(model=best[1], best_val=best[0], best_stage=best[2],
                        best_step=best[3], history=history)

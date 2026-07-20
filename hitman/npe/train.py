"""NPE training loop: event-grouped batches, adam + cosine schedule, early stopping.

Distinct from ``hitman.train.recipe`` (that is the NRE row-window loop with the marginal
permutation loss); NPE trains a density by forward-KL on event-grouped batches. The
jitted step takes ``data`` and ``spec`` as ARGUMENTS (never closure captures) so the
device-resident dataset rides the tracer -- a closure-captured DeviceData baked ~4 GB of
constants into the compile and hard-locked the box (2026-07-19). Drivers additionally
escalate jax's captured-constants warning to an error.
"""

import os
import time
from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from hitman.npe.batch import make_npe_batch
from hitman.npe.loss import npe_loss


@dataclass
class NpeFitResult:
    model: object                          # (encoder, flow)
    best_val: float
    best_step: int
    history: list = field(default_factory=list)   # (step, train_nll, val_nll)


def fit_npe(model, data, spec, *, key, val_batch, n_events=512, n_pad=320,
            time_sigma=50.0, lr=1e-3, max_steps=40_000, cosine_alpha=0.02,
            val_every=500, patience_steps=8_000, min_delta=1e-3,
            checkpoint_dir=None, snapshot_every=10, verbose=True):
    """Train ``model=(encoder, flow)`` by forward-KL on event-grouped batches.

    ``val_batch`` is a fixed held-out (hits, mask, charge, theta) tuple (one comparable
    yardstick across steps). ``spec`` selects the training-zone eligible events.
    Returns the GLOBAL best-on-val model.
    """
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    sched = optax.cosine_decay_schedule(lr, max_steps, alpha=cosine_alpha)
    optimizer = optax.adam(sched)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def train_step(model, opt_state, data, spec, key):
        def loss_fn(m):
            batch = make_npe_batch(data, spec, key, n_events, n_pad, time_sigma)
            return npe_loss(m, batch)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state)
        return eqx.apply_updates(model, updates), opt_state, loss

    @eqx.filter_jit
    def val_loss(model, batch):
        return npe_loss(model, batch)

    best = (np.inf, model, 0)
    gate, gate_step = np.inf, 0
    history = []
    t0 = time.time()
    for s in range(1, max_steps + 1):
        key, sk = jax.random.split(key)
        model, opt_state, tr = train_step(model, opt_state, data, spec, sk)
        if s % val_every == 0:
            v = float(val_loss(model, val_batch))
            history.append((s, float(tr), v))
            if v < best[0]:
                best = (v, model, s)
                if checkpoint_dir is not None:
                    eqx.tree_serialise_leaves(
                        os.path.join(checkpoint_dir, "best.eqx"), model)
            if checkpoint_dir is not None and (s // val_every) % snapshot_every == 0:
                eqx.tree_serialise_leaves(
                    os.path.join(checkpoint_dir, f"step{s:07d}.eqx"), model)
            if v < gate - min_delta:
                gate, gate_step = v, s
            if verbose:
                print(f"[npe] step {s:7d}  train {float(tr):.4f}  val {v:.4f}  "
                      f"(best {best[0]:.4f}, {time.time() - t0:.0f}s)", flush=True)
            if s - gate_step >= patience_steps:
                if verbose:
                    print(f"[npe] value-plateau stop at step {s}", flush=True)
                break
    return NpeFitResult(model=best[1], best_val=best[0], best_step=best[2],
                        history=history)

"""Event-batched exact-MLE training for the run20 spline model.

Batches are drawn PER EVENT (the conditioner + softmax need all 241 sensors per theta):
``build_event_spec`` lists the eligible events, ``make_event_batch`` gathers a padded
(pmt_id, t, mask, theta) batch on-device with the coherent per-event time augmentation.
The augmentation shifts each event's hit times and its theta_t together, which leaves the
TOF residual u = (t+dt) - (theta_t+dt) - n_eff d/c INVARIANT -- so it changes nothing in
this model; it is kept only for consistency with the rest of the campaign (and documented
as a no-op here). Two-stage (sgd -> cosine) loop mirrors ``train_recipe``; validation is
the exact-MLE loss on a FIXED held-out batch; best/snapshot checkpoints are written on
val improvement with a first-write assertion (a prior run lost its best model to a silent
checkpoint failure).
"""

import os
import time
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from hitman.splinemle.loss import splinemle_loss
from hitman.train.nwj import _jaxpr_max_bytes


class EventSpec(NamedTuple):
    offsets: jnp.ndarray  # (n_events + 1,) int32 -- store hit-offset table
    elig: jnp.ndarray     # (n_elig,) int32 -- drawable event indices (<= n_pad hits)


def build_event_spec(store, n_pad: int, lo: int = 0, hi: int = None) -> EventSpec:
    """Eligible = events in [lo, hi) with <= n_pad hits (over-pad events are dropped)."""
    hi = store.n_events if hi is None else hi
    nhit = np.diff(store.hit_offsets[lo:hi + 1])
    elig = lo + np.flatnonzero(nhit <= n_pad)
    return EventSpec(offsets=jnp.asarray(store.hit_offsets, jnp.int32),
                     elig=jnp.asarray(elig, jnp.int32))


def make_event_batch(data, spec: EventSpec, key, n_events: int, n_pad: int,
                     time_sigma: float = 50.0):
    """Sample an event-grouped padded batch -> (pmt_id (B,P), t (B,P), mask (B,P),
    theta (B,7)); all gathers on device. The coherent time shift (theta_t and t together)
    leaves u invariant -- a no-op for this model, kept for campaign consistency."""
    k_ev, k_aug = jax.random.split(key)
    ev = spec.elig[jax.random.randint(k_ev, (n_events,), 0, spec.elig.shape[0])]
    start = spec.offsets[ev]
    nh = spec.offsets[ev + 1] - start
    slot = jnp.arange(n_pad, dtype=jnp.int32)
    rows = jnp.minimum(start[:, None] + slot[None, :], data.n_hits - 1)
    mask = (slot[None, :] < nh[:, None]).astype(jnp.float32)
    t = data.t[rows]
    theta = data.hyp[ev]
    if time_sigma > 0:
        sh = time_sigma * jax.random.normal(k_aug, (n_events,))
        t = t + sh[:, None]
        theta = theta.at[:, 5].add(sh)
    pmt_ids = data.pmt_id[rows]
    return pmt_ids, t, mask, theta


def step_max_intermediate_gib(model, data, spec: EventSpec, *, n_events, n_pad,
                              w_count=1.0, time_sigma=50.0, n_chunk=1, key=None):
    """Static audit (jax.make_jaxpr, allocates nothing): largest intermediate tensor (GiB)
    in the value-and-grad training step at the exact given config. Reuses nwj's jaxpr walk;
    the CPU pre-flight aborts a GPU-intended run before an OOM launch."""
    if key is None:
        key = jax.random.PRNGKey(0)

    def loss(m):
        batch = make_event_batch(data, spec, key, n_events, n_pad, time_sigma)
        l, _ = splinemle_loss(m, batch, w_count=w_count, n_chunk=n_chunk)
        return l

    jaxpr = jax.make_jaxpr(eqx.filter_grad(loss))(model)
    return _jaxpr_max_bytes(jaxpr.jaxpr) / 2**30


class FitResult(NamedTuple):
    model: object
    best_val: float
    best_stage: str
    best_step: int
    history: list


def fit_splinemle(
    model,
    data,
    train_spec: EventSpec,
    val_batch,
    *,
    key,
    n_events: int = 2048,
    n_pad: int = 320,
    w_count: float = 1.0,
    n_chunk: int = 1,
    time_sigma: float = 50.0,
    sgd_lr: float = 1e-3,
    sgd_max_steps: int = 200_000,
    cosine_peak: float = 3e-4,
    cosine_steps: int = 30_000,
    val_every: int = 500,
    patience_steps: int = 10_000,
    min_delta: float = 2e-5,
    checkpoint_dir: str = None,
    snapshot_every: int = 20,
    verbose: bool = True,
) -> FitResult:
    """Two-stage exact-MLE training of a SplineMLE. ``val_batch`` is a fixed
    (pmt_id, t, mask, theta) tuple. Returns the GLOBAL best-on-val model."""
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    @eqx.filter_jit
    def val_fn(model):
        loss, _ = splinemle_loss(model, val_batch, w_count=w_count, n_chunk=n_chunk)
        return loss

    def make_step(opt):
        @eqx.filter_jit
        def step(model, opt_state, data, key):
            def loss_fn(m):
                batch = make_event_batch(data, train_spec, key, n_events, n_pad, time_sigma)
                l, aux = splinemle_loss(m, batch, w_count=w_count, n_chunk=n_chunk)
                return l, aux

            (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
            updates, opt_state = opt.update(grads, opt_state)
            return eqx.apply_updates(model, updates), opt_state, loss, aux

        return step

    best = (np.inf, model, "init", 0)
    history = []
    wrote_best = [False]
    gstep = 0

    def maybe_checkpoint(model, v, name, s):
        nonlocal best
        if v < best[0]:
            best = (v, model, name, s)
            if checkpoint_dir is not None:
                path = os.path.join(checkpoint_dir, "best.eqx")
                eqx.tree_serialise_leaves(path, model)
                if not wrote_best[0]:
                    assert os.path.exists(path) and os.path.getsize(path) > 0, (
                        f"best.eqx NOT written at {path} -- silent checkpoint failure")
                    if verbose:
                        print(f"[checkpoint] best.eqx written OK "
                              f"({os.path.getsize(path)} bytes) at {name} step {s}",
                              flush=True)
                    wrote_best[0] = True
        if (checkpoint_dir is not None and snapshot_every
                and (s // val_every) % snapshot_every == 0):
            eqx.tree_serialise_leaves(
                os.path.join(checkpoint_dir, f"{name}_step{s:07d}.eqx"), model)

    def run_stage(name, model, opt, max_steps, key):
        nonlocal gstep
        step_fn = make_step(opt)
        opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
        gate, gate_step = np.inf, 0
        t0 = time.time()
        for s in range(1, max_steps + 1):
            key, sk = jax.random.split(key)
            gstep += 1
            model, opt_state, _, _ = step_fn(model, opt_state, data, sk)
            if s % val_every == 0:
                v = float(val_fn(model))
                history.append({"stage": name, "step": s, "nll": v})
                maybe_checkpoint(model, v, name, s)
                if v < gate - min_delta:
                    gate, gate_step = v, s
                if verbose:
                    print(f"[{name}] step {s:7d}  val {v:.5f}  "
                          f"(best {best[0]:.5f}, {time.time()-t0:.0f}s)", flush=True)
                if s - gate_step >= patience_steps:
                    if verbose:
                        print(f"[{name}] value-plateau stop at step {s}", flush=True)
                    break
        return model, key

    model, key = run_stage("sgd", model, optax.adam(sgd_lr), sgd_max_steps, key)
    if cosine_steps and cosine_steps > 0:
        sched = optax.cosine_decay_schedule(cosine_peak, cosine_steps, alpha=0.01)
        model, key = run_stage("cosine", best[1], optax.adam(sched), cosine_steps, key)

    return FitResult(model=best[1], best_val=best[0], best_stage=best[2],
                     best_step=best[3], history=history)

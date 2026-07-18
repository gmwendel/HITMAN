"""Joint event-model training: hitnet + chargenet in one loop, one holdout, one target.

The all-sensor formulation is asymmetric by construction: hitnet trains on one row per
photon (~100x the events), chargenet on one row per event. Sharing an epoch clock would
under/over-train one net, so each keeps its own step budget per round; what is shared
is the validation holdout (split at an EVENT boundary, so the same events are held out
of both nets and no event straddles the split) and the stopping criterion: the
binary-cross-entropy of the *composed* event-level log-ratio

    log r_event(obs, theta) = sum_hits log r_hit + log r_charge

on held-out events — the quantity inference actually uses. Per round, three metrics are
reported: hitnet BCE, chargenet BCE, and the composed event-level BCE.
"""

import time
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from hitman.train.loop import _batch_loss
from hitman.train.resident import DeviceData, charge_batch, hit_batch


@dataclass
class EventFitResult:
    hitnet: object
    chargenet: object
    hit_val: list  # per-round hitnet BCE on val hits
    charge_val: list  # per-round chargenet BCE on val events
    event_val: list  # per-round composed event-level BCE (diagnostic; selector if opted in)
    best_hit_round: int  # round the returned hitnet was snapshotted (its own val optimum)
    best_charge_round: int  # same for chargenet
    final_event_bce: float  # composed BCE of the RETURNED pair (recomputed after selection)


def event_val_bce(
    hitnet, chargenet, data: DeviceData, hit_start: int, event_start: int, key,
    chunk_size: int = 2**19,
):
    """Composed event-level classifier BCE on the validation tail.

    Events [event_start:] are the holdout; because event_id is sorted, their hits are
    the contiguous tail [hit_start:]. Marginal pairs permute hypotheses among the
    validation events (all-sensor free shuffle at event granularity).

    The hit forward pass is evaluated in fixed-size chunks: at ~100 hits/event even a
    modest event holdout is millions of hit rows, and a single vmapped forward over
    them (x2, joint+marginal held live in one graph) allocates tens of GB of
    activations. Chunking bounds transient memory to ~chunk_size x width floats while
    the per-event sums accumulate in a (n_val,) buffer.
    """
    n_val = data.n_events - event_start
    charge = data.charge[event_start:]
    hyp = data.hyp[event_start:]
    perm = jax.random.permutation(key, n_val)

    n_hit_rows = data.n_hits - hit_start
    pad = (-n_hit_rows) % chunk_size
    rows = np.arange(hit_start, data.n_hits + pad, dtype=np.int32)
    rows[n_hit_rows:] = hit_start  # padding rows point at a valid hit, weighted 0
    weights = np.ones(len(rows), np.float32)
    weights[n_hit_rows:] = 0.0
    row_chunks = rows.reshape(-1, chunk_size)
    w_chunks = weights.reshape(-1, chunk_size)

    obs_style = getattr(hitnet, "obs_style", "xyz")

    @eqx.filter_jit
    def chunk_segsum(hitnet, data, rows, w, hyp_table):
        ids = data.pmt_id[rows]
        if obs_style == "id_t":
            obs = (ids, data.t[rows])
        else:
            obs = jnp.concatenate([data.pmt_pos[ids], data.t[rows, None]], axis=1)
        local = data.event_id[rows] - event_start
        logits = jax.vmap(hitnet)(obs, hyp_table[local]) * w
        return jax.ops.segment_sum(logits, local, num_segments=n_val)

    def composed(hyp_table):
        hit_term = jnp.zeros(n_val, jnp.float32)
        for rc, wc in zip(row_chunks, w_chunks):
            hit_term = hit_term + chunk_segsum(
                hitnet, data, jnp.asarray(rc), jnp.asarray(wc), hyp_table
            )
        return hit_term + jax.vmap(chargenet)(charge, hyp_table)

    l_joint = composed(hyp)
    l_marg = composed(hyp[perm])
    return 0.5 * (jnp.mean(jax.nn.softplus(-l_joint)) + jnp.mean(jax.nn.softplus(l_marg)))


def fit_event_model(
    hitnet,
    chargenet,
    data: DeviceData,
    *,
    key,
    val_fraction: float = 0.1,
    max_val_events: int = 2**16,
    max_val_rows: int = 2**16,
    hit_batch_size: int = 2**17,
    charge_batch_size: int = 2**14,
    charge_passes_per_round: int = 4,
    max_rounds: int = 1000,
    patience: int = 20,
    learning_rate: float = 1e-3,
    balance_weight: float = 0.0,
    selection: str = "independent",
    verbose: bool = True,
) -> EventFitResult:
    """Train both networks in one loop, reporting per-net and composed metrics each round.

    One round = one full pass over training hits (hitnet) + ``charge_passes_per_round``
    full passes over training events (chargenet) — the per-net step budgets that the
    photon-multiplicity asymmetry requires.

    ``selection="independent"`` (default): each net is snapshotted at its OWN validation
    optimum and training stops when both have exhausted ``patience`` — the two nets need
    not converge at the same round, and per-net convergence stays visible for debugging.
    ``selection="composed"``: snapshot the pair at the best composed event-level BCE
    (couples both nets to one round; use once the parts are individually trusted).
    """
    if selection not in ("independent", "composed"):
        raise ValueError(f"unknown selection mode {selection!r}")
    # Event-aligned split: holdout events and (contiguous, since event_id is sorted)
    # their hits. The composed metric is evaluated on at most max_val_events of them.
    n_events = data.n_events
    event_start = n_events - max(int(n_events * val_fraction), 1)
    hit_start = int(np.searchsorted(np.asarray(data.event_id), event_start))
    metric_event_start = max(event_start, n_events - max_val_events)
    metric_hit_start = int(np.searchsorted(np.asarray(data.event_id), metric_event_start))

    opt = optax.adam(learning_rate)
    states = {
        "hit": opt.init(eqx.filter(hitnet, eqx.is_inexact_array)),
        "charge": opt.init(eqx.filter(chargenet, eqx.is_inexact_array)),
    }

    def make_step(make_batch):
        @eqx.filter_jit
        def step(model, state, data, rows, key):
            k_aug, k_loss = jax.random.split(key)

            def loss_fn(model):
                obs, hyp = make_batch(data, rows, k_aug)
                return _batch_loss(model, obs, hyp, k_loss, balance_weight)

            loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
            updates, state = opt.update(grads, state)
            return eqx.apply_updates(model, updates), state, loss

        return step

    hit_step, charge_step = make_step(hit_batch), make_step(charge_batch)

    @eqx.filter_jit
    def val_bce(model, data, rows, key, make_batch):
        k_aug, k_loss = jax.random.split(key)
        obs, hyp = make_batch(data, rows, k_aug)
        return _batch_loss(model, obs, hyp, k_loss, balance_weight)

    def _capped(lo, hi):
        idx = np.arange(lo, hi, dtype=np.int32)
        if len(idx) > max_val_rows:
            idx = idx[:: len(idx) // max_val_rows + 1][:max_val_rows]
        return jnp.asarray(idx)

    hit_val_rows = _capped(metric_hit_start, data.n_hits)
    charge_val_rows = _capped(metric_event_start, n_events)

    def sweep(model, state, step_fn, n_rows, batch_size, key, passes=1):
        bs = min(batch_size, n_rows)
        for _ in range(passes):
            key, pk = jax.random.split(key)
            perm = jax.random.permutation(pk, n_rows)
            for s in range(max(n_rows // bs, 1)):
                rows = jax.lax.dynamic_slice_in_dim(perm, s * bs, bs)
                key, sk = jax.random.split(key)
                model, state, _ = step_fn(model, state, data, rows, sk)
        return model, state, key

    best_hit = (np.inf, 0, hitnet)
    best_charge = (np.inf, 0, chargenet)
    best_pair = (np.inf, 0, hitnet, chargenet)
    hit_hist, charge_hist, event_hist = [], [], []
    # One fixed marginal pairing for all validation metrics: a fresh permutation per
    # round adds ~0.1 of sampling noise to the composed BCE (large summed logits),
    # which would corrupt patience/selection decisions. Fixed pairing makes every
    # metric exactly comparable across rounds (and the final recomputation).
    key, metric_key = jax.random.split(key)
    kh = jax.random.fold_in(metric_key, 0)
    kc = jax.random.fold_in(metric_key, 1)
    ke = jax.random.fold_in(metric_key, 2)

    for rnd in range(max_rounds):
        t0 = time.time()
        key, k1, k2 = jax.random.split(key, 3)
        # A net that has exhausted its own patience stops training (frozen at best);
        # the other continues — they need not converge at the same round.
        hit_active = rnd - best_hit[1] < patience
        charge_active = rnd - best_charge[1] < patience
        if hit_active:
            hitnet, states["hit"], _ = sweep(
                hitnet, states["hit"], hit_step, hit_start, hit_batch_size, k1
            )
        if charge_active:
            chargenet, states["charge"], _ = sweep(
                chargenet, states["charge"], charge_step, event_start, charge_batch_size,
                k2, passes=charge_passes_per_round,
            )
        hv = float(val_bce(hitnet, data, hit_val_rows, kh, hit_batch))
        cv = float(val_bce(chargenet, data, charge_val_rows, kc, charge_batch))
        ev = float(event_val_bce(hitnet, chargenet, data, metric_hit_start, metric_event_start, ke))
        hit_hist.append(hv)
        charge_hist.append(cv)
        event_hist.append(ev)
        if hit_active and hv < best_hit[0]:
            best_hit = (hv, rnd, hitnet)
        if charge_active and cv < best_charge[0]:
            best_charge = (cv, rnd, chargenet)
        if ev < best_pair[0]:
            best_pair = (ev, rnd, hitnet, chargenet)
        if verbose:
            flags = f"{'h' if hit_active else '-'}{'c' if charge_active else '-'}"
            print(
                f"round {rnd:4d} [{flags}]  hit {hv:.5f}  charge {cv:.5f}  "
                f"EVENT {ev:.5f}  ({time.time() - t0:.1f}s)",
                flush=True,
            )
        if selection == "composed":
            if rnd - best_pair[1] >= patience:
                break
        elif not hit_active and not charge_active:
            break

    if selection == "composed":
        sel_hit, sel_charge = best_pair[2], best_pair[3]
        best_hit_round = best_charge_round = best_pair[1]
    else:
        sel_hit, sel_charge = best_hit[2], best_charge[2]
        best_hit_round, best_charge_round = best_hit[1], best_charge[1]

    final_ev = float(
        event_val_bce(sel_hit, sel_charge, data, metric_hit_start, metric_event_start, ke)
    )
    return EventFitResult(
        hitnet=sel_hit, chargenet=sel_charge,
        hit_val=hit_hist, charge_val=charge_hist, event_val=event_hist,
        best_hit_round=best_hit_round, best_charge_round=best_charge_round,
        final_event_bce=final_ev,
    )

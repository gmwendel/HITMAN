"""The ratio lane end to end: dataset contract, model buffers, training, bundles.

The synthetic problem below is a GROUPED one with a genuinely learnable ratio: each group
draws a hypothesis, and every row of the group observes a noisy function of it. That
structure is what makes the group-aware machinery testable -- on ungrouped data every
pairing agrees.
"""

import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hitman.ratio import (
    RatioBundle, RatioDataset, Standardizer, build_ratio_model, compute_support,
    concatenate, effective_sample_size, fit_ratio, hash_split, importance_weights,
    load_bundle, make_barrier, save_bundle, trust_radius,
)
from hitman.ratio.model import RatioModel

X_NAMES = ("obs_a", "obs_b")
TH_NAMES = ("hyp_u", "hyp_v")


def make_grouped_dataset(n_groups=200, rows_per_group=12, seed=0, weighted=False,
                         source_id=0):
    """theta ~ N(0, 1) per group; x = (theta_u + noise, independent noise)."""
    rng = np.random.default_rng(seed)
    theta_g = rng.normal(size=(n_groups, 2))
    gid = np.repeat(np.arange(n_groups), rows_per_group).astype(np.int64)
    theta = np.repeat(theta_g, rows_per_group, axis=0)
    n = gid.size
    x = np.stack([theta[:, 0] * 1.2 + rng.normal(0, 0.3, n), rng.normal(0, 1.0, n)], axis=1)
    split = hash_split(gid + 1)
    ds = RatioDataset.unweighted(
        x.astype(np.float32), theta.astype(np.float32), gid, split,
        x_names=X_NAMES, theta_names=TH_NAMES, source_id=source_id)
    if weighted:
        w = rng.uniform(0.5, 2.0, n).astype(np.float32)
        ds = RatioDataset(x=ds.x, theta=ds.theta, group_id=ds.group_id, weight=w,
                          log_q=ds.log_q, source_id=ds.source_id, split=ds.split,
                          x_names=X_NAMES, theta_names=TH_NAMES)
    return ds


# ---------------------------------------------------------------------------
# dataset contract
# ---------------------------------------------------------------------------
def test_dataset_requires_consistent_column_lengths():
    ds = make_grouped_dataset(n_groups=20)
    with pytest.raises(ValueError, match="rows"):
        RatioDataset(x=ds.x, theta=ds.theta[:-1], group_id=ds.group_id, weight=ds.weight,
                     log_q=ds.log_q, source_id=ds.source_id, split=ds.split)


def test_dataset_rejects_unknown_split_labels():
    ds = make_grouped_dataset(n_groups=20)
    bad = ds.split.copy()
    bad[0] = "trian"
    with pytest.raises(ValueError, match="unknown split labels"):
        RatioDataset(x=ds.x, theta=ds.theta, group_id=ds.group_id, weight=ds.weight,
                     log_q=ds.log_q, source_id=ds.source_id, split=bad)


def test_dataset_rejects_name_width_mismatch():
    ds = make_grouped_dataset(n_groups=20)
    with pytest.raises(ValueError, match="x_names"):
        RatioDataset(x=ds.x, theta=ds.theta, group_id=ds.group_id, weight=ds.weight,
                     log_q=ds.log_q, source_id=ds.source_id, split=ds.split,
                     x_names=("only_one",))


def test_hash_split_is_group_level_and_read_order_independent():
    """THE property a row-level split cannot have.

    Every row of a group lands on one side, and the assignment survives a reshuffle of
    the rows -- so a rerun that reads the pool in a different order, or selects a
    different subset, still puts the same group in the same split.
    """
    gid = np.repeat(np.arange(300), 7).astype(np.int64)
    split = hash_split(gid)
    for g in np.unique(gid):
        assert len(set(split[gid == g])) == 1

    perm = np.random.default_rng(0).permutation(gid.size)
    assert np.array_equal(hash_split(gid[perm]), split[perm])

    subset = np.where(gid % 3 == 0)[0]
    assert np.array_equal(hash_split(gid[subset]), split[subset])

    fracs = {s: float(np.mean(split == s)) for s in ("train", "val", "test")}
    assert 0.6 < fracs["train"] < 0.8 and 0.08 < fracs["val"] < 0.22


def test_unweighted_constructor_fills_the_mandatory_columns_explicitly():
    ds = make_grouped_dataset(n_groups=30)
    assert ds.weights_are_unit and ds.is_single_source
    assert ds.weight.shape == (ds.n_rows,) and np.all(ds.weight == 1.0)
    assert ds.log_q.shape == (ds.n_rows,)
    assert ds.source_id.shape == (ds.n_rows,)


def test_concatenate_offsets_group_ids_so_sources_cannot_collide():
    """Two pools both numbering groups from 0 must not share a group after combining --
    otherwise a 'different group' negative could come from the same group in another pool."""
    a = make_grouped_dataset(n_groups=40, seed=1, source_id=0)
    b = make_grouped_dataset(n_groups=40, seed=2, source_id=0)
    a = RatioDataset(**{**a.__dict__, "log_q": np.full(a.n_rows, -1.0)})
    b = RatioDataset(**{**b.__dict__, "log_q": np.full(b.n_rows, -2.0)})
    c = concatenate([a, b])
    assert c.n_rows == a.n_rows + b.n_rows
    assert set(np.unique(c.source_id).tolist()) == {0, 1}
    ga = set(np.unique(c.group_id[c.source_id == 0]).tolist())
    gb = set(np.unique(c.group_id[c.source_id == 1]).tolist())
    assert ga.isdisjoint(gb)


def test_concatenate_refuses_sources_with_no_recorded_proposal():
    a = make_grouped_dataset(n_groups=20, seed=1)
    b = make_grouped_dataset(n_groups=20, seed=2)
    with pytest.raises(ValueError, match="log_q identically zero"):
        concatenate([a, b])
    concatenate([a, b], require_proposal=False)  # explicit opt-out is allowed


def test_concatenate_refuses_differently_named_features():
    a = make_grouped_dataset(n_groups=20, seed=1)
    b = RatioDataset(**{**make_grouped_dataset(n_groups=20, seed=2).__dict__,
                        "x_names": ("other_a", "other_b")})
    with pytest.raises(ValueError, match="feature NAMES differ"):
        concatenate([a, b], require_proposal=False)


def test_effective_sample_size_is_one_for_equal_weights_and_falls_for_skew():
    n = 1000
    assert effective_sample_size(np.ones(n))["overall"] == pytest.approx(1.0)
    skew = np.ones(n)
    skew[0] = 1000.0
    assert effective_sample_size(skew)["overall"] < 0.6
    sid = np.repeat([0, 1], n // 2)
    per = effective_sample_size(skew, sid)["per_source"]
    assert per[1] == pytest.approx(1.0)
    assert per[0] < per[1]


def test_importance_weights_are_mean_one_and_clipped():
    rng = np.random.default_rng(0)
    log_q = rng.normal(size=5000)
    log_p = rng.normal(size=5000)
    w = importance_weights(log_p, log_q, w_max=10.0)
    assert w.mean() == pytest.approx(1.0, rel=1e-4)
    assert w.max() <= 10.0 + 1e-5


# ---------------------------------------------------------------------------
# model: standardization as buffers
# ---------------------------------------------------------------------------
def test_standardizer_whitens_the_data_it_was_fit_on():
    v = np.random.default_rng(0).normal(3.0, 7.0, size=(5000, 2))
    s = Standardizer.from_data(v, ("a", "b"))
    out = np.asarray(s(jnp.asarray(v)))
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(out.std(axis=0), 1.0, atol=1e-3)


def test_standardizer_buffers_receive_zero_gradient():
    """They serialize like weights but must never train: a drifting input convention
    would silently change what the recorded support and every consumer mean."""
    ds = make_grouped_dataset(n_groups=40)
    model = build_ratio_model(ds, key=jax.random.PRNGKey(0), width=8, depth=2)

    def loss(m):
        return jnp.sum(m.logit_batch(jnp.asarray(ds.x[:32]), jnp.asarray(ds.theta[:32])))

    g = jax.grad(loss)(model)
    assert np.allclose(np.asarray(g.x_std.mean), 0.0)
    assert np.allclose(np.asarray(g.x_std.scale), 0.0)
    assert np.allclose(np.asarray(g.theta_std.mean), 0.0)
    assert not np.allclose(np.asarray(g.net.layers[0].weight), 0.0)  # the net does train


def test_model_fits_standardizers_on_train_split_only():
    """Fitting on val/test would leak their distribution into the input convention."""
    ds = make_grouped_dataset(n_groups=200)
    model = build_ratio_model(ds, key=jax.random.PRNGKey(0), width=8, depth=2)
    tr = ds.split_view("train")
    assert np.allclose(np.asarray(model.x_std.mean), tr.x.mean(axis=0), atol=1e-4)
    assert not np.allclose(np.asarray(model.x_std.mean), ds.x.mean(axis=0), atol=1e-6)


def test_encoder_slot_is_reserved_not_implemented():
    ds = make_grouped_dataset(n_groups=20)
    x_std = Standardizer.from_data(ds.x, X_NAMES)
    th_std = Standardizer.from_data(ds.theta, TH_NAMES)
    with pytest.raises(NotImplementedError, match="reserved slot"):
        RatioModel(x_std=x_std, theta_std=th_std, key=jax.random.PRNGKey(0),
                   encoder=object())


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def test_fit_ratio_learns_a_grouped_ratio():
    ds = make_grouped_dataset(n_groups=300, rows_per_group=10, seed=3)
    model = build_ratio_model(ds, key=jax.random.PRNGKey(0), width=32, depth=2)
    res = fit_ratio(model, ds, key=jax.random.PRNGKey(1), steps=250, batch_size=256,
                    learning_rate=3e-3, val_every=50, residency="host", verbose=False)
    assert res.history[-1]["val_loss"] < res.history[0]["val_loss"]
    assert res.final_val_auc > 0.65, res.final_val_auc
    assert res.n_params > 0 and res.pairing == "group_shift"


def test_fit_ratio_refuses_a_group_aware_pairing_on_a_single_group_split():
    ds = make_grouped_dataset(n_groups=200)
    one = RatioDataset(**{**ds.__dict__, "group_id": np.zeros(ds.n_rows, np.int64)})
    model = build_ratio_model(one, key=jax.random.PRNGKey(0), width=8, depth=2)
    with pytest.raises(ValueError, match="< 2 groups|>= 2 distinct groups"):
        fit_ratio(model, one, key=jax.random.PRNGKey(0), steps=5, batch_size=32,
                  val_every=5, residency="host", verbose=False)


def test_fit_ratio_refuses_empty_splits():
    ds = make_grouped_dataset(n_groups=60)
    allt = RatioDataset(**{**ds.__dict__,
                           "split": np.array(["train"] * ds.n_rows, dtype="<U5")})
    model = build_ratio_model(allt, key=jax.random.PRNGKey(0), width=8, depth=2)
    with pytest.raises(ValueError, match="empty train .* or val"):
        fit_ratio(model, allt, key=jax.random.PRNGKey(0), steps=5, batch_size=32,
                  val_every=5, residency="host", verbose=False)


def test_host_pairing_forces_the_host_path():
    """group_derangement uses np.unique and cannot trace; asking for device must not
    silently produce a jit error later."""
    ds = make_grouped_dataset(n_groups=120, rows_per_group=8)
    model = build_ratio_model(ds, key=jax.random.PRNGKey(0), width=8, depth=2)
    res = fit_ratio(model, ds, key=jax.random.PRNGKey(0), pairing="group_derangement",
                    steps=20, batch_size=128, val_every=10, residency="device",
                    verbose=False)
    assert res.residency == "host"
    assert res.pairing == "group_derangement"


def test_unit_weights_are_a_no_op_against_the_unweighted_path():
    """Explicitly weighting by ones must reproduce the unweighted training exactly."""
    ds = make_grouped_dataset(n_groups=150, rows_per_group=8, seed=7)
    common = dict(key=jax.random.PRNGKey(0), steps=40, batch_size=128, val_every=20,
                  residency="host", verbose=False)
    a = fit_ratio(build_ratio_model(ds, key=jax.random.PRNGKey(0), width=8, depth=2),
                  ds, use_weights=False, **common)
    b = fit_ratio(build_ratio_model(ds, key=jax.random.PRNGKey(0), width=8, depth=2),
                  ds, use_weights=True, **common)
    for ha, hb in zip(a.history, b.history):
        assert ha["val_loss"] == pytest.approx(hb["val_loss"], abs=1e-6)


def test_weighted_dataset_is_detected_automatically():
    ds = make_grouped_dataset(n_groups=120, rows_per_group=8, weighted=True)
    assert not ds.weights_are_unit
    res = fit_ratio(build_ratio_model(ds, key=jax.random.PRNGKey(0), width=8, depth=2),
                    ds, key=jax.random.PRNGKey(0), steps=20, batch_size=128,
                    val_every=10, residency="host", verbose=False)
    assert res.weighted is True


def test_on_best_callback_fires_so_a_killed_run_leaves_a_record(tmp_path):
    ds = make_grouped_dataset(n_groups=150, rows_per_group=8)
    seen = []
    res = fit_ratio(build_ratio_model(ds, key=jax.random.PRNGKey(0), width=8, depth=2),
                    ds, key=jax.random.PRNGKey(0), steps=60, batch_size=128,
                    val_every=20, residency="host", verbose=False,
                    on_best=seen.append,
                    checkpoint_path=str(tmp_path / "best.eqx"))
    assert seen, "on_best never fired"
    assert os.path.exists(tmp_path / "best.eqx")
    assert seen[-1]["step"] == res.best_step


# ---------------------------------------------------------------------------
# support / guard
# ---------------------------------------------------------------------------
def test_support_records_extrema_exactly_and_barrier_is_zero_in_distribution():
    ds = make_grouped_dataset(n_groups=200)
    model = build_ratio_model(ds, key=jax.random.PRNGKey(0), width=8, depth=2)
    sup = compute_support(model, ds)
    assert sup.dim == 4 and sup.names == X_NAMES + TH_NAMES

    tr = ds.split_view("train")
    feat = np.asarray(jax.vmap(model.standardized)(jnp.asarray(tr.x), jnp.asarray(tr.theta)))
    assert np.allclose(sup.lo, feat.min(axis=0)) and np.allclose(sup.hi, feat.max(axis=0))

    d0 = trust_radius(sup, prob=0.999)
    barrier = make_barrier(sup, weight=1.0)
    typical = jnp.asarray(feat[np.argsort(sup.d_mahalanobis(feat))[: len(feat) // 2]])
    assert float(jnp.max(jax.vmap(lambda f: barrier(f, d0))(typical))) == 0.0

    far = jnp.asarray(sup.mean + 50.0 * np.sqrt(np.diag(sup.cov)))
    assert float(barrier(far.astype(jnp.float32), d0)) > 0.0


def test_trust_radius_is_anchored_so_an_ood_anchor_is_not_dragged_back():
    """The ratified behaviour: an operating point that is itself OOD must be REPORTED,
    not silently re-fitted by a guard anchored only on the training quantile."""
    ds = make_grouped_dataset(n_groups=200)
    model = build_ratio_model(ds, key=jax.random.PRNGKey(0), width=8, depth=2)
    sup = compute_support(model, ds)
    global_d0 = trust_radius(sup, prob=0.999)
    ood = sup.mean + 30.0 * np.sqrt(np.diag(sup.cov))
    anchored = trust_radius(sup, ood, prob=0.999, margin=2.0)
    assert anchored > global_d0
    barrier = make_barrier(sup, weight=1.0)
    assert float(barrier(jnp.asarray(ood, jnp.float32), anchored)) == 0.0


def test_support_flags_out_of_distribution_points():
    ds = make_grouped_dataset(n_groups=200)
    model = build_ratio_model(ds, key=jax.random.PRNGKey(0), width=8, depth=2)
    sup = compute_support(model, ds)
    far = sup.mean + 40.0 * np.sqrt(np.diag(sup.cov))
    f = sup.flag(far, prob=0.999)
    assert f["n_out_of_distribution"] == 1
    assert f["d_max"] > sup.dm_at(0.999)


# ---------------------------------------------------------------------------
# bundles -- the permanent fix for the silent-mismatch bug class
# ---------------------------------------------------------------------------
def test_bundle_roundtrips_weights_standardizers_and_support(tmp_path):
    ds = make_grouped_dataset(n_groups=200)
    model = build_ratio_model(ds, key=jax.random.PRNGKey(0), width=16, depth=2)
    sup = compute_support(model, ds)
    path = str(tmp_path / "bundle")
    save_bundle(path, RatioBundle(model=model, support=sup,
                                  provenance={"pool": "synthetic"}))
    b = load_bundle(path)

    x = jnp.asarray(ds.x[:64])
    th = jnp.asarray(ds.theta[:64])
    np.testing.assert_allclose(np.asarray(b.model.logit_batch(x, th)),
                               np.asarray(model.logit_batch(x, th)), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(b.model.x_std.mean), np.asarray(model.x_std.mean))
    assert b.x_names == X_NAMES and b.theta_names == TH_NAMES
    assert b.support is not None
    np.testing.assert_allclose(b.support.mean, sup.mean)
    assert b.provenance["pool"] == "synthetic"


def test_bundle_refuses_a_dataset_with_a_different_feature_layout(tmp_path):
    """THE bug class this exists to kill: same widths, different meaning."""
    ds = make_grouped_dataset(n_groups=120)
    model = build_ratio_model(ds, key=jax.random.PRNGKey(0), width=8, depth=2)
    path = str(tmp_path / "b")
    save_bundle(path, RatioBundle(model=model, support=compute_support(model, ds)))
    b = load_bundle(path)

    b.assert_compatible(ds)  # baseline: the honest dataset is accepted

    renamed = RatioDataset(**{**ds.__dict__, "x_names": ("pivot_u", "pivot_h")})
    with pytest.raises(ValueError, match="feature layout mismatch"):
        b.assert_compatible(renamed)

    reordered = RatioDataset(**{**ds.__dict__, "theta_names": TH_NAMES[::-1]})
    with pytest.raises(ValueError, match="feature layout mismatch"):
        b.assert_compatible(reordered)


def test_bundle_refuses_unnamed_features():
    ds = make_grouped_dataset(n_groups=60)
    model = build_ratio_model(ds, key=jax.random.PRNGKey(0), width=8, depth=2)
    b = RatioBundle(model=model)
    anon = RatioDataset(**{**ds.__dict__, "x_names": (), "theta_names": ()})
    with pytest.raises(ValueError, match="carries no feature names"):
        b.assert_compatible(anon)


def test_bundle_detects_tampering(tmp_path):
    ds = make_grouped_dataset(n_groups=60)
    model = build_ratio_model(ds, key=jax.random.PRNGKey(0), width=8, depth=2)
    path = str(tmp_path / "b")
    save_bundle(path, RatioBundle(model=model))
    with open(os.path.join(path, "bundle.json")) as f:
        meta = json.load(f)
    meta["architecture"]["width"] = 999
    with open(os.path.join(path, "bundle.json"), "w") as f:
        json.dump(meta, f)
    with pytest.raises(ValueError, match="sha256"):
        load_bundle(path)


def test_bundle_without_support_refuses_to_pretend_it_has_one():
    ds = make_grouped_dataset(n_groups=60)
    model = build_ratio_model(ds, key=jax.random.PRNGKey(0), width=8, depth=2)
    b = RatioBundle(model=model, support=None)
    with pytest.raises(ValueError, match="records no feature support"):
        b.require_support()


def test_validation_is_chunked_and_chunking_does_not_change_the_metrics():
    """Peak device residency during validation must be O(val_chunk_rows), not O(val).

    A single jitted forward over the whole validation subsample allocates a
    [rows, width] activation per layer PER CLASS, which OOMs under a fractional-VRAM
    ceiling long before the dataset does. Chunking must be numerically transparent:
    the loss is a chunk-size-weighted mean and the negatives are drawn once for the
    whole subsample, so a smaller chunk must give the same numbers.
    """
    ds = make_grouped_dataset(n_groups=200, rows_per_group=10, seed=21)
    common = dict(key=jax.random.PRNGKey(0), steps=30, batch_size=128, val_every=15,
                  residency="host", verbose=False)
    one_shot = fit_ratio(build_ratio_model(ds, key=jax.random.PRNGKey(0), width=8, depth=2),
                         ds, val_chunk_rows=10 ** 9, **common)
    chunked = fit_ratio(build_ratio_model(ds, key=jax.random.PRNGKey(0), width=8, depth=2),
                        ds, val_chunk_rows=97, **common)   # deliberately ragged
    assert len(one_shot.history) == len(chunked.history)
    for a, b in zip(one_shot.history, chunked.history):
        assert a["val_loss"] == pytest.approx(b["val_loss"], abs=1e-6), (a, b)
        assert a["val_auc"] == pytest.approx(b["val_auc"], abs=1e-6)
        assert a["val_self_norm"] == pytest.approx(b["val_self_norm"], rel=1e-5)


def test_auc_cap_limits_the_logits_pulled_to_host():
    """The host rank-sort, not the device forward, is validation's cost."""
    ds = make_grouped_dataset(n_groups=150, rows_per_group=10, seed=22)
    res = fit_ratio(build_ratio_model(ds, key=jax.random.PRNGKey(0), width=8, depth=2),
                    ds, key=jax.random.PRNGKey(0), steps=20, batch_size=128,
                    val_every=10, auc_max_rows=64, val_chunk_rows=32,
                    residency="host", verbose=False)
    assert all(np.isfinite(h["val_auc"]) for h in res.history)


def test_compute_support_is_chunked_and_chunking_changes_nothing():
    """Extrema must stay exact across chunk boundaries and the subsample must be the same
    rows regardless of chunk size -- otherwise the recorded support depends on a
    performance knob, which is the worst kind of silent dependence."""
    ds = make_grouped_dataset(n_groups=200, rows_per_group=11, seed=31)
    model = build_ratio_model(ds, key=jax.random.PRNGKey(0), width=8, depth=2)
    whole = compute_support(model, ds, chunk_rows=10 ** 9)
    ragged = compute_support(model, ds, chunk_rows=37)
    np.testing.assert_allclose(whole.lo, ragged.lo, rtol=1e-12)
    np.testing.assert_allclose(whole.hi, ragged.hi, rtol=1e-12)
    np.testing.assert_allclose(whole.mean, ragged.mean, rtol=1e-12)
    np.testing.assert_allclose(whole.cov_inv, ragged.cov_inv, rtol=1e-10)
    np.testing.assert_allclose(whole.dm_quantiles, ragged.dm_quantiles, rtol=1e-10)


def test_compute_support_extrema_are_exact_even_when_quantiles_subsample():
    """min/max come from every row; only the quantiles and covariance subsample."""
    ds = make_grouped_dataset(n_groups=120, rows_per_group=10, seed=32)
    model = build_ratio_model(ds, key=jax.random.PRNGKey(0), width=8, depth=2)
    sup = compute_support(model, ds, max_rows=50, chunk_rows=64)
    assert sup.n_rows_estimate == 50            # the subsample really is small
    assert sup.n_rows == ds.split_view("train").n_rows

    tr = ds.split_view("train")
    feat = np.asarray(jax.vmap(model.standardized)(jnp.asarray(tr.x), jnp.asarray(tr.theta)))
    np.testing.assert_allclose(sup.lo, feat.min(axis=0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(sup.hi, feat.max(axis=0), rtol=1e-5, atol=1e-6)


def test_split_view_is_a_view_when_the_split_is_contiguous():
    """Split-ordered data must not be copied to be split.

    At 10^8 rows, boolean-indexing every column materializes the whole split in host RAM
    only to hand it straight to the device. A contiguous split is a slice, and a slice of a
    memmap is free.
    """
    n = 400
    x = np.arange(2 * n, dtype=np.float32).reshape(n, 2)
    split = np.array(["train"] * 300 + ["val"] * 100, dtype="<U5")
    ds = RatioDataset.unweighted(x, x.copy(), np.repeat(np.arange(40), 10), split,
                                 x_names=("a", "b"), theta_names=("c", "d"))
    tr = ds.split_view("train")
    assert tr.n_rows == 300
    # shares_memory, not `.base`: numpy collapses base chains through reshape, so `.base`
    # points at the original buffer rather than the array we sliced.
    assert np.shares_memory(tr.x, ds.x), "contiguous split should slice (a view), not copy"
    assert np.shares_memory(tr.theta, ds.theta)
    np.testing.assert_array_equal(tr.x, x[:300])
    va = ds.split_view("val")
    assert np.shares_memory(va.x, ds.x)
    np.testing.assert_array_equal(va.x, x[300:])


def test_split_view_falls_back_to_masking_when_interleaved():
    """A hash-assigned split is interleaved; correctness must not depend on layout."""
    n = 300
    x = np.arange(2 * n, dtype=np.float32).reshape(n, 2)
    split = np.array(["train" if i % 3 else "val" for i in range(n)], dtype="<U5")
    ds = RatioDataset.unweighted(x, x.copy(), np.arange(n), split,
                                 x_names=("a", "b"), theta_names=("c", "d"))
    tr = ds.split_view("train")
    np.testing.assert_array_equal(tr.x, x[split == "train"])
    assert tr.n_rows == int((split == "train").sum())
    assert not np.shares_memory(tr.x, ds.x)   # interleaved -> a copy, necessarily

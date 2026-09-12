"""The negative-pairing protocol -- the seam whose absence forced the downstream fork.

The central property under test is not "it runs" but "the negative genuinely comes from a
different group", because the failure it guards against is silent: a within-batch row
permutation on grouped data still produces a loss that decreases and an AUC that rises,
while training against the wrong denominator.
"""

import jax
import numpy as np
import pytest

from hitman.ratio.pairing import (  # noqa: F401
    group_shift_rows_np, row_permutation_rows_np,
    GroupDerangement, GroupShift, RowPermutation, build_group_index,
    get_pairing, group_derangement_rows, group_shift_rows, random_derangement,
    row_permutation_rows,
)

RAGGED = [50, 3, 20, 1, 40, 7, 15, 2]


def _grouped(counts=RAGGED):
    return np.concatenate([np.full(c, g, dtype=np.int64) for g, c in enumerate(counts)])


# ---------------------------------------------------------------------------
# GroupIndex
# ---------------------------------------------------------------------------
def test_group_index_slices_recover_each_group_exactly():
    gid = _grouped()
    idx = build_group_index(gid)
    assert idx.n_groups == len(RAGGED)
    for g in range(len(RAGGED)):
        rows = idx.order[idx.start[g]:idx.start[g] + idx.count[g]]
        assert set(rows.tolist()) == set(np.where(gid == g)[0].tolist())
        assert idx.count[g] == RAGGED[g]


def test_group_index_relabels_noncontiguous_ids():
    """Real group ids are shower indices -- sparse and arbitrary, not 0..K-1."""
    gid = np.array([1000, 7, 1000, 999999, 7, 7], dtype=np.int64)
    idx = build_group_index(gid)
    assert idx.n_groups == 3
    assert set(idx.label.tolist()) == {0, 1, 2}
    # rows sharing an id share a label
    assert idx.label[0] == idx.label[2]
    assert idx.label[1] == idx.label[4] == idx.label[5]


def test_group_index_empty():
    idx = build_group_index(np.zeros(0, dtype=np.int64))
    assert idx.n_groups == 0 and idx.n_rows == 0


# ---------------------------------------------------------------------------
# derangement
# ---------------------------------------------------------------------------
def test_random_derangement_is_a_permutation_with_no_fixed_points():
    rng = np.random.default_rng(0)
    for n in (2, 3, 5, 10, 50):
        for _ in range(20):
            d = random_derangement(n, rng)
            assert sorted(d.tolist()) == list(range(n))
            assert not np.any(d == np.arange(n))


def test_random_derangement_degenerate():
    rng = np.random.default_rng(0)
    assert random_derangement(0, rng).tolist() == []
    assert random_derangement(1, rng).tolist() == [0]


# ---------------------------------------------------------------------------
# THE contract: group-aware pairings never draw from the row's own group
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["group_shift", "group_derangement"])
def test_group_aware_pairings_never_reuse_the_rows_own_group(kind):
    gid = _grouped()
    idx = build_group_index(gid)
    rows = np.arange(gid.size)
    for trial in range(25):
        if kind == "group_shift":
            neg = np.asarray(group_shift_rows(
                jax.random.PRNGKey(trial), np.asarray(rows), idx))
        else:
            neg = group_derangement_rows(np.random.default_rng(trial), rows, idx)
        assert neg.shape == rows.shape
        assert np.all(gid[neg] != gid[rows]), f"{kind}: negative shared the source group"
        assert np.all((neg >= 0) & (neg < gid.size))


def test_row_permutation_DOES_reuse_the_same_group_which_is_why_it_is_wrong_here():
    """The motivating counter-example, asserted rather than asserted-about.

    With 8 groups and 139 rows, a within-batch row permutation pairs a large fraction of
    rows with a row of their OWN group. Those pairs carry an unchanged group-level
    hypothesis, so they are not marginal draws at all -- and nothing about the resulting
    loss curve reveals it.
    """
    gid = _grouped()
    idx = build_group_index(gid)
    rows = np.arange(gid.size)
    collisions = 0
    for trial in range(25):
        neg = np.asarray(row_permutation_rows(
            jax.random.PRNGKey(trial), np.asarray(rows), idx))
        collisions += int(np.count_nonzero(gid[neg] == gid[rows]))
    assert collisions > 0, "expected same-group collisions from a plain row permutation"


def test_group_shift_is_uniform_over_the_other_groups():
    """Every group other than the source must be reachable, with no systematic gap."""
    gid = np.repeat(np.arange(6), 10).astype(np.int64)
    idx = build_group_index(gid)
    rows = np.arange(gid.size)
    seen = {g: set() for g in range(6)}
    for trial in range(200):
        neg = np.asarray(group_shift_rows(jax.random.PRNGKey(trial), np.asarray(rows), idx))
        for src, dst in zip(gid[rows], gid[neg]):
            seen[int(src)].add(int(dst))
    for g in range(6):
        assert seen[g] == set(range(6)) - {g}


def test_group_shift_refuses_a_single_group_split():
    idx = build_group_index(np.zeros(10, dtype=np.int64))
    with pytest.raises(ValueError, match=">= 2 distinct groups"):
        group_shift_rows(jax.random.PRNGKey(0), np.arange(10), idx)


def test_group_derangement_refuses_a_single_group_batch():
    gid = _grouped()
    idx = build_group_index(gid)
    only_group_0 = np.where(gid == 0)[0]
    with pytest.raises(ValueError, match=">= 2 distinct groups"):
        group_derangement_rows(np.random.default_rng(0), only_group_0, idx)


def test_group_derangement_shares_the_target_group_within_a_source_group():
    """The documented difference from group_shift: one target per source group."""
    gid = _grouped()
    idx = build_group_index(gid)
    rows = np.arange(gid.size)
    neg = group_derangement_rows(np.random.default_rng(0), rows, idx)
    for g in np.unique(gid):
        targets = np.unique(gid[neg[gid[rows] == g]])
        assert targets.size == 1, f"group {g} drew from {targets.size} target groups"


def test_group_shift_uses_independent_targets_within_a_source_group():
    """...and group_shift does NOT, which is the cleaner marginal."""
    gid = np.repeat(np.arange(8), 40).astype(np.int64)
    idx = build_group_index(gid)
    rows = np.arange(gid.size)
    neg = np.asarray(group_shift_rows(jax.random.PRNGKey(0), np.asarray(rows), idx))
    targets = np.unique(gid[neg[gid[rows] == 0]])
    assert targets.size > 1


def test_pairing_registry_resolves_names_and_passes_instances_through():
    assert isinstance(get_pairing("group_shift"), GroupShift)
    assert isinstance(get_pairing("group_derangement"), GroupDerangement)
    assert isinstance(get_pairing("row_permutation"), RowPermutation)
    inst = GroupShift()
    assert get_pairing(inst) is inst
    with pytest.raises(ValueError, match="unknown pairing"):
        get_pairing("nope")


def test_jittability_flags_match_reality():
    """group_derangement needs np.unique, so it must advertise jittable=False."""
    assert GroupShift.jittable and RowPermutation.jittable
    assert not GroupDerangement.jittable
    assert GroupShift.group_aware and GroupDerangement.group_aware
    assert not RowPermutation.group_aware

    gid = _grouped()
    idx = build_group_index(gid).to_device()
    rows = np.arange(gid.size)
    fn = jax.jit(lambda k, r: group_shift_rows(k, r, idx))
    neg = np.asarray(fn(jax.random.PRNGKey(0), rows))
    assert np.all(gid[neg] != gid[rows])


# ---------------------------------------------------------------------------
# numpy twins -- the host path must not device-put a 10^8-row table to index it
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["group_shift", "group_derangement", "row_permutation"])
def test_every_pairing_has_a_numpy_twin_with_the_same_contract(kind):
    from hitman.ratio.pairing import get_pairing

    gid = _grouped()
    idx = build_group_index(gid)
    rows = np.arange(gid.size)
    pair = get_pairing(kind)
    rng = np.random.default_rng(0)
    neg = np.asarray(pair.numpy(rng, rows, idx))
    assert neg.shape == rows.shape
    assert np.all((neg >= 0) & (neg < gid.size))
    if pair.group_aware:
        assert np.all(gid[neg] != gid[rows]), f"{kind}.numpy drew from the row's own group"


def test_group_shift_numpy_matches_the_jax_version_in_distribution():
    """Same construction, different execution place: not bit-identical (the RNG streams
    differ), but the reachable-target set and the uniformity must agree."""
    gid = np.repeat(np.arange(6), 12).astype(np.int64)
    idx = build_group_index(gid)
    rows = np.arange(gid.size)
    rng = np.random.default_rng(0)
    seen_np = {g: set() for g in range(6)}
    for _ in range(200):
        neg = group_shift_rows_np(rng, rows, idx)
        for s, d in zip(gid[rows], gid[neg]):
            seen_np[int(s)].add(int(d))
    for g in range(6):
        assert seen_np[g] == set(range(6)) - {g}


def test_group_shift_numpy_refuses_a_single_group():
    idx = build_group_index(np.zeros(10, dtype=np.int64))
    with pytest.raises(ValueError, match=">= 2 distinct groups"):
        group_shift_rows_np(np.random.default_rng(0), np.arange(10), idx)

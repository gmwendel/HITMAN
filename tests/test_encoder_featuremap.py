"""Proposal 5: injectable per-hit feature-map callable in the DeepSets encoder.

The WC default must be unchanged (same params + map => identical context); a non-WC mark
(e.g. continuous 3-column) plugs in via ``feature_map``/``n_hit_in`` and round-trips through
serialization (the map is a static field, like the MLP activation strings).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from hitman.npe.encoder import DeepSetsEncoder, N_HIT_IN, _hit_feature_map


def _mask_charge(P):
    mask = jnp.concatenate([jnp.ones(P // 2), jnp.zeros(P - P // 2)])
    charge = jnp.asarray([40.0, float(P // 2)])
    return mask, charge


def test_default_matches_explicit_wc_map():
    key = jax.random.PRNGKey(0)
    a = DeepSetsEncoder(key=key, context_dim=16, phi_dim=8, phi_hidden=8, rho_hidden=8)
    b = DeepSetsEncoder(key=key, context_dim=16, phi_dim=8, phi_hidden=8, rho_hidden=8,
                        feature_map=_hit_feature_map, n_hit_in=N_HIT_IN)
    assert a.n_hit_in == 5
    hits = jax.random.normal(jax.random.PRNGKey(1), (10, 4)) * 100.0
    mask, charge = _mask_charge(10)
    out_a = np.asarray(a(hits, mask, charge))
    out_b = np.asarray(b(hits, mask, charge))
    assert np.array_equal(out_a, out_b)
    assert out_a.shape == (16,)


def test_custom_feature_map_plugs_in():
    # a 3-column continuous mark (e.g. dt, alpha_r, alpha_t): different width, no WC scales
    def muon_map(hits):
        return hits[:, :3]                       # already normalized upstream

    enc = DeepSetsEncoder(key=jax.random.PRNGKey(2), context_dim=12, phi_dim=8,
                          phi_hidden=8, rho_hidden=8, feature_map=muon_map, n_hit_in=3)
    assert enc.n_hit_in == 3
    hits = jax.random.normal(jax.random.PRNGKey(3), (7, 3))
    mask, charge = _mask_charge(7)
    out = np.asarray(enc(hits, mask, charge))
    assert out.shape == (12,) and np.all(np.isfinite(out))


def test_feature_map_survives_serialization(tmp_path):
    def muon_map(hits):
        return hits[:, :3]

    enc = DeepSetsEncoder(key=jax.random.PRNGKey(4), context_dim=12, phi_dim=8,
                          phi_hidden=8, rho_hidden=8, feature_map=muon_map, n_hit_in=3)
    p = str(tmp_path / "enc.eqx")
    eqx.tree_serialise_leaves(p, enc)
    # template supplies the (static) map + width, leaves are restored
    skel = DeepSetsEncoder(key=jax.random.PRNGKey(9), context_dim=12, phi_dim=8,
                           phi_hidden=8, rho_hidden=8, feature_map=muon_map, n_hit_in=3)
    loaded = eqx.tree_deserialise_leaves(p, skel)
    hits = jax.random.normal(jax.random.PRNGKey(5), (7, 3))
    mask, charge = _mask_charge(7)
    assert np.array_equal(np.asarray(enc(hits, mask, charge)),
                          np.asarray(loaded(hits, mask, charge)))

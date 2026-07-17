"""Core batch container: flat hit arrays indexed back to events.

Hypotheses are stored once per event; hits reference their event through ``event_id``
(an int32 index array). Inside jit, ``hyp[event_id]`` is a fused XLA gather — the
per-hit hypothesis copies of HITMAN 1.x never materialize.
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class EventBatch(NamedTuple):
    """A batch of events in flat (structure-of-arrays) layout.

    Attributes
    ----------
    hyp : (n_events, 7) float32
        Event hypothesis/truth: x, y, z [mm], zenith, azimuth [rad], t [ns], E [MeV].
    charge : (n_events, 2) float32
        Chargenet observation: total charge, number of hits.
    hits : (n_hits, 4) float32
        Per-hit observation: sensor x, y, z [mm], hit time [ns].
        (The 1.x constant charge column, unused by the trafo, is dropped.)
    event_id : (n_hits,) int32
        Index of the owning event for each hit.
    """

    hyp: jnp.ndarray
    charge: jnp.ndarray
    hits: jnp.ndarray
    event_id: jnp.ndarray

    @property
    def n_events(self) -> int:
        return self.hyp.shape[0]

    @property
    def n_hits(self) -> int:
        return self.hits.shape[0]

    def to_device(self) -> "EventBatch":
        """Move all arrays onto the default JAX device (float32/int32)."""
        return EventBatch(
            hyp=jnp.asarray(self.hyp, jnp.float32),
            charge=jnp.asarray(self.charge, jnp.float32),
            hits=jnp.asarray(self.hits, jnp.float32),
            event_id=jnp.asarray(self.event_id, jnp.int32),
        )

    def select(self, event_indices: np.ndarray) -> "EventBatch":
        """Subset (numpy-side) to the given event indices, re-densifying event_id."""
        event_indices = np.asarray(event_indices)
        hyp = np.asarray(self.hyp)[event_indices]
        charge = np.asarray(self.charge)[event_indices]
        event_id = np.asarray(self.event_id)
        hits = np.asarray(self.hits)
        keep = np.isin(event_id, event_indices)
        remap = np.full(int(event_id.max()) + 1, -1, dtype=np.int32)
        remap[event_indices] = np.arange(len(event_indices), dtype=np.int32)
        return EventBatch(hyp=hyp, charge=charge, hits=hits[keep], event_id=remap[event_id[keep]])

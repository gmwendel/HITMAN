"""Core batch container: flat hit arrays indexed back to events.

Hypotheses are stored once per event; hits reference their event through ``event_id``
(an int32 index array). Inside jit, ``hyp[event_id]`` is a fused XLA gather — the
per-hit hypothesis copies of HITMAN 1.x never materialize.

Layout is detector-agnostic: the *meaning* of ``hyp``'s columns and ``hits``'s mark
columns is described by an injected :class:`hitman.spec.HypSpec` / :class:`hitman.spec.ObsSpec`
(water-Cherenkov is the default :data:`hitman.spec.WC_HYP_SPEC` / :data:`WC_OBS_SPEC`; a
continuous-mark detector such as the muon stations uses its own instance). ``EventBatch``
stores only arrays — the spec is a separate value object threaded through the APIs, never a
pytree leaf here — so a ``(n_events, K)`` hypothesis and a ``(n_hits, M)`` mark plug in with
no change to this container.
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class EventBatch(NamedTuple):
    """A batch of events in flat (structure-of-arrays) layout.

    Attributes
    ----------
    hyp : (n_events, K) float32
        Event hypothesis/truth. Column meaning is given by a :class:`hitman.spec.HypSpec`.
        Water-Cherenkov default (``WC_HYP_SPEC``, K=7): x, y, z [mm], zenith, azimuth [rad],
        t [ns], E [MeV].
    charge : (n_events, C) float32
        Per-event count/aggregate observation (``ObsSpec.count_names``). WC: total charge,
        number of hits.
    hits : (n_hits, M) float32
        Per-object continuous mark (``ObsSpec.mark_names``). WC (M=4): sensor x, y, z [mm],
        hit time [ns]. (The 1.x constant charge column, unused by the trafo, is dropped.)
    event_id : (n_hits,) int32
        Index of the owning event for each hit.
    pmt_id : (n_hits,) int32 or None
        Discrete per-object sensor index — present iff ``ObsSpec.has_sensor_index`` (the
        per-sensor EML formulation: in-sensor shuffling, per-sensor counts, geometry lookup).
        ``None`` for continuous-mark detectors with no sensor grid.
    """

    hyp: jnp.ndarray
    charge: jnp.ndarray
    hits: jnp.ndarray
    event_id: jnp.ndarray
    pmt_id: jnp.ndarray = None

    @property
    def n_events(self) -> int:
        return self.hyp.shape[0]

    @property
    def n_hyp(self) -> int:
        """Hypothesis dimension K (== ``HypSpec.dim`` of the governing spec)."""
        return self.hyp.shape[1]

    @property
    def n_marks(self) -> int:
        """Per-object continuous mark dimension M (== ``ObsSpec.n_marks``)."""
        return self.hits.shape[1]

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
            pmt_id=None if self.pmt_id is None else jnp.asarray(self.pmt_id, jnp.int32),
        )

    def select(self, event_indices: np.ndarray) -> "EventBatch":
        """Subset (numpy-side) to the given event indices, re-densifying event_id."""
        event_indices = np.asarray(event_indices)
        hyp = np.asarray(self.hyp)[event_indices]
        charge = np.asarray(self.charge)[event_indices]
        event_id = np.asarray(self.event_id)
        keep = np.isin(event_id, event_indices)
        remap = np.full(int(event_id.max()) + 1, -1, dtype=np.int32)
        remap[event_indices] = np.arange(len(event_indices), dtype=np.int32)
        return EventBatch(
            hyp=hyp,
            charge=charge,
            hits=np.asarray(self.hits)[keep],
            event_id=remap[event_id[keep]],
            pmt_id=None if self.pmt_id is None else np.asarray(self.pmt_id)[keep],
        )

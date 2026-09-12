"""Detector-agnostic observation/hypothesis specification protocol.

HITMAN's numeric core (closed-form densities, identity/receipt machinery, flows) is
already coordinate-agnostic; what leaks is the water-Cherenkov *layout* — the 7-parameter
hypothesis ``(x, y, z, zen, az, t, E)`` and the per-hit ``(sensor xyz, time)`` mark with a
discrete sensor index. This module lifts that layout into two small, injectable value
objects so a new detector can be plugged in without editing the core modules:

* :class:`HypSpec` — the named hypothesis dimensions, each with metadata (circular?
  positive? polar-cosine? units, prior box). Drives the flow domain, the receipt
  truth-length check, and the moment/identity parameter names.
* :class:`ObsSpec` — the named per-object (per-hit / per-muon) *mark* columns, the count
  columns, and whether a discrete sensor index is carried. Describes the payload of
  :class:`hitman.data.structures.EventBatch`.

The existing water-Cherenkov layout is one instantiation, :data:`WC_HYP_SPEC` /
:data:`WC_OBS_SPEC`, kept bit-for-bit consistent with :mod:`hitman.wc.nn.features` and
:mod:`hitman.npe.flow` (a test locks them together). :data:`MUON_HYP_SPEC` /
:data:`MUON_OBS_SPEC` are a second, deliberately different instantiation (continuous
angular marks, no sensor grid, a profile-summary hypothesis) that proves the protocol is
not secretly WC-shaped.

Design note (minimality). The spec carries exactly what the current consumers read — a
dim/name/box/circular/cos/positive record plus a ``direction()`` extractor and a mark/count
layout — and nothing speculative. It is pure metadata (a NamedTuple of Python scalars,
strings and tuples): specs are configuration, not traced data, so they are never threaded
through ``jax.jit``/``vmap`` as arrays. Numeric arrays continue to live in ``EventBatch``.
"""

import math
from typing import NamedTuple, Optional, Tuple

import jax.numpy as jnp

TWO_PI = 2.0 * math.pi


class DimSpec(NamedTuple):
    """One named hypothesis dimension with its prior/geometry metadata.

    Attributes
    ----------
    name : str
        Human-readable coordinate name (e.g. ``"zen"``, ``"E"``).
    lo, hi : float or None
        Prior box for a ``box`` coordinate (affine+logit domain in the flow). ``None`` for
        ``circular``/``cos`` coordinates, which are handled by ``period`` / the cosine map.
    circular : bool
        Periodic coordinate (azimuth, signed shower-plane angle). Uses ``period`` and a
        circular RQ spline; never boxed.
    positive : bool
        Physically constrained ``> 0`` (energy, width). Informational for downstream priors
        and reporting; the flow box still bounds it.
    cos : bool
        Polar angle in ``[0, pi]`` handled in ``cos()`` space by the flow domain (the
        ``sin`` Jacobian is folded in). Mutually exclusive with ``circular``/``box``.
    period : float
        Period for a ``circular`` coordinate (default ``2*pi``).
    units : str
        Units string, for reporting only.
    """

    name: str
    lo: Optional[float] = None
    hi: Optional[float] = None
    circular: bool = False
    positive: bool = False
    cos: bool = False
    period: float = TWO_PI
    units: str = ""

    @property
    def kind(self) -> str:
        """Flow-domain kind: ``'circ'``, ``'cos'`` or ``'box'``.

        Raises ``ValueError`` on a contradictory dim (``circular`` + ``cos``, a box on a
        circular/cos dim, or a box dim missing either bound) — every consumer reads the
        domain through here, so a misconfigured spec fails loudly at first use instead of
        silently resolving by flag precedence.
        """
        if self.circular and self.cos:
            raise ValueError(
                f"dim {self.name!r}: circular and cos are mutually exclusive")
        if (self.circular or self.cos) and not (self.lo is None and self.hi is None):
            raise ValueError(
                f"dim {self.name!r}: circular/cos dims must not set a (lo, hi) box")
        if self.circular:
            return "circ"
        if self.cos:
            return "cos"
        if self.lo is None or self.hi is None:
            raise ValueError(
                f"dim {self.name!r}: a box dim needs both lo and hi (got "
                f"lo={self.lo!r}, hi={self.hi!r})")
        return "box"


class HypSpec(NamedTuple):
    """A named, arbitrary-dimension hypothesis specification.

    ``dims`` is the ordered tuple of :class:`DimSpec`; the integer position of each dim is
    its index into the hypothesis vector ``theta``. ``zenith``/``azimuth`` name the two
    polar/azimuthal indices used by :meth:`direction` (leave ``None`` for detectors with no
    3-D direction, e.g. profile-summary hypotheses whose direction is carried differently).
    """

    dims: Tuple[DimSpec, ...]
    zenith: Optional[int] = None
    azimuth: Optional[int] = None

    # -- shape / naming --------------------------------------------------------
    @property
    def dim(self) -> int:
        return len(self.dims)

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(d.name for d in self.dims)

    def index(self, name: str) -> int:
        """Index of the named dimension (raises ``KeyError`` if absent)."""
        for i, d in enumerate(self.dims):
            if d.name == name:
                return i
        raise KeyError(f"no hypothesis dim named {name!r}; have {self.names}")

    # -- flow-domain views (consumed by hitman.npe.flow) -----------------------
    @property
    def kinds(self) -> Tuple[str, ...]:
        return tuple(d.kind for d in self.dims)

    @property
    def box(self) -> dict:
        """``{index: (lo, hi)}`` for the boxed dims only (matches ``flow.DEFAULT_BOX``)."""
        return {i: (d.lo, d.hi) for i, d in enumerate(self.dims) if d.kind == "box"}

    @property
    def cos_dims(self) -> Tuple[int, ...]:
        return tuple(i for i, d in enumerate(self.dims) if d.cos)

    @property
    def circular_dims(self) -> Tuple[int, ...]:
        return tuple(i for i, d in enumerate(self.dims) if d.circular)

    @property
    def positive_dims(self) -> Tuple[int, ...]:
        return tuple(i for i, d in enumerate(self.dims) if d.positive)

    @property
    def period(self) -> float:
        """Common circular period; defaults to ``2*pi`` when there are no circular dims."""
        periods = {d.period for d in self.dims if d.circular}
        if not periods:
            return TWO_PI
        if len(periods) > 1:
            raise ValueError(f"mixed circular periods {periods} unsupported by the flow")
        return periods.pop()

    # -- geometry --------------------------------------------------------------
    def direction(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Unit direction from the ``(zenith, azimuth)`` dims of ``theta``.

        Bit-for-bit ``hitman.wc.nn.features.direction`` when ``zenith``/``azimuth`` point at
        the WC ``(zen, az)`` columns. Raises if the spec declares no direction.
        """
        if self.zenith is None or self.azimuth is None:
            raise ValueError("this HypSpec declares no (zenith, azimuth) direction")
        zen, az = theta[self.zenith], theta[self.azimuth]
        s = jnp.sin(zen)
        return jnp.stack([s * jnp.cos(az), s * jnp.sin(az), jnp.cos(zen)])


class ObsSpec(NamedTuple):
    """Per-object (per-hit / per-muon) mark and count layout for an ``EventBatch``.

    Attributes
    ----------
    mark_names : tuple of str
        Column names of ``EventBatch.hits`` — the continuous per-object mark. For WC this
        is ``(sensor_x, sensor_y, sensor_z, time)``; for muon stations ``(dt_plane,
        alpha_r, alpha_t)``.
    count_names : tuple of str
        Column names of ``EventBatch.charge`` — the per-event count/aggregate observation.
        Required (no default): the count layout is as detector-specific as the marks.
    has_sensor_index : bool
        Whether a discrete per-object sensor index (``EventBatch.pmt_id``) is carried. True
        for detectors with a fixed sensor grid (WC softmax mark); False for continuous-mark
        detectors (muon angular marks).
    circular_marks : tuple of int
        Indices into ``mark_names`` that are periodic (e.g. an azimuthal mark), for the
        mark-density factors.
    """

    mark_names: Tuple[str, ...]
    count_names: Tuple[str, ...]
    has_sensor_index: bool = False
    circular_marks: Tuple[int, ...] = ()

    @property
    def n_marks(self) -> int:
        return len(self.mark_names)

    def index(self, name: str) -> int:
        for i, n in enumerate(self.mark_names):
            if n == name:
                return i
        raise KeyError(f"no mark named {name!r}; have {self.mark_names}")


# ---------------------------------------------------------------------------
# Default instantiations
# ---------------------------------------------------------------------------
# Water-Cherenkov (the existing HITMAN detector). Boxes / cos / circular are kept
# identical to hitman.npe.flow.DEFAULT_BOX / COS_DIMS / CIRCULAR_DIMS and the column order
# to hitman.wc.nn.features (X, Y, Z, ZENITH, AZIMUTH, TIME, ENERGY); test_spec locks them.
WC_HYP_SPEC = HypSpec(
    dims=(
        DimSpec("x", -950.0, 950.0, units="mm"),
        DimSpec("y", -950.0, 950.0, units="mm"),
        DimSpec("z", -980.0, 980.0, units="mm"),
        DimSpec("zen", cos=True, units="rad"),
        DimSpec("az", circular=True, units="rad"),
        DimSpec("t", -260.0, 260.0, units="ns"),
        # E: ``positive`` records the physical constraint; the (lo, hi) box is the flow's
        # padded PRIOR domain, which deliberately extends past the physical edges so the
        # boundary bins keep support (same convention as flow.DEFAULT_BOX since 1.x).
        DimSpec("E", -0.5, 10.5, positive=True, units="MeV"),
    ),
    zenith=3,
    azimuth=4,
)

WC_OBS_SPEC = ObsSpec(
    mark_names=("sensor_x", "sensor_y", "sensor_z", "time"),
    count_names=("total_charge", "n_hits"),
    has_sensor_index=True,
)

# Muon-station example (CORSIKA-8 clock-sync program; see docs/DESIGN_network.md). A
# deliberately different shape: continuous angular marks, NO sensor grid, a profile-summary
# hypothesis with a signed circular shower-plane angle and no (x, y, z, t) vertex. Provided
# so downstream code (and the protocol tests) exercise a non-WC instantiation.
MUON_HYP_SPEC = HypSpec(
    dims=(
        DimSpec("cos_zen", -1.0, 1.0, units=""),
        DimSpec("Xmu_max", 0.0, 1200.0, positive=True, units="g/cm^2"),
        DimSpec("width", 0.0, 400.0, positive=True, units="g/cm^2"),
        DimSpec("X1", 0.0, 800.0, positive=True, units="g/cm^2"),
        DimSpec("r", 0.0, 1000.0, positive=True, units="m"),
        DimSpec("psi", circular=True, units="rad"),
    ),
)

MUON_OBS_SPEC = ObsSpec(
    mark_names=("dt_plane", "alpha_r", "alpha_t"),
    count_names=("n_muons",),
    has_sensor_index=False,
)

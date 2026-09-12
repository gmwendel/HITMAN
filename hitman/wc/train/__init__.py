"""Water-Cherenkov training lanes: device-resident loops, moments/identities, NWJ, polish.

The generic ratio-estimation pieces (``nre_loss``, the host-path ``fit``) stayed in
``hitman.train``; the detector-agnostic ratio lane is ``hitman.ratio``.
"""

from hitman.wc.train.resident import (
    DeviceData, charge_batch, fit_resident, frame_hit_batch, hit_batch,
)
from hitman.wc.train.polish import exact_polish
from hitman.wc.train.recipe import RecipeResult, train_recipe
from hitman.wc.train.reweight import (
    BrightnessWeightTable, ChargeWeightTable, HitWeightTable,
    build_brightness_weights, build_charge_weights, build_hit_weights,
    make_brightness_weighted_hit_batch, make_weighted_charge_batch, make_weighted_hit_batch,
)
from hitman.wc.train.event import (
    EventFitResult, event_val_bce, fit_event_model, make_event_val_bce,
)

__all__ = [
    "DeviceData", "hit_batch", "charge_batch", "frame_hit_batch", "fit_resident",
    "exact_polish", "train_recipe", "RecipeResult",
    "HitWeightTable", "build_hit_weights", "make_weighted_hit_batch",
    "ChargeWeightTable", "build_charge_weights", "make_weighted_charge_batch",
    "BrightnessWeightTable", "build_brightness_weights", "make_brightness_weighted_hit_batch",
    "EventFitResult", "event_val_bce", "fit_event_model", "make_event_val_bce",
]

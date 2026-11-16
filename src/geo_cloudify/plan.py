"""Plain, JSON-serializable plan data passed between steps.

The plan is the contract between steps. Explore builds it; convert, verify and
upload only read it and make no decisions. Everything here must survive
`model_dump_json` then `model_validate_json` unchanged, so the plan can cross a
process boundary later (a queue worker) without loss.
"""

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _PlanModel(BaseModel):
    """Base for plan models: serialize NaN and infinity as strings, not null."""

    model_config = ConfigDict(ser_json_inf_nan="strings")


class Kind(str, Enum):
    """What kind of geospatial data a file is."""

    RASTER = "raster"
    OTHER = "other"  # not a recognized geospatial file; used for skipped files


class Action(str, Enum):
    """What the pipeline will do with a plan item."""

    CONVERT = "convert"
    PASS_THROUGH = "pass_through"
    SKIP = "skip"


class RasterFingerprint(_PlanModel):
    """Metadata about a raster, recorded in Explore for the report and Verify."""

    width: int
    height: int
    count: int                                 # number of bands
    dtype: str
    transform: list[float]                     # 6 affine coefficients, GDAL order
    crs: str | None = None                     # WKT, or None
    nodata: float | None = None
    color_table: list[list[int]] | None = None  # rows of [index, r, g, b, a]
    scales: list[float] = Field(default_factory=list)       # per band
    offsets: list[float] = Field(default_factory=list)      # per band
    units: list[str | None] = Field(default_factory=list)   # per band


class PlanItem(_PlanModel):
    """One entry in the plan: what a file is and what to do with it."""

    source: str
    kind: Kind
    action: Action
    files: list[str]                           # main file plus its companions
    settings: dict[str, Any] = Field(default_factory=dict)
    fingerprint: RasterFingerprint | None = None
    lossy: bool = False
    notes: list[str] = Field(default_factory=list)
    skip_reason: str | None = None


class JobPlan(_PlanModel):
    """The full plan for one job."""

    items: list[PlanItem] = Field(default_factory=list)


def to_json_safe_dict(plan: JobPlan) -> dict[str, Any]:
    """Return the plan as a JSON-safe dict, with NaN and infinity as strings.

    FastAPI's JSON encoder rejects raw float NaN, so routes use this instead of
    `model_dump(mode="json")`. Returns a plain dict.
    """
    return json.loads(plan.model_dump_json())

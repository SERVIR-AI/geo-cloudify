"""Unit tests for the plan data types."""

import json

from geoconvert.plan import Action, JobPlan, Kind, PlanItem, RasterFingerprint, to_json_safe_dict


def test_plan_round_trips_through_json():
    """A plan, including a NaN nodata, survives dump_json then validate_json unchanged."""
    item = PlanItem(
        source="/in/data.tif",
        kind=Kind.RASTER,
        action=Action.CONVERT,
        files=["/in/data.tif", "/in/data.tfw"],
        settings={"COMPRESS": "DEFLATE", "BLOCKSIZE": 512},
        fingerprint=RasterFingerprint(
            width=100,
            height=200,
            count=1,
            dtype="float32",
            transform=[0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
            crs="EPSG:4326",
            nodata=float("nan"),
            color_table=None,
            scales=[1.0],
            offsets=[0.0],
            units=[None],
        ),
        lossy=False,
        notes=["no declared nodata; left unset"],
    )
    plan = JobPlan(items=[item])

    json_text = plan.model_dump_json()
    assert '"NaN"' in json_text  # NaN must survive as "NaN", not null
    assert JobPlan.model_validate_json(json_text).model_dump_json() == json_text


def test_enums_serialize_to_their_string_values():
    """Kind and Action serialize to their string values in JSON mode."""
    item = PlanItem(source="x", kind=Kind.RASTER, action=Action.SKIP, files=["x"], skip_reason="not a raster")
    dumped = JobPlan(items=[item]).model_dump(mode="json")
    assert dumped["items"][0]["kind"] == "raster"
    assert dumped["items"][0]["action"] == "skip"


def test_to_json_safe_dict_serializes_nan_as_string():
    """The API-safe dict turns NaN into the string "NaN", not a float NaN."""
    item = PlanItem(
        source="x",
        kind=Kind.RASTER,
        action=Action.CONVERT,
        files=["x"],
        fingerprint=RasterFingerprint(
            width=1,
            height=1,
            count=1,
            dtype="float32",
            transform=[0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
            nodata=float("nan"),
        ),
    )
    data = to_json_safe_dict(JobPlan(items=[item]))
    assert data["items"][0]["fingerprint"]["nodata"] == "NaN"
    json.dumps(data, allow_nan=False)  # mirrors FastAPI's encoder; must not raise

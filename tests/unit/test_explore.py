"""Unit tests for the top-level explore step: grouping files, skips, plan order."""

import logging
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import RasterioError, RasterioIOError
from rasterio.transform import from_origin

import geo_cloudify.explore as geo_cloudify_explore
from geo_cloudify.explore import SKIP_REASON, UNSUPPORTED_REASON, build_plan
from geo_cloudify.plan import Action, Kind

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "data"


def _write_tif(path, *, width=600, height=600, dtype="uint8", crs="EPSG:4326", nodata=None, with_ovr=False):
    """Write a georeferenced GeoTIFF and return its path."""
    transform = from_origin(0, height, 1, 1)
    with rasterio.Env(TIFF_USE_OVR=True):
        with rasterio.open(
            path, "w", driver="GTiff", width=width, height=height, count=1,
            dtype=dtype, crs=crs, transform=transform, nodata=nodata,
        ) as dst:
            dst.write(np.zeros((1, height, width), dtype=dtype))
            if with_ovr:
                dst.build_overviews([2, 4], Resampling.nearest)
    return path


def _write_world_file(tif, suffix=".tfw"):
    """Write a world file next to a GeoTIFF and return its path."""
    path = tif.with_suffix(suffix)
    path.write_text("1.0\n0.0\n0.0\n-1.0\n0.5\n599.5\n")
    return path


# --- main files and companions ----------------------------------------------

def test_main_file_with_overview_gives_one_item(tmp_path):
    main = _write_tif(tmp_path / "a.tif", with_ovr=True)
    ovr = Path(str(main) + ".ovr")
    assert ovr.exists()

    plan = build_plan({"src": [main, ovr]}, None)

    assert len(plan.items) == 1
    item = plan.items[0]
    assert item.action is Action.CONVERT
    assert item.source == "src"
    assert item.files[0] == str(main)
    assert set(item.files) == {str(main), str(ovr)}


def test_detected_file_that_is_also_a_companion_is_not_a_main_file(tmp_path, monkeypatch):
    main = _write_tif(tmp_path / "a.tif")
    other = _write_tif(tmp_path / "b.tif")
    # Stand in for a `.tif.ovr` that GDAL accepts as a raster: a detected file that
    # is listed as a companion of another main file belongs to that file.
    monkeypatch.setattr(
        geo_cloudify_explore, "_companions", lambda dataset, path, by_resolved: [other] if path == main else []
    )

    plan = build_plan({"src": [main, other]}, None)

    assert len(plan.items) == 1
    assert plan.items[0].files[0] == str(main)
    assert set(plan.items[0].files) == {str(main), str(other)}


def test_world_file_is_attached_as_companion(tmp_path):
    main = _write_tif(tmp_path / "a.tif")
    tfw = _write_world_file(main)

    plan = build_plan({"src": [main, tfw]}, None)

    assert len(plan.items) == 1
    item = plan.items[0]
    assert item.action is Action.CONVERT
    assert set(item.files) == {str(main), str(tfw)}


@pytest.mark.parametrize("suffix", [".tfw", ".tifw", ".wld"])
def test_world_file_variants_are_attached(tmp_path, suffix):
    main = _write_tif(tmp_path / "a.tif")
    world = _write_world_file(main, suffix)

    plan = build_plan({"src": [main, world]}, None)

    assert len(plan.items) == 1
    assert set(plan.items[0].files) == {str(main), str(world)}


def test_relative_paths_match_gdal_absolute_paths(tmp_path, monkeypatch):
    _write_tif(tmp_path / "a.tif", with_ovr=True)
    monkeypatch.chdir(tmp_path)

    plan = build_plan({"src": [Path("a.tif"), Path("a.tif.ovr")]}, None)

    assert len(plan.items) == 1
    assert {Path(name).name for name in plan.items[0].files} == {"a.tif", "a.tif.ovr"}


def test_companion_from_another_source_is_not_attached(tmp_path):
    main = _write_tif(tmp_path / "a.tif", with_ovr=True)
    ovr = Path(str(main) + ".ovr")

    plan = build_plan({"one": [main], "two": [ovr]}, None)

    assert plan.items[0].source == "one"
    assert plan.items[0].files == [str(main)]
    assert plan.items[1].source == "two"
    assert plan.items[1].files == [str(ovr)]
    assert plan.items[1].action is Action.SKIP


# --- skips -------------------------------------------------------------------

def test_lone_world_file_is_skipped(tmp_path):
    tfw = _write_world_file(tmp_path / "a.tif")

    plan = build_plan({"src": [tfw]}, None)

    assert len(plan.items) == 1
    item = plan.items[0]
    assert item.kind is Kind.OTHER
    assert item.action is Action.SKIP
    assert item.files == [str(tfw)]
    assert item.skip_reason == UNSUPPORTED_REASON
    assert str(tmp_path) not in item.skip_reason


def test_ungeoreferenced_png_is_skipped(tmp_path):
    png = tmp_path / "preview.png"
    with rasterio.open(png, "w", driver="PNG", width=4, height=4, count=1, dtype="uint8") as dst:
        dst.write(np.zeros((1, 4, 4), dtype="uint8"))

    plan = build_plan({"src": [png]}, None)

    assert len(plan.items) == 1
    item = plan.items[0]
    assert item.kind is Kind.OTHER
    assert item.action is Action.SKIP
    assert item.files == [str(png)]
    assert item.skip_reason == SKIP_REASON


def test_text_file_is_skipped_with_a_plain_reason(tmp_path):
    text = tmp_path / "notes.txt"
    text.write_text("hello")

    plan = build_plan({"src": [text]}, None)

    assert len(plan.items) == 1
    assert plan.items[0].action is Action.SKIP
    assert plan.items[0].kind is Kind.OTHER
    assert plan.items[0].skip_reason == UNSUPPORTED_REASON
    assert str(tmp_path) not in plan.items[0].skip_reason


def test_corrupt_tif_is_skipped_and_others_are_still_planned(tmp_path):
    corrupt = tmp_path / "broken.tif"
    corrupt.write_bytes(b"this is not a tiff\n" * 100)
    good = _write_tif(tmp_path / "good.tif")

    plan = build_plan({"src": [corrupt, good]}, None)

    by_file = {item.files[0]: item for item in plan.items}
    assert by_file[str(good)].action is Action.CONVERT
    assert by_file[str(corrupt)].action is Action.SKIP
    # GDAL cannot tell a damaged tif from a vector file: same sentence, same reason.
    assert by_file[str(corrupt)].skip_reason == UNSUPPORTED_REASON
    assert str(tmp_path) not in by_file[str(corrupt)].skip_reason


def test_other_gdal_error_keeps_the_message_without_the_path(tmp_path, monkeypatch):
    unreadable = tmp_path / "locked.tif"
    unreadable.write_bytes(b"whatever")
    real_open = rasterio.open

    def fake_open(path, *args, **kwargs):
        """Fail the way GDAL does when the file exists but cannot be read."""
        if Path(path) == unreadable:
            raise RasterioIOError(f"'{path}' Permission denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(geo_cloudify_explore.rasterio, "open", fake_open)

    plan = build_plan({"src": [unreadable]}, None)

    assert plan.items[0].skip_reason == "could not read: Permission denied"
    assert str(tmp_path) not in plan.items[0].skip_reason


def test_read_error_after_detection_becomes_a_skip_and_a_warning(tmp_path, monkeypatch, caplog):
    bad = _write_tif(tmp_path / "bad.tif")
    good = _write_tif(tmp_path / "good.tif")
    real_explore = geo_cloudify_explore.cog_explore.explore

    def fake_explore(main, companions, source, cog_kwargs):
        """Fail while exploring `bad.tif`, otherwise defer to the real exploration."""
        if main == bad:
            raise RasterioIOError("simulated read failure")
        return real_explore(main, companions, source, cog_kwargs)

    monkeypatch.setattr(geo_cloudify_explore.cog_explore, "explore", fake_explore)
    caplog.set_level(logging.WARNING)

    plan = build_plan({"src": [bad, good]}, None)

    skipped = [item for item in plan.items if item.action is Action.SKIP]
    assert [item.files[0] for item in skipped] == [str(bad)]
    assert skipped[0].skip_reason == "could not read: simulated read failure"
    assert any(item.action is Action.CONVERT for item in plan.items)
    # A failure after detection is unusual, so it is a warning, not just a summary line.
    assert any("simulated read failure" in record.getMessage() for record in caplog.records)


def test_unexpected_error_propagates(tmp_path, monkeypatch):
    main = _write_tif(tmp_path / "a.tif")

    def boom(*args, **kwargs):
        """Raise an error that is not a rasterio error."""
        raise ValueError("unknown cog_kwargs option(s): NOPE")

    monkeypatch.setattr(geo_cloudify_explore.cog_explore, "explore", boom)

    with pytest.raises(ValueError, match="NOPE"):
        build_plan({"src": [main]}, {"src": {"NOPE": 1}})


def test_skips_are_summarised_in_one_info_line_per_source(tmp_path, caplog):
    first = tmp_path / "one.txt"
    first.write_text("hello")
    second = tmp_path / "two.txt"
    second.write_text("hello")
    caplog.set_level(logging.INFO)

    build_plan({"src": [first, second]}, None)

    summaries = [record for record in caplog.records if record.getMessage().startswith("source src:")]
    assert len(summaries) == 1
    assert summaries[0].levelno == logging.INFO
    assert "2 skipped" in summaries[0].getMessage()
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


# --- pass through and convert -----------------------------------------------

def test_already_cog_fixtures_pass_through():
    files = sorted((FIXTURES / "already_cog").glob("*.tif"))
    if not files:
        pytest.skip("fixture data not present")

    plan = build_plan({"src": files}, None)

    assert len(plan.items) == len(files)
    assert [item.files for item in plan.items] == [[str(path)] for path in files]
    assert all(item.action is Action.PASS_THROUGH for item in plan.items)


def test_netcdf_2d_converts():
    files = sorted((FIXTURES / "netcdf_2d").glob("*.nc"))
    if not files:
        pytest.skip("fixture data not present")

    plan = build_plan({"src": files}, None)

    assert len(plan.items) == len(files)
    assert all(item.action is Action.CONVERT for item in plan.items)
    assert all(item.kind is Kind.RASTER for item in plan.items)


def test_cog_kwargs_reach_every_item(tmp_path):
    main = _write_tif(tmp_path / "a.tif")

    plan = build_plan({"src": [main]}, {"src": {"nodata": -9999}})

    assert plan.items[0].settings["NODATA"] == -9999


# --- sources and order -------------------------------------------------------

def test_sources_stay_separate(tmp_path):
    first = _write_tif(tmp_path / "first.tif", with_ovr=True)
    second = _write_tif(tmp_path / "second.tif")

    plan = build_plan({"b-source": [second], "a-source": [first, Path(str(first) + ".ovr")]}, None)

    assert [item.source for item in plan.items] == ["a-source", "b-source"]
    assert plan.items[0].files[0] == str(first)
    assert set(plan.items[0].files) == {str(first), str(first) + ".ovr"}
    assert plan.items[1].files == [str(second)]


def test_same_file_under_two_sources_is_planned_twice(tmp_path):
    main = _write_tif(tmp_path / "a.tif")

    plan = build_plan({"one": [main], "two": [main]}, None)

    assert [item.source for item in plan.items] == ["one", "two"]
    assert all(item.files == [str(main)] for item in plan.items)


def test_items_are_sorted_by_source_then_main_file(tmp_path):
    zulu = _write_tif(tmp_path / "zulu.tif")
    alpha = _write_tif(tmp_path / "alpha.tif")
    notes = tmp_path / "notes.txt"
    notes.write_text("hello")

    plan = build_plan({"zz": [zulu], "aa": [zulu, alpha, notes]}, None)

    keys = [(item.source, item.files[0]) for item in plan.items]
    assert keys == sorted(keys)
    assert len(plan.items) == 4


def test_empty_request_gives_empty_plan():
    assert build_plan({}, None).items == []

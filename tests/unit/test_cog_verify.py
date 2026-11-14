"""Unit tests for checking a merged COG against its tiles (check_merged_cog)."""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import rasterio
import rasterio.shutil
from rasterio.enums import Resampling
from rasterio.io import DatasetReader
from rasterio.transform import from_origin

from geo_cloudify.cog import verify
from geo_cloudify.cog.merge import TileMismatchError, merge_to_cog
from geo_cloudify.cog.verify import MergeCheck, check_merged_cog

# A scaled-down RLCMS year: 30 m pixels, negative x origin, tiles of different sizes.
X0, Y0, PIXEL = -1260450.0, 3413550.0, 30.0
SIZES = {"nw": (530, 600), "ne": (530, 311), "sw": (257, 600), "se": (257, 311)}
FIXTURE_TILES = Path(__file__).parents[1] / "fixtures" / "data" / "tiles_to_mosaic"


def _write(path, data, *, x=X0, y=Y0, nodata=0, crs="EPSG:32648"):
    """Write `data` (bands, rows, cols) as a tiled GeoTIFF at (x, y) and return its path."""
    count, height, width = data.shape
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=count, dtype=data.dtype,
        crs=crs, transform=from_origin(x, y, PIXEL, PIXEL), nodata=nodata,
        tiled=True, blockxsize=256, blockysize=256,
    ) as dst:
        dst.write(data)
    return path


def _tile_data(dtype="uint8"):
    """Return {name: data} for four tiles with class values 0 to 22 (0 is common and is nodata)."""
    rng = np.random.default_rng(11)
    return {name: rng.integers(0, 23, size=(1, h, w)).astype(dtype) for name, (h, w) in SIZES.items()}


def _layout(folder, data, *, gap=0, dx=0.0, **kwargs):
    """Write the four tiles of `data` laid out like an RLCMS year and return their paths.

    `gap` leaves that many empty pixel columns between west and east tiles; `dx`
    shifts every tile east by that many metres; other kwargs go to `_write`.
    """
    folder.mkdir(parents=True, exist_ok=True)
    east_x = X0 + (600 + gap) * PIXEL
    south_y = Y0 - 530 * PIXEL
    origins = {"nw": (X0, Y0), "ne": (east_x, Y0), "sw": (X0, south_y), "se": (east_x, south_y)}
    return [
        _write(folder / f"{name}.tif", data[name], x=origins[name][0] + dx, y=origins[name][1], **kwargs)
        for name in SIZES
    ]


def _fields(check: MergeCheck) -> set[str]:
    """Return the fields named by the check's problems."""
    return {problem.field for problem in check.problems}


@pytest.fixture
def tiles(tmp_path):
    """Four good tiles."""
    return _layout(tmp_path / "tiles", _tile_data())


@pytest.fixture
def merged(tmp_path, tiles):
    """The four good tiles merged with mode overviews."""
    return merge_to_cog(tiles, tmp_path / "merged.tif", "mode")


# --- a good merge ---------------------------------------------------------------------


def test_good_merge_passes(tiles, merged):
    check = check_merged_cog(tiles, merged)
    assert isinstance(check, MergeCheck)
    assert check.problems == ()
    assert check.passed
    with rasterio.open(merged) as out:
        assert out.overviews(1)  # 911 x 787 is larger than one 512 block


@pytest.mark.parametrize("nodata", [0, None])
def test_good_merge_with_a_gap_passes(tmp_path, nodata):
    tiles = _layout(tmp_path / "tiles", _tile_data(), gap=7, nodata=nodata)
    output = merge_to_cog(tiles, tmp_path / "merged.tif", "nearest")
    check = check_merged_cog(tiles, output)
    assert check.passed, check.problems


def test_float_tiles_with_nan_nodata_pass(tmp_path):
    data = _tile_data("float32")
    for array in data.values():
        array[array == 0] = np.nan
    tiles = _layout(tmp_path / "tiles", data, gap=3, nodata=float("nan"))
    output = merge_to_cog(tiles, tmp_path / "merged.tif", "average")
    check = check_merged_cog(tiles, output)
    assert check.passed, check.problems


def test_small_image_needs_no_overviews(tmp_path):
    rng = np.random.default_rng(3)
    tiles = [
        _write(tmp_path / "a.tif", rng.integers(0, 23, size=(1, 40, 40)).astype("uint8")),
        _write(tmp_path / "b.tif", rng.integers(0, 23, size=(1, 40, 23)).astype("uint8"), x=X0 + 40 * PIXEL),
    ]
    output = merge_to_cog(tiles, tmp_path / "merged.tif", "mode")
    with rasterio.open(output) as out:
        assert out.overviews(1) == []
    assert check_merged_cog(tiles, output).passed


@pytest.mark.skipif(not FIXTURE_TILES.is_dir(), reason="test data not downloaded")
def test_real_tiles_merge_and_pass(tmp_path):
    tiles = sorted(FIXTURE_TILES.glob("*.tif"))
    output = merge_to_cog(tiles, tmp_path / "merged.tif", "mode")
    check = check_merged_cog(tiles, output)
    assert check.passed, check.problems


# --- valid COG and overviews ----------------------------------------------------------


def test_output_that_is_not_a_cog_fails(tmp_path, tiles, merged):
    striped = tmp_path / "striped.tif"
    rasterio.shutil.copy(str(merged), str(striped), driver="GTiff")
    check = check_merged_cog(tiles, striped)
    assert not check.passed
    assert "cog" in _fields(check)
    assert not _fields(check) & {"crs", "transform", "data type", "nodata", "pixel counts"}


def test_cog_warning_alone_fails(tiles, merged, monkeypatch):
    monkeypatch.setattr(verify, "cog_validate", lambda *args, **kwargs: (True, [], ["w"]))
    check = check_merged_cog(tiles, merged)
    assert [str(p) for p in check.problems] == ["cog: warning: w"]


def test_large_image_without_overviews_fails(tmp_path, tiles, merged):
    no_overviews = tmp_path / "no_overviews.tif"
    rasterio.shutil.copy(str(merged), str(no_overviews), driver="COG", OVERVIEWS="NONE")
    check = check_merged_cog(tiles, no_overviews)
    assert "overviews" in _fields(check)


def test_overviews_that_stop_before_one_block_fail(tmp_path, tiles, merged):
    with_one_level = tmp_path / "one_level.tif"
    rasterio.shutil.copy(str(merged), str(with_one_level), driver="GTiff", TILED="YES")
    with rasterio.open(with_one_level, "r+") as dst:
        dst.build_overviews([2], Resampling.nearest)
    short = tmp_path / "short.tif"
    # 128-pixel blocks: one overview level (456 x 394) is far from one block.
    rasterio.shutil.copy(str(with_one_level), str(short), driver="COG",
                         BLOCKSIZE=128, OVERVIEWS="FORCE_USE_EXISTING")
    with rasterio.open(short) as out:
        assert out.overviews(1) == [2]
    check = check_merged_cog(tiles, short)
    assert "overviews" in _fields(check)


# --- grid fields ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "change, field",
    [
        ({"crs": "EPSG:32647"}, "crs"),
        ({"dx": PIXEL}, "transform"),
        ({"dtype": "uint16"}, "data type"),
        ({"nodata": 255}, "nodata"),
    ],
)
def test_wrong_grid_field_fails_naming_the_field(tmp_path, tiles, change, field):
    change = dict(change)
    data = _tile_data(change.pop("dtype", "uint8"))
    other = _layout(tmp_path / "other", data, **change)
    output = merge_to_cog(other, tmp_path / "wrong.tif", "mode")
    check = check_merged_cog(tiles, output)
    assert field in _fields(check)
    assert any(str(problem).startswith(f"{field}:") for problem in check.problems)


def test_wrong_size_fails_naming_the_field(tmp_path, tiles):
    wider = _layout(tmp_path / "wider", _tile_data(), gap=7)
    output = merge_to_cog(wider, tmp_path / "wider.tif", "mode")
    check = check_merged_cog(tiles, output)
    assert "size" in _fields(check)


def test_wrong_band_count_fails_without_pixel_counts(tmp_path, tiles):
    data = {name: np.concatenate([array, array]) for name, array in _tile_data().items()}
    two_bands = _layout(tmp_path / "two_bands", data)
    output = merge_to_cog(two_bands, tmp_path / "two_bands.tif", "mode")
    check = check_merged_cog(tiles, output)
    assert "band count" in _fields(check)
    assert "pixel counts" not in _fields(check)


def test_every_problem_is_reported(tmp_path, tiles):
    other = _layout(tmp_path / "other", _tile_data(), crs="EPSG:32647", nodata=255)
    output = merge_to_cog(other, tmp_path / "wrong.tif", "mode")
    no_overviews = tmp_path / "no_overviews.tif"
    rasterio.shutil.copy(str(output), str(no_overviews), driver="COG", OVERVIEWS="NONE")
    check = check_merged_cog(tiles, no_overviews)
    assert {"crs", "nodata", "overviews"} <= _fields(check)


def test_tiles_that_do_not_match_each_other_are_refused(tmp_path, merged, tiles):
    odd = _write(tmp_path / "odd.tif", _tile_data()["se"], crs="EPSG:32647")
    with pytest.raises(TileMismatchError) as caught:
        check_merged_cog([*tiles[:3], odd], merged)
    assert caught.value.field == "crs"


# --- pixel counts ---------------------------------------------------------------------


def test_dropped_tile_is_caught(tmp_path, tiles):
    without_ne = [t for t in tiles if t.name != "ne.tif"]
    output = merge_to_cog(without_ne, tmp_path / "dropped.tif", "mode")
    check = check_merged_cog(tiles, output)
    assert _fields(check) == {"pixel counts"}


@pytest.mark.parametrize("new_value", [21, 0])  # 0 is nodata, and must still count
def test_changed_pixel_is_caught(tmp_path, tiles, new_value):
    data = _tile_data()
    old_value = 22
    data["sw"][0, 100, 200] = old_value
    original = _layout(tmp_path / "original", data)
    data["sw"][0, 100, 200] = new_value
    changed = _layout(tmp_path / "changed", data)
    output = merge_to_cog(changed, tmp_path / "changed.tif", "mode")
    check = check_merged_cog(original, output)
    assert _fields(check) == {"pixel counts"}
    message = str(check.problems[0])
    assert f"value {old_value}" in message and f"value {new_value}" in message


def test_overlapping_tiles_are_reported(tmp_path, tiles, merged):
    data = _tile_data()["nw"][:, :10, :10]
    extra = _write(tmp_path / "extra.tif", data, x=X0 + 5 * PIXEL, y=Y0 - 5 * PIXEL)
    check = check_merged_cog([*tiles, extra], merged)
    assert "overlap" in _fields(check)


def test_many_distinct_float_values_stay_bounded(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    data = {name: rng.random(size=(1, h, w)).astype("float32") for name, (h, w) in SIZES.items()}
    tiles = _layout(tmp_path / "tiles", data, nodata=None)
    output = merge_to_cog(tiles, tmp_path / "merged.tif", "average")
    monkeypatch.setattr(verify, "_MAX_DISTINCT_VALUES", 1000)
    monkeypatch.setattr(verify, "_WINDOW_PIXELS", 4096)
    sizes = []

    class RecordingCounter(Counter):
        def update(self, *args, **kwargs):
            super().update(*args, **kwargs)
            sizes.append(len(self))

    monkeypatch.setattr(verify, "Counter", RecordingCounter)
    check = check_merged_cog(tiles, output)
    assert [p.field for p in check.problems] == ["pixel counts"]
    assert "not checked" in check.problems[0].detail
    assert sizes and max(sizes) <= 1000 + 256 * 256  # the cap plus one 256 x 256 block window


def test_counts_are_made_in_windows(tmp_path, tiles, merged, monkeypatch):
    monkeypatch.setattr(verify, "_WINDOW_PIXELS", 4096)
    reads = []
    real_read = DatasetReader.read

    def recording_read(self, *args, **kwargs):
        reads.append((self.width, self.height, kwargs.get("window")))
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(DatasetReader, "read", recording_read)
    check = check_merged_cog(tiles, merged)
    assert check.passed, check.problems
    assert reads
    for width, height, window in reads:
        assert window is not None
        assert window.width * window.height < width * height

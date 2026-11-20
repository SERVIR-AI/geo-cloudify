"""Unit tests for checking a merged COG against its tiles (check_merged_cog, check_pixels)."""

import tracemalloc
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
from geo_cloudify.cog.verify import MergeCheck, check_merged_cog, check_pixels

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
    assert check.counts_checked
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
    assert check.counts_checked


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
    assert check.counts_checked
    message = str(check.problems[0])
    assert f"value {old_value}" in message and f"value {new_value}" in message


def test_overlapping_tiles_are_reported(tmp_path, tiles, merged):
    data = _tile_data()["nw"][:, :10, :10]
    extra = _write(tmp_path / "extra.tif", data, x=X0 + 5 * PIXEL, y=Y0 - 5 * PIXEL)
    check = check_merged_cog([*tiles, extra], merged)
    assert "overlap" in _fields(check)


def test_many_distinct_float_values_skip_the_count_check(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    data = {name: rng.random(size=(1, h, w)).astype("float32") for name, (h, w) in SIZES.items()}
    tiles = _layout(tmp_path / "tiles", data, nodata=None)
    output = merge_to_cog(tiles, tmp_path / "merged.tif", "average")
    # 256 x 256 windows: each holds fewer distinct values than the limit, so the tally grows.
    monkeypatch.setattr(verify, "_MAX_WINDOW_PIXELS", 4096)
    sizes = []

    class RecordingCounter(Counter):
        def update(self, *args, **kwargs):
            super().update(*args, **kwargs)
            sizes.append(len(self))

    monkeypatch.setattr(verify, "Counter", RecordingCounter)
    check = check_merged_cog(tiles, output)
    assert check.problems == ()
    assert not check.counts_checked
    assert sizes and max(sizes) <= 2 * 100_000


def test_distinct_value_limit_is_100000(tmp_path):
    exactly = np.arange(100_000, dtype="float32").reshape(1, 250, 400)
    one_more = (np.arange(251 * 400) % 100_001).astype("float32").reshape(1, 251, 400)
    at_limit = [_write(tmp_path / "at_limit.tif", exactly, nodata=None)]
    over_limit = [_write(tmp_path / "over_limit.tif", one_more, nodata=None)]
    at_output = merge_to_cog(at_limit, tmp_path / "at_limit_merged.tif", "average")
    over_output = merge_to_cog(over_limit, tmp_path / "over_limit_merged.tif", "average")
    assert check_merged_cog(at_limit, at_output).counts_checked
    assert not check_merged_cog(over_limit, over_output).counts_checked


def test_skipped_count_check_keeps_the_other_checks_and_bounded_memory(tmp_path):
    rng = np.random.default_rng(6)
    data = {name: rng.random(size=(1, h, w)).astype("float32") for name, (h, w) in SIZES.items()}
    tiles = _layout(tmp_path / "tiles", data, nodata=None)
    output = merge_to_cog(tiles, tmp_path / "merged.tif", "average")
    no_overviews = tmp_path / "no_overviews.tif"
    rasterio.shutil.copy(str(output), str(no_overviews), driver="COG", OVERVIEWS="NONE")
    # The nw tile (530 x 600) is one window with about 318,000 distinct values.
    # Held as Python ints in a Counter that is well over 20 MB.
    tracemalloc.start()
    try:
        check = check_merged_cog(tiles, no_overviews)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert "overviews" in _fields(check)
    assert "pixel counts" not in _fields(check)
    assert not check.counts_checked
    assert not check.passed
    assert peak < 16 * 1024 * 1024, peak


@pytest.mark.parametrize("window_pixels", [None, 4096])
def test_output_over_the_limit_fails_when_the_tiles_are_within_it(tmp_path, monkeypatch, window_pixels):
    rng = np.random.default_rng(12)
    classes = {name: rng.integers(0, 5, size=(1, h, w)).astype("float32") for name, (h, w) in SIZES.items()}
    noise = {name: rng.random(size=(1, h, w)).astype("float32") for name, (h, w) in SIZES.items()}
    tiles = _layout(tmp_path / "classes", classes, nodata=None)
    output = merge_to_cog(_layout(tmp_path / "noise", noise, nodata=None), tmp_path / "noise.tif", "average")
    if window_pixels:
        # Each 64 x 64 window holds fewer distinct values than the limit, so the tally grows.
        monkeypatch.setattr(verify, "_MAX_WINDOW_PIXELS", window_pixels)
    check = check_merged_cog(tiles, output)
    assert _fields(check) == {"pixel counts"}
    assert "band 1: output holds more than 100,000 distinct values, tiles hold 5" in str(check.problems[0])
    assert check.counts_checked
    assert not check.passed


@pytest.mark.parametrize("window_pixels", [None, 4096])
def test_tiles_over_the_limit_fail_when_the_output_is_within_it(tmp_path, monkeypatch, window_pixels):
    rng = np.random.default_rng(13)
    noise = {name: rng.random(size=(1, h, w)).astype("float32") for name, (h, w) in SIZES.items()}
    classes = {name: rng.integers(0, 5, size=(1, h, w)).astype("float32") for name, (h, w) in SIZES.items()}
    tiles = _layout(tmp_path / "noise", noise, nodata=None)
    output = merge_to_cog(_layout(tmp_path / "classes", classes, nodata=None), tmp_path / "classes.tif", "mode")
    if window_pixels:
        # Each 64 x 64 window holds fewer distinct values than the limit, so the tally grows.
        monkeypatch.setattr(verify, "_MAX_WINDOW_PIXELS", window_pixels)
    check = check_merged_cog(tiles, output)
    assert _fields(check) == {"pixel counts"}
    assert "band 1: tiles hold more than 100,000 distinct values, output holds 5" in str(check.problems[0])
    assert check.counts_checked
    assert not check.passed


def test_gap_fill_past_the_limit_fails_when_the_output_is_within_it(tmp_path):
    # 100,000 distinct values in the tiles, none of them 0; the gap adds 0 as one more.
    values = (np.arange(100_000, dtype="float32") + 1).reshape(1, 250, 400)
    tiles = [
        _write(tmp_path / "west.tif", values, nodata=None),
        _write(tmp_path / "east.tif", values, x=X0 + 401 * PIXEL, nodata=None),
    ]
    other = [_write(tmp_path / "other.tif", np.ones((1, 250, 801), dtype="float32"), nodata=None)]
    output = merge_to_cog(other, tmp_path / "other_merged.tif", "average")
    check = check_merged_cog(tiles, output)
    assert _fields(check) == {"pixel counts"}
    assert "band 1: tiles hold more than 100,000 distinct values, output holds 1" in str(check.problems[0])
    assert check.counts_checked


def test_gap_fill_past_the_limit_skips_the_count_check(tmp_path):
    # 100,000 distinct values in the tiles, none of them 0; the gap adds 0 as one more.
    values = (np.arange(100_000, dtype="float32") + 1).reshape(1, 250, 400)
    tiles = [
        _write(tmp_path / "west.tif", values, nodata=None),
        _write(tmp_path / "east.tif", values, x=X0 + 401 * PIXEL, nodata=None),
    ]
    output = merge_to_cog(tiles, tmp_path / "merged.tif", "average")
    check = check_merged_cog(tiles, output)
    assert check.problems == ()
    assert not check.counts_checked


def test_check_merged_cog_does_not_compare_pixels(tiles, merged, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("check_pixels was called")

    monkeypatch.setattr(verify, "check_pixels", refuse)
    check = check_merged_cog(tiles, merged)
    assert check.passed and check.counts_checked


def test_nothing_in_the_package_calls_compare_pixels():
    source = Path(verify.__file__).parents[1]
    callers = [
        path for path in source.rglob("*.py")
        if "check_pixels(" in path.read_text().replace("def check_pixels(", "")
    ]
    assert callers == []


def test_counts_are_made_in_windows(tmp_path, tiles, merged, monkeypatch):
    monkeypatch.setattr(verify, "_MAX_WINDOW_PIXELS", 4096)
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


# --- check_pixels -------------------------------------------------------------------


def _pixel_problems(problems) -> list[str]:
    """Return the problems as strings, after checking each is a "pixels" problem."""
    assert all(problem.field == "pixels" for problem in problems)
    return [str(problem) for problem in problems]


def test_compare_pixels_good_merge_has_no_problems(tiles, merged):
    assert check_pixels(tiles, merged) == ()


@pytest.mark.parametrize("window_pixels", [None, 4096])
def test_compare_pixels_finds_a_changed_pixel_and_its_place(tmp_path, monkeypatch, window_pixels):
    if window_pixels:
        # Small windows, so the changed pixel lies in a window that does not start at 0, 0.
        monkeypatch.setattr(verify, "_MAX_WINDOW_PIXELS", window_pixels)
    data = _tile_data()
    data["sw"][0, 100, 200] = 22
    data["se"][0, 120, 50] = 22
    original = _layout(tmp_path / "original", data)
    data["sw"][0, 100, 200] = 21
    data["se"][0, 120, 50] = 20
    changed = _layout(tmp_path / "changed", data)
    output = merge_to_cog(changed, tmp_path / "changed.tif", "mode")
    problems = _pixel_problems(check_pixels(original, output))
    assert len(problems) == 1
    assert "band 1: 2 pixels differ" in problems[0]
    # sw sits below nw (530 rows), so its pixel (200, 100) is at column 200, row 630.
    assert "column 200, row 630: expected 22, found 21" in problems[0]
    # se starts at column 600, past the first 512-pixel block, so its pixel (50, 120)
    # is at column 650, row 650.
    assert "column 650, row 650: expected 22, found 20" in problems[0]


def test_compare_pixels_finds_a_dropped_tile(tmp_path, tiles):
    without_ne = [t for t in tiles if t.name != "ne.tif"]
    output = merge_to_cog(without_ne, tmp_path / "dropped.tif", "mode")
    problems = _pixel_problems(check_pixels(tiles, output))
    assert len(problems) == 1 and "band 1:" in problems[0]


def test_compare_pixels_finds_a_gap_that_is_not_fill(tmp_path):
    tiles = _layout(tmp_path / "tiles", _tile_data(), gap=7)
    filler = _write(tmp_path / "filler.tif", np.full((1, 787, 7), 5, dtype="uint8"), x=X0 + 600 * PIXEL)
    output = merge_to_cog([*tiles, filler], tmp_path / "filled.tif", "mode")
    problems = _pixel_problems(check_pixels(tiles, output))
    assert len(problems) == 1
    assert f"{787 * 7} pixels differ" in problems[0]
    assert "column 600, row 0: expected 0, found 5" in problems[0]


def test_compare_pixels_finds_swapped_tiles_that_counts_miss(tmp_path):
    rng = np.random.default_rng(8)
    a = rng.integers(0, 23, size=(1, 300, 300)).astype("uint8")
    b = rng.integers(0, 23, size=(1, 300, 300)).astype("uint8")
    right = X0 + 300 * PIXEL
    tiles = [_write(tmp_path / "a.tif", a), _write(tmp_path / "b.tif", b, x=right)]
    swapped = [_write(tmp_path / "a_swapped.tif", a, x=right), _write(tmp_path / "b_swapped.tif", b)]
    output = merge_to_cog(swapped, tmp_path / "swapped.tif", "mode")
    check = check_merged_cog(tiles, output)
    assert check.passed and check.counts_checked
    assert len(_pixel_problems(check_pixels(tiles, output))) == 1


def test_compare_pixels_handles_many_distinct_float_values(tmp_path, monkeypatch):
    rng = np.random.default_rng(9)
    data = {name: rng.random(size=(1, h, w)).astype("float32") for name, (h, w) in SIZES.items()}
    tiles = _layout(tmp_path / "tiles", data, gap=3, nodata=None)
    output = merge_to_cog(tiles, tmp_path / "merged.tif", "average")
    monkeypatch.setattr(verify, "_MAX_WINDOW_PIXELS", 65536)
    # The image holds about 717,000 distinct values: over 20 MB as Python floats,
    # and about 12 MB in numpy arrays when read as one window.
    tracemalloc.start()
    try:
        problems = check_pixels(tiles, output)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert problems == ()
    assert peak < 8 * 1024 * 1024, peak
    data["se"][0, 5, 6] = 2.5
    changed = _layout(tmp_path / "changed", data, gap=3, nodata=None)
    assert len(check_pixels(changed, output)) == 1


def test_compare_pixels_treats_nan_as_equal_to_nan(tmp_path):
    data = _tile_data("float32")
    for array in data.values():
        array[array == 0] = np.nan
    tiles = _layout(tmp_path / "tiles", data, gap=3, nodata=float("nan"))
    output = merge_to_cog(tiles, tmp_path / "merged.tif", "average")
    assert check_pixels(tiles, output) == ()


def test_compare_pixels_reads_in_windows(tiles, merged, monkeypatch):
    monkeypatch.setattr(verify, "_MAX_WINDOW_PIXELS", 4096)
    reads = []
    real_read = DatasetReader.read

    def windowed_read(self, *args, **kwargs):
        if kwargs.get("window") is None:
            raise AssertionError("read without a window")
        reads.append((self.width, self.height, kwargs["window"]))
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(DatasetReader, "read", windowed_read)
    assert check_pixels(tiles, merged) == ()
    assert reads
    assert any(window.width * window.height < width * height for width, height, window in reads)
    assert all(window.width * window.height <= 512 * 512 for _, _, window in reads)


def test_compare_pixels_reports_a_different_grid_without_comparing(tmp_path, tiles):
    wider = _layout(tmp_path / "wider", _tile_data(), gap=7)
    output = merge_to_cog(wider, tmp_path / "wider.tif", "mode")
    problems = _pixel_problems(check_pixels(tiles, output))
    assert len(problems) == 1 and "not compared" in problems[0]

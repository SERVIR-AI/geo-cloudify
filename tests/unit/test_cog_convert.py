"""Unit tests for COG conversion (to_cog): valid COG, lossless pixels, metadata kept."""

import math
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rio_cogeo.cogeo import cog_validate

from geo_cloudify.cog.convert import to_cog
from geo_cloudify.cog.explore import explore, validate_cog_kwargs
from geo_cloudify.plan import Action, PlanItem

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "data"


def _geotiff(path, *, size=600, dtype="uint8", crs="EPSG:4326", nodata=None, colormap=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    data = rng.integers(0, 200, (1, size, size)).astype(dtype)
    with rasterio.open(
        path, "w", driver="GTiff", width=size, height=size, count=1, dtype=dtype,
        crs=crs, transform=from_origin(0, 10, 0.01, 0.01), nodata=nodata,
    ) as dst:
        dst.write(data)
        if colormap:
            dst.write_colormap(1, colormap)
    return path


def _item(main: Path, cog_kwargs=None) -> PlanItem:
    item = explore(main, [], "src", cog_kwargs)
    assert item.action is Action.CONVERT
    return item


def _same_nodata(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return (math.isnan(a) and math.isnan(b)) or a == b


def _assert_lossless(item: PlanItem, output: Path) -> None:
    """Check the output is a valid COG with the source's pixels and metadata."""
    is_valid, errors, _ = cog_validate(str(output))
    assert is_valid, errors
    expected_nodata = item.settings.get("NODATA", item.fingerprint.nodata)
    with rasterio.open(item.files[0]) as src, rasterio.open(output) as out:
        assert (out.width, out.height, out.count) == (src.width, src.height, src.count)
        assert out.dtypes == src.dtypes
        assert out.transform == src.transform
        assert out.crs == rasterio.crs.CRS.from_user_input(item.settings["CRS"])
        assert _same_nodata(out.nodata, expected_nodata)
        if item.fingerprint.color_table:
            assert out.colormap(1) == src.colormap(1)
        for _, window in out.block_windows(1):
            np.testing.assert_array_equal(out.read(window=window), src.read(window=window))


def test_striped_geotiff_becomes_lossless_cog(tmp_path):
    main = _geotiff(tmp_path / "job" / "src" / "r.tif")
    item = _item(main)
    output = to_cog(item, tmp_path / "job", tmp_path / "out")
    _assert_lossless(item, output)


def test_output_path_mirrors_job_folder(tmp_path):
    main = _geotiff(tmp_path / "job" / "zipname" / "r.tif")
    output = to_cog(_item(main), tmp_path / "job", tmp_path / "out")
    assert output == tmp_path / "out" / "zipname" / "r.tif"


def test_non_tif_name_keeps_its_suffix(tmp_path):
    main = _geotiff(tmp_path / "job" / "a.img")
    output = to_cog(_item(main), tmp_path / "job", tmp_path / "out")
    assert output == tmp_path / "out" / "a.img.tif"


def test_tif_and_other_with_same_stem_do_not_clash(tmp_path):
    first = to_cog(_item(_geotiff(tmp_path / "job" / "a.tif")), tmp_path / "job", tmp_path / "out")
    second = to_cog(_item(_geotiff(tmp_path / "job" / "a.img")), tmp_path / "job", tmp_path / "out")
    assert first != second


def test_existing_output_raises(tmp_path):
    item = _item(_geotiff(tmp_path / "job" / "a.tif"))
    to_cog(item, tmp_path / "job", tmp_path / "out")
    with pytest.raises(FileExistsError):
        to_cog(item, tmp_path / "job", tmp_path / "out")


def test_nan_nodata_is_kept(tmp_path):
    main = _geotiff(tmp_path / "job" / "r.tif", dtype="float32", nodata=float("nan"))
    item = _item(main)
    output = to_cog(item, tmp_path / "job", tmp_path / "out")
    _assert_lossless(item, output)
    with rasterio.open(output) as out:
        assert math.isnan(out.nodata)


def test_requested_nodata_is_set(tmp_path):
    main = _geotiff(tmp_path / "job" / "r.tif", dtype="int16")
    item = _item(main, {"nodata": -9999})
    output = to_cog(item, tmp_path / "job", tmp_path / "out")
    _assert_lossless(item, output)
    with rasterio.open(output) as out:
        assert out.nodata == -9999


def test_requested_nodata_replaces_declared(tmp_path):
    main = _geotiff(tmp_path / "job" / "r.tif", dtype="int16", nodata=0)
    item = _item(main, {"nodata": -9999})
    output = to_cog(item, tmp_path / "job", tmp_path / "out")
    _assert_lossless(item, output)
    with rasterio.open(output) as out:
        assert out.nodata == -9999


def test_unset_nodata_stays_unset(tmp_path):
    main = _geotiff(tmp_path / "job" / "r.tif")
    item = _item(main)
    output = to_cog(item, tmp_path / "job", tmp_path / "out")
    with rasterio.open(output) as out:
        assert out.nodata is None


def test_colour_table_is_kept(tmp_path):
    colormap = {i: (i, 255 - i, 0, 255) for i in range(256)}
    main = _geotiff(tmp_path / "job" / "r.tif", colormap=colormap)
    item = _item(main)
    output = to_cog(item, tmp_path / "job", tmp_path / "out")
    _assert_lossless(item, output)


def test_crs_from_settings_is_applied(tmp_path):
    main = _geotiff(tmp_path / "job" / "r.tif", crs="EPSG:4326")
    item = _item(main)
    item.settings["CRS"] = rasterio.crs.CRS.from_epsg(32617).to_wkt()
    output = to_cog(item, tmp_path / "job", tmp_path / "out")
    with rasterio.open(output) as out:
        assert out.crs.to_epsg() == 32617


BAD_VALUES = [
    ({"BLOCKSIZE": "big"}, "BLOCKSIZE"),
    ({"BLOCKSIZE": 0}, "BLOCKSIZE"),
    ({"BLOCKSIZE": 100}, "BLOCKSIZE"),
    ({"BLOCKSIZE": 8192}, "BLOCKSIZE"),
    ({"nodata": "abc"}, "NODATA"),
    ({"COMPRESS": "NOPE"}, "COMPRESS"),
    ({"OVERVIEW_COUNT": -1}, "OVERVIEW_COUNT"),
    ({"OVERVIEW_COUNT": "many"}, "OVERVIEW_COUNT"),
    ({"OVERVIEW_COUNT": 17}, "OVERVIEW_COUNT"),
    ({"OVERVIEW_COUNT": 2.5}, "OVERVIEW_COUNT"),
    ({"COMPRESS": "JXL"}, "COMPRESS"),
    ({"COMPRESS": "DEFLATE", "LEVEL": 13}, "LEVEL"),
    ({"COMPRESS": "ZSTD", "LEVEL": 23}, "LEVEL"),
    ({"COMPRESS": "LZMA", "LEVEL": 10}, "LEVEL"),
    ({"LEVEL": "high"}, "LEVEL"),
    ({"QUALITY": 0}, "QUALITY"),
    ({"QUALITY": 101}, "QUALITY"),
    ({"MAX_Z_ERROR": -0.1}, "MAX_Z_ERROR"),
    ({"MAX_Z_ERROR": "small"}, "MAX_Z_ERROR"),
]

GOOD_VALUES = [
    {"BLOCKSIZE": 256},
    {"COMPRESS": "zstd", "LEVEL": 22},
    {"COMPRESS": "LZW"},
    {"OVERVIEW_COUNT": 0},
    {"OVERVIEW_COUNT": 16},
    {"LEVEL": 12},
    {"COMPRESS": "WEBP", "QUALITY": 100},
    {"COMPRESS": "LERC", "MAX_Z_ERROR": 0},
]


@pytest.mark.parametrize(("cog_kwargs", "name"), BAD_VALUES)
def test_bad_value_is_rejected_before_gdal(tmp_path, monkeypatch, cog_kwargs, name):
    main = _geotiff(tmp_path / "job" / "r.tif")
    fingerprint_item = explore(main, [], "src", None)
    settings = {**fingerprint_item.settings, **{k.upper(): v for k, v in cog_kwargs.items()}}
    item = fingerprint_item.model_copy(update={"settings": settings})
    monkeypatch.setattr("rasterio.shutil.copy", lambda *a, **k: pytest.fail("GDAL was called"))
    with pytest.raises(ValueError, match=name):
        to_cog(item, tmp_path / "job", tmp_path / "out")


@pytest.mark.parametrize(("cog_kwargs", "name"), BAD_VALUES)
def test_bad_value_is_rejected_on_arrival(cog_kwargs, name):
    with pytest.raises(ValueError, match=name):
        validate_cog_kwargs(cog_kwargs)


@pytest.mark.parametrize("value", ["ALL_CPUS", 4])
def test_num_threads_is_not_accepted_from_caller(value):
    with pytest.raises(ValueError, match="unknown cog_kwargs option"):
        validate_cog_kwargs({"NUM_THREADS": value})


def test_level_without_compress_uses_configured_codec():
    # default codec is DEFLATE (1-12), so 20 is rejected on arrival, as at conversion
    with pytest.raises(ValueError, match="LEVEL must be from 1 to 12 for DEFLATE"):
        validate_cog_kwargs({"LEVEL": 20})


@pytest.mark.parametrize("cog_kwargs", GOOD_VALUES)
def test_good_values_are_accepted(cog_kwargs):
    validate_cog_kwargs(cog_kwargs)


def test_jpeg_is_lossy_and_writes_cog(tmp_path):
    main = _geotiff(tmp_path / "job" / "r.tif")
    item = _item(main, {"COMPRESS": "JPEG"})
    assert item.lossy
    output = to_cog(item, tmp_path / "job", tmp_path / "out")
    assert cog_validate(str(output))[0]


def test_non_convert_item_raises(tmp_path):
    item = PlanItem(source="src", kind="other", action=Action.SKIP, files=[str(tmp_path / "a.txt")])
    with pytest.raises(ValueError):
        to_cog(item, tmp_path, tmp_path / "out")


@pytest.mark.parametrize("folder", ["striped_geotiff", "tiled_geotiff_not_cog", "netcdf_2d"])
def test_fixture_files_become_lossless_cogs(folder, tmp_path):
    files = sorted((FIXTURES / folder).glob("*"))
    if not files:
        pytest.skip(f"fixture data missing: {folder}")
    for main in files:
        item = _item(main)
        output = to_cog(item, FIXTURES, tmp_path / "out")
        _assert_lossless(item, output)
        if folder == "netcdf_2d":
            with rasterio.open(output) as out:
                assert out.crs.to_epsg() == 4326

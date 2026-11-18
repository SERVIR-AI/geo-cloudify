"""Unit tests for COG conversion (to_cog): valid COG, lossless pixels, metadata kept, settings checked."""

import math
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rio_cogeo.cogeo import cog_validate

from geo_cloudify.cog.convert import to_cog

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "data"


def _geotiff(path, *, size=600, dtype="uint8", crs="EPSG:4326", nodata=None, colormap=None):
    """Write a small striped GeoTIFF (not a COG) and return its path."""
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


def _same_nodata(a, b) -> bool:
    """Return True when both nodata values are None, both NaN, or equal."""
    if a is None or b is None:
        return a is b
    return (math.isnan(a) and math.isnan(b)) or a == b


def _assert_lossless(source: Path, output: Path) -> None:
    """Check the output is a valid COG with the source's grid, nodata, colour table and pixels."""
    is_valid, errors, _ = cog_validate(str(output))
    assert is_valid, errors
    with rasterio.open(source) as src, rasterio.open(output) as out:
        assert (out.width, out.height, out.count) == (src.width, src.height, src.count)
        assert out.dtypes == src.dtypes
        assert out.transform == src.transform
        assert out.crs == src.crs
        assert _same_nodata(out.nodata, src.nodata)
        if src.colorinterp[0].name == "palette":
            assert out.colormap(1) == src.colormap(1)
        for _, window in out.block_windows(1):
            np.testing.assert_array_equal(out.read(window=window), src.read(window=window))


def test_striped_geotiff_becomes_lossless_cog(tmp_path):
    source = _geotiff(tmp_path / "src" / "r.tif")
    output = to_cog(source, tmp_path / "out" / "r.tif", "nearest")
    _assert_lossless(source, output)


def test_output_goes_where_asked(tmp_path):
    source = _geotiff(tmp_path / "src" / "r.tif")
    output = to_cog(source, tmp_path / "deep" / "folder" / "any-name.tif", "nearest")
    assert output == tmp_path / "deep" / "folder" / "any-name.tif"
    assert output.exists()


def test_existing_output_raises(tmp_path):
    source = _geotiff(tmp_path / "src" / "r.tif")
    to_cog(source, tmp_path / "out.tif", "nearest")
    with pytest.raises(FileExistsError):
        to_cog(source, tmp_path / "out.tif", "nearest")


def test_nothing_is_written_next_to_the_source(tmp_path):
    source = _geotiff(tmp_path / "src" / "r.tif")
    to_cog(source, tmp_path / "out.tif", "average")
    assert sorted(p.name for p in (tmp_path / "src").iterdir()) == ["r.tif"]


def test_nan_nodata_is_kept(tmp_path):
    source = _geotiff(tmp_path / "src" / "r.tif", dtype="float32", nodata=float("nan"))
    output = to_cog(source, tmp_path / "out.tif", "average")
    _assert_lossless(source, output)
    with rasterio.open(output) as out:
        assert math.isnan(out.nodata)


def test_declared_nodata_is_kept(tmp_path):
    source = _geotiff(tmp_path / "src" / "r.tif", dtype="int16", nodata=-9999)
    output = to_cog(source, tmp_path / "out.tif", "average")
    _assert_lossless(source, output)
    with rasterio.open(output) as out:
        assert out.nodata == -9999


def test_unset_nodata_stays_unset(tmp_path):
    source = _geotiff(tmp_path / "src" / "r.tif")
    output = to_cog(source, tmp_path / "out.tif", "nearest")
    with rasterio.open(output) as out:
        assert out.nodata is None


def test_colour_table_is_kept(tmp_path):
    colormap = {i: (i, 255 - i, 0, 255) for i in range(256)}
    source = _geotiff(tmp_path / "src" / "r.tif", colormap=colormap)
    output = to_cog(source, tmp_path / "out.tif", "nearest")
    _assert_lossless(source, output)


def test_crs_is_kept(tmp_path):
    source = _geotiff(tmp_path / "src" / "r.tif", crs="EPSG:32617")
    output = to_cog(source, tmp_path / "out.tif", "nearest")
    with rasterio.open(output) as out:
        assert out.crs.to_epsg() == 32617


def test_overviews_are_built(tmp_path):
    source = _geotiff(tmp_path / "src" / "r.tif", size=1200)
    output = to_cog(source, tmp_path / "out.tif", "mode")
    with rasterio.open(output) as out:
        assert out.overviews(1)


@pytest.mark.parametrize(("compress", "blocksize", "codec"), [("ZSTD", 256, "zstd"), ("lzw", 512, "lzw")])
def test_given_compress_and_blocksize_reach_the_output(tmp_path, compress, blocksize, codec):
    source = _geotiff(tmp_path / "src" / "r.tif")
    output = to_cog(source, tmp_path / "out.tif", "nearest", compress=compress, blocksize=blocksize)
    with rasterio.open(output) as out:
        assert out.compression.name.lower() == codec
        assert out.block_shapes[0] == (blocksize, blocksize)


def test_defaults_are_deflate_and_512(tmp_path):
    source = _geotiff(tmp_path / "src" / "r.tif", size=1200)
    output = to_cog(source, tmp_path / "out.tif", "nearest")
    with rasterio.open(output) as out:
        assert out.compression.name.lower() == "deflate"
        assert out.block_shapes[0] == (512, 512)


@pytest.mark.parametrize("compress", ["JPEG", "webp", "LERC", "LERC_DEFLATE", "LERC_ZSTD"])
def test_lossy_codec_is_refused_before_gdal(tmp_path, monkeypatch, compress):
    source = _geotiff(tmp_path / "src" / "r.tif")
    monkeypatch.setattr("rasterio.shutil.copy", lambda *a, **k: pytest.fail("GDAL was called"))
    with pytest.raises(ValueError, match="lossy"):
        to_cog(source, tmp_path / "out.tif", "nearest", compress=compress)


@pytest.mark.parametrize("compress", ["NOPE", "JXL", 5])
def test_unknown_codec_is_refused_before_gdal(tmp_path, monkeypatch, compress):
    source = _geotiff(tmp_path / "src" / "r.tif")
    monkeypatch.setattr("rasterio.shutil.copy", lambda *a, **k: pytest.fail("GDAL was called"))
    with pytest.raises(ValueError, match="compress"):
        to_cog(source, tmp_path / "out.tif", "nearest", compress=compress)


@pytest.mark.parametrize("blocksize", [0, 8, 100, 8192, "big", True, None, 512.5, 512.0, "512"])
def test_bad_block_size_is_refused_before_gdal(tmp_path, monkeypatch, blocksize):
    source = _geotiff(tmp_path / "src" / "r.tif")
    monkeypatch.setattr("rasterio.shutil.copy", lambda *a, **k: pytest.fail("GDAL was called"))
    with pytest.raises(ValueError, match="BLOCKSIZE"):
        to_cog(source, tmp_path / "out.tif", "nearest", blocksize=blocksize)


@pytest.mark.parametrize("method", ["bilinear", "MODE", "", None])
def test_bad_overview_resampling_is_refused_before_gdal(tmp_path, monkeypatch, method):
    source = _geotiff(tmp_path / "src" / "r.tif")
    monkeypatch.setattr("rasterio.shutil.copy", lambda *a, **k: pytest.fail("GDAL was called"))
    with pytest.raises(ValueError, match="overview_resampling"):
        to_cog(source, tmp_path / "out.tif", method)


def test_failed_write_leaves_no_output(tmp_path, monkeypatch):
    source = _geotiff(tmp_path / "src" / "r.tif")
    output = tmp_path / "out.tif"

    def half_write(src, dst, **kwargs):
        Path(dst).write_bytes(b"partial")
        raise rasterio.errors.RasterioIOError("disk full")

    monkeypatch.setattr("rasterio.shutil.copy", half_write)
    with pytest.raises(rasterio.errors.RasterioIOError):
        to_cog(source, output, "nearest")
    assert not output.exists()


@pytest.mark.parametrize(("folder", "method"), [("striped_geotiff", "average"), ("tiled_geotiff_not_cog", "nearest")])
def test_fixture_files_become_lossless_cogs(folder, method, tmp_path):
    files = sorted((FIXTURES / folder).glob("*.tif"))
    if not files:
        pytest.skip(f"fixture data missing: {folder}")
    for source in files:
        output = to_cog(source, tmp_path / "out" / source.name, method)
        _assert_lossless(source, output)

"""Unit tests for checking that tiles share one grid (find_union_grid) and merging them (merge_to_cog)."""

import hashlib
import math
import threading
from pathlib import Path

import numpy as np
import pytest
import rasterio
import rasterio.errors
import rasterio.shutil
from rasterio.crs import CRS
from rasterio.io import DatasetReader
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import Affine, from_origin
from rasterio.windows import Window
from rio_cogeo.cogeo import cog_validate

from geo_cloudify.cog.merge import (
    GeoCloudifyError, InvalidInputError, TileGrid, TileMismatchError, find_union_grid, merge_to_cog,
)

# A scaled-down RLCMS year: 30 m pixels, negative x origin, tiles of different sizes.
X0, Y0, PIXEL = -1260450.0, 3413550.0, 30.0


def _tile(path, *, x=X0, y=Y0, width=40, height=40, pixel=PIXEL, crs="EPSG:32648",
          dtype="uint8", count=1, nodata=0, transform=None):
    """Write a small GeoTIFF tile and return its path."""
    transform = transform or from_origin(x, y, pixel, pixel)
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=count, dtype=dtype,
        crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(np.ones((count, height, width), dtype=dtype))
    return path


def _four_tiles(tmp_path, **last_tile_kwargs):
    """Write four tiles laid out like an RLCMS year; extra kwargs change the last tile."""
    east_x = X0 + 40 * PIXEL
    south_y = Y0 - 40 * PIXEL
    return [
        _tile(tmp_path / "nw.tif", width=40, height=40),
        _tile(tmp_path / "ne.tif", x=east_x, width=23, height=40),
        _tile(tmp_path / "sw.tif", y=south_y, width=40, height=17),
        _tile(tmp_path / "se.tif", **{"x": east_x, "y": south_y, "width": 23, "height": 17,
                                      **last_tile_kwargs}),
    ]


def test_four_matching_tiles_give_the_union_grid(tmp_path):
    tiles = _four_tiles(tmp_path)
    grid = find_union_grid(tiles)
    assert isinstance(grid, TileGrid)
    assert grid.crs == CRS.from_epsg(32648)
    assert grid.transform == from_origin(X0, Y0, PIXEL, PIXEL)
    assert (grid.width, grid.height) == (63, 57)
    assert (grid.dtype, grid.count, grid.nodata) == ("uint8", 1, 0)


def test_union_starts_at_the_most_north_west_tile_in_any_order(tmp_path):
    tiles = _four_tiles(tmp_path)
    grid = find_union_grid(list(reversed(tiles)))
    assert grid.transform == from_origin(X0, Y0, PIXEL, PIXEL)
    assert (grid.width, grid.height) == (63, 57)


def test_union_covers_a_gap_between_tiles(tmp_path):
    a = _tile(tmp_path / "a.tif", width=10, height=10)
    b = _tile(tmp_path / "b.tif", x=X0 + 15 * PIXEL, y=Y0 - 5 * PIXEL, width=10, height=10)
    grid = find_union_grid([a, b])
    assert grid.transform == from_origin(X0, Y0, PIXEL, PIXEL)
    assert (grid.width, grid.height) == (25, 15)


def test_tiny_float_noise_in_origin_is_still_aligned(tmp_path):
    a = _tile(tmp_path / "a.tif", width=10, height=10)
    b = _tile(tmp_path / "b.tif", x=X0 + 10 * PIXEL + 1e-7, width=10, height=10)
    grid = find_union_grid([a, b])
    assert (grid.width, grid.height) == (20, 10)


def test_single_tile_is_accepted(tmp_path):
    tile = _tile(tmp_path / "one.tif", width=23, height=17, nodata=None)
    grid = find_union_grid([str(tile)])
    assert grid.transform == from_origin(X0, Y0, PIXEL, PIXEL)
    assert (grid.width, grid.height, grid.nodata) == (23, 17, None)


# EPSG:32648 written out by hand under another name. GeoTIFF rewrites any CRS to its
# EPSG form on save, so this text is kept by wrapping a tile in a VRT instead.
UTM48_OTHER_TEXT = (
    'PROJCS["My UTM 48",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,'
    '298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],'
    'PARAMETER["central_meridian",105],PARAMETER["scale_factor",0.9996],'
    'PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1]]'
)


def test_same_crs_written_differently_matches(tmp_path):
    a = _tile(tmp_path / "a.tif", width=10, height=10)
    _tile(tmp_path / "b.tif", width=10, height=10)
    b = tmp_path / "b.vrt"
    b.write_text(
        f'<VRTDataset rasterXSize="10" rasterYSize="10"><SRS>{UTM48_OTHER_TEXT}</SRS>'
        f"<GeoTransform>{X0 + 10 * PIXEL}, {PIXEL}, 0, {Y0}, 0, {-PIXEL}</GeoTransform>"
        '<VRTRasterBand dataType="Byte" band="1"><NoDataValue>0</NoDataValue><SimpleSource>'
        '<SourceFilename relativeToVRT="1">b.tif</SourceFilename><SourceBand>1</SourceBand>'
        "</SimpleSource></VRTRasterBand></VRTDataset>"
    )
    with rasterio.open(a) as src_a, rasterio.open(b) as src_b:
        assert src_a.crs.to_wkt() != src_b.crs.to_wkt()
    grid = find_union_grid([a, b])
    assert (grid.width, grid.height) == (20, 10)


@pytest.mark.parametrize("as_type", [str, lambda path: path], ids=["str", "PathLike"])
def test_single_path_instead_of_a_list_is_refused(tmp_path, as_type):
    tile = _tile(tmp_path / "a.tif", width=10, height=10)
    with pytest.raises(ValueError, match="not a single path"):
        find_union_grid(as_type(tile))


def test_empty_tile_list_is_refused():
    with pytest.raises(ValueError, match="no tiles"):
        find_union_grid([])


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"crs": "EPSG:32647"}, "crs"),
        ({"pixel": 20.0}, "pixel size"),
        ({"transform": Affine(PIXEL, 0.0, X0 + 40 * PIXEL, 0.0, -20.0, Y0 - 40 * PIXEL)},
         "pixel size"),
        ({"x": X0 + 40 * PIXEL + 10.0}, "grid alignment"),
        ({"y": Y0 - 40 * PIXEL - 15.0}, "grid alignment"),
        ({"dtype": "uint16"}, "data type"),
        ({"count": 2}, "band count"),
        ({"nodata": 255}, "nodata"),
        ({"nodata": None}, "nodata"),
    ],
)
def test_mismatched_tile_is_refused_naming_tile_and_field(tmp_path, change, field):
    tiles = _four_tiles(tmp_path, **change)
    with pytest.raises(TileMismatchError) as caught:
        find_union_grid(tiles)
    assert caught.value.tile == str(tiles[3])
    assert caught.value.field == field
    assert "se.tif" in str(caught.value) and field in str(caught.value)


def test_first_tile_without_nodata_and_second_with_is_refused(tmp_path):
    a = _tile(tmp_path / "a.tif", width=10, height=10, nodata=None)
    b = _tile(tmp_path / "b.tif", x=X0 + 10 * PIXEL, width=10, height=10, nodata=0)
    with pytest.raises(TileMismatchError) as caught:
        find_union_grid([a, b])
    assert (caught.value.tile, caught.value.field) == (str(b), "nodata")


def test_nan_and_number_nodata_are_refused(tmp_path):
    a = _tile(tmp_path / "a.tif", width=10, height=10, dtype="float32", nodata=float("nan"))
    b = _tile(tmp_path / "b.tif", x=X0 + 10 * PIXEL, width=10, height=10,
              dtype="float32", nodata=-9999)
    with pytest.raises(TileMismatchError, match="nodata"):
        find_union_grid([a, b])


def test_tile_without_crs_is_refused(tmp_path):
    a = _tile(tmp_path / "a.tif", width=10, height=10)
    b = _tile(tmp_path / "b.tif", x=X0 + 10 * PIXEL, width=10, height=10, crs=None)
    with pytest.raises(TileMismatchError) as caught:
        find_union_grid([a, b])
    assert (caught.value.tile, caught.value.field) == (str(b), "crs")


def test_rotated_tile_is_refused(tmp_path):
    rotated = Affine(PIXEL, 1.0, X0, 0.0, -PIXEL, Y0)
    tile = _tile(tmp_path / "r.tif", width=10, height=10, transform=rotated)
    with pytest.raises(TileMismatchError) as caught:
        find_union_grid([tile])
    assert caught.value.field == "orientation"


def test_only_headers_are_read(tmp_path, monkeypatch):
    tiles = _four_tiles(tmp_path)

    def no_read(*args, **kwargs):
        raise AssertionError("find_union_grid read pixels")

    monkeypatch.setattr(DatasetReader, "read", no_read)
    find_union_grid(tiles)


# --- merge_to_cog ---------------------------------------------------------------------

FIXTURE_TILES = Path(__file__).parents[1] / "fixtures" / "data" / "tiles_to_mosaic"


def _tile_with(path, data, *, x=X0, y=Y0, nodata=0, overviews=None):
    """Write `data` (bands, rows, cols) as a GeoTIFF tile at (x, y) and return its path.

    `overviews`, when given, is a Resampling method used to build overviews inside the tile.
    """
    count, height, width = data.shape
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=count, dtype=data.dtype,
        crs="EPSG:32648", transform=from_origin(x, y, PIXEL, PIXEL), nodata=nodata,
        tiled=True, blockxsize=256, blockysize=256,
    ) as dst:
        dst.write(data)
        if overviews is not None:
            dst.build_overviews([2, 4], overviews)
    return path


def _random_tiles(tmp_path, *, nodata=0, gap=0, dtype="uint8"):
    """Four tiles of different sizes, none a whole number of 512 blocks, with values 0 to 22.

    0 is common in the data, so it is both nodata and a real value. `gap` leaves that
    many empty pixel columns between the west and east tiles. A float `dtype` gives
    values with fractions, so float statistics are not whole numbers.
    """
    rng = np.random.default_rng(7)
    sizes = {"nw": (530, 600), "ne": (530, 311), "sw": (257, 600), "se": (257, 311)}
    east_x = X0 + (600 + gap) * PIXEL
    south_y = Y0 - 530 * PIXEL
    origins = {"nw": (X0, Y0), "ne": (east_x, Y0), "sw": (X0, south_y), "se": (east_x, south_y)}
    tiles = []
    for name, (height, width) in sizes.items():
        if np.dtype(dtype).kind == "f":
            data = rng.uniform(0, 23, size=(1, height, width)).astype(dtype)
            data[data < 1] = 0  # keep 0 common, as in the integer tiles
        else:
            data = rng.integers(0, 23, size=(1, height, width), dtype=dtype)
        x, y = origins[name]
        tiles.append(_tile_with(tmp_path / f"{name}.tif", data, x=x, y=y, nodata=nodata))
    return tiles


def _assert_tiles_placed(output, tiles):
    """Every full-resolution pixel of each tile equals the output pixel at its place.

    Reads one tile-sized window of the output at a time, never the whole output.
    """
    with rasterio.open(output) as out:
        for tile in tiles:
            with rasterio.open(tile) as src:
                col = round((src.transform.c - out.transform.c) / out.transform.a)
                row = round((src.transform.f - out.transform.f) / out.transform.e)
                window = Window(col, row, src.width, src.height)
                np.testing.assert_array_equal(out.read(window=window), src.read(), err_msg=str(tile))


def test_merge_places_every_pixel_and_keeps_the_grid(tmp_path):
    tiles = _random_tiles(tmp_path)
    output = merge_to_cog(tiles, tmp_path / "out" / "merged.tif", "mode")
    assert output == tmp_path / "out" / "merged.tif"
    _assert_tiles_placed(output, tiles)
    with rasterio.open(output) as out:
        assert out.crs == CRS.from_epsg(32648)
        assert out.transform == from_origin(X0, Y0, PIXEL, PIXEL)
        assert (out.width, out.height) == (911, 787)
        assert out.dtypes == ("uint8",)
        assert out.nodata == 0


def test_merged_output_is_a_valid_cog_with_overviews(tmp_path):
    output = merge_to_cog(_random_tiles(tmp_path), tmp_path / "merged.tif", "nearest")
    is_valid, errors, warnings = cog_validate(str(output), quiet=True)
    assert is_valid and not errors and not warnings
    with rasterio.open(output) as out:
        assert out.block_shapes == [(512, 512)]
        assert out.overviews(1)  # 911 x 787 is larger than one block
        assert out.compression is not None


def test_gap_between_tiles_is_filled_with_nodata(tmp_path):
    tiles = _random_tiles(tmp_path, gap=5)
    output = merge_to_cog(tiles, tmp_path / "merged.tif", "nearest")
    _assert_tiles_placed(output, tiles)
    with rasterio.open(output) as out:
        assert out.width == 600 + 5 + 311
        gap = out.read(1, window=Window(600, 0, 5, out.height))
        assert (gap == 0).all()


def test_tiles_without_nodata_give_an_output_without_nodata(tmp_path):
    tiles = _random_tiles(tmp_path, nodata=None)
    output = merge_to_cog(tiles, tmp_path / "merged.tif", "average")
    _assert_tiles_placed(output, tiles)
    with rasterio.open(output) as out:
        assert out.nodata is None
        assert out.nodatavals == (None,)


def test_nan_nodata_float_tiles_merge(tmp_path):
    data = np.arange(2 * 20 * 30, dtype="float32").reshape(2, 20, 30)
    data[:, 0, 0] = np.nan
    a = _tile_with(tmp_path / "a.tif", data, nodata=float("nan"))
    b = _tile_with(tmp_path / "b.tif", data, x=X0 + 30 * PIXEL, nodata=float("nan"))
    output = merge_to_cog([a, b], tmp_path / "merged.tif", "average")
    with rasterio.open(output) as out:
        assert out.count == 2 and out.dtypes == ("float32", "float32")
        assert math.isnan(out.nodata)
        merged = out.read()
    np.testing.assert_array_equal(merged, np.concatenate([data, data], axis=2))


def _class_pattern(size=1024):
    """A class raster where each 2 x 2 cell holds three 1s and one 5.

    The 5 moves around the four places from cell to cell, so a nearest overview
    picks up some 5s whichever pixel it samples, a mode overview is all 1 and an
    average overview is all 2.
    """
    data = np.ones((size, size), dtype="uint8")
    cells = size // 2
    which = np.indices((cells, cells)).sum(axis=0) % 4
    for place, (dr, dc) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        rows, cols = np.nonzero(which == place)
        data[rows * 2 + dr, cols * 2 + dc] = 5
    return data[np.newaxis]


@pytest.mark.parametrize(
    ("method", "expected"),
    [("mode", {1}), ("average", {2}), ("nearest", {1, 5})],
)
def test_overview_method_reaches_gdal_and_tile_overviews_are_not_reused(tmp_path, method, expected):
    # The tile's own overviews are built with another method, so reusing them would show.
    other = Resampling.cubic if method == "average" else Resampling.average
    tile = _tile_with(tmp_path / "t.tif", _class_pattern(), nodata=0, overviews=other)
    output = merge_to_cog([tile], tmp_path / "merged.tif", method)
    with rasterio.open(output, overview_level=0) as overview:
        assert (overview.width, overview.height) == (512, 512)
        assert set(np.unique(overview.read(1)).tolist()) == expected


@pytest.mark.parametrize("method", ["bilinear", "NEAREST", "", None, "mode "])
def test_unknown_overview_method_is_refused(tmp_path, method):
    tiles = _random_tiles(tmp_path)
    with pytest.raises(ValueError, match="overview_resampling"):
        merge_to_cog(tiles, tmp_path / "merged.tif", method)
    assert not (tmp_path / "merged.tif").exists()


def test_existing_output_is_refused_and_kept(tmp_path):
    tiles = _random_tiles(tmp_path)
    output = tmp_path / "merged.tif"
    output.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        merge_to_cog(tiles, output, "mode")
    assert output.read_bytes() == b"keep me"


def test_failed_write_leaves_no_file_behind(tmp_path, monkeypatch):
    tiles = _random_tiles(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    real_copy = rasterio.shutil.copy
    disk_full = rasterio.errors.RasterioIOError("disk full")

    def copy_then_fail(src, dst, **kwargs):
        real_copy(src, dst, **kwargs)
        raise disk_full

    monkeypatch.setattr(rasterio.shutil, "copy", copy_then_fail)
    with pytest.raises(GeoCloudifyError, match="merged.tif") as caught:
        merge_to_cog(tiles, out_dir / "merged.tif", "mode")
    assert caught.value.__cause__ is disk_full
    assert list(out_dir.iterdir()) == []


def test_mismatched_tiles_are_refused_before_writing(tmp_path):
    tiles = _four_tiles(tmp_path, nodata=255)
    with pytest.raises(TileMismatchError) as caught:
        merge_to_cog(tiles, tmp_path / "merged.tif", "mode")
    assert caught.value.field == "nodata"
    assert not (tmp_path / "merged.tif").exists()


def test_overlapping_tiles_are_refused_naming_both(tmp_path):
    a = _tile(tmp_path / "a.tif", width=10, height=10)
    b = _tile(tmp_path / "b.tif", x=X0 + 9 * PIXEL, y=Y0 - 9 * PIXEL, width=10, height=10)
    c = _tile(tmp_path / "c.tif", x=X0 + 20 * PIXEL, width=10, height=10)
    with pytest.raises(TileMismatchError) as caught:
        merge_to_cog([a, c, b], tmp_path / "merged.tif", "mode")
    assert caught.value.tile == str(b)
    assert caught.value.field == "overlap"
    assert "a.tif" in str(caught.value)
    assert not (tmp_path / "merged.tif").exists()


def test_url_tile_is_refused_before_it_is_opened(tmp_path, monkeypatch):
    local = _tile(tmp_path / "a.tif", width=10, height=10)
    url = "https://example.com/b.tif"
    real_open = rasterio.open
    opened = []

    def record_open(path, *args, **kwargs):
        opened.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(rasterio, "open", record_open)
    with pytest.raises(InvalidInputError, match="not a local file") as caught:
        merge_to_cog([local, url], tmp_path / "merged.tif", "mode")
    assert url in str(caught.value)
    assert url not in opened, "the URL tile was opened before it was refused"
    assert not (tmp_path / "merged.tif").exists()


def test_relative_tile_paths_are_found(tmp_path, monkeypatch):
    tiles = _random_tiles(tmp_path)
    monkeypatch.chdir(tmp_path)
    output = merge_to_cog([t.name for t in tiles], "merged.tif", "mode")
    _assert_tiles_placed(tmp_path / "merged.tif", tiles)
    assert output == Path("merged.tif")


@pytest.mark.parametrize(("dtype", "method"), [("uint8", "mode"), ("float32", "average")])
def test_same_tiles_give_the_same_bytes(tmp_path, dtype, method):
    tiles = _random_tiles(tmp_path, dtype=dtype)
    first = merge_to_cog(tiles, tmp_path / "first.tif", method)
    second = merge_to_cog(tiles, tmp_path / "second.tif", method)
    assert hashlib.sha256(first.read_bytes()).hexdigest() == hashlib.sha256(second.read_bytes()).hexdigest()


def test_colours_and_first_tile_metadata_are_kept(tmp_path):
    tiles = _random_tiles(tmp_path)
    colormap = {0: (0, 0, 0, 0), 1: (10, 120, 30, 255), 22: (200, 0, 0, 255)}
    for tile in tiles:
        with rasterio.open(tile, "r+") as dst:
            dst.write_colormap(1, colormap)
            dst.set_band_description(1, "land cover class")
    with rasterio.open(tiles[0], "r+") as dst:
        dst.update_tags(SOURCE="test")
        dst.update_tags(1, CLASSES="1-22")
    output = merge_to_cog(tiles, tmp_path / "merged.tif", "mode")
    with rasterio.open(output) as out:
        assert out.colorinterp[0].name == "palette"
        kept = out.colormap(1)
        assert all(kept[value] == rgba for value, rgba in colormap.items())
        assert out.descriptions[0] == "land cover class"
        assert out.tags()["SOURCE"] == "test"
        assert out.tags(1)["CLASSES"] == "1-22"


def test_scale_offset_and_units_are_kept(tmp_path):
    tiles = _random_tiles(tmp_path)
    for tile in tiles:
        with rasterio.open(tile, "r+") as dst:
            dst.scales = (0.5,)
            dst.offsets = (-3.0,)
            dst.units = ("degree Celsius",)
    output = merge_to_cog(tiles, tmp_path / "merged.tif", "average")
    with rasterio.open(output) as out:
        assert out.scales == (0.5,) and out.offsets == (-3.0,)
        assert out.units == ("degree Celsius",)


@pytest.mark.parametrize(("field", "change"), [
    ("scale", lambda dst: setattr(dst, "scales", (0.5,))),
    ("offset", lambda dst: setattr(dst, "offsets", (-3.0,))),
    ("units", lambda dst: setattr(dst, "units", ("m",))),
    ("description", lambda dst: dst.set_band_description(1, "other")),
    ("colour interpretation", lambda dst: setattr(dst, "colorinterp", [ColorInterp.red])),
    ("colour table", lambda dst: dst.write_colormap(1, {1: (200, 0, 0, 255)})),
])
def test_band_setting_that_differs_is_refused_naming_tile_and_field(tmp_path, field, change):
    tiles = _random_tiles(tmp_path)
    with rasterio.open(tiles[3], "r+") as dst:
        change(dst)
    with pytest.raises(TileMismatchError) as caught:
        merge_to_cog(tiles, tmp_path / "merged.tif", "mode")
    assert (caught.value.tile, caught.value.field) == (str(tiles[3]), field)
    assert "se.tif" in str(caught.value) and field in str(caught.value)
    assert not (tmp_path / "merged.tif").exists()


@pytest.mark.skipif(not FIXTURE_TILES.is_dir(), reason="test data not downloaded")
def test_real_gmw_tiles_merge(tmp_path):
    tiles = sorted(FIXTURE_TILES.glob("*.tif"))
    assert len(tiles) == 4
    output = merge_to_cog(tiles, tmp_path / "gmw.tif", "mode")
    _assert_tiles_placed(output, tiles)
    is_valid, errors, warnings = cog_validate(str(output), quiet=True)
    assert is_valid and not errors and not warnings
    with rasterio.open(output) as out, rasterio.open(tiles[0]) as first:
        assert out.crs == first.crs
        assert (out.width, out.height) == (9000, 9000)
        assert out.transform.a == first.transform.a and out.transform.e == first.transform.e
        assert (out.transform.c, out.transform.f) == (104.0, 10.0)
        assert out.dtypes == first.dtypes and out.nodata == first.nodata
        assert out.overviews(1)


# --- settings: defaults, lossless codecs only -------------------------------------------

def test_defaults_are_used_and_logged(tmp_path, caplog):
    tiles = _random_tiles(tmp_path)
    with caplog.at_level("INFO", logger="geo_cloudify.cog.merge"):
        merge_to_cog(tiles, tmp_path / "merged.tif", "mode")
    lines = [r.getMessage() for r in caplog.records
             if r.name == "geo_cloudify.cog.merge" and r.levelname == "INFO"]
    assert len(lines) == 1
    assert "DEFLATE" in lines[0] and "512" in lines[0] and "mode" in lines[0]


@pytest.mark.parametrize(("compress", "blocksize", "codec"), [
    ("ZSTD", 256, "zstd"), ("lzw", 1024, "lzw"), ("Deflate", 128, "deflate"),
])
def test_given_compress_and_blocksize_reach_the_output(tmp_path, caplog, compress, blocksize, codec):
    tiles = _random_tiles(tmp_path)
    with caplog.at_level("INFO", logger="geo_cloudify.cog.merge"):
        output = merge_to_cog(tiles, tmp_path / "merged.tif", "nearest",
                              compression_codec=compress, blocksize=blocksize)
    _assert_tiles_placed(output, tiles)
    with rasterio.open(output) as out:
        assert out.compression.name == codec
        assert out.block_shapes[0] == (blocksize, blocksize)
    assert any(compress.upper() in r.getMessage() and str(blocksize) in r.getMessage()
               for r in caplog.records if r.name == "geo_cloudify.cog.merge")


@pytest.mark.parametrize("compress", [
    "JPEG", "jpeg", "WEBP", "Webp", "LERC", "lerc", "LERC_DEFLATE", "lerc_deflate", "LERC_ZSTD", "Lerc_Zstd",
])
def test_lossy_codecs_are_refused_before_writing(tmp_path, compress):
    tiles = _random_tiles(tmp_path)
    out_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="(?i)lossy"):
        merge_to_cog(tiles, out_dir / "merged.tif", "mode", compression_codec=compress)
    assert not out_dir.exists()


@pytest.mark.parametrize("compress", ["FOO", "", None, 5, "DEFLATE "])
def test_unknown_codec_is_refused_before_writing(tmp_path, compress):
    tiles = _random_tiles(tmp_path)
    out_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="(?i)compress"):
        merge_to_cog(tiles, out_dir / "merged.tif", "mode", compression_codec=compress)
    assert not out_dir.exists()


@pytest.mark.parametrize("blocksize", [0, 8, 48, 100, 8192, "big", True, None, 512.5, 512.0, "512"])
def test_bad_block_size_is_refused_before_writing(tmp_path, blocksize):
    tiles = _random_tiles(tmp_path)
    out_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="BLOCKSIZE"):
        merge_to_cog(tiles, out_dir / "merged.tif", "mode", blocksize=blocksize)
    assert not out_dir.exists()


@pytest.mark.parametrize("blocksize", [np.int64(512), np.int32(256), np.uint16(512), np.int8(64)])
def test_numpy_integer_block_size_is_accepted(tmp_path, blocksize):
    tiles = _random_tiles(tmp_path)
    output = merge_to_cog(tiles, tmp_path / "merged.tif", "mode", blocksize=blocksize)
    with rasterio.open(output) as out:
        assert out.block_shapes[0] == (int(blocksize), int(blocksize))


@pytest.mark.parametrize("blocksize", [np.int64(100), np.int64(8192), np.float64(512.0), np.bool_(True)])
def test_bad_numpy_block_size_is_refused_before_writing(tmp_path, blocksize):
    tiles = _random_tiles(tmp_path)
    out_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="BLOCKSIZE"):
        merge_to_cog(tiles, out_dir / "merged.tif", "mode", blocksize=blocksize)
    assert not out_dir.exists()


# --- statistics are computed for the merged image -----------------------------------------

def test_statistics_are_computed_for_the_merged_image_not_copied(tmp_path):
    low = np.zeros((1, 300, 300), dtype="uint8")
    low[:, :, 1:] = np.random.default_rng(3).integers(1, 4, size=(1, 300, 299), dtype="uint8")
    high = low.copy()
    high[0, 151, 213] = 20  # one pixel, so statistics from an overview would miss it
    a = _tile_with(tmp_path / "a.tif", low)
    b = _tile_with(tmp_path / "b.tif", high, x=X0 + 300 * PIXEL)
    with rasterio.open(a, "r+") as dst:
        dst.update_tags(1, STATISTICS_MINIMUM="0", STATISTICS_MAXIMUM="3", STATISTICS_MEAN="1.5",
                        STATISTICS_STDDEV="1", STATISTICS_VALID_PERCENT="100", CLASSES="1-22")
        dst.update_tags(STATISTICS_MAXIMUM="3", SOURCE="test")
    out_dir = tmp_path / "out"
    output = merge_to_cog([a, b], out_dir / "merged.tif", "mode")
    with rasterio.open(output) as out:
        band_tags = out.tags(1)
        assert float(band_tags["STATISTICS_MAXIMUM"]) == 20
        assert float(band_tags["STATISTICS_MINIMUM"]) == 1  # nodata 0 is left out
        assert float(band_tags["STATISTICS_VALID_PERCENT"]) < 100
        assert band_tags["CLASSES"] == "1-22"
        assert "STATISTICS_MAXIMUM" not in out.tags()
        assert out.tags()["SOURCE"] == "test"
    assert [p.name for p in out_dir.iterdir()] == ["merged.tif"]


def _aux_xml(tile, nodata):
    """Write a `<tile>.aux.xml` next to `tile` holding statistics and a nodata value."""
    Path(f"{tile}.aux.xml").write_text(
        "<PAMDataset>\n  <PAMRasterBand band=\"1\">\n"
        f"    <NoDataValue>{nodata}</NoDataValue>\n"
        "    <Metadata>\n"
        "      <MDI key=\"STATISTICS_MAXIMUM\">3</MDI>\n"
        "      <MDI key=\"STATISTICS_MINIMUM\">1</MDI>\n"
        "    </Metadata>\n  </PAMRasterBand>\n</PAMDataset>\n"
    )


def _folder_bytes(folder):
    return {p.name: p.read_bytes() for p in folder.iterdir()}


def test_merge_leaves_the_tile_folder_unchanged(tmp_path):
    folder = tmp_path / "tiles"
    folder.mkdir()
    tiles = _random_tiles(folder)
    _aux_xml(tiles[1], 0)
    before = _folder_bytes(folder)
    merge_to_cog(tiles, tmp_path / "out" / "merged.tif", "mode")
    assert _folder_bytes(folder) == before


def test_other_threads_still_read_aux_xml_during_the_copy(tmp_path, monkeypatch):
    folder = tmp_path / "tiles"
    folder.mkdir()
    tiles = _random_tiles(folder)
    other = _tile(tmp_path / "other.tif", nodata=None)
    _aux_xml(other, 0)
    real_copy = rasterio.shutil.copy
    seen = {}

    def read_other():
        with rasterio.open(other) as src:
            seen["nodata"] = src.nodata

    def copy_and_read_from_another_thread(*args, **kwargs):
        seen["copy_on_main_thread"] = threading.current_thread() is threading.main_thread()
        reader = threading.Thread(target=read_other)
        reader.start()
        reader.join()
        with rasterio.open(other) as src:
            seen["copy_thread_nodata"] = src.nodata
        return real_copy(*args, **kwargs)

    monkeypatch.setattr(rasterio.shutil, "copy", copy_and_read_from_another_thread)
    assert threading.current_thread() is threading.main_thread()
    merge_to_cog(tiles, tmp_path / "out" / "merged.tif", "mode")
    assert seen == {"copy_on_main_thread": False, "nodata": 0, "copy_thread_nodata": None}
    with rasterio.open(other) as src:
        assert src.nodata == 0


def test_nodata_stored_only_in_an_aux_xml_still_reaches_the_output(tmp_path):
    folder = tmp_path / "tiles"
    folder.mkdir()
    tiles = _random_tiles(folder, nodata=None, gap=5)
    for tile in tiles:
        _aux_xml(tile, 0)
    before = _folder_bytes(folder)
    output = merge_to_cog(tiles, tmp_path / "out" / "merged.tif", "mode")
    with rasterio.open(output) as out:
        assert out.nodata == 0
        assert float(out.tags(1)["STATISTICS_VALID_PERCENT"]) < 100
    assert _folder_bytes(folder) == before


# --- holes with no nodata ----------------------------------------------------------------

def test_hole_without_nodata_is_refused_before_writing(tmp_path):
    tiles = _random_tiles(tmp_path, nodata=None, gap=5)
    out_dir = tmp_path / "out"
    with pytest.raises(InvalidInputError, match="nodata") as caught:
        merge_to_cog(tiles, out_dir / "merged.tif", "nearest")
    assert "3,935 pixels" in str(caught.value)
    assert not out_dir.exists()


def test_hole_count_for_a_missing_corner_tile(tmp_path):
    tiles = _random_tiles(tmp_path, nodata=None)[:3]  # se (257 x 311) left out
    with pytest.raises(InvalidInputError, match="79,927 pixels"):
        merge_to_cog(tiles, tmp_path / "merged.tif", "nearest")


# --- tiles with an internal mask ----------------------------------------------------------

def _masked_tile(path, *, x=X0):
    """Write a tile with an internal per-dataset mask and no nodata, and return its path."""
    data = np.full((1, 20, 20), 7, dtype="uint8")
    mask = np.full((20, 20), 255, dtype="uint8")
    mask[:5, :5] = 0
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(
            path, "w", driver="GTiff", width=20, height=20, count=1, dtype="uint8",
            crs="EPSG:32648", transform=from_origin(x, Y0, PIXEL, PIXEL),
        ) as dst:
            dst.write(data)
            dst.write_mask(mask)
    return path


def _alpha_tile(path, *, x=X0):
    """Write an RGBA tile whose fourth band is alpha, and return its path."""
    data = np.full((4, 20, 20), 9, dtype="uint8")
    data[3] = 255
    with rasterio.open(
        path, "w", driver="GTiff", width=20, height=20, count=4, dtype="uint8",
        crs="EPSG:32648", transform=from_origin(x, Y0, PIXEL, PIXEL), photometric="RGB", alpha="YES",
    ) as dst:
        dst.write(data)
    return path


@pytest.mark.parametrize("masked_first", [True, False])
def test_tile_with_internal_mask_is_refused(tmp_path, masked_first):
    masked = _masked_tile(tmp_path / "masked.tif")
    plain = _tile(tmp_path / "plain.tif", x=X0 + 20 * PIXEL, width=20, height=20, nodata=None)
    tiles = [masked, plain] if masked_first else [plain, masked]
    with pytest.raises(TileMismatchError) as caught:
        merge_to_cog(tiles, tmp_path / "merged.tif", "mode")
    assert caught.value.tile == str(masked)
    assert caught.value.field == "mask"
    assert "masked.tif" in str(caught.value)
    assert not (tmp_path / "merged.tif").exists()


def test_mask_is_found_without_reading_pixels(tmp_path, monkeypatch):
    masked = _masked_tile(tmp_path / "masked.tif")

    def no_read(self, *args, **kwargs):
        raise AssertionError("pixels were read")

    monkeypatch.setattr(DatasetReader, "read", no_read)
    monkeypatch.setattr(DatasetReader, "read_masks", no_read)
    monkeypatch.setattr(DatasetReader, "dataset_mask", no_read)
    with pytest.raises(TileMismatchError) as caught:
        find_union_grid([masked])
    assert caught.value.field == "mask"


def test_tiles_with_an_alpha_band_are_not_refused_for_a_mask(tmp_path):
    a = _alpha_tile(tmp_path / "a.tif")
    b = _alpha_tile(tmp_path / "b.tif", x=X0 + 20 * PIXEL)
    output = merge_to_cog([a, b], tmp_path / "merged.tif", "nearest")
    _assert_tiles_placed(output, [a, b])


def test_hole_in_alpha_tiles_is_refused_before_writing(tmp_path):
    a = _alpha_tile(tmp_path / "a.tif")
    b = _alpha_tile(tmp_path / "b.tif", x=X0 + 25 * PIXEL)
    out_dir = tmp_path / "out"
    with pytest.raises(InvalidInputError, match="100 pixels"):
        merge_to_cog([a, b], out_dir / "merged.tif", "nearest")
    assert not out_dir.exists()

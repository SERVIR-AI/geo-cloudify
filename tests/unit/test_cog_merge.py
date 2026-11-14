"""Unit tests for checking that tiles share one grid (check_tiles) and merging them (merge_to_cog)."""

import math
from pathlib import Path

import numpy as np
import pytest
import rasterio
import rasterio.errors
import rasterio.shutil
from rasterio.crs import CRS
from rasterio.io import DatasetReader
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_origin
from rasterio.windows import Window
from rio_cogeo.cogeo import cog_validate

from geo_cloudify.cog.merge import TileGrid, TileMismatchError, _gdal_path, check_tiles, merge_to_cog
from geo_cloudify.config import get_config
from geo_cloudify.settings import get_settings

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
    grid = check_tiles(tiles)
    assert isinstance(grid, TileGrid)
    assert grid.crs == CRS.from_epsg(32648)
    assert grid.transform == from_origin(X0, Y0, PIXEL, PIXEL)
    assert (grid.width, grid.height) == (63, 57)
    assert (grid.dtype, grid.count, grid.nodata) == ("uint8", 1, 0)


def test_union_starts_at_the_most_north_west_tile_in_any_order(tmp_path):
    tiles = _four_tiles(tmp_path)
    grid = check_tiles(list(reversed(tiles)))
    assert grid.transform == from_origin(X0, Y0, PIXEL, PIXEL)
    assert (grid.width, grid.height) == (63, 57)


def test_union_covers_a_gap_between_tiles(tmp_path):
    a = _tile(tmp_path / "a.tif", width=10, height=10)
    b = _tile(tmp_path / "b.tif", x=X0 + 15 * PIXEL, y=Y0 - 5 * PIXEL, width=10, height=10)
    grid = check_tiles([a, b])
    assert grid.transform == from_origin(X0, Y0, PIXEL, PIXEL)
    assert (grid.width, grid.height) == (25, 15)


def test_tiny_float_noise_in_origin_is_still_aligned(tmp_path):
    a = _tile(tmp_path / "a.tif", width=10, height=10)
    b = _tile(tmp_path / "b.tif", x=X0 + 10 * PIXEL + 1e-7, width=10, height=10)
    grid = check_tiles([a, b])
    assert (grid.width, grid.height) == (20, 10)


def test_single_tile_is_accepted(tmp_path):
    tile = _tile(tmp_path / "one.tif", width=23, height=17, nodata=None)
    grid = check_tiles([str(tile)])
    assert grid.transform == from_origin(X0, Y0, PIXEL, PIXEL)
    assert (grid.width, grid.height, grid.nodata) == (23, 17, None)


def test_tiles_without_nodata_give_a_grid_without_nodata(tmp_path):
    a = _tile(tmp_path / "a.tif", width=10, height=10, nodata=None)
    b = _tile(tmp_path / "b.tif", x=X0 + 10 * PIXEL, width=10, height=10, nodata=None)
    assert check_tiles([a, b]).nodata is None


def test_nan_nodata_on_every_tile_matches(tmp_path):
    a = _tile(tmp_path / "a.tif", width=10, height=10, dtype="float32", nodata=float("nan"))
    b = _tile(tmp_path / "b.tif", x=X0 + 10 * PIXEL, width=10, height=10,
              dtype="float32", nodata=float("nan"))
    assert math.isnan(check_tiles([a, b]).nodata)


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
    grid = check_tiles([a, b])
    assert (grid.width, grid.height) == (20, 10)


@pytest.mark.parametrize("as_type", [str, lambda path: path], ids=["str", "PathLike"])
def test_single_path_instead_of_a_list_is_refused(tmp_path, as_type):
    tile = _tile(tmp_path / "a.tif", width=10, height=10)
    with pytest.raises(ValueError, match="not a single path"):
        check_tiles(as_type(tile))


def test_empty_tile_list_is_refused():
    with pytest.raises(ValueError, match="no tiles"):
        check_tiles([])


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
        check_tiles(tiles)
    assert caught.value.tile == str(tiles[3])
    assert caught.value.field == field
    assert "se.tif" in str(caught.value) and field in str(caught.value)


def test_first_tile_without_nodata_and_second_with_is_refused(tmp_path):
    a = _tile(tmp_path / "a.tif", width=10, height=10, nodata=None)
    b = _tile(tmp_path / "b.tif", x=X0 + 10 * PIXEL, width=10, height=10, nodata=0)
    with pytest.raises(TileMismatchError) as caught:
        check_tiles([a, b])
    assert (caught.value.tile, caught.value.field) == (str(b), "nodata")


def test_nan_and_number_nodata_are_refused(tmp_path):
    a = _tile(tmp_path / "a.tif", width=10, height=10, dtype="float32", nodata=float("nan"))
    b = _tile(tmp_path / "b.tif", x=X0 + 10 * PIXEL, width=10, height=10,
              dtype="float32", nodata=-9999)
    with pytest.raises(TileMismatchError, match="nodata"):
        check_tiles([a, b])


def test_tile_without_crs_is_refused(tmp_path):
    a = _tile(tmp_path / "a.tif", width=10, height=10)
    b = _tile(tmp_path / "b.tif", x=X0 + 10 * PIXEL, width=10, height=10, crs=None)
    with pytest.raises(TileMismatchError) as caught:
        check_tiles([a, b])
    assert (caught.value.tile, caught.value.field) == (str(b), "crs")


def test_rotated_tile_is_refused(tmp_path):
    rotated = Affine(PIXEL, 1.0, X0, 0.0, -PIXEL, Y0)
    tile = _tile(tmp_path / "r.tif", width=10, height=10, transform=rotated)
    with pytest.raises(TileMismatchError) as caught:
        check_tiles([tile])
    assert caught.value.field == "orientation"


def test_only_headers_are_read(tmp_path, monkeypatch):
    tiles = _four_tiles(tmp_path)

    def no_read(*args, **kwargs):
        raise AssertionError("check_tiles read pixels")

    monkeypatch.setattr(DatasetReader, "read", no_read)
    check_tiles(tiles)


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


def _random_tiles(tmp_path, *, nodata=0, gap=0):
    """Four tiles of different sizes, none a whole number of 512 blocks, with values 0 to 22.

    0 is common in the data, so it is both nodata and a real value. `gap` leaves that
    many empty pixel columns between the west and east tiles.
    """
    rng = np.random.default_rng(7)
    sizes = {"nw": (530, 600), "ne": (530, 311), "sw": (257, 600), "se": (257, 311)}
    east_x = X0 + (600 + gap) * PIXEL
    south_y = Y0 - 530 * PIXEL
    origins = {"nw": (X0, Y0), "ne": (east_x, Y0), "sw": (X0, south_y), "se": (east_x, south_y)}
    tiles = []
    for name, (height, width) in sizes.items():
        data = rng.integers(0, 23, size=(1, height, width), dtype="uint8")
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

    def copy_then_fail(src, dst, **kwargs):
        real_copy(src, dst, **kwargs)
        raise rasterio.errors.RasterioIOError("disk full")

    monkeypatch.setattr(rasterio.shutil, "copy", copy_then_fail)
    with pytest.raises(rasterio.errors.RasterioIOError):
        merge_to_cog(tiles, out_dir / "merged.tif", "mode")
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


def test_relative_tile_paths_are_found(tmp_path, monkeypatch):
    tiles = _random_tiles(tmp_path)
    monkeypatch.chdir(tmp_path)
    output = merge_to_cog([t.name for t in tiles], "merged.tif", "mode")
    _assert_tiles_placed(tmp_path / "merged.tif", tiles)
    assert output == Path("merged.tif")


@pytest.fixture
def no_config_file(tmp_path, monkeypatch):
    """Run in a folder with no config.yaml and no GEO_CLOUDIFY_CONFIG, with fresh caches."""
    folder = tmp_path / "elsewhere"
    folder.mkdir()
    monkeypatch.chdir(folder)
    monkeypatch.delenv("GEO_CLOUDIFY_CONFIG", raising=False)
    get_settings.cache_clear()
    get_config.cache_clear()
    yield folder
    get_settings.cache_clear()
    get_config.cache_clear()


def test_merge_works_from_a_folder_without_a_config_file(tmp_path, no_config_file):
    tiles = _random_tiles(tmp_path)
    output = merge_to_cog(tiles, no_config_file / "merged.tif", "mode")
    _assert_tiles_placed(output, tiles)
    with rasterio.open(output) as out:
        assert out.compression.name == "deflate"
        assert out.block_shapes[0] == (512, 512)


def test_first_tile_colours_and_metadata_are_kept(tmp_path):
    tiles = _random_tiles(tmp_path)
    colormap = {0: (0, 0, 0, 0), 1: (10, 120, 30, 255), 22: (200, 0, 0, 255)}
    with rasterio.open(tiles[0], "r+") as dst:
        dst.write_colormap(1, colormap)
        dst.set_band_description(1, "land cover class")
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


def test_first_tile_scale_and_offset_are_kept(tmp_path):
    tiles = _random_tiles(tmp_path)
    with rasterio.open(tiles[0], "r+") as dst:
        dst.scales = (0.5,)
        dst.offsets = (-3.0,)
    output = merge_to_cog(tiles, tmp_path / "merged.tif", "average")
    with rasterio.open(output) as out:
        assert out.scales == (0.5,) and out.offsets == (-3.0,)


@pytest.mark.parametrize("tile, expected", [
    ("https://example.com/a/t.tif", "/vsicurl/https://example.com/a/t.tif"),
    ("http://example.com/t.tif", "/vsicurl/http://example.com/t.tif"),
    ("s3://bucket/key/t.tif", "/vsis3/bucket/key/t.tif"),
    ("gs://bucket/t.tif", "/vsigs/bucket/t.tif"),
    ("/vsicurl/https://example.com/t.tif", "/vsicurl/https://example.com/t.tif"),
    ("/vsimem/t.tif", "/vsimem/t.tif"),
    ("file:///data/t.tif", "/data/t.tif"),
])
def test_urls_become_gdal_paths(tile, expected):
    assert _gdal_path(tile) == expected


def test_relative_path_becomes_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _gdal_path("t.tif") == str(tmp_path / "t.tif")


def test_unsupported_url_scheme_is_refused():
    with pytest.raises(ValueError, match="ftpx"):
        _gdal_path("ftpx://example.com/t.tif")


def test_file_url_tiles_merge(tmp_path):
    tiles = _random_tiles(tmp_path)
    output = merge_to_cog([t.as_uri() for t in tiles], tmp_path / "merged.tif", "mode")
    _assert_tiles_placed(output, tiles)


def test_vsimem_tiles_merge(tmp_path):
    tiles = _random_tiles(tmp_path)
    mem_tiles = []
    for tile in tiles:
        mem = f"/vsimem/{tile.name}"
        rasterio.shutil.copy(str(tile), mem, driver="GTiff")
        mem_tiles.append(mem)
    try:
        output = merge_to_cog(mem_tiles, tmp_path / "merged.tif", "mode")
        _assert_tiles_placed(output, tiles)
    finally:
        for mem in mem_tiles:
            rasterio.shutil.delete(mem)


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

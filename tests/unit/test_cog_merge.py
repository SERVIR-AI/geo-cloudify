"""Unit tests for checking that tiles share one grid (check_tiles) before a merge."""

import math

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.io import DatasetReader
from rasterio.transform import Affine, from_origin

from geo_cloudify.cog.merge import TileGrid, TileMismatchError, check_tiles

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

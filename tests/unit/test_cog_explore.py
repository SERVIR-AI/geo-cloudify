"""Unit tests for COG exploration (detect, fingerprint, settings, explore)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from geo_cloudify.cog.explore import (
    _is_cube,
    _is_lossy,
    detect,
    explore,
    fingerprint,
    is_cog,
    resolve_settings,
)
from geo_cloudify.plan import Action, Kind, RasterFingerprint

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "data"


def _write_tif(path, *, count=1, dtype="uint8", crs="EPSG:4326", nodata=None, transform=None, width=2, height=3):
    transform = transform or from_origin(0, height, 1, 1)
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=count,
        dtype=dtype, crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(np.zeros((count, height, width), dtype=dtype))
    return path


def _fp(*, nodata=None, color_table=None, dtype="float32", crs=None):
    return RasterFingerprint(
        width=2, height=3, count=1, dtype=dtype,
        transform=[0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
        nodata=nodata, color_table=color_table, crs=crs,
    )


# --- detect ------------------------------------------------------------------

def test_detect_georeferenced_tif(tmp_path):
    assert detect(_write_tif(tmp_path / "a.tif")) is True


def test_detect_ungeoreferenced_raster(tmp_path):
    path = tmp_path / "plain.tif"
    with rasterio.open(path, "w", driver="GTiff", width=2, height=2, count=1, dtype="uint8") as dst:
        dst.write(np.zeros((1, 2, 2), dtype="uint8"))
    assert detect(path) is False


def test_detect_text_file(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("hello")
    assert detect(path) is False


def test_detect_single_grid_netcdf():
    path = FIXTURES / "netcdf_2d" / "biooracle_thetao_2010_costarica.nc"
    if not path.exists():
        pytest.skip("fixture data not present")
    assert detect(path) is True


# --- cube detection ----------------------------------------------------------

def test_is_cube_multiband_netcdf():
    assert _is_cube(SimpleNamespace(driver="netCDF", count=2, subdatasets=[])) is True


def test_is_cube_single_band_netcdf():
    assert _is_cube(SimpleNamespace(driver="netCDF", count=1, subdatasets=[])) is False


def test_is_cube_geotiff_multiband():
    assert _is_cube(SimpleNamespace(driver="GTiff", count=3, subdatasets=[])) is False


# --- is_cog ------------------------------------------------------------------

def test_is_cog_false_for_plain_tif(tmp_path):
    assert is_cog(_write_tif(tmp_path / "a.tif", width=600, height=600)) is False


def test_is_cog_true_for_cog_fixture():
    path = FIXTURES / "already_cog" / "Copernicus_DSM_COG_10_S03_00_W080_00_DEM.tif"
    if not path.exists():
        pytest.skip("fixture data not present")
    assert is_cog(path) is True


# --- fingerprint -------------------------------------------------------------

def test_fingerprint_records_metadata(tmp_path):
    path = _write_tif(tmp_path / "a.tif", dtype="uint8", nodata=0)
    fp = fingerprint(path)
    assert fp.width == 2
    assert fp.height == 3
    assert fp.count == 1
    assert fp.dtype == "uint8"
    assert fp.crs is not None
    assert fp.nodata == 0


# --- resolve_settings --------------------------------------------------------

def test_resolve_settings_defaults():
    settings = resolve_settings(_fp(), None)
    assert settings["COMPRESS"] == "DEFLATE"
    assert settings["BLOCKSIZE"] == 512
    assert settings["OVERVIEWS"] == "IGNORE_EXISTING"
    assert settings["RESAMPLING"] == "AVERAGE"
    assert settings["PREDICTOR"] == "YES"
    assert settings["NUM_THREADS"] == "ALL_CPUS"


def test_resolve_settings_color_table_uses_nearest():
    settings = resolve_settings(_fp(color_table=[[0, 0, 0, 0, 255]]), None)
    assert settings["RESAMPLING"] == "NEAREST"
    assert "PREDICTOR" not in settings


def test_resolve_settings_user_kwargs_win():
    settings = resolve_settings(_fp(), {"COMPRESS": "LZW", "BLOCKSIZE": 256})
    assert settings["COMPRESS"] == "LZW"
    assert settings["BLOCKSIZE"] == 256


def test_resolve_settings_lowercase_keys_are_accepted():
    assert resolve_settings(_fp(), {"nodata": -9999})["NODATA"] == -9999
    assert resolve_settings(_fp(), {"compress": "LZW"})["COMPRESS"] == "LZW"


def test_resolve_settings_integer_dtype_gets_average():
    settings = resolve_settings(_fp(dtype="uint8"), None)
    assert settings["RESAMPLING"] == "AVERAGE"
    assert settings["PREDICTOR"] == "YES"


def test_resolve_settings_keeps_crs():
    assert resolve_settings(_fp(crs="EPSG:4326"), None)["CRS"] == "EPSG:4326"


def test_resolve_settings_rejects_unknown_option():
    with pytest.raises(ValueError, match="unknown"):
        resolve_settings(_fp(), {"NOT_A_REAL_OPTION": "x"})


def test_resolve_settings_nodata_from_kwargs_wins():
    settings = resolve_settings(_fp(nodata=-9999), {"NODATA": -1})
    assert settings["NODATA"] == -1


def test_resolve_settings_nodata_from_file():
    assert resolve_settings(_fp(nodata=-9999), None)["NODATA"] == -9999


def test_resolve_settings_nodata_unset():
    assert "NODATA" not in resolve_settings(_fp(nodata=None), None)


# --- lossy -------------------------------------------------------------------

def test_lossy_jpeg_and_webp():
    assert _is_lossy({"COMPRESS": "JPEG"}) is True
    assert _is_lossy({"COMPRESS": "WEBP"}) is True
    assert _is_lossy({"COMPRESS": "DEFLATE"}) is False


def test_lossy_lerc_max_z_error():
    assert _is_lossy({"COMPRESS": "LERC", "MAX_Z_ERROR": 0}) is False
    assert _is_lossy({"COMPRESS": "LERC", "MAX_Z_ERROR": 0.5}) is True


def test_lossy_lerc_variants():
    assert _is_lossy({"COMPRESS": "LERC_DEFLATE", "MAX_Z_ERROR": 0.5}) is True
    assert _is_lossy({"COMPRESS": "LERC_ZSTD", "MAX_Z_ERROR": 0.5}) is True
    assert _is_lossy({"COMPRESS": "LERC_DEFLATE", "MAX_Z_ERROR": 0}) is False


def test_lossy_tiling_scheme():
    assert _is_lossy({"TILING_SCHEME": "GoogleMapsCompatible"}) is True


def test_lossy_unknown_max_z_error_is_lossy():
    assert _is_lossy({"COMPRESS": "LERC", "MAX_Z_ERROR": "not-a-number"}) is True


# --- explore -----------------------------------------------------------------

def test_explore_plain_tif_converts(tmp_path):
    main = _write_tif(tmp_path / "a.tif", width=600, height=600, dtype="float32")
    item = explore(main, [], "src", None)
    assert item.kind is Kind.RASTER
    assert item.action is Action.CONVERT
    assert item.files == [str(main)]
    assert item.lossy is False
    assert "no nodata value" in item.notes[0]
    assert item.settings["RESAMPLING"] == "AVERAGE"


def test_explore_already_cog_passes_through():
    main = FIXTURES / "already_cog" / "Copernicus_DSM_COG_10_S03_00_W080_00_DEM.tif"
    if not main.exists():
        pytest.skip("fixture data not present")
    item = explore(main, [], "src", None)
    assert item.action is Action.PASS_THROUGH
    assert item.fingerprint is not None


def test_explore_tiled_not_cog_uses_nearest():
    main = FIXTURES / "tiled_geotiff_not_cog" / "GMW_N10E106_2020_v3.tif"
    if not main.exists():
        pytest.skip("fixture data not present")
    item = explore(main, [], "src", None)
    assert item.action is Action.CONVERT
    assert item.settings["RESAMPLING"] == "NEAREST"

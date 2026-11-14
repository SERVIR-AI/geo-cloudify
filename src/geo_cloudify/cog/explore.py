"""COG exploration: detect rasters, fingerprint them, resolve COG settings.

It reads metadata only and makes every decision. Rasters that are already
valid COGs are passed through untouched.
"""

import logging
from pathlib import Path

import rasterio
from rasterio.crs import CRS
from rio_cogeo.cogeo import cog_validate

from geo_cloudify.config import get_config
from geo_cloudify.plan import Action, Kind, PlanItem, RasterFingerprint

logger = logging.getLogger(__name__)

EPSG_4326_WKT = CRS.from_epsg(4326).to_wkt()

# GDAL COG driver creation options a caller may set, plus NODATA (handled separately,
# not a driver option). NUM_THREADS is not one: it comes from config.
COG_DRIVER_OPTIONS = frozenset({
    "BLOCKSIZE", "COMPRESS", "LEVEL", "PREDICTOR",
    "RESAMPLING", "OVERVIEWS", "OVERVIEW_COUNT", "OVERVIEW_RESAMPLING",
    "BIGTIFF", "QUALITY", "MAX_Z_ERROR", "IN_MEMORY", "SPARSE_OK",
    "ADD_ALPHA", "ALIGNED_LEVELS", "TILING_SCHEME", "WARP_RESAMPLING",
    "WARP_MEMORY_LIMIT",
})
ALLOWED_KWARGS = COG_DRIVER_OPTIONS | {"NODATA"}

# COMPRESS values listed in the GDAL COG driver documentation.
COG_COMPRESS = frozenset(
    {"NONE", "LZW", "JPEG", "DEFLATE", "ZSTD", "WEBP", "LERC", "LERC_DEFLATE", "LERC_ZSTD", "LZMA"}
)
# LEVEL range per codec; any codec not listed uses the widest range.
LEVEL_RANGES = {"DEFLATE": (1, 12), "LERC_DEFLATE": (1, 12), "ZSTD": (1, 22), "LERC_ZSTD": (1, 22), "LZMA": (0, 9)}

CUBE_DRIVERS = frozenset({"netcdf", "hdf5", "hdf5image", "hdf4"})
DEFAULT_TRANSFORM = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def detect(path: Path) -> bool:
    """Return True if the file is a COG-bound raster: georeferenced and single-grid.

    Returns False if the file cannot be opened as a raster, is not georeferenced
    (no CRS and the default transform), or is a NetCDF/HDF5 cube (a time or depth
    dimension). Cubes are skipped here.
    """
    try:
        with rasterio.open(path) as ds:
            if ds.count < 1:
                return False
            if ds.crs is None and _is_default_transform(ds.transform):
                return False
            if _is_cube(ds):
                return False
            return True
    except rasterio.errors.RasterioError:
        return False


def is_cog(path: Path) -> bool:
    """Return True if the file already validates as a COG (no errors)."""
    _, errors, _ = cog_validate(str(path))
    return not errors


def fingerprint(path: Path) -> RasterFingerprint:
    """Return the raster's metadata fingerprint. Reads metadata only, no pixels."""
    with rasterio.open(path) as ds:
        return RasterFingerprint(
            width=ds.width,
            height=ds.height,
            count=ds.count,
            dtype=ds.dtypes[0],
            transform=list(ds.transform.to_gdal()),
            crs=ds.crs.to_wkt() if ds.crs else None,
            nodata=ds.nodata,
            color_table=_color_table(ds),
            scales=list(ds.scales),
            offsets=list(ds.offsets),
            units=list(ds.units),
        )


def resolve_settings(fingerprint: RasterFingerprint, cog_kwargs: dict | None) -> dict:
    """Return the resolved conversion settings: COG driver options, NODATA and CRS.

    Resolution order: defaults, then data-derived values, then user kwargs.
    User kwargs always win. Option names are case-insensitive (uppercased).

    Raises:
        ValueError: if cog_kwargs contains an option the COG driver does not know.
    """
    # Key is capitalized to handle the case where it is all lowercase/first letter capitalized
    kwargs = {str(key).upper(): value for key, value in (cog_kwargs or {}).items()}
    validate_cog_kwargs(kwargs)
    nodata = kwargs.pop("NODATA", None)
    if nodata is None:
        nodata = fingerprint.nodata

    # COG driver creation options (GDAL accepts these keys).
    settings = {
        "COMPRESS": get_config().cog.compress,    # codec for the tiles, e.g. DEFLATE
        "BLOCKSIZE": get_config().cog.blocksize,  # tile size in pixels; 512 is the COG convention
        # Drop any overviews the source already has and rebuild them in COG
        # order, because existing ones are often in the wrong order.
        "OVERVIEWS": "IGNORE_EXISTING",
        "NUM_THREADS": get_config().cog.num_threads,  # set by the service, not the caller
    }
    # How the zoomed-out overview levels are built: nearest keeps class values
    # intact (colour table), average suits continuous data.
    if fingerprint.color_table:
        settings["RESAMPLING"] = "NEAREST"
    else:
        settings["RESAMPLING"] = "AVERAGE"
        settings["PREDICTOR"] = "YES"  # lossless differencing, shrinks the tiles

    settings.update(kwargs)  # user kwargs always win
    # NODATA and CRS are not COG driver options; they are instructions for the
    # convert step, which sets them on the output band and dataset.
    if nodata is not None:
        settings["NODATA"] = nodata
    if fingerprint.crs is not None:
        settings["CRS"] = fingerprint.crs
    return settings


def explore(main: Path, companions: list[Path], source: str, cog_kwargs: dict | None) -> PlanItem:
    """Decide what happens to one raster and return its plan entry.

    Call this only for files where `detect()` returned True. The result is one of:

    - PASS_THROUGH: the file is already a valid COG. It is uploaded as is, with
      no settings, conversion or verification.
    - CONVERT: the file needs converting. The entry holds the resolved COG
      settings (including NODATA and CRS for convert to apply), the source
      fingerprint for verify, the `lossy` flag, and notes for the caller.

    Two cases are noted on the entry and logged as warnings, never guessed:
    a NetCDF with no CRS (assumed EPSG:4326) and a raster with no nodata value
    (left unset; the caller can set `nodata` in `cog_kwargs`).

    Args:
        main: the raster file to explore.
        companions: files that belong with it (for example `.tfw`, `.aux.xml`).
        source: the source string from the request that produced this file.
        cog_kwargs: the request's COG options, or None.

    Returns:
        The PlanItem for this raster.

    Raises:
        ValueError: if cog_kwargs contains an option the COG driver does not know.
    """
    files = [str(main)] + [str(p) for p in companions]
    # Metadata only: no pixels are read, so this stays quick on large files.
    fp = fingerprint(main)

    # Already a COG: nothing to convert, so no settings and no verify.
    if is_cog(main):
        return PlanItem(
            source=source,
            kind=Kind.RASTER,
            action=Action.PASS_THROUGH,
            files=files,
            fingerprint=fp,
            notes=["already a valid COG"],
        )

    notes: list[str] = []
    # A COG needs a CRS. NetCDF climate and ocean data without one is almost
    # always plain latitude/longitude, so assume EPSG:4326 and say so.
    # Setting it on the fingerprint means settings["CRS"] tells convert to assign it.
    if fp.crs is None and _driver(main) == "netcdf":
        fp.crs = EPSG_4326_WKT
        notes.append("no CRS; assumed EPSG:4326")
        logger.warning("source %s has no CRS; assumed EPSG:4326", source)

    # Defaults, then values from the data, then the request's cog_kwargs (which win).
    settings = resolve_settings(fp, cog_kwargs)

    # No nodata from the request or the file. Never guess one (a guess could mark
    # real values as empty); tell the caller instead.
    if "NODATA" not in settings:
        notes.append("no nodata value (not declared and not requested); set `nodata` in `cog_kwargs` if needed")
        logger.warning("source %s has no nodata value", source)

    return PlanItem(
        source=source,
        kind=Kind.RASTER,
        action=Action.CONVERT,
        files=files,
        settings=settings,
        fingerprint=fp,
        # True when the settings change the data on purpose (lossy compression or
        # reprojection); verify then reports "lossy by request" instead of checking pixels.
        lossy=_is_lossy(settings),
        notes=notes,
    )


def _driver(path: Path) -> str:
    """Return the GDAL driver name for the file, in lowercase."""
    with rasterio.open(path) as ds:
        return (ds.driver or "").lower()


def _is_default_transform(transform) -> bool:
    """Return True if the transform is GDAL's default (identity) geotransform."""
    return transform.to_gdal() == DEFAULT_TRANSFORM


def _is_cube(ds) -> bool:
    """Return True for a NetCDF/HDF5 file with more than one band or subdatasets."""
    if (ds.driver or "").lower() not in CUBE_DRIVERS:
        return False
    return ds.count > 1 or bool(ds.subdatasets)


def _color_table(ds) -> list[list[int]] | None:
    """Return the colour table as rows of [index, r, g, b, a], or None."""
    try:
        colormap = ds.colormap(1)
    except ValueError:
        return None
    if not colormap:
        return None
    rows = []
    for index, rgba in sorted(colormap.items()):
        r, g, b = rgba[0], rgba[1], rgba[2]
        alpha = rgba[3] if len(rgba) > 3 else 255
        rows.append([index, r, g, b, alpha])
    return rows


def validate_cog_kwargs(cog_kwargs: dict) -> None:
    """Check cog_kwargs names (case-insensitive) and values. Returns None.

    Raises:
        ValueError: if a key is not an option the COG driver knows, or a value
            fails `check_cog_values`.
    """
    unknown = {str(key).upper() for key in cog_kwargs} - ALLOWED_KWARGS
    if unknown:
        raise ValueError(f"unknown cog_kwargs option(s): {', '.join(sorted(unknown))}")
    check_cog_values({str(key).upper(): value for key, value in cog_kwargs.items()})


def check_cog_values(settings: dict) -> None:
    """Check COG option values before GDAL sees them. Keys must be uppercase. Returns None.

    GDAL does not reject every bad value: it reads a BLOCKSIZE of "big" as 0 and
    can then try to allocate huge buffers, and it silently replaces an unknown
    COMPRESS. Used on arrival (`validate_cog_kwargs`) and by `cog.convert.to_cog`.

    Raises:
        ValueError: if a value is out of range or not a number where one is needed.
    """
    if "BLOCKSIZE" in settings:
        size = _as_int(settings["BLOCKSIZE"])
        if size is None or not (16 <= size <= 4096 and size & (size - 1) == 0):
            raise ValueError(f"BLOCKSIZE must be a power of two from 16 to 4096, got {settings['BLOCKSIZE']!r}")
    if "NODATA" in settings and _as_float(settings["NODATA"]) is None:
        raise ValueError(f"NODATA must be a number, got {settings['NODATA']!r}")
    # Without COMPRESS, conversion uses the configured codec, so LEVEL is checked against it.
    compress = str(settings.get("COMPRESS", get_config().cog.compress)).upper()
    if "COMPRESS" in settings and compress not in COG_COMPRESS:
        raise ValueError(f"COMPRESS must be one of {', '.join(sorted(COG_COMPRESS))}, got {settings['COMPRESS']!r}")
    if "OVERVIEW_COUNT" in settings:
        count = _as_int(settings["OVERVIEW_COUNT"])
        if count is None or not 0 <= count <= 16:
            raise ValueError(f"OVERVIEW_COUNT must be a whole number from 0 to 16, got {settings['OVERVIEW_COUNT']!r}")
    if "NUM_THREADS" in settings and str(settings["NUM_THREADS"]).upper() != "ALL_CPUS":
        threads = _as_int(settings["NUM_THREADS"])
        if threads is None or threads < 1:
            raise ValueError(f"NUM_THREADS must be ALL_CPUS or a whole number of 1 or more, got {settings['NUM_THREADS']!r}")
    if "LEVEL" in settings:
        low, high = LEVEL_RANGES.get(compress, (0, 22))
        level = _as_int(settings["LEVEL"])
        if level is None or not low <= level <= high:
            raise ValueError(f"LEVEL must be from {low} to {high} for {compress}, got {settings['LEVEL']!r}")
    if "QUALITY" in settings:
        quality = _as_int(settings["QUALITY"])
        if quality is None or not 1 <= quality <= 100:
            raise ValueError(f"QUALITY must be from 1 to 100, got {settings['QUALITY']!r}")
    if "MAX_Z_ERROR" in settings:
        error = _as_float(settings["MAX_Z_ERROR"])
        if error is None or error < 0:
            raise ValueError(f"MAX_Z_ERROR must be a number of 0 or more, got {settings['MAX_Z_ERROR']!r}")


def _as_int(value) -> int | None:
    """Return value as a whole number, or None if it is not one."""
    if isinstance(value, bool):
        return None
    try:
        number = float(str(value))
    except ValueError:
        return None
    return int(number) if number.is_integer() else None


def _as_float(value) -> float | None:
    """Return value as a number, or None if it is not one."""
    if isinstance(value, bool):
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


def _is_lossy(settings: dict) -> bool:
    """Return True if the resolved settings use lossy compression or reprojection."""
    compress = str(settings.get("COMPRESS", "")).upper()
    if compress in ("JPEG", "WEBP"):
        return True
    if settings.get("TILING_SCHEME"):
        return True
    if compress in ("LERC", "LERC_DEFLATE", "LERC_ZSTD"):
        try:
            return float(settings.get("MAX_Z_ERROR", 0)) > 0
        except (TypeError, ValueError):
            return True  # can't read the error bound; assume lossy
    return False

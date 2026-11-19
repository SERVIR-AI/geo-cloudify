"""COG conversion: write one raster as one Cloud Optimized GeoTIFF.

GDAL's COG driver copies the source's pixels, CRS, transform, nodata and colour
table unchanged and builds fresh overviews. Only lossless codecs are accepted,
so the full-resolution pixels are always identical to the source. No config
file is read: the settings are the arguments.
"""

import logging
from os import PathLike
from pathlib import Path

import rasterio
import rasterio.shutil

from geo_cloudify.cog.merge import OVERVIEW_METHODS, validate_cog_settings

logger = logging.getLogger(__name__)


def to_cog(
    source: str | PathLike,
    output: str | PathLike,
    overview_resampling: str,
    *,
    compression_codec: str = "DEFLATE",
    blocksize: int = 512,
) -> Path:
    """Convert one raster to a COG and return the output path.

    The output keeps the source's CRS, transform, data type, nodata (none if the
    source has none) and colour table. Overviews already in the source are not
    reused; new ones are built with `overview_resampling`. A source without a
    colour table also gets `PREDICTOR=YES`, a lossless step that makes the
    tiles smaller.

    Args:
        source: the raster to convert, as a local path or a GDAL-readable path.
        output: where to write the COG. Its folder is created if missing.
        overview_resampling: "nearest" or "mode" for class data, "average" for
            continuous data. Lower case, nothing else is accepted.
        compression_codec: a lossless codec: "DEFLATE" (default), "LZW", "ZSTD",
            "LZMA" or "NONE", in any case.
        blocksize: the COG's block size in pixels, an integer (`int` or a
            numpy integer) that is a multiple of 16 from 16 to 4096 (default 512).

    Returns:
        The output path, as given, as a Path.

    Raises:
        ValueError: if `overview_resampling` is not one of the allowed values,
            `compression_codec` is lossy or unknown, or `blocksize` is not allowed.
            These are checked before GDAL runs.
        FileExistsError: if the output file already exists.
        rasterio.errors.RasterioError: if GDAL cannot read the source or write
            the output. A failed write leaves no output behind.
    """
    # Check the arguments before touching any file.
    if overview_resampling not in OVERVIEW_METHODS:
        raise ValueError(
            f"overview_resampling must be one of {', '.join(OVERVIEW_METHODS)}, got {overview_resampling!r}"
        )
    codec, blocksize = validate_cog_settings(compression_codec, blocksize)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    cog_creation_options = {
        "COMPRESS": codec,
        "BLOCKSIZE": blocksize,
        "NUM_THREADS": "ALL_CPUS",
        "OVERVIEW_RESAMPLING": overview_resampling.upper(),
        "OVERVIEWS": "IGNORE_EXISTING",  # the source's own overviews may use another method
    }
    with rasterio.open(source) as dataset:
        has_colour_table = dataset.colorinterp[0].name == "palette"
    # Differencing would mix palette indexes, so it is only used without a colour table.
    if not has_colour_table:
        cog_creation_options["PREDICTOR"] = "YES"
    logger.info(
        "converting %s to %s: compression_codec=%s blocksize=%d overview_resampling=%s",
        source, output, codec, blocksize, overview_resampling,
    )
    try:
        rasterio.shutil.copy(str(source), str(output), driver="COG", **cog_creation_options)
    except Exception:
        output.unlink(missing_ok=True)  # never leave a half-written COG behind
        raise
    return output

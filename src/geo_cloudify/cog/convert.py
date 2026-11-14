"""COG conversion: write one plan entry as one Cloud Optimized GeoTIFF.

Follows the plan entry built by `cog/explore.py` and makes no decisions of its
own. The source is first copied to a small VRT (a text file that points at the
source pixels), nodata and CRS are set on the VRT, and GDAL's COG driver writes
the output from it. Pixels are never rewritten in between, so the copy is
lossless unless the settings ask for lossy compression.
"""

import tempfile
from pathlib import Path

import rasterio
import rasterio.shutil
from rasterio.crs import CRS

from geo_cloudify.cog.explore import check_cog_values
from geo_cloudify.plan import Action, PlanItem

# Values in item.settings that are instructions for this step, not GDAL options.
_INSTRUCTIONS = ("NODATA", "CRS")


def output_path(main: Path, job_dir: Path, out_dir: Path) -> Path:
    """Return where the COG for `main` goes: its path under `job_dir`, moved to `out_dir`.

    A `.tif` or `.tiff` name is kept; any other name gets `.tif` added
    (`a.nc` becomes `a.nc.tif`), so `a.tif` and `a.nc` never share an output.

    Raises:
        ValueError: if `main` is not inside `job_dir`.
    """
    relative = main.resolve().relative_to(job_dir.resolve())
    if relative.suffix.lower() not in (".tif", ".tiff"):
        relative = relative.with_name(relative.name + ".tif")
    return out_dir / relative


def to_cog(item: PlanItem, job_dir: Path, out_dir: Path) -> Path:
    """Convert one CONVERT plan entry to a COG and return the output path.

    Raises:
        ValueError: if the item is not a CONVERT item, its file is not inside `job_dir`,
            or a setting value fails `cog.explore.check_cog_values` (for example
            `BLOCKSIZE: "big"`). Values are checked before GDAL runs.
        FileExistsError: if the output file already exists.
        rasterio.errors.RasterioError: if GDAL fails to read the source or write the output.
    """
    if item.action is not Action.CONVERT:
        raise ValueError(f"to_cog needs a convert item, got {item.action.value}")
    check_cog_values(item.settings)
    main = Path(item.files[0])
    output = output_path(main, job_dir, out_dir)
    if output.exists():
        raise FileExistsError(f"output already exists: {output.name}")
    output.parent.mkdir(parents=True, exist_ok=True)

    options = {key: value for key, value in item.settings.items() if key not in _INSTRUCTIONS}
    with tempfile.TemporaryDirectory() as tmp:
        vrt = Path(tmp) / "source.vrt"
        rasterio.shutil.copy(str(main), str(vrt), driver="VRT")
        with rasterio.open(vrt, "r+") as dataset:
            if "NODATA" in item.settings:
                dataset.nodata = float(item.settings["NODATA"])
            if "CRS" in item.settings:
                dataset.crs = CRS.from_user_input(item.settings["CRS"])
        try:
            rasterio.shutil.copy(str(vrt), str(output), driver="COG", **options)
        except Exception:
            output.unlink(missing_ok=True)  # never leave a half-written COG behind
            raise
    return output

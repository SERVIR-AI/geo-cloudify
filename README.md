# geo-cloudify

Converts geospatial files to cloud-optimized formats. Rasters become Cloud Optimized GeoTIFFs (COGs). The import name is `geo_cloudify`.

## Install

- Library only (COG conversion, no web server): `uv sync --extra cog`, or `pip install "geo-cloudify[cog] @ git+https://github.com/SERVIR-AI/geo-cloudify.git"`.
- With the FastAPI service: `uv sync --extra cog --extra api`. Everything: `uv sync --all-extras`.
- Start the service: `uv run uvicorn geo_cloudify.api:app`.

## Which rasters are converted, passed through or skipped

Each downloaded file is checked in this order. The first rule that matches decides what happens to it.

| Check | Result | Examples |
|---|---|---|
| The file cannot be opened as a raster by GDAL | Skipped | Text files, Shapefiles, GeoJSON, a world file whose raster is missing |
| The raster has no location: no CRS and no geotransform | Skipped | A PNG preview image; a raster located only by ground control points (GCPs) |
| The file is NetCDF or HDF with more than one band, or with subdatasets | Skipped (a data cube, with a time or depth dimension) | A NetCDF with several time steps |
| The file already passes COG validation (`rio_cogeo.cog_validate`, no errors) | Passed through unchanged | ESA WorldCover and Copernicus DEM tiles; any raster that fits in one 512x512 block |
| Anything else | Converted to a COG | Striped or untiled GeoTIFFs, tiled GeoTIFFs in the wrong order, ERDAS `.img`, a NetCDF with a single 2D grid |

Multi-band GeoTIFFs are converted like any other raster. Only NetCDF and HDF files are treated as cubes when they have more than one band, because GDAL reads each time or depth step as a band.

A skipped file carries one reason in the job plan:

- A file GDAL cannot open as a raster gets `not a raster GDAL can open (a vector, a text file, or a damaged file; vectors come in a later version)`. GDAL uses one sentence for all of these, so the reason names the likely causes rather than repeating it.
- A file GDAL opens but that is not a COG-bound raster gets `not a COG-bound raster (not georeferenced, or a NetCDF/HDF cube)`. `detect()` answers yes or no, not why, so the reason names both causes.
- A file that fails later, after GDAL accepted it, gets `could not read: <GDAL's message>`.

Skip reasons never contain the server's folder path. Skips are summarised in one log line per source; only a failure after GDAL accepted a file is logged as a warning.

Companion files travel with their raster: the files GDAL reports it opened (for example `.ovr`, `.msk`, `.aux.xml`, `.rrd`) and same-stem world files (`.tfw`, `.tifw`, `.wld`), as long as they came from the same source. The raster and its companions are one plan entry. Any file that is neither a raster nor a companion of one, such as a world file whose raster is missing, is skipped with a reason.

## What a conversion keeps and changes

- **CRS**: the source CRS is kept. A NetCDF with no CRS is assumed to be EPSG:4326; this is logged and noted on the job's plan.
- **Nodata**: taken from `cog_kwargs` if given, otherwise from the file. If neither has one, it is left unset, and the plan notes that it can be set with `nodata` in `cog_kwargs`. The service never guesses a value (for example, an undeclared -9999).
- **Overviews**: rebuilt. Rasters with a colour table use nearest resampling; all others use average. Pass `RESAMPLING=NEAREST` for class maps without a colour table.
- **Defaults** (from `config.yaml`): `COMPRESS=DEFLATE`, `BLOCKSIZE=512`. `PREDICTOR=YES` is added for rasters without a colour table.

## `cog_kwargs`

Set per source in the `POST /jobs` body. A source is either a plain path or URL (no options) or an object with `source` and `cog_kwargs`:

```json
{"sources": [
  {"source": "https://example.com/tmin.tif", "cog_kwargs": {"COMPRESS": "ZSTD", "nodata": -9999}},
  "/data/thetao.nc"
]}
```

- A source's `cog_kwargs` applies only to the files that source produced (every file from a zip or folder shares it). Other sources keep their own settings, including their declared nodata.

- Any GDAL COG driver creation option except `NUM_THREADS`, plus `NODATA`. Names are case-insensitive. An unknown name is rejected with HTTP 422 and no job is created.
- `NUM_THREADS` is set by the service (`cog.num_threads` in `config.yaml`, default `ALL_CPUS`) and is not accepted in requests.
- Values are checked on arrival too. A bad value is rejected with HTTP 422:

  | Option | Allowed values |
  |---|---|
  | `COMPRESS` | `NONE`, `LZW`, `JPEG`, `DEFLATE`, `ZSTD`, `WEBP`, `LERC`, `LERC_DEFLATE`, `LERC_ZSTD`, `LZMA` |
  | `BLOCKSIZE` | a power of two from 16 to 4096 |
  | `NODATA` | a number |
  | `OVERVIEW_COUNT` | a whole number from 0 to 16 |
  | `LEVEL` | DEFLATE and LERC_DEFLATE 1-12, ZSTD and LERC_ZSTD 1-22, LZMA 0-9, others 0-22. Without `COMPRESS`, checked against the default codec |
  | `QUALITY` | 1 to 100 |
  | `MAX_Z_ERROR` | a number of 0 or more |

- Values in `cog_kwargs` override the defaults and the values derived from the data.
- These settings mark a conversion as lossy (reported as "lossy by request", not checked pixel by pixel):
  - `COMPRESS=JPEG` or `COMPRESS=WEBP`
  - `COMPRESS=LERC`, `LERC_DEFLATE` or `LERC_ZSTD` with `MAX_Z_ERROR` above 0
  - `TILING_SCHEME` (reprojects the raster)

## Job status

`GET /jobs/{id}` returns `status`, `error` (the reason when the whole job fails), per-source `results`, and `plan` (one item per file: convert, pass through or skip, with the resolved settings). NaN and infinity appear as the strings `"NaN"` and `"Infinity"`.

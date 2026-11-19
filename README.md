# geo-cloudify

Converts geospatial files to cloud-optimized formats and checks the result. Rasters become Cloud Optimized GeoTIFFs (COGs). The import name is `geo_cloudify`.

## Install

- With uv: `uv sync --extra cog`.
- With pip: `pip install "geo-cloudify[cog] @ git+https://github.com/SERVIR-AI/geo-cloudify.git"`.

The functions below read no config file and no `.env`; their settings are their arguments. Each logs one info line with the settings used, and GDAL uses all CPUs (`NUM_THREADS=ALL_CPUS`).

## Convert one raster to a COG

```python
from geo_cloudify.cog.convert import to_cog

to_cog("input.tif", "output.tif", "average")
```

`to_cog(source, output, overview_resampling, *, compression_codec="DEFLATE", blocksize=512)` returns the output path.

- `source`: a local path or a GDAL-readable path.
- `overview_resampling`: `nearest` or `mode` for class data, `average` for continuous data. Nothing else is accepted.
- `compression_codec`: a lossless codec, `DEFLATE` (default), `LZW`, `ZSTD`, `LZMA` or `NONE`, in any case. Lossy codecs (`JPEG`, `WEBP`, `LERC`, `LERC_DEFLATE`, `LERC_ZSTD`) are refused with a `ValueError`.
- `blocksize`: the COG block size in pixels, an integer (`int` or a numpy integer) that is a multiple of 16 from 16 to 4096 (default 512). Strings, floats and bools are refused.
- Keeps the source's CRS, transform, data type, nodata and colour table. It never sets a nodata the source does not have.
- Builds overviews fresh with the given method; overviews already in the source are not reused. Adds `PREDICTOR=YES` when the source has no colour table.
- Refuses to overwrite an existing output. A failed write leaves no output.
- Raises `ValueError` for a bad argument before GDAL runs, and `FileExistsError` when the output exists.

## Merge tiles into one COG

Large exports come as several tiles on one grid. `merge_to_cog` merges them into one COG, and `check_merged_cog` checks it.

```python
from geo_cloudify.cog.merge import merge_to_cog
from geo_cloudify.cog.verify import check_merged_cog

tiles = ["tile_0.tif", "tile_1.tif", "tile_2.tif", "tile_3.tif"]
merge_to_cog(tiles, "merged.tif", "mode")
result = check_merged_cog(tiles, "merged.tif")
if not result.passed:
    for problem in result.problems:
        print(problem.field, problem.detail)
```

`merge_to_cog(tiles, output, overview_resampling, *, compression_codec="DEFLATE", blocksize=512)` returns the output path.

- `tiles`: a list or tuple of local paths, GDAL `/vsi...` paths, or URLs (`http(s)`, `ftp`, `s3`, `gs`, `az`, `oss`, `file`). Reading from cloud storage needs that provider's credentials set up for GDAL. A single path instead of a list is refused.
- `overview_resampling`, `compression_codec` and `blocksize`: as for `to_cog`.
- Refuses tiles that differ in CRS, pixel size, grid alignment (an origin offset that is not a whole number of pixels), data type, band count or nodata, and tiles that overlap. Also refuses a tile that marks empty pixels with a mask (such as an internal GeoTIFF mask) instead of a nodata value or an alpha band. The error (`TileMismatchError`) names the tile and the field (`mask` for a mask).
- Places tiles pixel for pixel, never resampled. The output covers the union of the tiles; gaps are filled with the nodata value. When the tiles have no nodata, gaps are filled with 0 and a warning gives the number of pixels filled, since they cannot be told apart from real 0 values. Tiles with an alpha band get no warning: the alpha band is 0 there, so gaps are transparent.
- Keeps the tiles' CRS, grid, data type and nodata. It never sets a nodata the tiles do not have.
- Writes a tiled, compressed COG with overviews built fresh by the given method. Band statistics (`STATISTICS_*`) are computed from the tiles' pixels; the tiles' own statistics are not copied. Gaps count as not valid when the tiles have a nodata; when they have none, the 0 fill is left out of the statistics. Colour table, band descriptions, scale, offset and other metadata come from the first tile.
- Refuses to overwrite an existing output. A failed write leaves no output and no temporary files. Nothing is written next to the tiles (no `.aux.xml`). Other threads still read `.aux.xml` files while a merge runs.
- Works in blocks, so a whole raster is never held in memory.

`check_merged_cog(tiles, output)` returns a `MergeCheck` with every problem found (`passed` is true when there are none). It checks that the output:

- is a valid COG (`rio_cogeo.cog_validate`; errors and warnings both fail);
- has overviews down to one block, or none when it fits in one block;
- has the tiles' CRS, transform, size, band count, data type and nodata;
- comes from tiles that do not overlap;
- holds, per band, the same count of each pixel value as the tiles together (plus the gap fill). This catches dropped, doubled or changed tiles. Counts are made at full resolution, in windows.

Pixels are verified by these value counts only, which assumes class data. When a band holds more than 100,000 distinct values, the count check is skipped: no problem is added, the other checks still run, and `counts_checked` is false. When a band goes over the limit in only one of the tiles or the output, that is a "pixel counts" problem. A caller that needs the count proof reads `counts_checked` as well as `passed`.

`check_pixels(tiles, output)` (in `geo_cloudify.cog.verify`) compares every pixel of the output with the tiles, for any data type, in windows. It returns one `pixels` problem per band that differs, with the first differing places. `check_merged_cog` does not run it.

It raises `TileMismatchError` when the tiles do not share one grid, and `ValueError` for complex data or when `tiles` is empty or a single path, instead of returning a problem.

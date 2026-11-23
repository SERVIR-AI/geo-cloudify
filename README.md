# geo-cloudify

Converts geospatial files to cloud-optimized formats and checks the result. Rasters become Cloud Optimized GeoTIFFs (COGs). The import name is `geo_cloudify`.

## Install

- With uv: `uv sync`.
- With pip: `pip install "geo-cloudify @ git+https://github.com/SERVIR-AI/geo-cloudify.git"`.

The functions below read no config file and no `.env`; their settings are their arguments. Each logs one info line with the settings used, and GDAL uses all CPUs (`NUM_THREADS=ALL_CPUS`).

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
- `overview_resampling`: `nearest` or `mode` for class data, `average` for continuous data. Nothing else is accepted.
- `compression_codec`: a lossless codec, `DEFLATE` (default), `LZW`, `ZSTD`, `LZMA` or `NONE`, in any case. Lossy codecs (`JPEG`, `WEBP`, `LERC`, `LERC_DEFLATE`, `LERC_ZSTD`) are refused with an `InvalidInputError`.
- `blocksize`: the COG block size in pixels, an integer (`int` or a numpy integer) that is a power of two from 16 to 4096 (default 512). Strings, floats and bools are refused.
- Refuses tiles that differ in CRS, pixel size, grid alignment (an origin offset that is not a whole number of pixels), data type, band count or nodata, and tiles that overlap. Also refuses a tile that marks empty pixels with a mask (such as an internal GeoTIFF mask) instead of a nodata value or an alpha band. Also refuses a tile whose band scale, offset, units, description, colour interpretation or colour table differs from the first tile's. The error (`TileMismatchError`) names the tile and the field (`mask` for a mask).
- Places tiles pixel for pixel, never resampled. The output covers the union of the tiles; gaps are filled with the nodata value. When the tiles leave gaps and have no nodata, the merge is refused with an `InvalidInputError` that gives the number of uncovered pixels, even when the tiles have an alpha band.
- Keeps the tiles' CRS, grid, data type and nodata. It never sets a nodata the tiles do not have.
- Writes a tiled, compressed COG with overviews built fresh by the given method. Band statistics (`STATISTICS_*`) are computed from the tiles' pixels; the tiles' own statistics are not copied. Gaps count as not valid. Colour table, band descriptions, scale, offset, units and other metadata come from the first tile; metadata tags of the other tiles are not compared.
- Refuses to overwrite an existing output. A failed write leaves no output and no temporary files. Nothing is written next to the tiles (no `.aux.xml`). Other threads still read `.aux.xml` files while a merge runs.
- Works in blocks, so a whole raster is never held in memory.

`check_merged_cog(tiles, output)` returns a `MergeCheck` with every problem found (`passed` is true when there are none). It checks that the output:

- is a valid COG (`rio_cogeo.cog_validate`; errors and warnings both fail);
- has overviews down to one block, or none when it fits in one block;
- has the tiles' CRS, transform, size, band count, data type and nodata;
- comes from tiles that do not overlap;
- holds, in every band, the same value as the tiles at every full-resolution pixel, and the gap fill where no tile covers it. This catches dropped, swapped or changed tiles and pixels, for any data type. Pixels are compared in windows, and each band that differs gives one `pixels` problem with the first differing places.

It raises `TileMismatchError` when the tiles do not share one grid, and `InvalidInputError` for complex data or when `tiles` is empty or a single path, instead of returning a problem.

Every error these functions raise is a `GeoCloudifyError` (from `geo_cloudify.cog.merge`), so a caller can catch one type. A GDAL or rasterio failure is raised as a `GeoCloudifyError` with the original error as its cause. `InvalidInputError` and `TileMismatchError` are also `ValueError`s, and `OutputExistsError` (an existing output) is also a `FileExistsError`.

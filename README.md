# geo-cloudify

Converts geospatial files to cloud-optimized formats and checks the result. Rasters become Cloud Optimized GeoTIFFs (COGs). The import name is `geo_cloudify`.

## Install

- With uv: `uv sync`.
- With pip: `pip install "geo-cloudify @ git+https://github.com/SERVIR-AI/geo-cloudify.git@v0.1.0"`.

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

- `tiles`: a list or tuple of local paths.
- `overview_resampling`: `nearest` or `mode` for class data, `average` for continuous data.
- `compression_codec`: a lossless codec, `DEFLATE` (default), `LZW`, `ZSTD`, `LZMA` or `NONE`.
- `blocksize`: the COG block size in pixels, a power of two from 16 to 4096 (default 512).

`check_merged_cog(tiles, output)` returns a `MergeCheck` with every problem found (`passed` is true when there are none).

Every error these functions raise is a `GeoCloudifyError` (from `geo_cloudify.cog.merge`).
The rules each function applies, and the errors it raises, are in its docstring, for example `help(merge_to_cog)`.

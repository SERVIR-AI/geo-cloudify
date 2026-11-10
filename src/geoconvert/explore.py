"""Top-level explore step: detect kind, group companion files, decide the action.

Not implemented yet. This module owns what every format shares: detecting the kind
(raster, cube, vector) by opening the file, grouping companion files, deciding whether
to convert, pass through or skip, and building the job plan. Format-specific work goes
to `cog/explore.py` (and later `zarr/`, `geoparquet/`).
"""

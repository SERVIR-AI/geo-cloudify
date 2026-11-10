"""COG exploration: detect rasters, fingerprint them, resolve COG settings.

Not implemented yet. This module detects a raster by opening it, records its
fingerprint (CRS, transform, dtype, nodata, colour table, checksums) and resolves the
COG settings (defaults, then data-derived values, then user kwargs).
"""

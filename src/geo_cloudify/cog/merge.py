"""COG merge: combine tiles that share one grid into one Cloud Optimized GeoTIFF.

`check_tiles` reads only the tiles' headers and refuses tiles that would be
changed or misplaced by a merge: a different CRS, pixel size, grid alignment,
data type, band count or nodata. When they match, it returns the grid that
covers all of them.

`merge_to_cog` checks the tiles, refuses overlapping tiles, writes a small VRT
(a text file that places each tile on the union grid, pixel for pixel) and lets
GDAL's COG driver write one COG from it. Pixels are copied, never resampled,
and GDAL reads in blocks, so a whole raster is never held in memory. The merge
needs no config file: without one it uses the `CogConfig` defaults.
"""

import math
import os
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import rasterio
import rasterio.dtypes
import rasterio.shutil
from rasterio.crs import CRS
from rasterio.transform import Affine

from geo_cloudify.config import CogConfig, get_config
from geo_cloudify.settings import get_settings

# How far (in pixels) an origin offset may be from a whole number and still count as aligned.
ALIGN_TOLERANCE = 1e-6
# How far (relative) two pixel sizes may differ and still count as equal.
PIXEL_SIZE_TOLERANCE = 1e-9
# URL schemes a tile may use, and the GDAL virtual file system prefix each one maps to.
_VSI_PREFIXES = {
    "http": "/vsicurl/",
    "https": "/vsicurl/",
    "ftp": "/vsicurl/",
    "s3": "/vsis3/",
    "gs": "/vsigs/",
    "az": "/vsiaz/",
    "oss": "/vsioss/",
}


class TileMismatchError(ValueError):
    """A tile cannot be merged with the others.

    Attributes:
        tile: the tile as the caller gave it (path or URL, as a string).
        field: what differs: "crs", "orientation", "pixel size", "grid alignment",
            "band count", "data type", "nodata" or "overlap".
    """

    def __init__(self, tile: str, field: str, detail: str) -> None:
        """Build the error for `tile` and `field`, with `detail` saying how it differs.

        Args:
            tile: the tile as the caller gave it (path or URL).
            field: the field that differs (see the class docstring for values).
            detail: a short plain explanation, added to the message.

        Returns:
            None.

        Raises:
            Nothing.
        """
        super().__init__(f"tile {tile}: {field} differs: {detail}")
        self.tile = tile
        self.field = field


@dataclass(frozen=True)
class TileGrid:
    """The grid that covers all tiles: the shape a merged output takes.

    Attributes:
        crs: the tiles' shared CRS.
        transform: north-up transform whose origin is the union's top-left corner.
        width: union width in pixels.
        height: union height in pixels.
        dtype: the tiles' shared data type, for example "uint8".
        count: the tiles' shared band count.
        nodata: the tiles' shared nodata, or None when the tiles have none.
    """

    crs: CRS
    transform: Affine
    width: int
    height: int
    dtype: str
    count: int
    nodata: float | None


@dataclass(frozen=True)
class _Header:
    """What `check_tiles` needs from one tile's header.

    Attributes:
        name: the tile as the caller gave it (path or URL, as a string).
        crs: the tile's CRS, or None when it has none.
        transform: the tile's affine transform.
        width: width in pixels.
        height: height in pixels.
        dtypes: one data type per band, so its length is the band count.
        nodatavals: one nodata value per band (None where a band has none).
    """

    name: str
    crs: CRS | None
    transform: Affine
    width: int
    height: int
    dtypes: tuple[str, ...]
    nodatavals: tuple[float | None, ...]


def same_nodata(a: float | None, b: float | None) -> bool:
    """Return True when two nodata values are equal, treating NaN as equal to NaN.

    Args:
        a: a nodata value or None.
        b: a nodata value or None.

    Returns:
        True if both are None, both are NaN, or both are the same number.

    Raises:
        Nothing.
    """
    if a is None or b is None:
        return a is None and b is None
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return a == b


def _read_header(tile: str | PathLike) -> _Header:
    """Open `tile` and return its header. No pixels are read.

    Args:
        tile: a local path or a URL GDAL can open.

    Returns:
        The tile's header.

    Raises:
        rasterio.errors.RasterioIOError: if GDAL cannot open the tile.
    """
    with rasterio.open(tile) as dataset:
        return _Header(
            name=str(tile),
            crs=dataset.crs,
            transform=dataset.transform,
            width=dataset.width,
            height=dataset.height,
            dtypes=tuple(dataset.dtypes),
            nodatavals=tuple(dataset.nodatavals),
        )


def _check_own_header(header: _Header) -> None:
    """Refuse a tile that cannot be placed on any grid by itself.

    A tile must have a CRS, a north-up transform with no rotation, and one data
    type and one nodata shared by all its bands.

    Args:
        header: the tile's header.

    Returns:
        None.

    Raises:
        TileMismatchError: naming the tile and the field ("crs", "orientation",
            "data type" or "nodata").
    """
    if header.crs is None:
        raise TileMismatchError(header.name, "crs", "the tile has no CRS")
    t = header.transform
    if t.b != 0 or t.d != 0 or t.a <= 0 or t.e >= 0:
        raise TileMismatchError(
            header.name, "orientation", f"transform {tuple(t)[:6]} is not north-up without rotation"
        )
    if len(set(header.dtypes)) > 1:
        raise TileMismatchError(header.name, "data type", f"bands differ: {header.dtypes}")
    if any(not same_nodata(v, header.nodatavals[0]) for v in header.nodatavals):
        raise TileMismatchError(header.name, "nodata", f"bands differ: {header.nodatavals}")


def _pixel_offset(header: _Header, ref: _Header) -> tuple[int, int]:
    """Return where `header`'s top-left pixel sits on `ref`'s grid, as (column, row).

    Args:
        header: the tile to place; it already matches `ref` in CRS and pixel size.
        ref: the reference tile whose top-left pixel is (0, 0).

    Returns:
        Whole-number column and row offsets. Negative means west of or north of `ref`.

    Raises:
        TileMismatchError: naming the tile and "grid alignment" when an offset is
            not a whole number of pixels.
    """
    t, r = header.transform, ref.transform
    col = (t.c - r.c) / r.a
    row = (t.f - r.f) / r.e  # r.e is negative, so a tile further north gets a negative row
    for axis, offset in (("x", col), ("y", row)):
        if abs(offset - round(offset)) > ALIGN_TOLERANCE:
            raise TileMismatchError(
                header.name, "grid alignment",
                f"origin is {offset:.6f} pixels from {ref.name} along {axis}, not a whole number",
            )
    return round(col), round(row)


def check_tiles(tiles: Sequence[str | PathLike]) -> TileGrid:
    """Check that tiles share one grid and return the grid that covers them all.

    The first tile is the reference. Every other tile must have the same CRS
    (compared as a CRS, not as text), pixel size, data type, band count and
    nodata (a tile with nodata and a tile without do not match), and its origin
    must sit a whole number of pixels from the reference origin. Only headers
    are read, never pixels. Overlap and gaps are not checked here.

    Args:
        tiles: a list or tuple of one or more local paths or URLs GDAL can open.
            A single path given as a bare string or path object is refused.

    Returns:
        The union grid: shared CRS, dtype, band count and nodata, and the
        transform, width and height that cover every tile.

    Raises:
        ValueError: if `tiles` is empty, or is a single string or path object
            instead of a sequence of them.
        TileMismatchError: naming the first tile that differs and the field
            ("crs", "orientation", "pixel size", "grid alignment", "band count",
            "data type" or "nodata").
        rasterio.errors.RasterioIOError: if GDAL cannot open a tile.
    """
    return place_tiles(tiles)[0]


def place_tiles(tiles: Sequence[str | PathLike]) -> tuple[TileGrid, list[tuple[_Header, int, int]]]:
    """Run the `check_tiles` checks and also say where each tile sits on the union grid.

    Args:
        tiles: a list or tuple of one or more local paths or URLs GDAL can open.

    Returns:
        The union grid, and for each tile in the order given: its header and the
        column and row of its top-left pixel on the union grid (0 or more).

    Raises:
        ValueError: as `check_tiles`.
        TileMismatchError: as `check_tiles`.
        rasterio.errors.RasterioIOError: if GDAL cannot open a tile.
    """
    if isinstance(tiles, (str, bytes, PathLike)):
        raise ValueError(f"tiles must be a list of paths, not a single path: {tiles!r}")
    if not tiles:
        raise ValueError("no tiles given")
    headers = [_read_header(tile) for tile in tiles]
    ref = headers[0]
    offsets = []
    for header in headers:
        _check_own_header(header)
        if header.crs != ref.crs:
            raise TileMismatchError(header.name, "crs", f"{header.crs} vs {ref.crs} in {ref.name}")
        t, r = header.transform, ref.transform
        if not (math.isclose(t.a, r.a, rel_tol=PIXEL_SIZE_TOLERANCE)
                and math.isclose(t.e, r.e, rel_tol=PIXEL_SIZE_TOLERANCE)):
            raise TileMismatchError(
                header.name, "pixel size", f"({t.a}, {-t.e}) vs ({r.a}, {-r.e}) in {ref.name}"
            )
        if len(header.dtypes) != len(ref.dtypes):
            raise TileMismatchError(
                header.name, "band count", f"{len(header.dtypes)} vs {len(ref.dtypes)} in {ref.name}"
            )
        if header.dtypes[0] != ref.dtypes[0]:
            raise TileMismatchError(
                header.name, "data type", f"{header.dtypes[0]} vs {ref.dtypes[0]} in {ref.name}"
            )
        if not same_nodata(header.nodatavals[0], ref.nodatavals[0]):
            raise TileMismatchError(
                header.name, "nodata", f"{header.nodatavals[0]} vs {ref.nodatavals[0]} in {ref.name}"
            )
        offsets.append(_pixel_offset(header, ref))

    left = min(col for col, _ in offsets)
    top = min(row for _, row in offsets)
    right = max(col + h.width for (col, _), h in zip(offsets, headers))
    bottom = max(row + h.height for (_, row), h in zip(offsets, headers))
    # Take the origin from the westmost and northmost tiles' own headers, not from
    # the reference plus a pixel count, so float error does not move it.
    west = next(h for (col, _), h in zip(offsets, headers) if col == left)
    north = next(h for (_, row), h in zip(offsets, headers) if row == top)
    grid = TileGrid(
        crs=ref.crs,
        transform=Affine(ref.transform.a, 0.0, west.transform.c, 0.0, ref.transform.e, north.transform.f),
        width=right - left,
        height=bottom - top,
        dtype=ref.dtypes[0],
        count=len(ref.dtypes),
        nodata=ref.nodatavals[0],
    )
    return grid, [(h, col - left, row - top) for (col, row), h in zip(offsets, headers)]


def check_no_overlap(placed: Sequence[tuple[_Header, int, int]]) -> None:
    """Refuse tiles whose pixels would land on the same place in the union grid.

    Args:
        placed: each tile's header with its column and row on the union grid,
            as returned by `place_tiles`.

    Returns:
        None.

    Raises:
        TileMismatchError: naming the later of two overlapping tiles, with field
            "overlap" and the earlier tile in the message.
    """
    for i, (h, col, row) in enumerate(placed):
        for other, o_col, o_row in placed[:i]:
            if (col < o_col + other.width and o_col < col + h.width
                    and row < o_row + other.height and o_row < row + h.height):
                raise TileMismatchError(h.name, "overlap", f"its pixels overlap {other.name}")


def _gdal_path(tile: str) -> str:
    """Return the name GDAL needs to open `tile` from a VRT in another folder.

    Args:
        tile: a local path (absolute or relative to the current folder), a
            "file://" URL, a GDAL path such as "/vsicurl/...", or a URL with
            scheme "http", "https", "ftp", "s3", "gs", "az" or "oss".

    Returns:
        An absolute local path, the GDAL path unchanged, or the URL as a GDAL
        virtual file path (for example "s3://b/k.tif" gives "/vsis3/b/k.tif").

    Raises:
        ValueError: if `tile` is a URL with any other scheme.
    """
    if tile.startswith("/vsi"):
        return tile
    scheme, sep, rest = tile.partition("://")
    if not sep:
        return os.path.abspath(tile)
    scheme = scheme.lower()
    if scheme == "file":
        return os.path.abspath(rest)
    if scheme in _VSI_PREFIXES:
        return _VSI_PREFIXES[scheme] + (tile if _VSI_PREFIXES[scheme] == "/vsicurl/" else rest)
    raise ValueError(
        f"tile {tile}: URL scheme {scheme!r} is not supported; use one of "
        f"{', '.join(sorted(_VSI_PREFIXES))}, a local path or a /vsi path"
    )


@dataclass(frozen=True)
class _BandLook:
    """How one band is shown and described, copied from the first tile to the output.

    Attributes:
        color_interp: GDAL colour interpretation name, for example "Gray" or "Palette".
        colormap: palette entries as {value: (r, g, b, a)}, or None when the band has none.
        description: the band description, or "" when it has none.
        scale: the band's scale (1.0 when unset).
        offset: the band's offset (0.0 when unset).
        tags: the band's metadata in the default domain.
    """

    color_interp: str
    colormap: dict[int, tuple[int, int, int, int]] | None
    description: str
    scale: float
    offset: float
    tags: dict[str, str]


def _read_look(tile: str | PathLike) -> tuple[dict[str, str], list[_BandLook]]:
    """Return a tile's dataset metadata and, per band, how it is shown and described.

    No pixels are read.

    Args:
        tile: a local path or a URL GDAL can open.

    Returns:
        The dataset metadata in the default domain, and one `_BandLook` per band.

    Raises:
        rasterio.errors.RasterioIOError: if GDAL cannot open the tile.
    """
    with rasterio.open(tile) as dataset:
        bands = []
        for band in range(1, dataset.count + 1):
            try:
                colormap = dataset.colormap(band)
            except ValueError:  # rasterio raises this when the band has no colour table
                colormap = None
            bands.append(_BandLook(
                color_interp=dataset.colorinterp[band - 1].name.capitalize(),
                colormap=colormap,
                description=dataset.descriptions[band - 1] or "",
                scale=dataset.scales[band - 1],
                offset=dataset.offsets[band - 1],
                tags=dataset.tags(band),
            ))
        return dataset.tags(), bands


def _add_metadata(parent: ET.Element, tags: dict[str, str]) -> None:
    """Add a VRT `<Metadata>` element holding `tags` to `parent`, if there are any.

    Args:
        parent: the `VRTDataset` or `VRTRasterBand` element.
        tags: metadata keys and values in the default domain.

    Returns:
        None.

    Raises:
        Nothing.
    """
    if not tags:
        return
    metadata = ET.SubElement(parent, "Metadata")
    for key, value in tags.items():
        ET.SubElement(metadata, "MDI", key=key).text = value


def _write_vrt(
    grid: TileGrid,
    placed: Sequence[tuple[_Header, int, int]],
    path: Path,
    tags: dict[str, str],
    looks: Sequence[_BandLook],
) -> None:
    """Write a VRT that places every tile on the union grid, pixel for pixel.

    Each tile becomes one source per band whose source and destination windows
    have the same size, so GDAL copies pixels without resampling. Places no tile
    covers read as the grid's nodata (or 0 when the tiles have no nodata). A
    nodata value is written only when the tiles have one.

    Args:
        grid: the union grid from `place_tiles`.
        placed: each tile's header with its column and row on the grid.
        path: where to write the VRT file.
        tags: dataset metadata (default domain) to put on the VRT.
        looks: one `_BandLook` per band: colour interpretation, colour table,
            description, scale, offset and band metadata to put on that band.

    Returns:
        None.

    Raises:
        OSError: if the file cannot be written.
    """
    root = ET.Element("VRTDataset", rasterXSize=str(grid.width), rasterYSize=str(grid.height))
    ET.SubElement(root, "SRS").text = grid.crs.to_wkt()
    ET.SubElement(root, "GeoTransform").text = ", ".join(repr(v) for v in grid.transform.to_gdal())
    _add_metadata(root, tags)
    data_type = rasterio.dtypes.typename_fwd[rasterio.dtypes.dtype_rev[grid.dtype]]
    for band, look in zip(range(1, grid.count + 1), looks):
        band_el = ET.SubElement(root, "VRTRasterBand", dataType=data_type, band=str(band))
        _add_metadata(band_el, look.tags)
        if look.description:
            ET.SubElement(band_el, "Description").text = look.description
        if grid.nodata is not None:
            ET.SubElement(band_el, "NoDataValue").text = repr(float(grid.nodata))
        ET.SubElement(band_el, "ColorInterp").text = look.color_interp
        if look.colormap:
            table = ET.SubElement(band_el, "ColorTable")
            for index in range(max(look.colormap) + 1):
                rgba = look.colormap.get(index, (0, 0, 0, 0))
                ET.SubElement(table, "Entry", **{f"c{i + 1}": str(v) for i, v in enumerate(rgba)})
        if look.offset != 0.0:
            ET.SubElement(band_el, "Offset").text = repr(float(look.offset))
        if look.scale != 1.0:
            ET.SubElement(band_el, "Scale").text = repr(float(look.scale))
        for header, col, row in placed:
            source = ET.SubElement(band_el, "SimpleSource")
            ET.SubElement(source, "SourceFilename", relativeToVRT="0").text = _gdal_path(header.name)
            ET.SubElement(source, "SourceBand").text = str(band)
            size = {"xSize": str(header.width), "ySize": str(header.height)}
            ET.SubElement(source, "SrcRect", xOff="0", yOff="0", **size)
            ET.SubElement(source, "DstRect", xOff=str(col), yOff=str(row), **size)
    ET.ElementTree(root).write(path, encoding="utf-8")


# Overview methods the caller may choose: nearest or mode for class data, average for continuous data.
OVERVIEW_METHODS = ("nearest", "mode", "average")


def _cog_config() -> CogConfig:
    """Return the COG settings from the config file, or the defaults when there is no file.

    The merge is called as a plain function from any folder, where the
    service's `config.yaml` is usually absent, so a missing file is not an error.

    Returns:
        The `cog` section of the config file named in settings if that file
        exists, otherwise `CogConfig()` with its defaults.

    Raises:
        pydantic.ValidationError: if the config file exists but has unknown keys or invalid values.
        yaml.YAMLError: if the config file exists but is not valid YAML.
    """
    if not get_settings().config_path.is_file():
        return CogConfig()
    return get_config().cog


def merge_to_cog(
    tiles: Sequence[str | PathLike], output: str | PathLike, overview_resampling: str
) -> Path:
    """Merge tiles that share one grid into one COG and return the output path.

    The output keeps the tiles' CRS, pixel grid, data type, band count and
    nodata (none if the tiles have none), and covers the union of the tiles.
    Gaps between tiles are filled with nodata (0 when there is none). Pixels are
    copied without resampling. The COG is tiled and compressed (codec and block
    size from the config file, or the `CogConfig` defaults when there is no
    config file), and its overviews are built fresh with `overview_resampling`;
    overviews already inside the tiles are not reused. Colour interpretation,
    colour table, band descriptions, scale, offset and metadata (default domain)
    are copied from the first tile; the other tiles' values for these are not
    compared. It is written to a temporary folder next to `output` and moved into place
    only when complete, so a failure leaves no output and no temporary files.

    Args:
        tiles: a list or tuple of one or more local paths or URLs GDAL can open.
        output: where to write the COG. Its folder is created if missing.
        overview_resampling: "nearest" or "mode" for class data, "average" for
            continuous data. Lower case, nothing else is accepted.

    Returns:
        The output path, as given, as a Path.

    Raises:
        ValueError: if `overview_resampling` is not one of the allowed values,
            `tiles` is empty or a single path instead of a list, or a tile is a
            URL whose scheme is not supported (see `_gdal_path`).
        FileExistsError: if `output` already exists. It is left untouched.
        TileMismatchError: naming the tile and the field when tiles differ (see
            `check_tiles`) or two tiles overlap (field "overlap").
        rasterio.errors.RasterioError: if GDAL cannot read a tile or write the output.
        OSError: if the output folder or temporary files cannot be written.
        pydantic.ValidationError: if a config file exists but has invalid values.
        yaml.YAMLError: if a config file exists but is not valid YAML.
    """
    if overview_resampling not in OVERVIEW_METHODS:
        raise ValueError(
            f"overview_resampling must be one of {', '.join(OVERVIEW_METHODS)}, got {overview_resampling!r}"
        )
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    grid, placed = place_tiles(tiles)
    check_no_overlap(placed)
    tags, looks = _read_look(tiles[0])

    config = _cog_config()
    options = {
        "COMPRESS": config.compress,
        "BLOCKSIZE": config.blocksize,
        "NUM_THREADS": config.num_threads,
        "OVERVIEW_RESAMPLING": overview_resampling.upper(),
        "OVERVIEWS": "IGNORE_EXISTING",  # the tiles' own overviews may use another method
        "BIGTIFF": "IF_SAFER",  # a merged year can pass 4 GB uncompressed
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=f".{output.name}.") as tmp:
        vrt = Path(tmp) / "tiles.vrt"
        partial = Path(tmp) / output.name
        _write_vrt(grid, placed, vrt, tags, looks)
        rasterio.shutil.copy(str(vrt), str(partial), driver="COG", **options)
        if output.exists():
            raise FileExistsError(f"output already exists: {output}")
        os.replace(partial, output)
    return output

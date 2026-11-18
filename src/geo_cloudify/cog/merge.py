"""COG merge: combine tiles that share one grid into one Cloud Optimized GeoTIFF.

`check_tiles` reads only the tiles' headers and refuses tiles that would be
changed or misplaced by a merge: a different CRS, pixel size, grid alignment,
data type, band count or nodata, or an internal mask. When they match, it
returns the grid that covers all of them.

`merge_to_cog` checks the tiles, refuses overlapping tiles, writes a small VRT
(a text file that places each tile on the union grid, pixel for pixel) and lets
GDAL's COG driver write one COG from it. Pixels are copied, never resampled,
and GDAL reads in blocks, so a whole raster is never held in memory. The merge
reads no config file and no `.env`: its settings are its arguments.
"""

import logging
import math
import numbers
import os
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import rasterio
import rasterio.dtypes
import rasterio.shutil
from rasterio.crs import CRS
from rasterio.enums import MaskFlags
from rasterio.transform import Affine

# COMPRESS values listed in the GDAL COG driver documentation.
COG_COMPRESS = frozenset(
    {"NONE", "LZW", "JPEG", "DEFLATE", "ZSTD", "WEBP", "LERC", "LERC_DEFLATE", "LERC_ZSTD", "LZMA"}
)

logger = logging.getLogger(__name__)

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
            "band count", "data type", "nodata", "mask" or "overlap".
    """

    def __init__(self, tile: str, field: str, detail: str) -> None:
        """Build the error for `tile` and `field`, with `detail` saying how it differs.

        Args:
            tile: the tile as the caller gave it (path or URL).
            field: the field that differs (see the class docstring for values).
            detail: a short plain explanation, added to the message.
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
        has_mask: True when a band's empty pixels are marked by a mask that is
            neither a nodata value nor an alpha band (for example an internal
            GeoTIFF mask).
    """

    name: str
    crs: CRS | None
    transform: Affine
    width: int
    height: int
    dtypes: tuple[str, ...]
    nodatavals: tuple[float | None, ...]
    has_mask: bool = False


def same_nodata(a: float | None, b: float | None) -> bool:
    """Return True when two nodata values are equal, treating NaN as equal to NaN.

    Args:
        a: a nodata value or None.
        b: a nodata value or None.

    Returns:
        True if both are None, both are NaN, or both are the same number.
    """
    if a is None or b is None:
        return a is None and b is None
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return a == b


def _has_mask(flags: Sequence[MaskFlags]) -> bool:
    """Return True when a band's mask flags show a mask that a value copy would lose.

    A band with no mask has `all_valid`, a band masked by nodata has `nodata`,
    and a colour band masked by an alpha band has `alpha`. Any other set of flags
    (a per-dataset mask, such as an internal GeoTIFF mask, or a per-band mask)
    is a mask.

    Args:
        flags: the band's mask flags, as in `rasterio` `dataset.mask_flag_enums`.

    Returns:
        True for a mask that is not all-valid, nodata or alpha.
    """
    return not ({MaskFlags.all_valid, MaskFlags.nodata, MaskFlags.alpha} & set(flags))


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
            has_mask=any(_has_mask(flags) for flags in dataset.mask_flag_enums),
        )


def _check_own_header(header: _Header) -> None:
    """Refuse a tile that cannot be placed on any grid by itself.

    A tile must have a CRS, a north-up transform with no rotation, one data
    type and one nodata shared by all its bands, and no mask other than nodata
    or an alpha band (the merge copies values only, so masked pixels would come
    out as real data).

    Args:
        header: the tile's header.

    Raises:
        TileMismatchError: naming the tile and the field ("crs", "orientation",
            "data type", "nodata" or "mask").
    """
    if header.crs is None:
        raise TileMismatchError(header.name, "crs", "the tile has no CRS")
    t = header.transform
    # b and d are the rotation terms; a and e are the pixel width and height.
    # North-up means no rotation, a positive width and a negative height.
    if t.b != 0 or t.d != 0 or t.a <= 0 or t.e >= 0:
        raise TileMismatchError(
            header.name, "orientation", f"transform {tuple(t)[:6]} is not north-up without rotation"
        )
    if len(set(header.dtypes)) > 1:
        raise TileMismatchError(header.name, "data type", f"bands differ: {header.dtypes}")
    if any(not same_nodata(v, header.nodatavals[0]) for v in header.nodatavals):
        raise TileMismatchError(header.name, "nodata", f"bands differ: {header.nodatavals}")
    if header.has_mask:
        raise TileMismatchError(
            header.name, "mask",
            "the tile marks empty pixels with a mask, not a nodata value or an alpha band",
        )


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
    # Distance between the two origins, in pixels.
    col = (t.c - r.c) / r.a
    row = (t.f - r.f) / r.e  # r.e is negative, so a tile further north gets a negative row
    # Each offset must be a whole number of pixels, within a tiny tolerance.
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
    must sit a whole number of pixels from the reference origin. A tile whose
    empty pixels are marked by a mask (not a nodata value or an alpha band) is
    refused. Only headers are read, never pixels. Overlap and gaps are not
    checked here.

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
            "data type", "nodata" or "mask").
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
    # Check each tile on its own, then against the first tile.
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

    # The union box in pixels, on the first tile's grid: left and top are the smallest
    # offsets, right and bottom are the largest tile edges.
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
    # Offsets are given from the union's top-left corner.
    return grid, [(h, col - left, row - top) for (col, row), h in zip(offsets, headers)]


def check_no_overlap(placed: Sequence[tuple[_Header, int, int]]) -> None:
    """Refuse tiles whose pixels would land on the same place in the union grid.

    Args:
        placed: each tile's header with its column and row on the union grid,
            as returned by `place_tiles`.

    Raises:
        TileMismatchError: naming the later of two overlapping tiles, with field
            "overlap" and the earlier tile in the message.
    """
    # Compare each tile with the earlier ones. Two boxes overlap when they overlap on
    # both the column axis and the row axis.
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
    # Already a GDAL virtual path.
    if tile.startswith("/vsi"):
        return tile
    # Split "scheme://rest"; `sep` is empty when there is no scheme (a plain local path).
    scheme, sep, rest = tile.partition("://")
    if not sep:
        return os.path.abspath(tile)
    scheme = scheme.lower()
    if scheme == "file":
        return os.path.abspath(rest)
    # "/vsicurl/" takes the whole URL; the other prefixes take the part after "://".
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
        tags: the band's metadata in the default domain, without `STATISTICS_*` tags.
    """

    color_interp: str
    colormap: dict[int, tuple[int, int, int, int]] | None
    description: str
    scale: float
    offset: float
    tags: dict[str, str]


def _without_statistics(tags: dict[str, str]) -> dict[str, str]:
    """Return `tags` without the ones whose name starts with "STATISTICS_" (any case).

    Such tags describe one tile, not the merged image.

    Args:
        tags: metadata keys and values.

    Returns:
        A new dict holding every other tag.
    """
    return {key: value for key, value in tags.items() if not key.upper().startswith("STATISTICS_")}


def _read_look(tile: str | PathLike) -> tuple[dict[str, str], list[_BandLook]]:
    """Return a tile's dataset metadata and, per band, how it is shown and described.

    `STATISTICS_*` tags are left out at dataset and band level. No pixels are read.

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
                tags=_without_statistics(dataset.tags(band)),
            ))
        return _without_statistics(dataset.tags()), bands


def _add_metadata(parent: ET.Element, tags: dict[str, str]) -> None:
    """Add a VRT `<Metadata>` element holding `tags` to `parent`, if there are any.

    Args:
        parent: the `VRTDataset` or `VRTRasterBand` element.
        tags: metadata keys and values in the default domain.
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

    Raises:
        OSError: if the file cannot be written.
    """
    # Dataset level: size, CRS, transform and metadata of the union grid.
    root = ET.Element("VRTDataset", rasterXSize=str(grid.width), rasterYSize=str(grid.height))
    ET.SubElement(root, "SRS").text = grid.crs.to_wkt()
    ET.SubElement(root, "GeoTransform").text = ", ".join(repr(v) for v in grid.transform.to_gdal())
    _add_metadata(root, tags)
    data_type = rasterio.dtypes.typename_fwd[rasterio.dtypes.dtype_rev[grid.dtype]]
    # One band at a time, with the look and metadata copied from the first tile.
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
            # The table lists every index from 0 to the largest; unused ones are transparent black.
            for index in range(max(look.colormap) + 1):
                rgba = look.colormap.get(index, (0, 0, 0, 0))
                ET.SubElement(table, "Entry", **{f"c{i + 1}": str(v) for i, v in enumerate(rgba)})
        if look.offset != 0.0:
            ET.SubElement(band_el, "Offset").text = repr(float(look.offset))
        if look.scale != 1.0:
            ET.SubElement(band_el, "Scale").text = repr(float(look.scale))
        # One source per tile: read the whole tile (SrcRect) and place it at its offset
        # (DstRect) at the same size, so nothing is resampled.
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
# Codecs that change pixel values. LERC is lossless only with a zero error allowance,
# which the merge does not take, so every LERC variant is refused.
LOSSY_CODECS = frozenset({"JPEG", "WEBP", "LERC", "LERC_DEFLATE", "LERC_ZSTD"})


def cog_options(compress: str, blocksize: int) -> tuple[str, int]:
    """Check the codec and block size for a merge and return them as GDAL takes them.

    Args:
        compress: a lossless codec the COG driver knows: "DEFLATE", "LZW",
            "ZSTD", "LZMA" or "NONE", in any case.
        blocksize: a whole-number integer type (`int` or a numpy integer such
            as `np.int64`; not a bool, float or string) that is a multiple of 16
            from 16 to 4096.

    Returns:
        The codec in upper case and the block size as a plain int.

    Raises:
        ValueError: if `compress` is not a string, is a lossy codec ("JPEG",
            "WEBP", "LERC", "LERC_DEFLATE", "LERC_ZSTD", in any case) or is not
            a codec the COG driver knows, or `blocksize` is not an integer that
            is a multiple of 16 from 16 to 4096.
    """
    if not isinstance(compress, str):
        raise ValueError(f"compress must be a codec name, got {compress!r}")
    codec = compress.upper()
    if codec in LOSSY_CODECS:
        raise ValueError(f"compress {compress!r} is lossy; use one of DEFLATE, LZW, ZSTD, LZMA or NONE")
    if codec not in COG_COMPRESS:
        raise ValueError(f"compress must be one of DEFLATE, LZW, ZSTD, LZMA or NONE, got {compress!r}")
    if not isinstance(blocksize, numbers.Integral) or isinstance(blocksize, bool):
        raise ValueError(f"BLOCKSIZE must be an integer, got {blocksize!r}")
    blocksize = int(blocksize)
    if not (blocksize % 16 == 0 and 16 <= blocksize <= 4096):
        raise ValueError(f"BLOCKSIZE must be a multiple of 16 from 16 to 4096, got {blocksize!r}")
    return codec, blocksize


def _copy_without_pam(vrt: Path, partial: Path, options: dict) -> None:
    """Write the COG from the VRT with GDAL's `.aux.xml` files (PAM) turned off.

    With PAM on, `STATISTICS=YES` would save a `<tile>.aux.xml` next to each
    tile. The VRT already holds every tile setting read with PAM on (nodata,
    metadata). Call this only from a thread that is not the main thread:
    rasterio applies the setting to the whole process on the main thread, and
    only to the current thread elsewhere, so other threads keep reading
    `.aux.xml` files during the copy.

    Args:
        vrt: the VRT written by `_write_vrt`.
        partial: where to write the COG.
        options: COG driver creation options, as GDAL takes them.

    Raises:
        rasterio.errors.RasterioError: if GDAL cannot read a tile or write the output.
    """
    # PAM off: GDAL writes no `.aux.xml` files next to the tiles.
    with rasterio.Env(GDAL_PAM_ENABLED="NO"):
        rasterio.shutil.copy(str(vrt), str(partial), driver="COG", **options)


def _hole_pixels(grid: TileGrid, placed: Sequence[tuple[_Header, int, int]]) -> int:
    """Return how many pixels of the union grid no tile covers.

    Args:
        grid: the union grid from `place_tiles`.
        placed: each tile's header with its place on the grid; the tiles must
            not overlap (see `check_no_overlap`).

    Returns:
        The number of uncovered pixels in one band (0 when the tiles fill the grid).
    """
    return grid.width * grid.height - sum(h.width * h.height for h, _, _ in placed)


def merge_to_cog(
    tiles: Sequence[str | PathLike],
    output: str | PathLike,
    overview_resampling: str,
    *,
    compress: str = "DEFLATE",
    blocksize: int = 512,
) -> Path:
    """Merge tiles that share one grid into one COG and return the output path.

    The output:
    - keeps the tiles' CRS, pixel grid, data type, band count and nodata (none if
      the tiles have none), and covers the union of the tiles;
    - has places no tile covers filled with nodata. When the tiles have no nodata
      they are filled with 0 and, unless the first tile has an alpha band (which
      is also 0 there, so those places are transparent), a warning is logged with
      the number of such pixels, since they cannot be told apart from real 0 values;
    - holds copied pixels, never resampled;
    - is tiled with `blocksize` and compressed with `compress`;
    - has overviews built fresh with `overview_resampling`; overviews already
      inside the tiles are not reused;
    - has band statistics (`STATISTICS_*` tags) computed by GDAL from the tiles'
      pixels; those stored in the tiles are not copied. Places no tile covers
      count as not valid when the tiles have a nodata, and are left out when they
      do not (0 fill is not in the minimum, mean or valid percent);
    - gets colour interpretation, colour table, band descriptions, scale, offset
      and other metadata (default domain) from the first tile; the other tiles'
      values for these are not compared.

    How it runs:
    - GDAL does not save statistics or other `.aux.xml` files next to the tiles,
      so the tiles' folder is left as it was. The COG is written in a separate
      thread with `.aux.xml` reading turned off for that thread only, so other
      threads still read `.aux.xml` files while the merge runs.
    - GDAL uses all CPUs (`NUM_THREADS=ALL_CPUS`).
    - No config file and no `.env` is read. One info line logs the settings used.
    - The COG is written to a temporary folder next to `output` and moved into
      place only when complete, so a failure leaves no output and no temporary
      files.

    Args:
        tiles: a list or tuple of one or more local paths or URLs GDAL can open.
        output: where to write the COG. Its folder is created if missing.
        overview_resampling: "nearest" or "mode" for class data, "average" for
            continuous data. Lower case, nothing else is accepted.
        compress: a lossless codec: "DEFLATE" (default), "LZW", "ZSTD",
            "LZMA" or "NONE", in any case.
        blocksize: the COG's block size in pixels, an integer (`int` or a
            numpy integer) that is a multiple of 16 from 16 to 4096 (default 512).

    Returns:
        The output path, as given, as a Path.

    Raises:
        ValueError: if `overview_resampling` is not one of the allowed values,
            `compress` is lossy or unknown, `blocksize` is not allowed,
            `tiles` is empty or a single path instead of a list, or a tile is a
            URL whose scheme is not supported (see `_gdal_path`). Nothing is
            written.
        FileExistsError: if `output` already exists. It is left untouched.
        TileMismatchError: naming the tile and the field when tiles differ (see
            `check_tiles`), a tile has a mask (field "mask") or two tiles
            overlap (field "overlap").
        rasterio.errors.RasterioError: if GDAL cannot read a tile or write the output.
        OSError: if the output folder or temporary files cannot be written.
    """
    # Check the arguments before touching any file.
    if overview_resampling not in OVERVIEW_METHODS:
        raise ValueError(
            f"overview_resampling must be one of {', '.join(OVERVIEW_METHODS)}, got {overview_resampling!r}"
        )
    codec, blocksize = cog_options(compress, blocksize)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    # The tiles must share one grid and not overlap; the look comes from the first tile.
    grid, placed = place_tiles(tiles)
    check_no_overlap(placed)
    tags, looks = _read_look(tiles[0])

    # COG driver options for the final write.
    options = {
        "COMPRESS": codec,
        "BLOCKSIZE": blocksize,
        "NUM_THREADS": "ALL_CPUS",
        "OVERVIEW_RESAMPLING": overview_resampling.upper(),
        "OVERVIEWS": "IGNORE_EXISTING",  # the tiles' own overviews may use another method
        "BIGTIFF": "IF_SAFER",  # a merged year can pass 4 GB uncompressed
        "STATISTICS": "YES",
    }
    logger.info(
        "merging %d tiles into %s: compress=%s blocksize=%d overview_resampling=%s num_threads=%s",
        len(placed), output, codec, blocksize, overview_resampling, options["NUM_THREADS"],
    )
    # Without a nodata, gaps are filled with 0 and look like real 0 values: warn.
    holes = _hole_pixels(grid, placed)
    has_alpha = any(look.color_interp == "Alpha" for look in looks)
    if grid.nodata is None and holes and not has_alpha:
        logger.warning(
            "%s pixels per band are not covered by any tile and the tiles have no nodata: "
            "they are filled with 0 and cannot be told apart from real 0 values",
            f"{holes:,}",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    # Build in a hidden temporary folder next to the output, so the final move stays on one
    # disk and a failure leaves nothing behind.
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=f".{output.name}.") as tmp:
        vrt = Path(tmp) / "tiles.vrt"
        partial = Path(tmp) / output.name
        _write_vrt(grid, placed, vrt, tags, looks)
        # One worker thread, so the PAM setting applies to this write only
        # (see `_copy_without_pam`).
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(_copy_without_pam, vrt, partial, options).result()
        # Check again in case something wrote the output meanwhile.
        if output.exists():
            raise FileExistsError(f"output already exists: {output}")
        os.replace(partial, output)
    return output

"""COG merge: combine tiles that share one grid into one Cloud Optimized GeoTIFF.

`check_tiles` reads only the tiles' headers and refuses tiles that would be
changed or misplaced by a merge: a different CRS, pixel size, grid alignment,
data type, band count or nodata. When they match, it returns the grid that
covers all of them.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike

import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

# How far (in pixels) an origin offset may be from a whole number and still count as aligned.
_ALIGN_TOLERANCE = 1e-6
# How far (relative) two pixel sizes may differ and still count as equal.
_PIXEL_SIZE_TOLERANCE = 1e-9


class TileMismatchError(ValueError):
    """A tile cannot be merged with the others.

    Attributes:
        tile: the tile as the caller gave it (path or URL, as a string).
        field: what differs: "crs", "orientation", "pixel size", "grid alignment",
            "band count", "data type" or "nodata".
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


def _same_nodata(a: float | None, b: float | None) -> bool:
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
    if any(not _same_nodata(v, header.nodatavals[0]) for v in header.nodatavals):
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
        if abs(offset - round(offset)) > _ALIGN_TOLERANCE:
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
        if not (math.isclose(t.a, r.a, rel_tol=_PIXEL_SIZE_TOLERANCE)
                and math.isclose(t.e, r.e, rel_tol=_PIXEL_SIZE_TOLERANCE)):
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
        if not _same_nodata(header.nodatavals[0], ref.nodatavals[0]):
            raise TileMismatchError(
                header.name, "nodata", f"{header.nodatavals[0]} vs {ref.nodatavals[0]} in {ref.name}"
            )
        offsets.append(_pixel_offset(header, ref))

    left = min(col for col, _ in offsets)
    top = min(row for _, row in offsets)
    right = max(col + h.width for (col, _), h in zip(offsets, headers))
    bottom = max(row + h.height for (_, row), h in zip(offsets, headers))
    return TileGrid(
        crs=ref.crs,
        transform=ref.transform @ Affine.translation(left, top),
        width=right - left,
        height=bottom - top,
        dtype=ref.dtypes[0],
        count=len(ref.dtypes),
        nodata=ref.nodatavals[0],
    )

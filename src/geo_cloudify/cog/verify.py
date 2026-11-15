"""COG verification: check that a merged COG is a faithful copy of its tiles.

`check_merged_cog` runs four checks on the output of `merge_to_cog` and returns
every problem it finds, not only the first:

- it is a valid COG (`rio_cogeo.cog_validate`; errors and warnings both count);
- it has overviews down to one block, or none when it fits in one block;
- its CRS, transform, size, band count, data type and nodata match the tiles;
- the count of each pixel value in each band equals the sum over the tiles,
  plus the fill in gaps between tiles. This catches dropped, doubled or changed pixels.

Pixels are read at full resolution, in block-aligned windows, so a whole raster
is never held in memory. Byte and 16-bit data are counted in a fixed-size
histogram; wider types (such as float32) are counted only up to
`_MAX_DISTINCT_VALUES` distinct values per band, and past that the pixel-count
check is skipped.

`compare_pixels` compares every full-resolution pixel of the output with what
the tiles hold at that place, in windows, for any real data type.
"""

import math
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from os import PathLike

import numpy as np
import rasterio
from rasterio.io import DatasetReader
from rasterio.transform import Affine
from rasterio.windows import Window
from rio_cogeo.cogeo import cog_validate

from geo_cloudify.cog.merge import (
    ALIGN_TOLERANCE,
    PIXEL_SIZE_TOLERANCE,
    TileGrid,
    TileMismatchError,
    check_no_overlap,
    place_tiles,
    same_nodata,
)

# About how many pixels one read may hold. Windows are rounded to whole blocks.
_WINDOW_PIXELS = 1 << 24
# How many differing values one pixel-count problem lists.
_MAX_LISTED_VALUES = 10
# How many distinct values per band the pixel-count check keeps for types wider than 16 bits.
_MAX_DISTINCT_VALUES = 100_000
# How many differing pixels one "pixels" problem lists.
_MAX_LISTED_PIXELS = 5


class _TooManyValues(Exception):
    """Raised when a band holds more distinct values than `_MAX_DISTINCT_VALUES`."""

    def __init__(self, band: int) -> None:
        """Record which band went over the limit.

        Args:
            band: the 1-based band number.

        Returns:
            None.

        Raises:
            Nothing.
        """
        super().__init__(f"band {band}")
        self.band = band


@dataclass(frozen=True)
class Problem:
    """One thing wrong with a merged COG.

    Attributes:
        field: what is wrong: "cog", "overviews", "crs", "transform", "size",
            "band count", "data type", "nodata", "pixel counts", "overlap"
            or "pixels" (from `compare_pixels`).
        detail: a short plain explanation.
    """

    field: str
    detail: str

    def __str__(self) -> str:
        """Return the problem as "field: detail".

        Returns:
            The field and detail joined by ": ".

        Raises:
            Nothing.
        """
        return f"{self.field}: {self.detail}"


@dataclass(frozen=True)
class MergeCheck:
    """The result of checking a merged COG against its tiles.

    Attributes:
        output: the checked file, as the caller gave it (a path or a URL).
        problems: every problem found, in the order the checks ran; empty when all pass.
        counts_checked: True only when the pixel counts were compared. False when
            they were skipped: too many distinct values in a band, or the band
            count or data type already differs.
    """

    output: str | PathLike
    problems: tuple[Problem, ...]
    counts_checked: bool

    @property
    def passed(self) -> bool:
        """Return True when no problem was found.

        A pass does not mean the pixel counts were compared; see `counts_checked`.

        Returns:
            True if `problems` is empty, otherwise False.

        Raises:
            Nothing.
        """
        return not self.problems


def check_merged_cog(tiles: Sequence[str | PathLike], output: str | PathLike) -> MergeCheck:
    """Check that `output` is a valid COG holding exactly the pixels of `tiles`.

    The expected grid is the union of the tiles, as `check_tiles` gives it. The
    checks are: a valid COG (any `cog_validate` error or warning is a problem);
    overviews whose smallest level fits in one block (an image that fits in one
    block needs none, and every band must have the same levels); CRS (compared
    as a CRS), transform (with a tiny float tolerance), size, band count, data
    type and nodata equal to the tiles'; no two tiles overlap; and, per band, the
    count of each value equals the sum over the tiles at full resolution, where
    the expected count of the fill value (the nodata, or 0 when there is none)
    also includes the gap pixels no tile covers. NaN values count as one value.

    Pixels are verified by these value counts only, which assumes class data
    (few distinct values). Pixel counts are skipped when the band count or data
    type already differs, and when a band of a type wider than 16 bits holds
    more than 100,000 distinct values (in one window alone, or in total) in
    both the tiles and the output. A skipped count check adds no problem, the
    other checks still run, and `counts_checked` is False. When only one side,
    the tiles or the output, goes over the limit in a band, the counts cannot
    match, so that is a "pixel counts" problem and `counts_checked` is True.
    `compare_pixels` compares every pixel with the tiles and works for any
    data; this function does not call it.

    Args:
        tiles: a list or tuple of one or more local paths or URLs GDAL can open:
            the tiles the output was merged from.
        output: a local path or URL of the merged COG.

    Returns:
        A MergeCheck listing every problem found (empty when all checks pass),
        with `counts_checked` True only when the pixel counts were compared.

    Raises:
        ValueError: if `tiles` is empty or a single path instead of a list, or
            the tiles hold complex numbers (pixel counts are not supported for them).
        TileMismatchError: if the tiles do not share one grid (see `check_tiles`).
        rasterio.errors.RasterioIOError: if GDAL cannot open a tile or the output.
        rasterio.errors.RasterioError: if GDAL cannot read their pixels.
    """
    grid, placed = place_tiles(tiles)
    if np.dtype(grid.dtype).kind == "c":
        raise ValueError(f"pixel counts are not supported for complex data ({grid.dtype})")
    problems: list[Problem] = []
    with rasterio.open(output) as out:
        problems += _check_cog(output)
        problems += _check_overviews(out, output)
        problems += _check_grid(out, grid)
        same_layout = out.count == grid.count and set(out.dtypes) == {grid.dtype}
    try:
        check_no_overlap(placed)
    except TileMismatchError as error:
        problems.append(Problem("overlap", str(error)))
    counts_checked = False
    if same_layout:
        count_problems = _check_counts([h.name for h, _, _ in placed], output, grid)
        if count_problems is not None:
            problems += count_problems
            counts_checked = True
    return MergeCheck(output=output, problems=tuple(problems), counts_checked=counts_checked)


def _check_cog(output: str | PathLike) -> list[Problem]:
    """Return a problem for each error and each warning `cog_validate` gives.

    Args:
        output: a local path or URL of the file to check.

    Returns:
        One "cog" problem per error or warning; empty when the file is a clean COG.

    Raises:
        Nothing: `cog_validate` reports a file it cannot read as an error.
    """
    is_valid, errors, warnings = cog_validate(str(output), quiet=True)
    problems = [Problem("cog", f"error: {e}") for e in errors]
    problems += [Problem("cog", f"warning: {w}") for w in warnings]
    if not is_valid and not errors:
        problems.append(Problem("cog", "not a valid COG"))
    return problems


def _check_overviews(out: DatasetReader, output: str | PathLike) -> list[Problem]:
    """Return a problem when the overviews do not go down to one block.

    Args:
        out: the open output.
        output: its path or URL, used to open the smallest overview level.

    Returns:
        "overviews" problems: bands with different overview levels, or a smallest
        level (the full image when there are no overviews) larger than one block.

    Raises:
        rasterio.errors.RasterioIOError: if GDAL cannot open an overview level.
    """
    problems = []
    factors = out.overviews(1)
    for band in range(2, out.count + 1):
        if out.overviews(band) != factors:
            problems.append(Problem(
                "overviews", f"band {band} has levels {out.overviews(band)}, band 1 has {factors}"
            ))
    if factors:
        with rasterio.open(output, overview_level=len(factors) - 1) as smallest:
            width, height = smallest.width, smallest.height
    else:
        width, height = out.width, out.height
    block_height, block_width = out.block_shapes[0]
    if width > block_width or height > block_height:
        problems.append(Problem(
            "overviews",
            f"the smallest level is {width} x {height} pixels, larger than one "
            f"{block_width} x {block_height} block (levels: {factors or 'none'})",
        ))
    return problems


def _same_transform(a: Affine, b: Affine) -> bool:
    """Return True when two transforms place pixels in the same place, allowing float noise.

    Args:
        a: one transform.
        b: the expected transform (north-up, no rotation).

    Returns:
        True if the pixel sizes match within a relative 1e-9, the rotation terms
        are zero and the origins are within a millionth of a pixel.

    Raises:
        Nothing.
    """
    tol_x, tol_y = ALIGN_TOLERANCE * abs(b.a), ALIGN_TOLERANCE * abs(b.e)
    return (
        math.isclose(a.a, b.a, rel_tol=PIXEL_SIZE_TOLERANCE)
        and math.isclose(a.e, b.e, rel_tol=PIXEL_SIZE_TOLERANCE)
        and abs(a.b) <= tol_x and abs(a.d) <= tol_y
        and abs(a.c - b.c) <= tol_x and abs(a.f - b.f) <= tol_y
    )


def _check_grid(out: DatasetReader, grid: TileGrid) -> list[Problem]:
    """Return a problem for each grid field of the output that differs from the tiles'.

    Args:
        out: the open output.
        grid: the tiles' union grid.

    Returns:
        Problems with field "crs", "transform", "size", "band count", "data type"
        or "nodata", one per field that differs.

    Raises:
        Nothing.
    """
    problems = []
    if out.crs is None or out.crs != grid.crs:
        problems.append(Problem("crs", f"output {out.crs}, tiles {grid.crs}"))
    if not _same_transform(out.transform, grid.transform):
        problems.append(Problem(
            "transform", f"output {tuple(out.transform)[:6]}, tiles {tuple(grid.transform)[:6]}"
        ))
    if (out.width, out.height) != (grid.width, grid.height):
        problems.append(Problem(
            "size", f"output {out.width} x {out.height}, tiles cover {grid.width} x {grid.height}"
        ))
    if out.count != grid.count:
        problems.append(Problem("band count", f"output {out.count}, tiles {grid.count}"))
    if set(out.dtypes) != {grid.dtype}:
        problems.append(Problem("data type", f"output {out.dtypes}, tiles {grid.dtype}"))
    if any(not same_nodata(v, grid.nodata) for v in out.nodatavals):
        problems.append(Problem("nodata", f"output {out.nodatavals}, tiles {grid.nodata}"))
    return problems


def _windows(dataset: DatasetReader) -> Iterator[Window]:
    """Yield block-aligned windows that cover `dataset`, each about `_WINDOW_PIXELS` or less.

    A window is never smaller than one block, so a dataset with very large
    blocks gives windows of one block each.

    Args:
        dataset: an open raster.

    Yields:
        Windows in row order that together cover every pixel once.

    Raises:
        Nothing.
    """
    block_height, block_width = dataset.block_shapes[0]
    rows = block_height * max(1, _WINDOW_PIXELS // (block_height * dataset.width))
    if rows * dataset.width <= _WINDOW_PIXELS:
        cols = dataset.width
    else:
        cols = block_width * max(1, _WINDOW_PIXELS // (rows * block_width))
    for row in range(0, dataset.height, rows):
        for col in range(0, dataset.width, cols):
            yield Window(col, row, min(cols, dataset.width - col), min(rows, dataset.height - row))


def _as_bits(data: np.ndarray) -> np.ndarray:
    """Return `data` as unsigned integers of the same width, with every NaN made the same.

    Counting the bit patterns compares values exactly, and makes all NaNs one value.

    Args:
        data: an array of a real (integer or float) data type.

    Returns:
        The same values viewed as uint8, uint16, uint32 or uint64.

    Raises:
        Nothing.
    """
    if data.dtype.kind == "f":
        data = np.where(np.isnan(data), np.array(np.nan, dtype=data.dtype), data)
    return data.view(f"u{data.dtype.itemsize}")


def _count_values(source: str | PathLike) -> list[Counter]:
    """Count each pixel value in each band of `source`, at full resolution, window by window.

    Args:
        source: a local path or URL GDAL can open, with a real data type.

    Byte and 16-bit bands are counted in a fixed-size histogram. Wider bands
    are counted in a Counter. A window's distinct values are checked against
    `_MAX_DISTINCT_VALUES` before they are listed or added, so the Counter never
    holds more than twice that many.

    Returns:
        One Counter per band, mapping each value's bit pattern (see `_as_bits`)
        to how many pixels hold it.

    Raises:
        _TooManyValues: if a band wider than 16 bits holds more than
            `_MAX_DISTINCT_VALUES` distinct values.
        rasterio.errors.RasterioIOError: if GDAL cannot open `source`.
        rasterio.errors.RasterioError: if GDAL cannot read its pixels.
    """
    with rasterio.open(source) as dataset:
        windows = list(_windows(dataset))
        result = []
        for band in range(1, dataset.count + 1):
            itemsize = np.dtype(dataset.dtypes[band - 1]).itemsize
            histogram = np.zeros(1 << (8 * itemsize), dtype=np.int64) if itemsize <= 2 else None
            counts: Counter = Counter()
            for window in windows:
                bits = _as_bits(dataset.read(band, window=window)).ravel()
                if histogram is not None:
                    histogram += np.bincount(bits, minlength=histogram.size)
                else:
                    bits.sort()
                    change = bits[1:] != bits[:-1]
                    if int(np.count_nonzero(change)) + 1 > _MAX_DISTINCT_VALUES:
                        raise _TooManyValues(band)
                    starts = np.concatenate(([0], np.flatnonzero(change) + 1))
                    n = np.diff(np.append(starts, bits.size))
                    counts.update(dict(zip(bits[starts].tolist(), n.tolist())))
                    if len(counts) > _MAX_DISTINCT_VALUES:
                        raise _TooManyValues(band)
            if histogram is not None:
                counts.update({int(v): int(histogram[v]) for v in np.flatnonzero(histogram)})
            result.append(counts)
        return result


def _expected_counts(tiles: Sequence[str], grid: TileGrid) -> list[Counter] | int:
    """Sum the per-band value counts of the tiles, adding the gap fill on the union grid.

    Args:
        tiles: the tiles, as the caller gave them.
        grid: the tiles' union grid.

    Returns:
        One Counter per band, mapping each value's bit pattern to how many
        pixels of the union grid should hold it; or, when a band holds more
        than `_MAX_DISTINCT_VALUES` distinct values, that 1-based band number.

    Raises:
        rasterio.errors.RasterioIOError: if GDAL cannot open a tile.
        rasterio.errors.RasterioError: if GDAL cannot read its pixels.
    """
    expected = [Counter() for _ in range(grid.count)]
    tile_pixels = 0
    for tile in tiles:
        with rasterio.open(tile) as dataset:
            tile_pixels += dataset.width * dataset.height
        try:
            tile_counts = _count_values(tile)
        except _TooManyValues as error:
            return error.band
        for band, (band_counts, counts) in enumerate(zip(expected, tile_counts), start=1):
            band_counts.update(counts)
            if len(band_counts) > _MAX_DISTINCT_VALUES:
                return band
    gap = grid.width * grid.height - tile_pixels
    if gap > 0:
        fill = np.array([0 if grid.nodata is None else grid.nodata], dtype=np.dtype(grid.dtype))
        fill_bits = int(_as_bits(fill)[0])
        for band_counts in expected:
            band_counts[fill_bits] += gap
    for band, band_counts in enumerate(expected, start=1):
        if len(band_counts) > _MAX_DISTINCT_VALUES:
            return band
    return expected


def _check_counts(
    tiles: Sequence[str], output: str | PathLike, grid: TileGrid
) -> list[Problem] | None:
    """Compare per-band value counts of the output with the sum over the tiles.

    Args:
        tiles: the tiles, as the caller gave them.
        output: the merged output; it has the grid's band count and data type.
        grid: the tiles' union grid.

    Returns:
        One "pixel counts" problem per band whose counts differ, listing up to
        `_MAX_LISTED_VALUES` differing values (empty when all match); or None,
        meaning not checked, when both the tiles (with the gap fill) and the
        output hold more than `_MAX_DISTINCT_VALUES` distinct values in a band.
        When only one side goes over the limit, one "pixel counts" problem for
        the first band that does.

    Raises:
        rasterio.errors.RasterioIOError: if GDAL cannot open a tile or the output.
        rasterio.errors.RasterioError: if GDAL cannot read their pixels.
    """
    dtype = np.dtype(grid.dtype)
    expected = _expected_counts(tiles, grid)
    if isinstance(expected, int):
        try:
            output_counts = _count_values(output)
        except _TooManyValues:
            return None
        held = len(output_counts[expected - 1])
        return [
            Problem(
                "pixel counts",
                f"band {expected}: tiles hold more than {_MAX_DISTINCT_VALUES:,} "
                f"distinct values, output holds {held:,}",
            )
        ]

    try:
        output_counts = _count_values(output)
    except _TooManyValues as error:
        held = len(expected[error.band - 1])
        return [
            Problem(
                "pixel counts",
                f"band {error.band}: output holds more than {_MAX_DISTINCT_VALUES:,} "
                f"distinct values, tiles hold {held:,}",
            )
        ]
    problems = []
    for band, (want, got) in enumerate(zip(expected, output_counts), start=1):
        differ = sorted(v for v in want.keys() | got.keys() if want[v] != got[v])
        if not differ:
            continue
        listed = "; ".join(
            f"value {_value(v, dtype)}: output {got[v]}, tiles {want[v]}"
            for v in differ[:_MAX_LISTED_VALUES]
        )
        more = f" (and {len(differ) - _MAX_LISTED_VALUES} more)" if len(differ) > _MAX_LISTED_VALUES else ""
        problems.append(Problem("pixel counts", f"band {band}: {listed}{more}"))
    return problems


def compare_pixels(tiles: Sequence[str | PathLike], output: str | PathLike) -> tuple[Problem, ...]:
    """Compare every full-resolution pixel of `output` with what the tiles hold there.

    For each window of the output and each band, the expected values are built
    from the tiles placed on their union grid, with the fill value (the nodata,
    or 0 when there is none) where no tile covers the area, and compared with
    the output value for value. Values are compared by their bits, so NaN
    equals NaN. Works for any real data type, one band window at a time, so
    memory does not grow with the image.

    Args:
        tiles: a list or tuple of one or more local paths or URLs GDAL can open:
            the tiles the output was merged from.
        output: a local path or URL of the merged raster.

    Returns:
        One "pixels" problem per band that differs, giving how many pixels
        differ and up to `_MAX_LISTED_PIXELS` of them as column, row, expected
        and found value. A single "pixels" problem saying "not compared" when the
        output's size, transform, band count or data type differs from the
        tiles'. Empty when every pixel matches.

    Raises:
        ValueError: if `tiles` is empty or a single path instead of a list, or
            the tiles hold complex numbers.
        TileMismatchError: if the tiles do not share one grid (see `check_tiles`),
            or two tiles overlap (field "overlap").
        rasterio.errors.RasterioIOError: if GDAL cannot open a tile or the output.
        rasterio.errors.RasterioError: if GDAL cannot read their pixels.
    """
    grid, placed = place_tiles(tiles)
    dtype = np.dtype(grid.dtype)
    if dtype.kind == "c":
        raise ValueError(f"pixel comparison is not supported for complex data ({grid.dtype})")
    check_no_overlap(placed)
    fill = 0 if grid.nodata is None else grid.nodata
    with ExitStack() as stack:
        out = stack.enter_context(rasterio.open(output))
        different = _grid_difference(out, grid)
        if different:
            return (Problem("pixels", f"not compared: {different}"),)
        sources = [(stack.enter_context(rasterio.open(h.name)), col, row) for h, col, row in placed]
        windows = list(_windows(out))
        problems = []
        for band in range(1, grid.count + 1):
            differing = 0
            listed: list[str] = []
            for window in windows:
                expected = np.full((window.height, window.width), fill, dtype=dtype)
                for source, col, row in sources:
                    _paste(expected, window, source, band, col, row)
                found = out.read(band, window=window)
                mismatch = _as_bits(expected) != _as_bits(found)
                differing += int(np.count_nonzero(mismatch))
                if len(listed) < _MAX_LISTED_PIXELS and differing:
                    listed += _first_places(mismatch, expected, found, window, _MAX_LISTED_PIXELS - len(listed))
            if differing:
                noun = "pixel differs" if differing == 1 else "pixels differ"
                problems.append(Problem(
                    "pixels", f"band {band}: {differing} {noun}; first: {'; '.join(listed)}"
                ))
    return tuple(problems)


def _grid_difference(out: DatasetReader, grid: TileGrid) -> str | None:
    """Return why the output's pixels cannot be lined up with the tiles', or None.

    Args:
        out: the open output.
        grid: the tiles' union grid.

    Returns:
        A short plain reason when the size, transform, band count or data type
        differs from the grid; None when they all match.

    Raises:
        Nothing.
    """
    if (out.width, out.height) != (grid.width, grid.height):
        return f"output is {out.width} x {out.height}, tiles cover {grid.width} x {grid.height}"
    if not _same_transform(out.transform, grid.transform):
        return f"output transform {tuple(out.transform)[:6]}, tiles {tuple(grid.transform)[:6]}"
    if out.count != grid.count:
        return f"output has {out.count} bands, tiles {grid.count}"
    if set(out.dtypes) != {grid.dtype}:
        return f"output data type {out.dtypes}, tiles {grid.dtype}"
    return None


def _paste(
    expected: np.ndarray, window: Window, source: DatasetReader, band: int, col: int, row: int
) -> None:
    """Copy the part of one tile that falls inside `window` into `expected`.

    Args:
        expected: the window's expected values, shape (window height, window width).
        window: a window of the union grid.
        source: the open tile.
        band: the band number, from 1.
        col: the tile's left column on the union grid.
        row: the tile's top row on the union grid.

    Returns:
        None. `expected` is changed in place; nothing happens when the tile
        does not reach the window.

    Raises:
        rasterio.errors.RasterioError: if GDAL cannot read the tile's pixels.
    """
    w_col, w_row = int(window.col_off), int(window.row_off)
    left, right = max(w_col, col), min(w_col + window.width, col + source.width)
    top, bottom = max(w_row, row), min(w_row + window.height, row + source.height)
    if left >= right or top >= bottom:
        return
    part = Window(left - col, top - row, right - left, bottom - top)
    expected[top - w_row:bottom - w_row, left - w_col:right - w_col] = source.read(band, window=part)


def _first_places(
    mismatch: np.ndarray, expected: np.ndarray, found: np.ndarray, window: Window, limit: int
) -> list[str]:
    """Describe the first differing pixels of one window, in row order.

    Searches row by row, so memory stays at one row however many pixels differ.

    Args:
        mismatch: True where the window's pixels differ.
        expected: the window's expected values.
        found: the window's values in the output.
        window: the window on the union grid, to give full-grid places.
        limit: the most pixels to describe (1 or more).

    Returns:
        Up to `limit` strings "column C, row R: expected E, found F".

    Raises:
        Nothing.
    """
    places: list[str] = []
    for r in range(mismatch.shape[0]):
        for c in np.flatnonzero(mismatch[r])[:limit - len(places)]:
            places.append(
                f"column {int(window.col_off) + int(c)}, row {int(window.row_off) + r}: "
                f"expected {expected[r, c].item()}, found {found[r, c].item()}"
            )
        if len(places) >= limit:
            break
    return places


def _value(bits: int, dtype: np.dtype) -> int | float:
    """Return the pixel value whose bit pattern is `bits`, for messages.

    Args:
        bits: a bit pattern from `_as_bits`.
        dtype: the data type the pattern belongs to.

    Returns:
        The value as a Python int or float.

    Raises:
        Nothing.
    """
    return np.array([bits], dtype=f"u{dtype.itemsize}").view(dtype)[0].item()

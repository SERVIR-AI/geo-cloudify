"""COG verification: check that a merged COG is a faithful copy of its tiles.

`check_merged_cog` runs these checks on the output of `merge_to_cog` and returns
every problem it finds, not only the first:

- it is a valid COG (`rio_cogeo.cog_validate`; errors and warnings both count);
- it has overviews down to one block, or none when it fits in one block;
- its CRS, transform, size, band count, data type and nodata match the tiles;
- no two tiles overlap;
- the count of each pixel value in each band equals the sum over the tiles,
  plus the fill in gaps between tiles. This catches dropped, doubled or changed pixels.

Pixels are read at full resolution, in block-aligned windows, so a whole raster
is never held in memory. Byte and 16-bit data are counted in a fixed-size
histogram; wider types (such as float32) are counted only up to
`_MAX_DISTINCT_VALUES` distinct values per band, and past that the pixel-count
check is skipped.

`check_pixels` compares every full-resolution pixel of the output with what
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
    _validate_and_place_tiles,
    nodata_equal,
)

# About how many pixels one read may hold. Windows are rounded to whole blocks.
_MAX_WINDOW_PIXELS = 1 << 24
# How many differing values one pixel-count problem lists.
_MAX_LISTED_VALUES = 10
# How many distinct values per band the pixel-count check keeps for types wider than 16 bits.
_MAX_DISTINCT_VALUES = 100_000
# How many differing pixels one "pixels" problem lists.
_MAX_LISTED_PIXELS = 5


class _TooManyValuesError(Exception):
    """Raised when a band holds more distinct values than `_MAX_DISTINCT_VALUES`."""

    def __init__(self, band: int) -> None:
        """Record which band went over the limit.

        Args:
            band: the 1-based band number.
        """
        super().__init__(f"band {band}")
        self.band = band


@dataclass(frozen=True)
class Problem:
    """One thing wrong with a merged COG.

    Attributes:
        field: what is wrong: "cog", "overviews", "crs", "transform", "size",
            "band count", "data type", "nodata", "pixel counts", "overlap"
            or "pixels" (from `check_pixels`).
        detail: a short plain explanation.
    """

    field: str
    detail: str

    def __str__(self) -> str:
        """Return the problem as "field: detail".

        Returns:
            The field and detail joined by ": ".
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
        """
        return not self.problems


def check_merged_cog(tiles: Sequence[str | PathLike], output: str | PathLike) -> MergeCheck:
    """Check that `output` is a valid COG holding exactly the pixels of `tiles`.

    The expected grid is the union of the tiles, as `find_union_grid` gives it. The
    checks are:
    - a valid COG (any `cog_validate` error or warning is a problem);
    - overviews whose smallest level fits in one block (an image that fits in one
      block needs none, and every band must have the same levels);
    - CRS (compared as a CRS), transform (with a tiny float tolerance), size,
      band count, data type and nodata equal to the tiles';
    - no two tiles overlap;
    - per band, the count of each value equals the sum over the tiles at full
      resolution, where the expected count of the fill value (the nodata, or 0
      when there is none) also includes the gap pixels no tile covers. NaN
      values count as one value.

    Pixels are verified by these value counts only, which assumes class data
    (few distinct values). Pixel counts are skipped when the band count or data
    type already differs, and when a band of a type wider than 16 bits holds
    more than 100,000 distinct values (in one window alone, or in total) in
    both the tiles and the output. A skipped count check adds no problem, the
    other checks still run, and `counts_checked` is False. When only one side,
    the tiles or the output, goes over the limit in a band, the counts cannot
    match, so that is a "pixel counts" problem and `counts_checked` is True.
    `check_pixels` compares every pixel with the tiles and works for any
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
        TileMismatchError: if the tiles do not share one grid (see `find_union_grid`).
        rasterio.errors.RasterioIOError: if GDAL cannot open a tile or the output.
        rasterio.errors.RasterioError: if GDAL cannot read their pixels.
    """
    grid, placed = _validate_and_place_tiles(tiles)
    if np.dtype(grid.dtype).kind == "c":
        raise ValueError(f"pixel counts are not supported for complex data ({grid.dtype})")
    problems: list[Problem] = []
    # Checks on the output file itself: COG validity, overviews and grid fields.
    with rasterio.open(output) as output_dataset:
        problems += _check_cog_validity(output)
        problems += _check_overviews(output_dataset, output)
        problems += _check_grid(output_dataset, grid)
        same_count_and_dtype = output_dataset.count == grid.count and set(output_dataset.dtypes) == {grid.dtype}
    # Overlapping tiles are reported as a problem here, not raised.
    try:
        check_no_overlap(placed)
    except TileMismatchError as error:
        problems.append(Problem("overlap", str(error)))
    # Compare pixel counts only when the output has the same band count and data type.
    counts_checked = False
    if same_count_and_dtype:
        count_problems = _check_counts([h.tile for h, _, _ in placed], output, grid)
        if count_problems is not None:
            problems += count_problems
            counts_checked = True
    return MergeCheck(output=output, problems=tuple(problems), counts_checked=counts_checked)


def _check_cog_validity(output: str | PathLike) -> list[Problem]:
    """Return a problem for each error and each warning `cog_validate` gives.

    Args:
        output: a local path or URL of the file to check.

    Returns:
        One "cog" problem per error or warning; empty when the file is a clean COG.
    """
    is_valid, errors, warnings = cog_validate(str(output), quiet=True)
    problems = [Problem("cog", f"error: {e}") for e in errors]
    problems += [Problem("cog", f"warning: {w}") for w in warnings]
    if not is_valid and not errors:
        problems.append(Problem("cog", "not a valid COG"))
    return problems


def _check_overviews(output_dataset: DatasetReader, output: str | PathLike) -> list[Problem]:
    """Return a problem when the overviews do not go down to one block.

    Args:
        output_dataset: the open output.
        output: its path or URL, used to open the smallest overview level.

    Returns:
        "overviews" problems: bands with different overview levels, or a smallest
        level (the full image when there are no overviews) larger than one block.

    Raises:
        rasterio.errors.RasterioIOError: if GDAL cannot open an overview level.
    """
    problems = []
    factors = output_dataset.overviews(1)
    # Every band must have the same overview levels as band 1.
    for band in range(2, output_dataset.count + 1):
        if output_dataset.overviews(band) != factors:
            problems.append(Problem(
                "overviews", f"band {band} has levels {output_dataset.overviews(band)}, band 1 has {factors}"
            ))
    # Size of the smallest level; with no overviews that is the full image.
    if factors:
        with rasterio.open(output, overview_level=len(factors) - 1) as smallest:
            width, height = smallest.width, smallest.height
    else:
        width, height = output_dataset.width, output_dataset.height
    block_height, block_width = output_dataset.block_shapes[0]
    # The smallest level must fit in one block.
    if width > block_width or height > block_height:
        problems.append(Problem(
            "overviews",
            f"the smallest level is {width} x {height} pixels, larger than one "
            f"{block_width} x {block_height} block (levels: {factors or 'none'})",
        ))
    return problems


def _same_transform(actual: Affine, expected: Affine) -> bool:
    """Return True when two transforms place pixels in the same place, allowing float noise.

    Args:
        actual: one transform.
        expected: the expected transform (north-up, no rotation).

    Returns:
        True if the pixel sizes match within a relative 1e-9, the rotation terms
        are zero and the origins are within a millionth of a pixel.
    """
    # Origin and rotation may be off by a millionth of a pixel, in map units.
    tolerance_x, tolerance_y = ALIGN_TOLERANCE * abs(expected.a), ALIGN_TOLERANCE * abs(expected.e)
    return (
        math.isclose(actual.a, expected.a, rel_tol=PIXEL_SIZE_TOLERANCE)
        and math.isclose(actual.e, expected.e, rel_tol=PIXEL_SIZE_TOLERANCE)
        and abs(actual.b) <= tolerance_x and abs(actual.d) <= tolerance_y
        and abs(actual.c - expected.c) <= tolerance_x and abs(actual.f - expected.f) <= tolerance_y
    )


def _check_grid(output_dataset: DatasetReader, grid: TileGrid) -> list[Problem]:
    """Return a problem for each grid field of the output that differs from the tiles'.

    Args:
        output_dataset: the open output.
        grid: the tiles' union grid.

    Returns:
        Problems with field "crs", "transform", "size", "band count", "data type"
        or "nodata", one per field that differs.
    """
    problems = []
    if output_dataset.crs is None or output_dataset.crs != grid.crs:
        problems.append(Problem("crs", f"output {output_dataset.crs}, tiles {grid.crs}"))
    if not _same_transform(output_dataset.transform, grid.transform):
        problems.append(Problem(
            "transform", f"output {tuple(output_dataset.transform)[:6]}, tiles {tuple(grid.transform)[:6]}"
        ))
    if (output_dataset.width, output_dataset.height) != (grid.width, grid.height):
        problems.append(Problem(
            "size", f"output {output_dataset.width} x {output_dataset.height}, tiles cover {grid.width} x {grid.height}"
        ))
    if output_dataset.count != grid.count:
        problems.append(Problem("band count", f"output {output_dataset.count}, tiles {grid.count}"))
    if set(output_dataset.dtypes) != {grid.dtype}:
        problems.append(Problem("data type", f"output {output_dataset.dtypes}, tiles {grid.dtype}"))
    if any(not nodata_equal(v, grid.nodata) for v in output_dataset.nodatavals):
        problems.append(Problem("nodata", f"output {output_dataset.nodatavals}, tiles {grid.nodata}"))
    return problems


def _windows(dataset: DatasetReader) -> Iterator[Window]:
    """Yield block-aligned windows that cover `dataset`, each about `_MAX_WINDOW_PIXELS` or less.

    A window is never smaller than one block, so a dataset with very large
    blocks gives windows of one block each.

    Args:
        dataset: an open raster.

    Yields:
        Windows in row order that together cover every pixel once.
    """
    block_height, block_width = dataset.block_shapes[0]
    # Whole blocks of rows that fit the pixel budget at full width (at least one block).
    rows = block_height * max(1, _MAX_WINDOW_PIXELS // (block_height * dataset.width))
    # If a full-width window fits the budget, windows span the whole width.
    if rows * dataset.width <= _MAX_WINDOW_PIXELS:
        cols = dataset.width
    else:
        # Very wide image: cut the width into whole blocks as well.
        cols = block_width * max(1, _MAX_WINDOW_PIXELS // (rows * block_width))
    # Walk the image row by row; windows at the edges are cut to fit.
    for row in range(0, dataset.height, rows):
        for col in range(0, dataset.width, cols):
            yield Window(col, row, min(cols, dataset.width - col), min(rows, dataset.height - row))


def _as_unsigned_bits(data: np.ndarray) -> np.ndarray:
    """Return `data` as unsigned integers of the same width, with every NaN made the same.

    Counting the bit patterns compares values exactly, and makes all NaNs one value.

    Args:
        data: an array of a real (integer or float) data type.

    Returns:
        The same values viewed as uint8, uint16, uint32 or uint64.
    """
    if data.dtype.kind == "f":
        # Replace every NaN with one NaN, so all NaNs share one bit pattern.
        data = np.where(np.isnan(data), np.array(np.nan, dtype=data.dtype), data)
    # Read the same bytes as unsigned integers of the same width.
    return data.view(f"u{data.dtype.itemsize}")


def _count_values(raster_path: str | PathLike) -> list[Counter]:
    """Count each pixel value in each band of `raster_path`, at full resolution, window by window.

    Byte and 16-bit bands are counted in a fixed-size histogram. Wider bands
    are counted in a Counter. A window's distinct values are checked against
    `_MAX_DISTINCT_VALUES` before they are listed or added, so the Counter never
    holds more than twice that many.

    Args:
        raster_path: a local path or URL GDAL can open, with a real data type.

    Returns:
        One Counter per band, mapping each value's bit pattern (see `_as_unsigned_bits`)
        to how many pixels hold it.

    Raises:
        _TooManyValuesError: if a band wider than 16 bits holds more than
            `_MAX_DISTINCT_VALUES` distinct values.
        rasterio.errors.RasterioIOError: if GDAL cannot open `raster_path`.
        rasterio.errors.RasterioError: if GDAL cannot read its pixels.
    """
    with rasterio.open(raster_path) as dataset:
        windows = list(_windows(dataset))
        per_band_counts = []
        for band in range(1, dataset.count + 1):
            itemsize = np.dtype(dataset.dtypes[band - 1]).itemsize
            # 8 and 16-bit bands: count every value with a fixed-size histogram.
            histogram = np.zeros(1 << (8 * itemsize), dtype=np.int64) if itemsize <= 2 else None
            counts: Counter = Counter()
            for window in windows:
                bits = _as_unsigned_bits(dataset.read(band, window=window)).ravel()
                if histogram is not None:
                    histogram += np.bincount(bits, minlength=histogram.size)
                else:
                    # Wider bands: sort so equal values sit side by side, then
                    # count each run of equal values instead of looping per pixel.
                    bits.sort()
                    # True wherever a new value starts.
                    is_new_value = bits[1:] != bits[:-1]
                    # Stop early if this window alone has too many distinct values.
                    if int(np.count_nonzero(is_new_value)) + 1 > _MAX_DISTINCT_VALUES:
                        raise _TooManyValuesError(band)
                    # Start index and length of each run, added to the running counts.
                    run_starts = np.concatenate(([0], np.flatnonzero(is_new_value) + 1))
                    n = np.diff(np.append(run_starts, bits.size))
                    counts.update(dict(zip(bits[run_starts].tolist(), n.tolist())))
                    if len(counts) > _MAX_DISTINCT_VALUES:
                        raise _TooManyValuesError(band)
            if histogram is not None:
                counts.update({int(v): int(histogram[v]) for v in np.flatnonzero(histogram)})
            per_band_counts.append(counts)
        return per_band_counts


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
    # Add up each tile's value counts; give up when a band has too many distinct values.
    for tile in tiles:
        with rasterio.open(tile) as dataset:
            tile_pixels += dataset.width * dataset.height
        try:
            tile_counts = _count_values(tile)
        except _TooManyValuesError as error:
            return error.band
        for band, (band_counts, counts) in enumerate(zip(expected, tile_counts), start=1):
            band_counts.update(counts)
            if len(band_counts) > _MAX_DISTINCT_VALUES:
                return band
    # Pixels no tile covers hold the fill value: the nodata, or 0 when there is none.
    gap = grid.width * grid.height - tile_pixels
    if gap > 0:
        fill = np.array([0 if grid.nodata is None else grid.nodata], dtype=np.dtype(grid.dtype))
        # Counts are keyed by bit pattern, so turn the fill value into one the same way.
        fill_bits = int(_as_unsigned_bits(fill)[0])
        for band_counts in expected:
            band_counts[fill_bits] += gap
    # The gap fill may add one more value, so check the limit again.
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
    # The tiles hold too many distinct values (`expected` is the band number). If the
    # output does too, skip the check; if not, the counts cannot match.
    if isinstance(expected, int):
        try:
            output_counts = _count_values(output)
        except _TooManyValuesError:
            return None
        held = len(output_counts[expected - 1])
        return [
            Problem(
                "pixel counts",
                f"band {expected}: tiles hold more than {_MAX_DISTINCT_VALUES:,} "
                f"distinct values, output holds {held:,}",
            )
        ]

    # The tiles are within the limit but the output is not: the counts cannot match.
    try:
        output_counts = _count_values(output)
    except _TooManyValuesError as error:
        held = len(expected[error.band - 1])
        return [
            Problem(
                "pixel counts",
                f"band {error.band}: output holds more than {_MAX_DISTINCT_VALUES:,} "
                f"distinct values, tiles hold {held:,}",
            )
        ]
    # Per band, list the values whose counts differ.
    problems = []
    for band, (want, got) in enumerate(zip(expected, output_counts), start=1):
        differ = sorted(v for v in want.keys() | got.keys() if want[v] != got[v])
        if not differ:
            continue
        listed = "; ".join(
            f"value {_value_from_bits(v, dtype)}: output {got[v]}, tiles {want[v]}"
            for v in differ[:_MAX_LISTED_VALUES]
        )
        more = f" (and {len(differ) - _MAX_LISTED_VALUES} more)" if len(differ) > _MAX_LISTED_VALUES else ""
        problems.append(Problem("pixel counts", f"band {band}: {listed}{more}"))
    return problems


def check_pixels(tiles: Sequence[str | PathLike], output: str | PathLike) -> tuple[Problem, ...]:
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
        TileMismatchError: if the tiles do not share one grid (see `find_union_grid`),
            or two tiles overlap (field "overlap").
        rasterio.errors.RasterioIOError: if GDAL cannot open a tile or the output.
        rasterio.errors.RasterioError: if GDAL cannot read their pixels.
    """
    grid, placed = _validate_and_place_tiles(tiles)
    dtype = np.dtype(grid.dtype)
    if dtype.kind == "c":
        raise ValueError(f"pixel comparison is not supported for complex data ({grid.dtype})")
    check_no_overlap(placed)
    fill = 0 if grid.nodata is None else grid.nodata
    # ExitStack keeps the output and every tile open until the comparison ends.
    with ExitStack() as stack:
        output_dataset = stack.enter_context(rasterio.open(output))
        grid_mismatch = _grid_mismatch_reason(output_dataset, grid)
        if grid_mismatch:
            return (Problem("pixels", f"not compared: {grid_mismatch}"),)
        placed_datasets = [(stack.enter_context(rasterio.open(h.tile)), col, row) for h, col, row in placed]
        windows = list(_windows(output_dataset))
        problems = []
        for band in range(1, grid.count + 1):
            differing_count = 0
            first_differences: list[str] = []
            for window in windows:
                # Start the window as all fill, then paste in each tile that reaches it.
                expected = np.full((window.height, window.width), fill, dtype=dtype)
                for tile_dataset, col, row in placed_datasets:
                    _paste(expected, window, tile_dataset, band, col, row)
                found = output_dataset.read(band, window=window)
                # True wherever the output differs from what is expected.
                differs = _as_unsigned_bits(expected) != _as_unsigned_bits(found)
                differing_count += int(np.count_nonzero(differs))
                # Keep the first few differing places for the message.
                if len(first_differences) < _MAX_LISTED_PIXELS and differing_count:
                    first_differences += _describe_first_differences(differs, expected, found, window, _MAX_LISTED_PIXELS - len(first_differences))
            if differing_count:
                noun = "pixel differs" if differing_count == 1 else "pixels differ"
                problems.append(Problem(
                    "pixels", f"band {band}: {differing_count} {noun}; first: {'; '.join(first_differences)}"
                ))
    return tuple(problems)


def _grid_mismatch_reason(output_dataset: DatasetReader, grid: TileGrid) -> str | None:
    """Return why the output's pixels cannot be lined up with the tiles', or None.

    Args:
        output_dataset: the open output.
        grid: the tiles' union grid.

    Returns:
        A short plain reason when the size, transform, band count or data type
        differs from the grid; None when they all match.
    """
    if (output_dataset.width, output_dataset.height) != (grid.width, grid.height):
        return f"output is {output_dataset.width} x {output_dataset.height}, tiles cover {grid.width} x {grid.height}"
    if not _same_transform(output_dataset.transform, grid.transform):
        return f"output transform {tuple(output_dataset.transform)[:6]}, tiles {tuple(grid.transform)[:6]}"
    if output_dataset.count != grid.count:
        return f"output has {output_dataset.count} bands, tiles {grid.count}"
    if set(output_dataset.dtypes) != {grid.dtype}:
        return f"output data type {output_dataset.dtypes}, tiles {grid.dtype}"
    return None


def _paste(
    expected: np.ndarray, window: Window, tile_dataset: DatasetReader, band: int, col: int, row: int
) -> None:
    """Copy the part of one tile that falls inside `window` into `expected`, in place.

    Nothing is copied when the tile does not reach the window.

    Args:
        expected: the window's expected values, shape (window height, window width).
        window: a window of the union grid.
        tile_dataset: the open tile.
        band: the band number, from 1.
        col: the tile's left column on the union grid.
        row: the tile's top row on the union grid.


    Raises:
        rasterio.errors.RasterioError: if GDAL cannot read the tile's pixels.
    """
    window_col, window_row = int(window.col_off), int(window.row_off)
    # The part of the tile inside the window, as a box on the union grid.
    left, right = max(window_col, col), min(window_col + window.width, col + tile_dataset.width)
    top, bottom = max(window_row, row), min(window_row + window.height, row + tile_dataset.height)
    if left >= right or top >= bottom:
        return
    # The same box in the tile's own pixels, to read.
    tile_window = Window(left - col, top - row, right - left, bottom - top)
    # The same box in the window's pixels, to write.
    expected[top - window_row:bottom - window_row, left - window_col:right - window_col] = tile_dataset.read(band, window=tile_window)


def _describe_first_differences(
    differs: np.ndarray, expected: np.ndarray, found: np.ndarray, window: Window, limit: int
) -> list[str]:
    """Describe the first differing pixels of one window, in row order.

    Searches row by row, so memory stays at one row however many pixels differ.

    Args:
        differs: True where the window's pixels differ.
        expected: the window's expected values.
        found: the window's values in the output.
        window: the window on the union grid, to give full-grid places.
        limit: the most pixels to describe (1 or more).

    Returns:
        Up to `limit` strings "column C, row R: expected E, found F".
    """
    places: list[str] = []
    for row in range(differs.shape[0]):
        # Columns of differing pixels in this row, only as many as are still needed.
        for col in np.flatnonzero(differs[row])[:limit - len(places)]:
            places.append(
                f"column {int(window.col_off) + int(col)}, row {int(window.row_off) + row}: "
                f"expected {expected[row, col].item()}, found {found[row, col].item()}"
            )
        if len(places) >= limit:
            break
    return places


def _value_from_bits(bits: int, dtype: np.dtype) -> int | float:
    """Return the pixel value whose bit pattern is `bits`, for messages.

    Args:
        bits: a bit pattern from `_as_unsigned_bits`.
        dtype: the data type the pattern belongs to.

    Returns:
        The value as a Python int or float.
    """
    # Put the bits in an unsigned array, then read the same bytes as the real type.
    return np.array([bits], dtype=f"u{dtype.itemsize}").view(dtype)[0].item()

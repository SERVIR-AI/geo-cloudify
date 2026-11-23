"""COG verification: check that a merged COG is a faithful copy of its tiles.

`check_merged_cog` runs these checks on the output of `merge_to_cog` and returns
every problem it finds, not only the first:

- it is a valid COG (`rio_cogeo.cog_validate`; errors and warnings both count);
- it has overviews down to one block, or none when it fits in one block;
- its CRS, transform, size, band count, data type and nodata match the tiles;
- no two tiles overlap;
- every full-resolution pixel equals what the tiles hold at that place, or the
  fill value in gaps between tiles. This catches dropped, swapped or changed pixels.

Pixels are read in block-aligned windows, so a whole raster is never held in memory.
"""

import math
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
    _GDAL_ERRORS,
    PIXEL_SIZE_TOLERANCE,
    GeoCloudifyError,
    InvalidInputError,
    TileGrid,
    TileMismatchError,
    check_no_overlap,
    _TileHeader,
    _validate_and_place_tiles,
    nodata_equal,
)

# About how many pixels one read may hold. Windows are rounded to whole blocks.
_MAX_WINDOW_PIXELS = 1 << 24
# How many differing pixels one "pixels" problem lists.
_MAX_LISTED_PIXELS = 5


@dataclass(frozen=True)
class Problem:
    """One thing wrong with a merged COG.

    Attributes:
        field: what is wrong: "cog", "overviews", "crs", "transform", "size",
            "band count", "data type", "nodata", "overlap" or "pixels".
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
    """

    output: str | PathLike
    problems: tuple[Problem, ...]

    @property
    def passed(self) -> bool:
        """Return True when no problem was found.

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
    - every pixel of every band, at full resolution, equals the value the tiles
      hold at that place, or the fill value (the nodata, or 0 when there is none)
      where no tile covers it. Values are compared by their bits, so NaN equals
      NaN. Works for any real data type, one band window at a time.

    Pixels are not compared when the tiles overlap, or when the output's size,
    transform, band count or data type differs from the tiles'. Then a "pixels"
    problem says "not compared" and why.

    Args:
        tiles: a list or tuple of one or more local paths or URLs GDAL can open:
            the tiles the output was merged from.
        output: a local path or URL of the merged COG.

    Returns:
        A MergeCheck listing every problem found (empty when all checks pass).
        A changed, missing or misplaced pixel gives one "pixels" problem per band
        that differs, with how many pixels differ and the first few places.

    Raises:
        InvalidInputError: if `tiles` is empty or a single path instead of a list, or
            the tiles hold complex numbers (the pixel comparison does not support them).
        TileMismatchError: if the tiles do not share one grid (see `find_union_grid`).
        GeoCloudifyError: if GDAL cannot open or read a tile or the output, with
            the GDAL or rasterio error as its cause.
    """
    grid, placed = _validate_and_place_tiles(tiles)
    if np.dtype(grid.dtype).kind == "c":
        raise InvalidInputError(f"pixel comparison is not supported for complex data ({grid.dtype})")
    problems: list[Problem] = []
    try:
        with rasterio.open(output) as output_dataset:
            # Checks on the output file itself: COG validity, overviews and grid fields.
            problems += _check_cog_validity(output)
            problems += _check_overviews(output_dataset, output)
            problems += _check_grid(output_dataset, grid)
            # Overlapping tiles are reported as a problem here, not raised. With an
            # overlap there is no one expected value, so pixels are not compared.
            try:
                check_no_overlap(placed)
            except TileMismatchError as error:
                problems.append(Problem("overlap", str(error)))
                problems.append(Problem("pixels", "not compared: tiles overlap"))
            else:
                problems += _check_pixels(placed, grid, output_dataset)
    except _GDAL_ERRORS as error:
        raise GeoCloudifyError(f"output {output}: cannot check against its tiles: {error}") from error
    return MergeCheck(output=output, problems=tuple(problems))


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

    Comparing the bit patterns compares values exactly, and makes all NaNs one value.

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


def _check_pixels(
    placed: Sequence[tuple[_TileHeader, int, int]], grid: TileGrid, output_dataset: DatasetReader
) -> list[Problem]:
    """Compare every full-resolution pixel of the output with what the tiles hold there.

    Args:
        placed: each tile's header with its column and row on the union grid,
            as returned by `_validate_and_place_tiles`; no two tiles overlap.
        grid: the tiles' union grid, of a real data type.
        output_dataset: the open output.

    Returns:
        One "pixels" problem per band that differs, giving how many pixels
        differ and up to `_MAX_LISTED_PIXELS` of them as column, row, expected
        and found value. A single "pixels" problem saying "not compared" when the
        output's size, transform, band count or data type differs from the
        tiles'. Empty when every pixel matches.

    Raises:
        rasterio.errors.RasterioIOError: if GDAL cannot open a tile.
        rasterio.errors.RasterioError: if GDAL cannot read the pixels of a tile or the output.
    """
    grid_mismatch = _grid_mismatch_reason(output_dataset, grid)
    if grid_mismatch:
        return [Problem("pixels", f"not compared: {grid_mismatch}")]
    dtype = np.dtype(grid.dtype)
    fill = 0 if grid.nodata is None else grid.nodata
    # ExitStack keeps every tile open until the comparison ends.
    with ExitStack() as stack:
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
    return problems


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

"""Top-level explore step: detect the kind, group companions, decide the action.

This module owns what every format shares: detecting whether a file can be
converted by opening it, keeping companion files together with their main file,
deciding between convert, pass through and skip, and building the plan. The
format-specific work is handed to `cog/explore.py` (and later `zarr/` and
`geoparquet/`). Exploration reads metadata only: it writes no files and no job
status, so `jobs.py` stays the only writer of status.
"""

import logging
from pathlib import Path

import rasterio
from rasterio.io import DatasetReader

from geoconvert.cog import explore as cog_explore
from geoconvert.plan import Action, JobPlan, Kind, PlanItem

logger = logging.getLogger(__name__)

# One reason for every file GDAL opens but that is not a COG-bound raster.
# `detect()` answers yes or no, not why, so the reason names the two causes.
SKIP_REASON = "not a COG-bound raster (not georeferenced, or a NetCDF/HDF cube)"

# GDAL's sentence for a file it cannot open at all. It is the same text for a vector
# file, a text file and a corrupt raster, and it would read as if a Shapefile were an
# unsupported format, so it is replaced with a plain reason (GDAL 3.12).
GDAL_NOT_RECOGNIZED = "not recognized as being in a supported file format"
UNSUPPORTED_REASON = (
    "not a raster GDAL can open (a vector, a text file, or a damaged file; "
    "vectors come in a later version)"
)
READ_ERROR_PREFIX = "could not read: "

# World files that GDAL only reads when the raster has no georeferencing of its own.
WORLD_FILE_SUFFIXES = (".tfw", ".tifw", ".wld")

try:
    # GDAL's own errors come from rasterio's private module and are not subclasses
    # of RasterioError, so both families are caught. The fallback keeps the module
    # importable if a rasterio upgrade moves or renames it.
    from rasterio._err import CPLE_BaseError

    FILE_ERRORS: tuple[type[BaseException], ...] = (rasterio.errors.RasterioError, CPLE_BaseError)
except ImportError:  # pragma: no cover - depends on the rasterio version
    FILE_ERRORS = (rasterio.errors.RasterioError,)


def build_plan(
    files_by_source: dict[str, list[Path]], cog_kwargs_by_source: dict[str, dict | None] | None
) -> JobPlan:
    """Group files per source and return the job plan: what to convert, keep or skip.

    Files are handled per source, so an item never mixes two sources and a
    companion is only attached to a main file of the same source. Items are
    sorted by source, then by main file path, so the same input gives the same
    plan.

    Args:
        files_by_source: files that downloaded fine, keyed by the source string
            from the request.
        cog_kwargs_by_source: COG options per source string, or None. A source
            that is missing or maps to None uses no options. Each source's options
            are passed unchanged to the COG explorer, which validates the names.

    Returns:
        The plan for the whole job.

    Raises:
        ValueError: if a file needs COG settings and its source's options have an unknown
            option name.
        Exception: any error that is not a rasterio read error propagates, so a
            fault in the code fails the job instead of being reported as a skip.
    """
    items: list[PlanItem] = []
    for source in sorted(files_by_source):
        cog_kwargs = (cog_kwargs_by_source or {}).get(source)
        items.extend(_plan_source(source, files_by_source[source], cog_kwargs))
    items.sort(key=lambda item: (item.source, item.files[0]))
    return JobPlan(items=items)


def _plan_source(source: str, files: list[Path], cog_kwargs: dict | None) -> list[PlanItem]:
    """Return the plan items for one source's files.

    Raises:
        ValueError: if cog_kwargs has an unknown option name.
        Exception: any error that is not a rasterio read error propagates.
    """
    # Keys are resolved so GDAL's absolute paths match however the request wrote them.
    by_resolved: dict[Path, Path] = {path.resolve(): path for path in files}
    ordered = sorted(by_resolved.values())

    items: list[PlanItem] = []
    detected: list[Path] = []
    companions: dict[Path, list[Path]] = {}
    read_errors: dict[Path, str] = {}  # resolved path -> GDAL's message
    handled: set[Path] = set()  # resolved paths already placed in an item

    # Pass 1: find the main files and their companions.
    for path in ordered:
        try:
            # Open once first, so a file GDAL cannot read at all is reported with
            # GDAL's own message; `detect()` would swallow that error into False.
            with rasterio.open(path) as dataset:
                found = _companions(dataset, path, by_resolved)
            if not cog_explore.detect(path):
                continue
            companions[path] = found
            detected.append(path)
        except FILE_ERRORS as exc:
            # Recorded, not reported yet: a world file GDAL cannot open is fine
            # when it turns out to be a companion of a raster.
            read_errors[path.resolve()] = str(exc)

    # Pass 2: a `.tif.ovr` can look like a raster itself. A file that belongs to
    # another main file is a companion, never a main file of its own.
    companion_paths = {path.resolve() for found in companions.values() for path in found}
    mains = [path for path in detected if path.resolve() not in companion_paths]

    # Pass 3: explore each main file, which decides convert or pass through.
    for main in mains:
        try:
            items.append(cog_explore.explore(main, companions[main], source, cog_kwargs))
            handled.add(main.resolve())
            handled.update(path.resolve() for path in companions[main])
        except FILE_ERRORS as exc:
            items.append(_skip_item(source, main, _plain_reason(main, str(exc))))
            handled.add(main.resolve())
            logger.warning("source %s: cannot explore %s: %s", source, main, exc)

    # Pass 4: everything left over is not part of a conversion. A file GDAL could
    # not open keeps a short version of its message, so the caller sees why.
    for path in ordered:
        resolved = path.resolve()
        if resolved in handled:
            continue
        message = read_errors.get(resolved)
        items.append(_skip_item(source, path, _plain_reason(path, message) if message else SKIP_REASON))

    _log_summary(source, len(ordered), items)
    return items


def _log_summary(source: str, file_count: int, items: list[PlanItem]) -> None:
    """Log one line per source: what was planned and how many files could not be read."""
    converted = sum(1 for item in items if item.action is Action.CONVERT)
    passed = sum(1 for item in items if item.action is Action.PASS_THROUGH)
    skipped = [item for item in items if item.action is Action.SKIP]
    unreadable = sum(1 for item in skipped if item.skip_reason != SKIP_REASON)
    logger.info(
        "source %s: %d file(s): %d to convert, %d passed through, %d skipped (%d could not be read)",
        source,
        file_count,
        converted,
        passed,
        len(skipped),
        unreadable,
    )


def _plain_reason(path: Path, message: str) -> str:
    """Return a short skip reason from GDAL's message, without the server's path.

    GDAL says the same thing for a vector, a text file and a damaged raster, so that
    case gets one plain reason. Any other error keeps GDAL's message, so it can be
    acted on.
    """
    detail = _without_path(path, message)
    if GDAL_NOT_RECOGNIZED in detail.lower():
        return UNSUPPORTED_REASON
    return READ_ERROR_PREFIX + (detail or "unknown GDAL error")


def _without_path(path: Path, message: str) -> str:
    """Return GDAL's message with this file's path removed, including quotes."""
    for candidate in (str(path), str(path.resolve())):
        message = message.replace(candidate, "")
    return message.strip().strip("'\"`").strip(" :").strip()


def _companions(dataset: DatasetReader, path: Path, by_resolved: dict[Path, Path]) -> list[Path]:
    """Return the companions of one open raster that belong to the same source, sorted.

    GDAL's own list (`dataset.files`) covers the files it opened, for example
    `.ovr`, `.msk` and `.aux.xml`. World files are added by name, because GDAL
    does not list them when the raster carries its own georeferencing.
    """
    listed = [Path(name) for name in dataset.files]
    candidates = listed + [path.with_suffix(suffix) for suffix in WORLD_FILE_SUFFIXES]
    resolved_main = path.resolve()

    found = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved != resolved_main and resolved in by_resolved:
            found.add(by_resolved[resolved])
    return sorted(found)


def _skip_item(source: str, path: Path, reason: str) -> PlanItem:
    """Return a skip entry for one file that will not be converted."""
    return PlanItem(
        source=source,
        kind=Kind.OTHER,
        action=Action.SKIP,
        files=[str(path)],
        skip_reason=reason,
    )

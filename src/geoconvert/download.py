"""Download step: put every source into one job folder and return the local file paths.

Sources can be local paths (files or folders) or http(s) URLs. Archives (zip, tar) are
unpacked into a folder named after the archive, and the archive itself is removed.
Originals are never changed.
"""

import logging
import shutil
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

from geoconvert.config import get_config

logger = logging.getLogger(__name__)

ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")
CHUNK = 1024 * 1024
# Longest wait for the server to connect or send the next bytes. Not a limit on the whole download.
STALL_TIMEOUT_SECONDS = 60


class DownloadError(Exception):
    """A source could not be fetched or unpacked."""


def fetch_sources(sources: list[str], job_id: str, download_dir: str | Path | None = None) -> list[Path]:
    """Fetch all sources into `<download_dir>/<job_id>/` and return the files added by this call, sorted.

    `download_dir` defaults to `download.dir` from the config. When the job folder
    already holds files from earlier calls, only the newly added files are returned,
    so each source is attributed only its own files.

    Raises:
        DownloadError: if any source cannot be fetched or unpacked.
    """
    base = Path(download_dir) if download_dir is not None else get_config().download.dir
    job_dir = base / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    before = {p for p in job_dir.rglob("*") if p.is_file()}

    for source in sources:
        scheme = urlparse(source).scheme
        if scheme in ("http", "https"):
            _download_url(source, job_dir)
        elif scheme in ("", "file") or len(scheme) == 1:  # one letter: a Windows drive
            _copy_local(source, job_dir)
        else:
            raise DownloadError(f"{source}: scheme '{scheme}' is not supported")

    for archive in sorted(p for p in job_dir.rglob("*") if p.is_file() and _is_archive(p)):
        _unpack(archive)

    after = {p for p in job_dir.rglob("*") if p.is_file()}
    return sorted(after - before)


def _copy_local(source: str, job_dir: Path) -> None:
    """Copy a local file or folder into the job folder. Originals are never changed.

    Raises:
        DownloadError: if the source does not exist or its name is already taken.
    """
    src = Path(unquote(urlparse(source).path) if source.startswith("file:") else source).expanduser()
    if not src.exists():
        raise DownloadError(f"{source}: not found")
    target = _available_target(job_dir / src.name)
    if src.is_dir():
        shutil.copytree(src, target)
    else:
        shutil.copy2(src, target)


def _download_url(url: str, job_dir: Path) -> None:
    """Download a URL into the job folder, retrying transient failures.

    Raises:
        DownloadError: if the download still fails after retries, or is not retryable.
    """
    download = get_config().download
    for attempt in range(1, download.retry_attempts + 1):
        try:
            _download_once(url, job_dir)
            return
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            # Give up on permanent errors, or when this was the last attempt.
            if not _is_transient(exc) or attempt == download.retry_attempts:
                if isinstance(exc, urllib.error.HTTPError):
                    raise DownloadError(f"{url}: HTTP {exc.code} {exc.reason}") from exc
                raise DownloadError(f"{url}: {exc.reason}") from exc
            logger.warning("retrying %s after error: %s", url, exc)
            time.sleep(download.retry_delay)


def _download_once(url: str, job_dir: Path) -> None:
    """Download a URL in one attempt, streaming to a `.part` file then renaming.

    Raises:
        DownloadError: if the target name is already taken in the job folder.
    """
    with urllib.request.urlopen(url, timeout=STALL_TIMEOUT_SECONDS) as response:
        name = response.headers.get_filename() or Path(unquote(urlparse(response.url).path)).name
        target = _available_target(job_dir / (Path(name).name or "download"))
        partial = target.with_name(target.name + ".part")
        try:
            with partial.open("wb") as out:
                shutil.copyfileobj(response, out, CHUNK)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        partial.rename(target)


def _is_transient(exc: BaseException) -> bool:
    """Return True for failures worth retrying (HTTP 5xx and network errors)."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code >= 500
    return isinstance(exc, urllib.error.URLError)


def _unpack(archive: Path) -> None:
    """Unpack an archive into a folder named after it, then remove the archive.

    Raises:
        DownloadError: if an entry would escape the folder or the archive is unreadable.
    """
    out = _available_target(archive.parent / _archive_stem(archive))
    out.mkdir()
    try:
        if archive.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive) as zf:
                for member in zf.namelist():
                    # Reject entries that would land outside the unpack folder.
                    if not (out / member).resolve().is_relative_to(out.resolve()):
                        raise DownloadError(f"{archive.name}: '{member}' would unpack outside the folder")
                zf.extractall(out)
        else:
            with tarfile.open(archive) as tf:
                try:
                    tf.extractall(out, filter="data")
                except tarfile.FilterError as e:
                    raise DownloadError(f"{archive.name}: '{e.tarinfo.name}' would unpack outside the folder") from e
    except (zipfile.BadZipFile, tarfile.ReadError) as e:
        raise DownloadError(f"{archive.name}: not a readable archive ({e})") from e
    archive.unlink()


def _is_archive(path: Path) -> bool:
    """Return True if the path looks like a supported archive."""
    return path.name.lower().endswith(ARCHIVE_SUFFIXES)


def _archive_stem(path: Path) -> str:
    """Return the file name without its archive suffix."""
    name = path.name
    for suffix in sorted(ARCHIVE_SUFFIXES, key=len, reverse=True):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _available_target(target: Path) -> Path:
    """Return the path if nothing exists there yet, else raise.

    Raises:
        DownloadError: if a file or folder already exists at that path.
    """
    if target.exists():
        raise DownloadError(f"{target.name} already exists in the job folder (two sources with the same name)")
    return target

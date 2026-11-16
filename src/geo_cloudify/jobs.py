"""Job orchestration: run the steps in order and track job status.

`jobs.py` is the only code that writes job status. Each step returns its result
and records nothing itself; `run_job()` takes the result and updates the status.
"""

import logging
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from geo_cloudify import download, explore
from geo_cloudify.cog import convert
from geo_cloudify.plan import Action, JobPlan

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Lifecycle of a job, from creation to completion."""

    PENDING = "pending"
    DOWNLOADING = "downloading"
    CONVERTING = "converting"
    DONE = "done"
    FAILED = "failed"


class SourceStatus(str, Enum):
    """Outcome of one source within a job."""

    OK = "ok"
    FAILED = "failed"


@dataclass
class SourceOutcome:
    """Result of fetching one source."""

    source: str
    status: SourceStatus
    files: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class ItemOutcome:
    """Result of converting one plan item: its main file, and the output or the error."""

    file: str
    status: SourceStatus
    output: str | None = None
    error: str | None = None


@dataclass
class Job:
    """A conversion job: its sources, options, status, per-source results and plan.

    `error` holds the reason when the whole job fails.
    """

    id: str
    status: JobStatus = JobStatus.PENDING
    sources: list[str] = field(default_factory=list)
    results: list[SourceOutcome] = field(default_factory=list)
    cog_kwargs_by_source: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    plan: JobPlan | None = None
    conversions: list[ItemOutcome] = field(default_factory=list)
    error: str | None = None


class JobStore:
    """In-memory job store. Replace with a queue or database when jobs move off-process."""

    def __init__(self) -> None:
        """Create an empty store."""
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, sources: list[str], cog_kwargs_by_source: dict[str, dict[str, Any] | None] | None = None) -> Job:
        """Register a new job for the given sources and per-source COG options and return it.

    Returns:
        The new job, stored under a fresh ID.
    """
        job = Job(id=uuid.uuid4().hex, sources=sources, cog_kwargs_by_source=cog_kwargs_by_source or {})
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        """Return the job with the given ID, or None if it does not exist.

    Returns:
        The job, or None.
    """
        with self._lock:
            return self._jobs.get(job_id)


# Single store shared by the routes and the worker. Replaced by a queue or DB later.
store = JobStore()


def run_job(job: Job) -> None:
    """Run the steps for one job, in order. Only this function writes job status.

    Does not raise. Each source that fails to download is recorded as a failed
    result and the job continues. After downloads, the plan is built from the
    sources that downloaded, then each CONVERT item is converted into
    `<job folder>/output/`. A conversion that fails on its file is recorded on
    that item and the rest continue. An unexpected error marks the whole job
    failed and is kept in `job.error`.
    """
    job.status = JobStatus.DOWNLOADING
    for source in job.sources:
        try:
            paths = download.fetch_sources([source], job.id)
        except download.DownloadError as exc:
            logger.warning("source %s failed: %s", source, exc)
            job.results.append(SourceOutcome(source=source, status=SourceStatus.FAILED, error=str(exc)))
            continue
        except Exception as exc:  # unexpected error: stop the job, keep the reason
            logger.exception("job %s failed on source %s", job.id, source)
            job.status = JobStatus.FAILED
            job.error = f"unexpected error: {exc}"
            job.results.append(SourceOutcome(source=source, status=SourceStatus.FAILED, error=job.error))
            return
        job.results.append(SourceOutcome(source=source, status=SourceStatus.OK, files=[str(p) for p in paths]))

    # Only sources that downloaded take part in the plan.
    files_by_source = {
        result.source: [Path(f) for f in result.files] for result in job.results if result.status is SourceStatus.OK
    }
    try:
        job.plan = explore.build_plan(files_by_source, job.cog_kwargs_by_source)
    except Exception as exc:  # unexpected error: stop the job, keep the reason
        logger.exception("job %s failed during exploration", job.id)
        job.status = JobStatus.FAILED
        job.error = f"unexpected error: {exc}"
        return

    job.status = JobStatus.CONVERTING
    job_dir = download.get_config().download.dir / job.id
    for item in job.plan.items:
        if item.action is not Action.CONVERT:
            continue
        try:
            output = convert.to_cog(item, job_dir, job_dir / "output")
        except (*explore.FILE_ERRORS, ValueError, FileExistsError) as exc:
            # A problem with this file only: record it and convert the next item.
            logger.warning("conversion of %s failed: %s", item.files[0], exc)
            job.conversions.append(ItemOutcome(file=item.files[0], status=SourceStatus.FAILED, error=str(exc)))
            continue
        except Exception as exc:  # unexpected error: stop the job, keep the reason
            logger.exception("job %s failed converting %s", job.id, item.files[0])
            job.status = JobStatus.FAILED
            job.error = f"unexpected error: {exc}"
            return
        job.conversions.append(ItemOutcome(file=item.files[0], status=SourceStatus.OK, output=str(output)))
    job.status = JobStatus.DONE

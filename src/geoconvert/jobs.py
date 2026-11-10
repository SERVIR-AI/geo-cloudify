"""Job orchestration: run the steps in order and track job status.

`jobs.py` is the only code that writes job status. Each step returns its result
and records nothing itself; `run_job()` takes the result and updates the status.
"""

import logging
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum

from geoconvert import download

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Lifecycle of a job, from creation to completion."""

    PENDING = "pending"
    DOWNLOADING = "downloading"
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
class Job:
    """A conversion job: its sources, its status, and per-source results."""

    id: str
    status: JobStatus = JobStatus.PENDING
    sources: list[str] = field(default_factory=list)
    results: list[SourceOutcome] = field(default_factory=list)


class JobStore:
    """In-memory job store. Replace with a queue or database when jobs move off-process."""

    def __init__(self) -> None:
        """Create an empty store."""
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, sources: list[str]) -> Job:
        """Register a new job for the given sources and return it."""
        job = Job(id=uuid.uuid4().hex, sources=sources)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        """Return the job with the given ID, or None if it does not exist."""
        with self._lock:
            return self._jobs.get(job_id)


# Single store shared by the routes and the worker. Replaced by a queue or DB later.
store = JobStore()


def run_job(job: Job) -> None:
    """Run the steps for one job, in order. Only this function writes job status.

    Does not raise. Each source that fails to download is recorded as a failed
    result and the job continues; an unexpected error marks the whole job failed.
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
            job.results.append(SourceOutcome(source=source, status=SourceStatus.FAILED, error=f"unexpected error: {exc}"))
            return
        job.results.append(SourceOutcome(source=source, status=SourceStatus.OK, files=[str(p) for p in paths]))
    job.status = JobStatus.DONE

"""Job routes: create a job and read its status."""

from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, ConfigDict, field_validator

from geoconvert.cog.explore import validate_cog_kwargs
from geoconvert.jobs import run_job, store
from geoconvert.plan import to_json_safe_dict

router = APIRouter()


class SourceRequest(BaseModel):
    """One source in the request: a path or URL with its own COG options."""

    model_config = ConfigDict(extra="forbid")

    source: str
    cog_kwargs: dict[str, Any] | None = None

    @field_validator("cog_kwargs")
    @classmethod
    def _check_cog_kwargs(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        """Return the options unchanged if every name is known.

        Raises:
            ValueError: for an unknown option name (FastAPI returns HTTP 422).
        """
        if value:
            validate_cog_kwargs(value)
        return value


class CreateJobRequest(BaseModel):
    """Request body for creating a job. A source is a plain string or a SourceRequest."""

    model_config = ConfigDict(extra="forbid")

    sources: list[str | SourceRequest]


@router.post("/jobs")
def create_job(request: CreateJobRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    """Create a job and run it in the background; return its ID straight away.

    Extra keys in the request body and unknown cog_kwargs options are rejected (HTTP 422).
    """
    items = [SourceRequest(source=s) if isinstance(s, str) else s for s in request.sources]
    job = store.create([i.source for i in items], {i.source: i.cog_kwargs for i in items})
    background_tasks.add_task(run_job, job)
    return {"job_id": job.id}


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    """Return the job status, error, per-source results, plan and conversions.

    Raises:
        HTTPException: 404 if the job does not exist.
    """
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "job_id": job.id,
        "status": job.status.value,
        "error": job.error,
        "results": [
            {
                "source": result.source,
                "status": result.status.value,
                "files": result.files,
                "error": result.error,
            }
            for result in job.results
        ],
        "plan": to_json_safe_dict(job.plan) if job.plan else None,
        "conversions": [
            {"file": c.file, "status": c.status.value, "output": c.output, "error": c.error}
            for c in job.conversions
        ],
    }

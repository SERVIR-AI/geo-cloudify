"""Job routes: create a job and read its status."""

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, ConfigDict

from geoconvert.jobs import run_job, store

router = APIRouter()


class CreateJobRequest(BaseModel):
    """Request body for creating a job."""

    model_config = ConfigDict(extra="forbid")

    sources: list[str]


@router.post("/jobs")
def create_job(request: CreateJobRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    """Create a job and run it in the background; return its ID straight away."""
    job = store.create(request.sources)
    background_tasks.add_task(run_job, job)
    return {"job_id": job.id}


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    """Return the job status and per-source results."""
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "job_id": job.id,
        "status": job.status.value,
        "results": [
            {
                "source": result.source,
                "status": result.status.value,
                "files": result.files,
                "error": result.error,
            }
            for result in job.results
        ],
    }

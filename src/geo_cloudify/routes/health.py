"""Health check route."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Return a simple liveness check."""
    return {"status": "ok"}

"""FastAPI app, routes, request/response models."""

from fastapi import FastAPI

app = FastAPI(title="geoconvert")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

"""FastAPI app entrypoint: build the app and mount the routers."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from geo_cloudify.config import get_config
from geo_cloudify.routes import health, jobs
from geo_cloudify.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load settings and config once at startup, so a bad file stops the app early.

    Raises:
        FileNotFoundError: if the config file is missing.
        pydantic.ValidationError: if the config has unknown keys.
    """
    get_settings()
    get_config()
    yield


app = FastAPI(title="geo-cloudify", lifespan=lifespan)
app.include_router(health.router)
app.include_router(jobs.router)

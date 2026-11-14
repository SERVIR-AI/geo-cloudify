"""Deployment settings, read from environment variables or `.env`."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Deployment settings from environment variables or `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Path to the YAML file with tunable parameters (see config.py)
    config_path: Path = Field(default=Path("config.yaml"), validation_alias="GEO_CLOUDIFY_CONFIG")


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings, read once from the environment and `.env`."""
    return Settings()

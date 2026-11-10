"""Tunable parameters, read from a YAML file."""

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from geoconvert.settings import get_settings


class DownloadConfig(BaseModel):
    """Tunable parameters for the download step."""

    model_config = ConfigDict(extra="forbid")

    dir: Path = Path("tmp/downloads")
    retry_delay: float = 10.0  # seconds to wait between download attempts
    retry_attempts: int = 3    # total attempts, including the first


class Config(BaseModel):
    """All tunable parameters, grouped by module."""

    model_config = ConfigDict(extra="forbid")

    download: DownloadConfig = DownloadConfig()


def load_config(path: str | Path) -> Config:
    """Read and validate the YAML config. Relative paths in it are resolved from the file's folder."""
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text()) or {}
    config = Config.model_validate(data)
    if not config.download.dir.is_absolute():
        config.download.dir = (path.parent / config.download.dir).resolve()
    return config


@lru_cache
def get_config() -> Config:
    """Return the cached config, loaded once from the file named in settings."""
    return load_config(get_settings().config_path)

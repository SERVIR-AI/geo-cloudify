"""Unit tests for settings and config loading."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from geoconvert.config import load_config
from geoconvert.settings import Settings

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_repo_config_file_loads():
    config = load_config(REPO_ROOT / "config.yaml")
    assert config.download.dir == REPO_ROOT / "tmp" / "downloads"


def test_relative_download_dir_is_resolved_from_config_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("download:\n  dir: downloads\n")
    config = load_config(path)
    assert config.download.dir == tmp_path / "downloads"


def test_empty_config_uses_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("")
    config = load_config(path)
    assert config.download.dir == tmp_path / "tmp" / "downloads"


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("download:\n  folder: x\n")
    with pytest.raises(ValidationError):
        load_config(path)


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_config_path_read_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("GEOCONVERT_CONFIG", str(tmp_path / "other.yaml"))
    settings = Settings(_env_file=None)
    assert settings.config_path == tmp_path / "other.yaml"


def test_download_retry_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("download:\n  dir: downloads\n")
    config = load_config(path)
    assert config.download.retry_delay == 10.0
    assert config.download.retry_attempts == 3


def test_download_retry_override(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("download:\n  retry_delay: 1.5\n  retry_attempts: 5\n")
    config = load_config(path)
    assert config.download.retry_delay == 1.5
    assert config.download.retry_attempts == 5


def test_cog_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("")
    config = load_config(path)
    assert config.cog.compress == "DEFLATE"
    assert config.cog.blocksize == 512
    assert config.cog.num_threads == "ALL_CPUS"


def test_cog_override(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("cog:\n  compress: LZW\n  blocksize: 256\n  num_threads: 2\n")
    config = load_config(path)
    assert config.cog.compress == "LZW"
    assert config.cog.blocksize == 256
    assert config.cog.num_threads == 2

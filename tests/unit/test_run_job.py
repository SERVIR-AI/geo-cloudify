"""Unit tests for job orchestration (run_job)."""

import logging

from geoconvert.config import Config, DownloadConfig
from geoconvert.jobs import Job, JobStatus, SourceStatus, run_job


def test_failed_source_is_recorded_and_logged(tmp_path, monkeypatch, caplog):
    config = Config(download=DownloadConfig(dir=tmp_path / "downloads"))
    monkeypatch.setattr("geoconvert.download.get_config", lambda: config)
    caplog.set_level(logging.WARNING)

    job = Job(id="j1", sources=[str(tmp_path / "missing.tif")])
    run_job(job)

    assert job.status is JobStatus.DONE
    assert len(job.results) == 1
    assert job.results[0].status is SourceStatus.FAILED
    assert "not found" in job.results[0].error
    assert any("missing.tif" in record.getMessage() for record in caplog.records)

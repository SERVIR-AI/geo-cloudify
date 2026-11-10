"""Integration tests for the job routes (POST /jobs, GET /jobs/{id})."""

from pathlib import Path

from fastapi.testclient import TestClient

from geoconvert.api import app
from geoconvert.config import Config, DownloadConfig


def _client(tmp_path, monkeypatch):
    config = Config(download=DownloadConfig(dir=tmp_path / "downloads"))
    monkeypatch.setattr("geoconvert.download.get_config", lambda: config)
    return TestClient(app)


def _write(path: Path, text: str = "data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_create_job_returns_id(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    src = _write(tmp_path / "in" / "a.tif")
    response = client.post("/jobs", json={"sources": [str(src)]})
    assert response.status_code == 200
    assert response.json()["job_id"]


def test_get_job_returns_status_and_downloaded_files(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    src = _write(tmp_path / "in" / "a.tif")
    job_id = client.post("/jobs", json={"sources": [str(src)]}).json()["job_id"]
    response = client.get(f"/jobs/{job_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "done"
    assert len(body["results"]) == 1
    result = body["results"][0]
    assert result["status"] == "ok"
    assert Path(result["files"][0]).name == "a.tif"


def test_bad_source_is_reported_and_job_continues(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    good = _write(tmp_path / "in" / "a.tif")
    missing = tmp_path / "in" / "nope.tif"
    job_id = client.post("/jobs", json={"sources": [str(good), str(missing)]}).json()["job_id"]
    body = client.get(f"/jobs/{job_id}").json()
    assert body["status"] == "done"
    assert [r["status"] for r in body["results"]] == ["ok", "failed"]
    assert "not found" in body["results"][1]["error"]
    assert body["results"][0]["files"]


def test_unknown_job_is_404(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    response = client.get("/jobs/does-not-exist")
    assert response.status_code == 404


def test_extra_request_key_is_rejected(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    response = client.post("/jobs", json={"sources": [], "destination": "s3://bucket/x"})
    assert response.status_code == 422


def test_empty_sources_completes(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    job_id = client.post("/jobs", json={"sources": []}).json()["job_id"]
    body = client.get(f"/jobs/{job_id}").json()
    assert body["status"] == "done"
    assert body["results"] == []

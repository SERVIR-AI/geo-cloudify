"""Integration tests for the job routes (POST /jobs, GET /jobs/{id})."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from fastapi.testclient import TestClient

from geoconvert.api import app
from geoconvert.config import Config, DownloadConfig
from geoconvert.jobs import store


def _client(tmp_path, monkeypatch):
    config = Config(download=DownloadConfig(dir=tmp_path / "downloads"))
    monkeypatch.setattr("geoconvert.download.get_config", lambda: config)
    return TestClient(app)


def _geotiff(path: Path, size: int = 600, dtype: str = "uint8", nodata=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff", "width": size, "height": size, "count": 1, "dtype": dtype,
        "crs": "EPSG:4326", "transform": from_origin(0, 10, 0.01, 0.01), "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.ones((1, size, size), dtype=dtype))
    return path


def _plan_items(client, body: dict) -> list[dict]:
    job_id = client.post("/jobs", json=body).json()["job_id"]
    response = client.get(f"/jobs/{job_id}")
    assert response.status_code == 200
    return response.json()["plan"]["items"]


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


def test_each_source_lists_only_its_own_files(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    a = _write(tmp_path / "in" / "a.tif")
    b = _write(tmp_path / "in" / "b.nc")
    job_id = client.post("/jobs", json={"sources": [str(a), str(b)]}).json()["job_id"]
    results = client.get(f"/jobs/{job_id}").json()["results"]
    assert [Path(r["files"][0]).name for r in results] == ["a.tif", "b.nc"]
    assert [len(r["files"]) for r in results] == [1, 1]


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


def test_bad_archive_source_does_not_break_later_source(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    bad = tmp_path / "in" / "bad.zip"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"not a zip")
    good = _write(tmp_path / "in" / "good.tif")
    job_id = client.post("/jobs", json={"sources": [str(bad), str(good)]}).json()["job_id"]
    results = client.get(f"/jobs/{job_id}").json()["results"]
    assert [r["status"] for r in results] == ["failed", "ok"]
    assert "not a readable archive" in results[0]["error"]


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


def test_geotiff_gives_one_convert_item(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    src = _geotiff(tmp_path / "in" / "r.tif")
    items = _plan_items(client, {"sources": [str(src)]})
    assert len(items) == 1
    assert items[0]["action"] == "convert"


def test_nan_nodata_is_returned_as_string(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    src = _geotiff(tmp_path / "in" / "r.tif", dtype="float32", nodata=float("nan"))
    item = _plan_items(client, {"sources": [str(src)]})[0]
    assert item["fingerprint"]["nodata"] == "NaN"
    assert item["settings"]["NODATA"] == "NaN"


def test_cog_kwargs_nodata_lands_in_settings(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    src = _geotiff(tmp_path / "in" / "r.tif", dtype="int16")
    body = {"sources": [{"source": str(src), "cog_kwargs": {"nodata": -9999}}]}
    item = _plan_items(client, body)[0]
    assert item["settings"]["NODATA"] == -9999


def test_each_source_gets_its_own_cog_kwargs(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    a = _geotiff(tmp_path / "in" / "a.tif", dtype="int16")
    b = _geotiff(tmp_path / "in" / "b.tif", dtype="int16")
    body = {"sources": [
        {"source": str(a), "cog_kwargs": {"nodata": -9999}},
        {"source": str(b), "cog_kwargs": {"NODATA": -1}},
    ]}
    nodata = {Path(i["files"][0]).name: i["settings"]["NODATA"] for i in _plan_items(client, body)}
    assert nodata == {"a.tif": -9999, "b.tif": -1}


def test_source_without_options_keeps_declared_nodata(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    a = _geotiff(tmp_path / "in" / "a.tif", dtype="int16")
    b = _geotiff(tmp_path / "in" / "b.tif", dtype="int16", nodata=-32768)
    body = {"sources": [{"source": str(a), "cog_kwargs": {"nodata": -9999}}, str(b)]}
    nodata = {Path(i["files"][0]).name: i["settings"]["NODATA"] for i in _plan_items(client, body)}
    assert nodata == {"a.tif": -9999, "b.tif": -32768}


def test_job_wide_cog_kwargs_is_rejected(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    response = client.post("/jobs", json={"sources": [], "cog_kwargs": {"nodata": -9999}})
    assert response.status_code == 422


def test_extra_key_in_source_item_is_rejected(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    response = client.post("/jobs", json={"sources": [{"source": "a.tif", "zarr": {}}]})
    assert response.status_code == 422


def test_unknown_cog_option_is_422_and_creates_no_job(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    monkeypatch.setattr("geoconvert.jobs.store._jobs", {})
    body = {"sources": [{"source": "a.tif", "cog_kwargs": {"COMPRES": "LZW"}}]}
    response = client.post("/jobs", json=body)
    assert response.status_code == 422
    assert "unknown cog_kwargs option(s): COMPRES" in response.text
    assert store._jobs == {}


def test_text_file_named_tif_is_a_skip_item(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    src = _write(tmp_path / "in" / "a.tif")
    items = _plan_items(client, {"sources": [str(src)]})
    assert [i["action"] for i in items] == ["skip"]

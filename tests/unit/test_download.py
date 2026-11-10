"""Unit tests for the download step (fetch_sources)."""

import functools
import http.server
import io
import tarfile
import threading
import zipfile
from pathlib import Path

import pytest

from geoconvert.download import DownloadError, fetch_sources

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "data"


def _write(path: Path, text: str = "data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# --- local paths -------------------------------------------------------------

def test_local_file_is_copied_and_original_kept(tmp_path):
    src = _write(tmp_path / "in" / "a.tif")
    out = fetch_sources([str(src)], "job", tmp_path)
    assert out == [tmp_path / "job" / "a.tif"]
    assert out[0].read_text() == "data"
    assert src.exists()


def test_local_folder_is_copied_with_all_files(tmp_path):
    _write(tmp_path / "in" / "rivers" / "rivers.shp")
    _write(tmp_path / "in" / "rivers" / "rivers.dbf")
    _write(tmp_path / "in" / "rivers" / "sub" / "x.tif")
    out = fetch_sources([str(tmp_path / "in" / "rivers")], "job", tmp_path)
    names = [p.relative_to(tmp_path / "job").as_posix() for p in out]
    assert names == ["rivers/rivers.dbf", "rivers/rivers.shp", "rivers/sub/x.tif"]


def test_several_sources_in_one_call(tmp_path):
    a = _write(tmp_path / "in" / "a.tif")
    b = _write(tmp_path / "in" / "b.nc")
    out = fetch_sources([str(a), str(b)], "job", tmp_path)
    assert [p.name for p in out] == ["a.tif", "b.nc"]


def test_second_call_returns_only_new_files(tmp_path):
    a = _write(tmp_path / "in" / "a.tif")
    b = _write(tmp_path / "in" / "b.nc")
    first = fetch_sources([str(a)], "job", tmp_path)
    second = fetch_sources([str(b)], "job", tmp_path)
    assert first == [tmp_path / "job" / "a.tif"]
    assert second == [tmp_path / "job" / "b.nc"]


def test_real_fixture_is_copied_byte_for_byte(tmp_path):
    src = FIXTURES / "tiled_geotiff_not_cog" / "GMW_N10E106_2020_v3.tif"
    if not src.exists():
        pytest.skip("fixture data not present")
    [out] = fetch_sources([str(src)], "job", tmp_path)
    assert out.read_bytes() == src.read_bytes()


def test_missing_local_path_raises(tmp_path):
    with pytest.raises(DownloadError, match="not found"):
        fetch_sources([str(tmp_path / "nope.tif")], "job", tmp_path)


def test_same_file_name_twice_raises(tmp_path):
    a = _write(tmp_path / "one" / "a.tif")
    b = _write(tmp_path / "two" / "a.tif")
    with pytest.raises(DownloadError, match="already exists"):
        fetch_sources([str(a), str(b)], "job", tmp_path)


def test_default_download_dir_comes_from_config(tmp_path, monkeypatch):
    from geoconvert.config import Config, DownloadConfig

    config = Config(download=DownloadConfig(dir=tmp_path / "from_config"))
    monkeypatch.setattr("geoconvert.download.get_config", lambda: config)
    src = _write(tmp_path / "in" / "a.tif")
    out = fetch_sources([str(src)], "job-1")
    assert out == [tmp_path / "from_config" / "job-1" / "a.tif"]


def test_unsupported_scheme_raises(tmp_path):
    with pytest.raises(DownloadError, match="not supported"):
        fetch_sources(["s3://bucket/a.tif"], "job", tmp_path)


# --- archives ----------------------------------------------------------------

def test_zip_is_unpacked_and_archive_removed(tmp_path):
    zpath = tmp_path / "in" / "tiles.zip"
    zpath.parent.mkdir()
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("tiles/a.tif", "a")
        zf.writestr("tiles/b.tif", "b")
    out = fetch_sources([str(zpath)], "job", tmp_path)
    names = [p.relative_to(tmp_path / "job").as_posix() for p in out]
    assert names == ["tiles/tiles/a.tif", "tiles/tiles/b.tif"]
    assert not (tmp_path / "job" / "tiles.zip").exists()


@pytest.mark.parametrize("suffix,mode", [(".tar", "w"), (".tar.gz", "w:gz"), (".tgz", "w:gz")])
def test_tar_is_unpacked(tmp_path, suffix, mode):
    tpath = tmp_path / "in" / f"cube{suffix}"
    tpath.parent.mkdir()
    with tarfile.open(tpath, mode) as tf:
        data = b"nc"
        info = tarfile.TarInfo("cube.nc")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    out = fetch_sources([str(tpath)], "job", tmp_path)
    assert [p.relative_to(tmp_path / "job").as_posix() for p in out] == ["cube/cube.nc"]


def test_zip_inside_a_copied_folder_is_unpacked(tmp_path):
    folder = tmp_path / "in" / "delivery"
    folder.mkdir(parents=True)
    with zipfile.ZipFile(folder / "shp.zip", "w") as zf:
        zf.writestr("r.shp", "x")
    out = fetch_sources([str(folder)], "job", tmp_path)
    assert [p.relative_to(tmp_path / "job").as_posix() for p in out] == ["delivery/shp/r.shp"]


def test_zip_with_path_outside_folder_is_rejected(tmp_path):
    zpath = tmp_path / "in" / "evil.zip"
    zpath.parent.mkdir()
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("../../escaped.txt", "x")
    with pytest.raises(DownloadError, match="outside"):
        fetch_sources([str(zpath)], "job", tmp_path)
    assert not (tmp_path / "escaped.txt").exists()


def test_tar_with_path_outside_folder_is_rejected(tmp_path):
    tpath = tmp_path / "in" / "evil.tar"
    tpath.parent.mkdir()
    with tarfile.open(tpath, "w") as tf:
        info = tarfile.TarInfo("../escaped.txt")
        info.size = 1
        tf.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(DownloadError, match="outside"):
        fetch_sources([str(tpath)], "job", tmp_path)


# --- URLs (served by a local test server, no internet) -----------------------

@pytest.fixture
def http_server(tmp_path):
    root = tmp_path / "served"
    root.mkdir()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a, **k: None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield root, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_url_is_downloaded(tmp_path, http_server):
    root, base = http_server
    _write(root / "chirts.tif", "pixels")
    [out] = fetch_sources([f"{base}/chirts.tif"], "job", tmp_path)
    assert out == tmp_path / "job" / "chirts.tif"
    assert out.read_text() == "pixels"


def test_url_zip_is_unpacked(tmp_path, http_server):
    root, base = http_server
    with zipfile.ZipFile(root / "gmw.zip", "w") as zf:
        zf.writestr("t.tif", "x")
    out = fetch_sources([f"{base}/gmw.zip"], "job", tmp_path)
    assert [p.relative_to(tmp_path / "job").as_posix() for p in out] == ["gmw/t.tif"]


def test_url_not_found_raises(tmp_path, http_server):
    _, base = http_server
    with pytest.raises(DownloadError, match="404"):
        fetch_sources([f"{base}/missing.tif"], "job", tmp_path)


# --- retries -----------------------------------------------------------------


def _serve_with(root: Path, handler):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(handler, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_url_500_is_retried_until_success(tmp_path, monkeypatch):
    from geoconvert.config import Config, DownloadConfig

    config = Config(download=DownloadConfig(dir=tmp_path, retry_delay=0.0, retry_attempts=3))
    monkeypatch.setattr("geoconvert.download.get_config", lambda: config)

    root = tmp_path / "served"
    root.mkdir()
    _write(root / "a.tif", "pixels")
    attempts = {"n": 0}

    class FlakyHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            attempts["n"] += 1
            if attempts["n"] < 3:
                self.send_error(500)
                return
            super().do_GET()

        def log_message(self, *args, **kwargs):
            pass

    server, base = _serve_with(root, FlakyHandler)
    try:
        out = fetch_sources([f"{base}/a.tif"], "job", tmp_path)
    finally:
        server.shutdown()
    assert out == [tmp_path / "job" / "a.tif"]
    assert out[0].read_text() == "pixels"
    assert attempts["n"] == 3


def test_url_500_exhausts_retries_then_raises(tmp_path, monkeypatch):
    from geoconvert.config import Config, DownloadConfig

    config = Config(download=DownloadConfig(dir=tmp_path, retry_delay=0.0, retry_attempts=3))
    monkeypatch.setattr("geoconvert.download.get_config", lambda: config)

    root = tmp_path / "served"
    root.mkdir()
    attempts = {"n": 0}

    class FailingHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            attempts["n"] += 1
            self.send_error(503)

        def log_message(self, *args, **kwargs):
            pass

    server, base = _serve_with(root, FailingHandler)
    try:
        with pytest.raises(DownloadError, match="503"):
            fetch_sources([f"{base}/a.tif"], "job", tmp_path)
    finally:
        server.shutdown()
    assert attempts["n"] == 3


def test_url_404_is_not_retried(tmp_path, monkeypatch):
    from geoconvert.config import Config, DownloadConfig

    config = Config(download=DownloadConfig(dir=tmp_path, retry_delay=0.0, retry_attempts=3))
    monkeypatch.setattr("geoconvert.download.get_config", lambda: config)

    root = tmp_path / "served"
    root.mkdir()
    attempts = {"n": 0}

    class NotFoundHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            attempts["n"] += 1
            self.send_error(404)

        def log_message(self, *args, **kwargs):
            pass

    server, base = _serve_with(root, NotFoundHandler)
    try:
        with pytest.raises(DownloadError, match="404"):
            fetch_sources([f"{base}/missing.tif"], "job", tmp_path)
    finally:
        server.shutdown()
    assert attempts["n"] == 1

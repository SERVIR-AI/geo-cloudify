# /// script
# requires-python = ">=3.12"
# dependencies = ["rasterio>=1.4", "rio-cogeo>=7.0", "numpy>=2.0"]
# ///
"""Demo: send rasters to a running geo_cloudify API, then check every COG it wrote.

Usage:
    uv run scripts/demo_cog_api.py [--api URL] [--sidecar-sample] [SOURCE ...] [--kwargs SOURCE JSON ...]

Start the API first, in another terminal:  uv run uvicorn geo_cloudify.api:app

With no SOURCE, a sample set from tests/fixtures/data is sent:
  - a striped CHIRTS GeoTIFF with {"nodata": -9999}
  - two mangrove tiles that are tiled but not in COG order (colour table)
  - a Copernicus DEM tile that is already a COG (passed through)
  - two single-grid Bio-ORACLE NetCDFs (skipped: NetCDF goes to Zarr, a later version)
--sidecar-sample also builds a raster whose georeferencing lives only in its
companion files (.tfw world file, .aux.xml with the CRS and nodata, external .ovr)
and sends it, to show those files being combined into one COG.

The script then:
  1. POSTs the job and polls GET /jobs/{id}, printing each status change
  2. prints the plan: what is converted, passed through or skipped, and why
  3. checks each COG itself (the service's own verify step is not built yet):
     - rio-cogeo says it is a valid COG
     - it opens on its own: CRS, transform, data type and colour table are inside
       the file, equal to the source as GDAL reads it (main file plus companions)
     - nodata is the value the plan asked for
     - every full-resolution pixel equals the source, compared in fixed full-width
       strips (overviews are new data and are not compared)
  4. prints a report: sizes (source = main file + companions), and the verdict

Exits non-zero if any conversion fails or any check fails.
"""

import argparse
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from rio_cogeo.cogeo import cog_validate

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "data"
STRIP_ROWS = 512


# --- talking to the API ------------------------------------------------------

def call(method: str, url: str, body: dict | None = None) -> dict:
    """Send one request and return the JSON reply. Exits with a clear message on HTTP errors."""
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {url} -> HTTP {e.code}: {e.read().decode()[:500]}")
    except urllib.error.URLError as e:
        sys.exit(f"cannot reach the API at {url} ({e.reason}). Start it with: uv run uvicorn geo_cloudify.api:app")


def run_job(api: str, sources: list) -> dict:
    """POST the job, poll until it finishes, print status changes, return the final job."""
    job_id = call("POST", f"{api}/jobs", {"sources": sources})["job_id"]
    print(f"job {job_id} created")
    last, start = None, time.time()
    while True:
        job = call("GET", f"{api}/jobs/{job_id}")
        if job["status"] != last:
            print(f"  {time.time() - start:6.1f} s  {job['status']}")
            last = job["status"]
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.5)


# --- sample inputs ------------------------------------------------------------

def default_sources() -> list:
    """The sample set: every raster in tests/fixtures/data, one folder per source.

    Folders are sent as sources, so each source's cog_kwargs applies to every file
    in it (both CHIRTS months get nodata -9999).
    """
    folders = ["striped_geotiff", "tiled_geotiff_not_cog", "tiles_to_mosaic", "already_cog", "netcdf_2d"]
    missing = [name for name in folders if not (FIXTURES / name).is_dir() or not any((FIXTURES / name).iterdir())]
    if missing:
        sys.exit("fixture data missing (see CLAUDE.md, 'Test data'): " + ", ".join(missing))
    sources: list = [{"source": str(FIXTURES / "striped_geotiff"), "cog_kwargs": {"nodata": -9999}}]
    return sources + [str(FIXTURES / name) for name in folders[1:]]


def build_sidecar_sample(folder: Path) -> Path:
    """Write a raster whose location lives only in companion files. Returns the folder.

    From a mangrove tile: a GeoTIFF with no CRS and no geotransform inside it, plus
    a .tfw world file (the geotransform), a .tif.aux.xml (the CRS and nodata) and a
    .tif.ovr (external overviews).
    """
    src = FIXTURES / "tiled_geotiff_not_cog" / "GMW_N10E106_2020_v3.tif"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    out = folder / "GMW_N10E106_sidecars.tif"
    with rasterio.open(src) as ds:
        profile = ds.profile.copy()
        for key in ("crs", "transform", "nodata"):
            profile.pop(key, None)
        profile.update(driver="GTiff", tiled=False, compress="lzw")
        colormap = ds.colormap(1)
        data = ds.read()
        transform, crs, nodata = ds.transform, ds.crs, ds.nodata
    with rasterio.Env(GDAL_PAM_ENABLED="NO"):  # keep georeferencing out of the main file
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(data)
            dst.write_colormap(1, colormap)
    # World file: pixel size x, rotation, rotation, pixel size y, centre of the top-left pixel.
    centre = transform * Affine.translation(0.5, 0.5)
    out.with_suffix(".tfw").write_text("\n".join(f"{v:.12f}" for v in (transform.a, transform.d, transform.b, transform.e, centre.c, centre.f)) + "\n")
    Path(str(out) + ".aux.xml").write_text(
        "<PAMDataset>\n"
        f"  <SRS dataAxisToSRSAxisMapping=\"2,1\">{crs.to_wkt()}</SRS>\n"
        f"  <PAMRasterBand band=\"1\"><NoDataValue>{nodata:g}</NoDataValue></PAMRasterBand>\n"
        "</PAMDataset>\n"
    )
    with rasterio.Env(TIFF_USE_OVR="YES"):
        with rasterio.open(out, "r+") as ds:
            ds.build_overviews([2, 4, 8, 16])
    with rasterio.open(out) as ds:  # the sample must only be readable through its companions
        assert ds.crs is not None and ds.transform.almost_equals(transform, precision=1e-9), "sidecar sample did not georeference"
    return folder


# --- checking one COG ---------------------------------------------------------

def check(item: dict, output: Path) -> tuple[list[str], list[str]]:
    """Compare one COG with its source. Returns (checks passed, problems)."""
    passed, problems = [], []
    source = Path(item["files"][0])
    settings = item["settings"]

    valid, errors, _ = cog_validate(str(output))
    (passed if valid else problems).append("valid COG" if valid else f"not a valid COG: {errors}")

    with rasterio.open(source) as a, rasterio.open(output) as b:
        same = {
            "size": (a.width, a.height, a.count) == (b.width, b.height, b.count),
            "data type": a.dtypes == b.dtypes,
            "transform": a.transform.almost_equals(b.transform, precision=1e-12),
            "colour table": _colormap(a) == _colormap(b),
        }
        if a.crs is not None:
            same["CRS"] = a.crs == b.crs
        elif b.crs is not None:
            passed.append(f"CRS assigned: {b.crs.to_string()} (source had none)")
        else:
            problems.append("no CRS in the output")
        for name, ok in same.items():
            (passed if ok else problems).append(f"{name} equal" if ok else f"{name} differs")

        want = settings.get("NODATA")
        want = None if want is None else float("nan") if str(want).lower() == "nan" else float(want)
        got = b.nodata
        nodata_ok = (want is None and got is None) or (want is not None and got is not None and (got == want or (np.isnan(got) and np.isnan(want))))
        label = f"nodata {got!r}" + (f" (source: {a.nodata!r})" if a.nodata != got else "")
        (passed if nodata_ok else problems).append(label if nodata_ok else f"nodata {got!r}, plan asked for {want!r}")

        if item.get("lossy"):
            # Not lossless by design; measure how far the values moved instead.
            worst = 0.0
            if same["size"]:
                for row in range(0, a.height, STRIP_ROWS):
                    window = rasterio.windows.Window(0, row, a.width, min(STRIP_ROWS, a.height - row))
                    xa = a.read(window=window).astype("float64")
                    xb = b.read(window=window).astype("float64")
                    valid = np.isfinite(xa) & (xa != b.nodata if b.nodata is not None else True)
                    if valid.any():
                        worst = max(worst, float(np.abs(xa - xb)[valid].max()))
            limit = settings.get("MAX_Z_ERROR")
            passed.append(f"lossy by request: largest pixel change {worst:.6g}" + (f" (MAX_Z_ERROR {limit})" if limit is not None else ""))
        elif same["size"]:
            for row in range(0, a.height, STRIP_ROWS):
                window = rasterio.windows.Window(0, row, a.width, min(STRIP_ROWS, a.height - row))
                xa, xb = a.read(window=window), b.read(window=window)
                if xa.tobytes() != xb.tobytes():
                    problems.append(f"pixels differ in rows {row}-{row + window.height - 1}")
                    break
            else:
                passed.append(f"all {a.width * a.height * a.count:,} pixels equal")
        overviews = b.overviews(1)
        passed.append(f"{len(overviews)} internal overview level(s)" if overviews else "no overviews (small raster)")

    extra = [p.name for p in output.parent.iterdir() if p.name.startswith(output.name) and p != output]
    (problems if extra else passed).append(f"files next to the COG: {extra}" if extra else "self-contained (no files next to it)")
    return passed, problems


def _colormap(ds):
    """Return band 1's colour table as {index: (r, g, b)}, or None.

    Alpha is left out on purpose: a TIFF palette stores only red, green and blue.
    GDAL reports alpha itself (0 for the nodata entry, but only when nodata is a
    TIFF tag, not when it comes from .aux.xml), so alpha can differ while the file
    stores exactly the same colours.
    """
    try:
        return {index: tuple(rgba[:3]) for index, rgba in ds.colormap(1).items()}
    except ValueError:
        return None


# --- report -------------------------------------------------------------------

def _in_job(path: Path, job_id: str) -> str:
    """Return the file's path inside the job folder, for example tiles_to_mosaic/a.tif."""
    parts = path.parts
    return str(Path(*parts[parts.index(job_id) + 1:])) if job_id in parts else path.name


def mb(n: int) -> float:
    return round(n / 1e6, 3)


def report(job: dict) -> bool:
    """Print the plan, per-item checks and the summary table. Returns True if all passed."""
    print(f"\nstatus: {job['status']}" + (f"   error: {job['error']}" if job["error"] else ""))
    for r in job["results"]:
        if r["status"] != "ok":
            print(f"  source failed: {r['source']}: {r['error']}")
    items = (job.get("plan") or {}).get("items", [])
    conversions = {c["file"]: c for c in job.get("conversions", [])}

    rows, all_ok = [], job["status"] == "done"
    print("\nPLAN AND RESULTS")
    for item in items:
        files = [Path(f) for f in item["files"]]
        main, companions = files[0], files[1:]
        action = item["action"]
        name = _in_job(main, job["job_id"])
        print(f"\n  {name}   [{action}]")
        if companions:
            print("    companions combined into the COG:")
            for c in companions:
                print(f"      {c.name:<40} {mb(c.stat().st_size):>10} MB")
        if action == "skip":
            print(f"    reason: {item['skip_reason']}")
            rows.append((name, action, mb(main.stat().st_size), None, "-"))
            continue
        for note in item.get("notes", []):
            print(f"    note: {note}")
        source_bytes = sum(f.stat().st_size for f in files)
        if action == "pass_through":
            print("    already a valid COG: uploaded as is, not converted")
            rows.append((name, action, mb(source_bytes), mb(source_bytes), "-"))
            continue
        shown = {k: v for k, v in item["settings"].items() if k != "CRS"}
        print(f"    settings: {json.dumps(shown)}")
        outcome = conversions.get(item["files"][0])
        if outcome is None or outcome["status"] != "ok":
            print(f"    CONVERSION FAILED: {outcome['error'] if outcome else 'no result recorded'}")
            rows.append((name, action, mb(source_bytes), None, "FAIL"))
            all_ok = False
            continue
        output = Path(outcome["output"])
        print(f"    output: {output}")
        passed, problems = check(item, output)
        for p in passed:
            print(f"      ok    {p}")
        for p in problems:
            print(f"      FAIL  {p}")
        verdict = "lossy" if item.get("lossy") and not problems else ("PASS" if not problems else "FAIL")
        all_ok &= not problems
        rows.append((name + (f" +{len(companions)}" if companions else ""), action, mb(source_bytes), mb(output.stat().st_size), verdict))

    print("\n" + "=" * 104)
    print("REPORT: rasters -> COG (through the geo_cloudify API)")
    print("=" * 104)
    print(f"{'file (+companions)':<56}{'action':>13}{'source MB':>11}{'COG MB':>10}{'ratio':>7}{'lossless':>9}")
    print("-" * 104)
    for name, action, src, out, verdict in rows:
        name = name if len(name) <= 55 else "..." + name[-52:]
        ratio = f"{src / out:.1f}x" if out else "-"
        print(f"{name:<56}{action:>13}{src:>11}{(out if out is not None else '-'):>10}{ratio:>7}{verdict:>9}")
    print("-" * 104)
    converted = [r for r in rows if r[1] == "convert"]
    counts = {a: sum(r[1] == a for r in rows) for a in ("convert", "pass_through", "skip")}
    print(f"{counts['convert']} converted ({sum(r[4] == 'PASS' for r in converted)} lossless), "
          f"{counts['pass_through']} passed through, {counts['skip']} skipped. "
          f"Converted: {sum(r[2] for r in converted):.2f} MB -> {sum(r[3] or 0 for r in converted):.2f} MB.")
    print("Lossless = valid COG; same size, data type, transform, CRS and colour table; nodata as planned;")
    print("every full-resolution pixel equal to the source (main file read together with its companions).")
    return all_ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("sources", nargs="*", help="local paths or URLs (default: the sample set)")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--kwargs", nargs=2, action="append", metavar=("SOURCE", "JSON"), default=[],
                    help='cog_kwargs for one source, for example --kwargs /data/a.tif \'{"nodata": -9999}\'')
    ap.add_argument("--sidecar-sample", action="store_true", help="also send a raster georeferenced only by companion files")
    args = ap.parse_args()

    options = {src: json.loads(js) for src, js in args.kwargs}
    sources = [{"source": s, "cog_kwargs": options[s]} if s in options else s for s in args.sources] or default_sources()
    if args.sidecar_sample:
        folder = build_sidecar_sample(REPO / "tmp" / "demo" / "sidecar_sample")
        print(f"sidecar sample written to {folder}: " + ", ".join(sorted(p.name for p in folder.iterdir())))
        sources.append(str(folder))

    call("GET", f"{args.api}/health")
    job = run_job(args.api, sources)
    return 0 if report(job) else 1


if __name__ == "__main__":
    sys.exit(main())

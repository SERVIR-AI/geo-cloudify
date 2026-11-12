# /// script
# requires-python = ">=3.12"
# dependencies = ["geopandas>=1.0", "pyogrio>=0.10", "pyarrow>=17", "shapely>=2.0"]
# ///
"""Proof of concept: convert vector files to GeoParquet and check nothing was lost.

Usage:
    uv run scripts/vector_to_geoparquet.py <path-or-url> [<path-or-url> ...] [--out DIR]

For each input (a file, a folder, a .zip, or an http(s) URL to any of those):
  1. download / unpack into a temp folder
  2. find every vector layer GDAL can read (Shapefile, GeoJSON, GeoPackage, FlatGeobuf, KML, ...)
  3. write one GeoParquet per layer: ZSTD compression, bbox covering column, rows sorted
     along a Hilbert curve (so readers like DuckDB can skip row groups by location)
  4. read the GeoParquet back and compare it with the source layer:
     same CRS, same columns and dtypes, same row count, every attribute value equal,
     every geometry byte-for-byte equal (WKB). Rows are matched through the sort order.

Prints one report line per layer and exits non-zero if any layer fails the check.
Not production code: a blueprint for the geoconvert GeoParquet step.
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse, unquote

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyogrio

SHAPEFILE_PARTS = {".shp", ".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx", ".qix"}
VECTOR_SUFFIXES = {".shp", ".geojson", ".json", ".gpkg", ".fgb", ".kml", ".gml", ".gdb", ".tab", ".mif"}
ROW_GROUP_SIZE = 50_000


def fetch(source: str, work: Path) -> Path:
    """Download a URL or point at a local path; unpack zips. Returns a file or folder."""
    if urlparse(source).scheme in ("http", "https"):
        with urllib.request.urlopen(source, timeout=120) as r:
            name = r.headers.get_filename() or Path(unquote(urlparse(r.url).path)).name or "download"
            path = work / name
            with path.open("wb") as f:
                shutil.copyfileobj(r, f)
    else:
        path = Path(source).expanduser().resolve()
    if path.suffix.lower() == ".zip":
        out = work / path.stem
        with zipfile.ZipFile(path) as z:
            z.extractall(out)
        return out
    return path


def find_layers(path: Path) -> list[tuple[Path, str]]:
    """Return (file, layer name) for every vector layer under path."""
    files = [path] if path.is_file() or path.suffix.lower() == ".gdb" else sorted(
        p for p in path.rglob("*") if p.suffix.lower() in VECTOR_SUFFIXES and (p.is_file() or p.suffix.lower() == ".gdb")
    )
    layers = []
    for f in files:
        try:
            for name, geom_type in pyogrio.list_layers(f):
                if geom_type is not None:  # skip attribute-only tables
                    layers.append((f, name))
        except Exception as e:  # not a vector file GDAL can read
            print(f"  skip {f.name}: {e}", file=sys.stderr)
    return layers


def convert_layer(src: Path, layer: str, out_dir: Path) -> dict:
    """Convert one layer and verify it. Returns a report dict."""
    t0 = time.time()
    gdf = gpd.read_file(src, layer=layer, engine="pyogrio")
    read_s = time.time() - t0

    # Spatial sort: Hilbert distance of each feature's bbox centre. Keep the permutation
    # so the check can compare row i of the output with the right source row.
    order = np.argsort(gdf.geometry.hilbert_distance(), kind="stable") if len(gdf) else np.arange(0)
    sorted_gdf = gdf.iloc[order]

    # Name after the file and its format, so a.shp and a.geojson do not overwrite each other.
    name = f"{src.stem}.{src.suffix.lstrip('.').lower()}"
    if layer != src.stem:
        name += f".{layer}"
    out = out_dir / f"{name}.parquet"
    t0 = time.time()
    sorted_gdf.to_parquet(
        out,
        compression="zstd",
        write_covering_bbox=True,
        schema_version="1.1.0",
        row_group_size=ROW_GROUP_SIZE,
        index=False,
    )
    write_s = time.time() - t0

    geometry_types = sorted(set(gdf.geom_type.dropna()))
    crs = gdf.crs.to_string() if gdf.crs else None
    del gdf, sorted_gdf
    problems = verify(src, layer, order, out)
    if src.suffix.lower() == ".shp":
        src_bytes = sum(p.stat().st_size for p in src.parent.glob(src.stem + ".*") if p.suffix.lower() in SHAPEFILE_PARTS)
    else:
        src_bytes = _size(src)
    return {
        "source": str(src),
        "layer": layer,
        "output": str(out),
        "rows": len(order),
        "geometry_types": geometry_types,
        "crs": crs,
        "source_mb": round(src_bytes / 1e6, 2),
        "output_mb": round(out.stat().st_size / 1e6, 2),
        "read_s": round(read_s, 2),
        "write_s": round(write_s, 2),
        "lossless": not problems,
        "problems": problems,
    }


def verify(src: Path, layer: str, order: np.ndarray, out: Path) -> list[str]:
    """Re-read the source layer and compare it with the written GeoParquet. Returns a list of problems."""
    problems = []
    source = gpd.read_file(src, layer=layer, engine="pyogrio")

    # GeoParquet metadata: the 'geo' key, its version, CRS and the bbox covering.
    meta = pq.read_metadata(out).metadata or {}
    if b"geo" not in meta:
        return ["no 'geo' metadata: not a GeoParquet file"]
    geo = json.loads(meta[b"geo"])
    col = geo["columns"][geo["primary_column"]]
    if "covering" not in col:
        problems.append("no bbox covering in geo metadata")

    back = gpd.read_parquet(out)
    expected = source.iloc[order].reset_index(drop=True)

    if len(back) != len(expected):
        return problems + [f"row count {len(back)} != {len(expected)}"]
    if (back.crs is None) != (expected.crs is None) or (back.crs and not back.crs.equals(expected.crs)):
        problems.append(f"CRS changed: {expected.crs} -> {back.crs}")

    back_attrs = back.drop(columns=[back.geometry.name, "bbox"], errors="ignore")
    exp_attrs = expected.drop(columns=[expected.geometry.name])
    if list(back_attrs.columns) != list(exp_attrs.columns):
        problems.append(f"columns changed: {list(exp_attrs.columns)} -> {list(back_attrs.columns)}")
    else:
        for c in exp_attrs.columns:
            if str(back_attrs[c].dtype) != str(exp_attrs[c].dtype):
                problems.append(f"column {c}: dtype {exp_attrs[c].dtype} -> {back_attrs[c].dtype}")
            try:
                pd.testing.assert_series_equal(back_attrs[c], exp_attrs[c], check_dtype=False, check_exact=True, check_names=False)
            except AssertionError as e:
                problems.append(f"column {c}: values differ ({str(e).splitlines()[0]})")

    # Geometry: compare WKB bytes, so any change in a coordinate shows up.
    wkb_back = back.geometry.to_wkb()
    wkb_exp = expected.geometry.to_wkb()
    diff = int((wkb_back.values != wkb_exp.values).sum())
    if diff:
        problems.append(f"{diff} geometries differ")
    return problems


def _size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--out", default="tmp/poc/geoparquet")
    ap.add_argument("--json", action="store_true", help="print one JSON line per layer instead of the readable report")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    with tempfile.TemporaryDirectory() as tmp:
        for source in args.sources:
            if not args.json:
                print(f"\n== {source}")
            path = fetch(source, Path(tmp))
            for f, layer in find_layers(path):
                report = convert_layer(f, layer, out_dir)
                reports.append(report)
                print(json.dumps(report) if args.json else _describe(report))
    if not args.json:
        _print_summary(reports)
    return 0 if reports and all(r["lossless"] for r in reports) else 1


def _describe(r: dict) -> str:
    """One readable block for a converted layer."""
    lines = [
        f"  {Path(r['source']).name} [{r['layer']}]",
        f"    rows: {r['rows']:,}   geometry: {', '.join(r['geometry_types']) or 'none'}   CRS: {r['crs']}",
        f"    size: {r['source_mb']} MB -> {r['output_mb']} MB   read {r['read_s']} s, write {r['write_s']} s",
        f"    output: {r['output']}",
        f"    lossless: {'PASS' if r['lossless'] else 'FAIL'}",
    ]
    lines += [f"      - {p}" for p in r["problems"]]
    return "\n".join(lines)


def _print_summary(reports: list[dict]) -> None:
    """Print the final table: one row per layer, then totals."""
    print("\n" + "=" * 92)
    print("REPORT: vector -> GeoParquet")
    print("=" * 92)
    print(f"{'file [layer]':<46}{'rows':>10}{'source MB':>11}{'output MB':>11}{'ratio':>7}{'lossless':>9}")
    print("-" * 92)
    # A multi-layer file (for example a GeoPackage) is one source file shared by
    # several outputs, so its size is shown once, on a file total row.
    layers_per_file: dict[str, list[dict]] = {}
    for r in reports:
        layers_per_file.setdefault(r["source"], []).append(r)
    for source, rows in layers_per_file.items():
        shared = len(rows) > 1
        for r in rows:
            name = f"{Path(r['source']).name} [{r['layer']}]"
            name = name if len(name) <= 45 else name[:42] + "..."
            src_mb = "shared" if shared else r["source_mb"]
            ratio = "-" if shared or not r["output_mb"] else f"{r['source_mb'] / r['output_mb']:.1f}x"
            print(f"{name:<46}{r['rows']:>10,}{src_mb:>11}{r['output_mb']:>11}{ratio:>7}{('PASS' if r['lossless'] else 'FAIL'):>9}")
        if shared:
            out_mb = round(sum(r["output_mb"] for r in rows), 2)
            name = f"  = {Path(source).name} (all {len(rows)} layers)"
            ratio = f"{rows[0]['source_mb'] / out_mb:.1f}x" if out_mb else "-"
            ok = "PASS" if all(r["lossless"] for r in rows) else "FAIL"
            print(f"{name[:45]:<46}{sum(r['rows'] for r in rows):>10,}{rows[0]['source_mb']:>11}{out_mb:>11}{ratio:>7}{ok:>9}")
    print("-" * 92)
    passed = sum(r["lossless"] for r in reports)
    src = sum(rows[0]["source_mb"] for rows in layers_per_file.values())
    out = sum(r["output_mb"] for r in reports)
    print(f"{len(reports)} layer(s) from {len(layers_per_file)} file(s), {passed} lossless, {len(reports) - passed} failed. "
          f"Total {src:.2f} MB -> {out:.2f} MB.")
    print("Lossless = same CRS, columns, dtypes and row count; every attribute value equal; every geometry equal as WKB bytes.")


if __name__ == "__main__":
    sys.exit(main())

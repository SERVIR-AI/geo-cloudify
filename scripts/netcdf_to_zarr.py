# /// script
# requires-python = ">=3.12"
# dependencies = ["xarray>=2025.1", "zarr>=3.0", "netCDF4>=1.7", "h5netcdf>=1.4", "scipy>=1.14", "numpy>=2.0"]
# ///
"""Proof of concept: convert NetCDF/HDF5 data cubes to Zarr and check nothing was lost.

Usage:
    uv run scripts/netcdf_to_zarr.py <path-or-url> [<path-or-url> ...] [--out DIR] [--chunk-mb N]

For each input (a .nc / .nc4 / .h5 file, local or an http(s) URL):
  1. download it into a temp folder
  2. open it with xarray WITHOUT decoding (no masking, no scale/offset, no time decoding),
     so the stored numbers and their encoding attributes are carried over exactly
  3. choose chunks: one chunk per step along the outer dimensions (time, depth, ...) and the
     full spatial grid, split further only if a chunk would be larger than --chunk-mb
  4. write a Zarr v3 store with consolidated metadata (zstd compression)
  5. reopen the Zarr store, also without decoding, and compare it with the source:
     same dimensions and sizes, same variables (data and coordinates), same dtypes,
     every value equal bit for bit (NaN counts as equal to NaN), same global and
     per-variable attributes

Prints one report line per input and exits non-zero if any input fails the check.
Not production code: a blueprint for the geo_cloudify Zarr step.
"""

import argparse
import json
import math
import shutil
import sys
import tempfile
import time
import urllib.request
import warnings
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import xarray as xr
import zarr

# Known, harmless warnings: consolidated metadata is an xarray/zarr-python extension to
# Zarr v3, and very old calendars decode to cftime objects instead of numpy datetimes.
warnings.filterwarnings("ignore", message="Consolidated metadata is currently not part")
warnings.filterwarnings("ignore", message="Unable to decode time axis")

SPATIAL_HINTS = {"lat", "latitude", "lon", "longitude", "x", "y", "rlat", "rlon", "nlat", "nlon"}


def fetch(source: str, work: Path) -> Path:
    """Download a URL into work, or return the local path."""
    if urlparse(source).scheme not in ("http", "https"):
        return Path(source).expanduser().resolve()
    with urllib.request.urlopen(source, timeout=300) as r:
        name = r.headers.get_filename() or Path(unquote(urlparse(r.url).path)).name or "download.nc"
        path = work / name
        with path.open("wb") as f:
            shutil.copyfileobj(r, f)
    return path


def open_raw(path) -> xr.Dataset:
    """Open without any decoding, so values and encoding attributes stay as stored."""
    return xr.open_dataset(path, decode_cf=False, mask_and_scale=False, decode_times=False, decode_coords=False)


def choose_chunks(var: xr.DataArray, chunk_mb: float) -> tuple[int, ...]:
    """Pick chunks of about chunk_mb (uncompressed): full spatial grid, steps grouped along time.

    Start with one step along the outer dimensions (time, depth, ...) and the full grid
    along spatial ones. If that is larger than chunk_mb, halve the spatial sides until it
    fits. If it is smaller, group more steps along the first outer dimension until the
    chunk reaches about chunk_mb, so a long time series is not split into thousands of
    tiny files.
    """
    if var.ndim and all(d.lower() not in SPATIAL_HINTS for d in var.dims):  # e.g. a 1D time series
        return tuple(var.shape)
    chunks = [1 if d.lower() not in SPATIAL_HINTS else n for d, n in zip(var.dims, var.shape)]
    limit = chunk_mb * 1e6
    while math.prod(chunks) * var.dtype.itemsize > limit:
        spatial = [i for i, d in enumerate(var.dims) if d.lower() in SPATIAL_HINTS and chunks[i] > 1]
        if not spatial:
            break
        for i in spatial:
            chunks[i] = math.ceil(chunks[i] / 2)
    outer = [i for i, d in enumerate(var.dims) if d.lower() not in SPATIAL_HINTS]
    if outer:
        i = outer[0]
        size = math.prod(chunks) * var.dtype.itemsize
        chunks[i] = min(var.shape[i], max(1, int(limit // size)))
    return tuple(max(1, c) for c in chunks)


def convert(src: Path, out_dir: Path, chunk_mb: float) -> dict:
    """Convert one NetCDF file to Zarr and verify it. Returns a report dict."""
    ds = open_raw(src)
    out = out_dir / f"{src.stem}.zarr"
    if out.exists():
        shutil.rmtree(out)

    compressor = zarr.codecs.ZstdCodec(level=3)
    encoding = {}
    for name, var in ds.variables.items():
        enc = {"compressors": [compressor]}
        if var.ndim:
            enc["chunks"] = choose_chunks(var, chunk_mb)
        if "_FillValue" not in var.attrs:
            # xarray otherwise adds _FillValue=NaN to every float variable, which the
            # source never had. None keeps the attributes exactly as in the source.
            enc["_FillValue"] = None
        encoding[name] = enc
    # Drop netCDF-only encodings (contiguous, zlib, chunksizes, ...) so the Zarr writer
    # only sees the choices above. Attributes such as scale_factor/_FillValue are kept as
    # plain attributes because the file was opened without decoding.
    for var in ds.variables.values():
        var.encoding = {}

    t0 = time.time()
    ds.to_zarr(out, mode="w", zarr_format=3, consolidated=True, encoding=encoding)
    write_s = time.time() - t0

    problems = verify(src, out)
    notes = decoded_notes(src, out) if not problems else []
    return {
        "source": str(src),
        "output": str(out),
        "format": _format(src),
        "dims": {k: int(v) for k, v in ds.sizes.items()},
        "data_vars": list(ds.data_vars),
        "chunks": {n: list(e["chunks"]) for n, e in encoding.items() if "chunks" in e and n in ds.data_vars},
        "source_mb": round(src.stat().st_size / 1e6, 3),
        "output_mb": round(sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6, 3),
        "output_files": sum(1 for f in out.rglob("*") if f.is_file()),
        "write_s": round(write_s, 2),
        "lossless": not problems,
        "problems": problems,
        "notes": notes,
    }


def decoded_notes(src: Path, out: Path) -> list[str]:
    """Check what a normal reader sees (CF decoding on). Returns notes, not failures.

    Zarr stores attributes as JSON, so a float32 scale_factor comes back as a float64.
    xarray then decodes packed integers to float64 instead of float32. The stored numbers
    are identical (checked in verify); here we confirm the decoded values still match the
    source once cast back to the source's decoded dtype, and report any dtype widening.
    """
    notes = []
    a = xr.open_dataset(src)
    b = xr.open_zarr(out, consolidated=True)
    for name in a.data_vars:
        da, db = a[name], b[name]
        if da.dtype != db.dtype:
            notes.append(f"{name}: decoded dtype {da.dtype} -> {db.dtype} (stored values identical)")
        if da.ndim and da.dtype.kind in "fc":
            same = all(
                np.array_equal(da[i].values if da.ndim > 1 else da.values,
                               (db[i].values if db.ndim > 1 else db.values).astype(da.dtype), equal_nan=True)
                for i in range(da.shape[0] if da.ndim > 1 else 1)
            )
            if not same:
                notes.append(f"{name}: decoded values differ from the source")
        elif not da.equals(db.load()):
            notes.append(f"{name}: decoded values differ from the source")
    return notes


def verify(src: Path, out: Path) -> list[str]:
    """Reopen source and Zarr without decoding and compare everything. Returns problems."""
    problems = []
    a = open_raw(src)
    b = xr.open_zarr(out, consolidated=True, decode_cf=False, mask_and_scale=False, decode_times=False, decode_coords=False)

    if dict(a.sizes) != dict(b.sizes):
        problems.append(f"dims {dict(a.sizes)} -> {dict(b.sizes)}")
    if set(a.variables) != set(b.variables):
        problems.append(f"variables {sorted(a.variables)} -> {sorted(b.variables)}")
    if _attrs(a.attrs) != _attrs(b.attrs):
        problems.append("global attributes differ")

    for name in sorted(set(a.variables) & set(b.variables)):
        va, vb = a[name], b[name]
        if va.dims != vb.dims:
            problems.append(f"{name}: dims {va.dims} -> {vb.dims}")
            continue
        if va.dtype != vb.dtype:
            problems.append(f"{name}: dtype {va.dtype} -> {vb.dtype}")
        if _attrs(va.attrs) != _attrs(vb.attrs):
            problems.append(f"{name}: attributes differ: {_diff(va.attrs, vb.attrs)}")
        # Bit-for-bit comparison, one step of the first dimension at a time to bound memory.
        steps = va.shape[0] if va.ndim > 1 else 1
        for i in range(steps):
            xa = (va[i] if va.ndim > 1 else va).values
            xb = (vb[i] if vb.ndim > 1 else vb).values
            if xa.dtype.kind in "fc":
                same = np.array_equal(xa.view(np.uint8) if xa.dtype.kind == "f" else xa, xb.view(np.uint8) if xb.dtype.kind == "f" else xb) \
                    or np.array_equal(xa, xb, equal_nan=True)
            else:
                same = np.array_equal(xa, xb)
            if not same:
                problems.append(f"{name}: values differ (step {i})")
                break
    return problems


def _attrs(attrs: dict) -> dict:
    """Normalise attribute values (numpy scalars/arrays to plain Python) for comparison."""
    out = {}
    for k, v in attrs.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, float) and math.isnan(v):
            v = "NaN"
        out[k] = v
    return out


def _diff(a: dict, b: dict) -> dict:
    na, nb = _attrs(a), _attrs(b)
    return {k: (na.get(k), nb.get(k)) for k in set(na) | set(nb) if na.get(k) != nb.get(k)}


def _format(path: Path) -> str:
    head = path.open("rb").read(8)
    if head.startswith(b"CDF\x01") or head.startswith(b"CDF\x02"):
        return "NetCDF3"
    if head.startswith(b"\x89HDF"):
        return "NetCDF4/HDF5"
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--out", default="tmp/poc/zarr")
    ap.add_argument("--chunk-mb", type=float, default=16.0, help="largest uncompressed chunk, in MB")
    ap.add_argument("--json", action="store_true", help="print one JSON line per input instead of the readable report")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    with tempfile.TemporaryDirectory() as tmp:
        for source in args.sources:
            if not args.json:
                print(f"\n== {source if len(source) <= 120 else source[:117] + '...'}")
            report = convert(fetch(source, Path(tmp)), out_dir, args.chunk_mb)
            reports.append(report)
            print(json.dumps(report) if args.json else _describe(report))
    if not args.json:
        _print_summary(reports)
    return 0 if reports and all(r["lossless"] for r in reports) else 1


def _describe(r: dict) -> str:
    """One readable block for a converted file."""
    dims = ", ".join(f"{k}={v}" for k, v in r["dims"].items())
    chunks = "; ".join(f"{k} {v}" for k, v in r["chunks"].items())
    lines = [
        f"  {Path(r['source']).name} ({r['format']})",
        f"    dims: {dims}",
        f"    variables: {', '.join(r['data_vars'])}",
        f"    chunks: {chunks}",
        f"    size: {r['source_mb']} MB -> {r['output_mb']} MB in {r['output_files']} files   write {r['write_s']} s",
        f"    output: {r['output']}",
        f"    lossless: {'PASS' if r['lossless'] else 'FAIL'}",
    ]
    lines += [f"      - {p}" for p in r["problems"]]
    lines += [f"      note: {n}" for n in r["notes"]]
    return "\n".join(lines)


def _print_summary(reports: list[dict]) -> None:
    """Print the final table: one row per input, then totals."""
    print("\n" + "=" * 92)
    print("REPORT: NetCDF -> Zarr")
    print("=" * 92)
    print(f"{'file':<40}{'format':>13}{'steps':>7}{'source MB':>11}{'output MB':>11}{'ratio':>7}{'lossless':>9}")
    print("-" * 92)
    for r in reports:
        name = Path(r["source"]).name
        name = name if len(name) <= 39 else name[:36] + "..."
        steps = next(iter(r["dims"].values()), 0)
        ratio = f"{r['source_mb'] / r['output_mb']:.1f}x" if r["output_mb"] else "-"
        print(f"{name:<40}{r['format']:>13}{steps:>7}{r['source_mb']:>11}{r['output_mb']:>11}{ratio:>7}{('PASS' if r['lossless'] else 'FAIL'):>9}")
    print("-" * 92)
    passed = sum(r["lossless"] for r in reports)
    src = sum(r["source_mb"] for r in reports)
    out = sum(r["output_mb"] for r in reports)
    print(f"{len(reports)} file(s), {passed} lossless, {len(reports) - passed} failed. "
          f"Total {src:.2f} MB -> {out:.2f} MB.")
    print("Lossless = same dims, variables, dtypes and attributes; every stored value equal bit for bit (no decoding).")
    if any(r["notes"] for r in reports):
        print("Notes are about the decoded view (what xarray shows by default); they do not affect the stored values.")


if __name__ == "__main__":
    sys.exit(main())

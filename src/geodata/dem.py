"""
Digital Elevation Model acquisition, reprojection and validation.

Primary source: Copernicus DEM GLO-30 (Cloud Optimized GeoTIFF on AWS Open
Data, no credentials required). Only the window covering the study area is
read through GDAL's ``/vsicurl/`` driver, then cached locally.

A user-supplied higher-resolution DEM (e.g. national survey or LiDAR GeoTIFF)
can be used instead through :func:`prepare_dem_grid` with ``src_path``.

All hydraulic computations use a metric grid in the local UTM zone.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.fill import fillnodata
from rasterio.merge import merge
from rasterio.transform import Affine, from_origin, rowcol, xy
from rasterio.warp import reproject, transform_bounds
from pyproj import Transformer

BBox = Tuple[float, float, float, float]  # (west, south, east, north) in degrees

GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    GDAL_HTTP_MAX_RETRY="3",
    GDAL_HTTP_RETRY_DELAY="1",
    VSI_CACHE="TRUE",
)


class DemError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------
def validate_bbox(bbox: Sequence[float]) -> BBox:
    if len(bbox) != 4:
        raise DemError("bbox must be (west, south, east, north)")
    w, s, e, n = (float(v) for v in bbox)
    if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
        raise DemError(f"Invalid bbox {bbox}: expected west<east and south<north in degrees")
    return w, s, e, n


def utm_epsg_for(lon: float, lat: float) -> int:
    """EPSG code of the WGS84/UTM zone containing (lon, lat)."""
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def copernicus_tile_names(bbox: BBox) -> List[str]:
    """Names of the 1°x1° GLO-30 tiles intersecting the bbox."""
    w, s, e, n = bbox
    names = []
    for lat in range(int(math.floor(s)), int(math.ceil(n))):
        for lon in range(int(math.floor(w)), int(math.ceil(e))):
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            names.append(
                f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"
            )
    return names


def _bbox_key(bbox: BBox, extra: str = "") -> str:
    raw = ",".join(f"{v:.6f}" for v in bbox) + extra
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class DemGrid:
    """Elevation on a regular metric grid (north-up)."""

    elevation: np.ndarray            # (rows, cols) float64, metres
    transform: Affine                # pixel -> projected coordinates
    crs: str                         # e.g. "EPSG:32640"
    valid: np.ndarray                # bool mask of cells with real data
    bbox_lonlat: BBox                # requested study area
    source: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.elevation.shape

    @property
    def cell_size(self) -> float:
        return float(self.transform.a)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        rows, cols = self.shape
        left, top = self.transform.c, self.transform.f
        return left, top + rows * self.transform.e, left + cols * self.transform.a, top

    def bounds_lonlat(self) -> BBox:
        """Geographic envelope of the projected grid."""
        l, b, r, t = self.bounds
        return tuple(transform_bounds(self.crs, "EPSG:4326", l, b, r, t, densify_pts=21))  # type: ignore

    def _to_proj(self) -> Transformer:
        return Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)

    def _to_geo(self) -> Transformer:
        return Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)

    def lonlat_to_rowcol(self, lon: float, lat: float) -> Tuple[int, int]:
        x, y = self._to_proj().transform(lon, lat)
        r, c = rowcol(self.transform, x, y)
        return int(r), int(c)

    def rowcol_to_lonlat(self, row: float, col: float) -> Tuple[float, float]:
        x, y = xy(self.transform, row, col, offset="center")
        lon, lat = self._to_geo().transform(x, y)
        return float(lon), float(lat)

    def contains(self, row: int, col: int) -> bool:
        rows, cols = self.shape
        return 0 <= row < rows and 0 <= col < cols

    # -- persistence -----------------------------------------------------
    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        rows, cols = self.shape
        nodata = -9999.0
        data = np.where(self.valid, self.elevation, nodata).astype("float32")
        with rasterio.open(
            folder / "dem.tif", "w", driver="GTiff", height=rows, width=cols, count=1,
            dtype="float32", crs=self.crs, transform=self.transform, nodata=nodata,
            compress="deflate", tiled=True, blockxsize=256, blockysize=256,
        ) as dst:
            dst.write(data, 1)
        meta = {"crs": self.crs, "bbox_lonlat": list(self.bbox_lonlat), "source": self.source}
        (folder / "dem_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path) -> "DemGrid":
        with rasterio.open(folder / "dem.tif") as src:
            arr = src.read(1).astype("float64")
            valid = arr != src.nodata
            transform, crs = src.transform, src.crs.to_string()
        meta = json.loads((folder / "dem_meta.json").read_text(encoding="utf-8"))
        return cls(arr, transform, crs, valid, tuple(meta["bbox_lonlat"]), meta.get("source", {}))


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------
def fetch_copernicus_dem(bbox: Sequence[float], cache_dir: Path,
                         base_url: str = "https://copernicus-dem-30m.s3.amazonaws.com",
                         margin_deg: float = 0.005) -> Path:
    """
    Read the GLO-30 window covering ``bbox`` (+ small margin) from the public
    COGs and cache it as a geographic GeoTIFF. Returns the cached path.
    """
    bbox = validate_bbox(bbox)
    w, s, e, n = bbox
    padded = (w - margin_deg, s - margin_deg, e + margin_deg, n + margin_deg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"cop30_{_bbox_key(padded)}.tif"
    if out.exists():
        return out

    datasets, missing = [], []
    with rasterio.Env(**GDAL_ENV):
        for name in copernicus_tile_names(padded):
            url = f"/vsicurl/{base_url}/{name}/{name}.tif"
            try:
                datasets.append(rasterio.open(url))
            except rasterio.errors.RasterioIOError:
                missing.append(name)  # ocean tiles are not published
        if not datasets:
            raise DemError(f"No Copernicus GLO-30 tiles available for {bbox} (missing: {missing})")
        try:
            mosaic, transform = merge(datasets, bounds=padded, nodata=-32767.0)
            crs = datasets[0].crs
        finally:
            for ds in datasets:
                ds.close()

    with rasterio.open(
        out, "w", driver="GTiff", height=mosaic.shape[1], width=mosaic.shape[2], count=1,
        dtype=mosaic.dtype, crs=crs, transform=transform, nodata=-32767.0, compress="deflate",
    ) as dst:
        dst.write(mosaic[0], 1)
    meta = {"tiles": copernicus_tile_names(padded), "missing_tiles": missing, "bbox": padded}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def prepare_dem_grid(src_path: Path, bbox: Sequence[float], resolution_m: float = 30.0,
                     dst_epsg: Optional[int] = None, max_cells: Optional[int] = None,
                     source_id: str = "copernicus_glo30",
                     fill_max_search_px: int = 20) -> DemGrid:
    """
    Reproject a DEM (any CRS) onto a north-up UTM grid covering ``bbox`` at
    ``resolution_m``. Small voids are filled by inverse-distance interpolation
    (rasterio.fill); the fraction filled is recorded in the metadata.
    """
    bbox = validate_bbox(bbox)
    if resolution_m <= 0:
        raise DemError("resolution_m must be positive")
    w, s, e, n = bbox
    epsg = dst_epsg or utm_epsg_for((w + e) / 2, (s + n) / 2)
    dst_crs = CRS.from_epsg(epsg)

    l, b, r, t = transform_bounds("EPSG:4326", dst_crs, w, s, e, n, densify_pts=21)
    l = math.floor(l / resolution_m) * resolution_m
    t = math.ceil(t / resolution_m) * resolution_m
    cols = int(math.ceil((r - l) / resolution_m))
    rows = int(math.ceil((t - b) / resolution_m))
    if max_cells and rows * cols > max_cells:
        raise DemError(
            f"Domain {rows}x{cols}={rows*cols:,} cells exceeds max_cells={max_cells:,}. "
            f"Use a smaller area or a coarser resolution (e.g. {resolution_m*2:.0f} m)."
        )
    dst_transform = from_origin(l, t, resolution_m, resolution_m)
    dst = np.full((rows, cols), np.nan, dtype="float64")

    with rasterio.open(src_path) as src:
        src_nodata = src.nodata
        src_res = abs(src.res[0])
        src_crs = src.crs
        if src_crs is None:
            raise DemError(f"{src_path} has no CRS — cannot georeference")
        # Average when coarsening, bilinear otherwise
        native_m = src_res * 111_320 if src_crs.is_geographic else src_res
        resampling = Resampling.average if resolution_m > 1.5 * native_m else Resampling.bilinear
        reproject(
            source=rasterio.band(src, 1), destination=dst,
            src_nodata=src_nodata, dst_nodata=np.nan,
            dst_transform=dst_transform, dst_crs=dst_crs, resampling=resampling,
        )

    valid = np.isfinite(dst)
    void_frac = 1.0 - valid.mean()
    if valid.sum() == 0:
        raise DemError("DEM has no valid data inside the study area")
    filled = dst.copy()
    if void_frac > 0:
        sentinel = -32768.0
        tmp = np.where(valid, dst, sentinel).astype("float32")
        tmp = fillnodata(tmp, mask=valid.astype("uint8"), max_search_distance=fill_max_search_px)
        filled = tmp.astype("float64")
        filled[tmp == sentinel] = np.nan
    final_valid = np.isfinite(filled)

    source = {
        "source_id": source_id,
        "source_file": str(src_path),
        "native_resolution": f"{src_res} {'deg' if src_crs.is_geographic else 'm'}",
        "native_crs": src_crs.to_string(),
        "target_crs": f"EPSG:{epsg}",
        "resolution_m": resolution_m,
        "resampling": resampling.name,
        "void_fraction_before_fill": round(float(void_frac), 6),
        "void_fraction_after_fill": round(float(1.0 - final_valid.mean()), 6),
    }
    return DemGrid(np.where(final_valid, filled, 0.0), dst_transform, f"EPSG:{epsg}",
                   final_valid, bbox, source)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_dem(grid: DemGrid) -> Dict[str, Any]:
    """Return a structured quality report; ``ok`` is False on blocking issues."""
    issues: List[str] = []
    warnings: List[str] = []
    crs = CRS.from_string(grid.crs)
    if not crs.is_projected:
        issues.append("Grid CRS is not projected (metres required for hydraulics).")
    if abs(grid.transform.a + grid.transform.e) > 1e-9 or grid.transform.b or grid.transform.d:
        issues.append("Grid is not square / north-up.")

    z = grid.elevation[grid.valid]
    zmin, zmax = float(z.min()), float(z.max())
    if zmin < -450 or zmax > 8900:
        issues.append(f"Elevation range {zmin:.1f}..{zmax:.1f} m is physically implausible.")
    void = 1.0 - float(grid.valid.mean())
    if void > 0.05:
        issues.append(f"{void:.1%} of cells have no data after filling.")
    elif void > 0:
        warnings.append(f"{void:.2%} of cells have no data and are excluded from computation.")

    gy, gx = np.gradient(grid.elevation, grid.cell_size)
    slope = np.hypot(gx, gy)[grid.valid]
    flat_frac = float((slope < 0.001).mean())
    if flat_frac > 0.3:
        warnings.append(f"{flat_frac:.0%} of cells are near-flat (<0.1%): flow paths there are poorly constrained at this resolution.")
    sea_frac = float((z <= 0.5).mean())
    if sea_frac > 0.01:
        warnings.append(f"{sea_frac:.1%} of cells are at or below ~0 m (likely sea). Consider enabling the sea sink.")

    # Georeferencing round trip at the centre cell
    rows, cols = grid.shape
    lon, lat = grid.rowcol_to_lonlat(rows // 2, cols // 2)
    r2, c2 = grid.lonlat_to_rowcol(lon, lat)
    if (r2, c2) != (rows // 2, cols // 2):
        issues.append("Georeferencing round-trip mismatch at grid centre.")

    return {
        "ok": not issues,
        "issues": issues,
        "warnings": warnings,
        "rows": rows, "cols": cols, "cells": rows * cols,
        "cell_size_m": grid.cell_size,
        "area_km2": round(rows * cols * grid.cell_size ** 2 / 1e6, 3),
        "crs": grid.crs,
        "elevation_min_m": round(zmin, 2),
        "elevation_max_m": round(zmax, 2),
        "elevation_mean_m": round(float(z.mean()), 2),
        "slope_median_pct": round(float(np.median(slope)) * 100, 2),
        "flat_fraction": round(flat_frac, 4),
        "sea_fraction": round(sea_frac, 4),
        "void_fraction": round(void, 6),
    }

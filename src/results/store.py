"""
Simulation Results Storage and GeoTIFF Exporter.

Manages run directories under ``data/runs/<run_id>/``:
- Preserves full run parameters, scenario, and metrics in ``summary.json``
- Exports peak raster metrics as Cloud-Optimized GeoTIFFs (max_depth.tif, max_speed.tif, arrival_time.tif)
- Reads time-series frame arrays (depth, u, v) on demand
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import rasterio
from rasterio.transform import Affine

from ..geodata.dem import DemGrid


class RunStore:
    """
    Manages persistence and retrieval of simulation runs.
    """

    def __init__(self, runs_root: Path):
        self.runs_root = runs_root
        self.runs_root.mkdir(parents=True, exist_ok=True)

    def run_path(self, run_id: str) -> Path:
        return self.runs_root / run_id

    def exists(self, run_id: str) -> bool:
        return (self.run_path(run_id) / "summary.json").exists()

    def list_runs(self) -> List[Dict[str, Any]]:
        runs = []
        for folder in sorted(self.runs_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            summary_file = folder / "summary.json"
            if folder.is_dir() and summary_file.exists():
                try:
                    data = json.loads(summary_file.read_text(encoding="utf-8"))
                    runs.append({
                        "run_id": folder.name,
                        "status": data.get("status", "unknown"),
                        "engine": data.get("engine", "unknown"),
                        "simulated_duration_hours": data.get("simulated_duration_hours", 0),
                        "peak_depth_overall_m": data.get("peak_depth_overall_m", 0),
                        "total_snapshots": data.get("total_snapshots", 0),
                        "domain_id": data.get("domain_id", ""),
                        "created_at": folder.stat().st_mtime
                    })
                except Exception:
                    continue
        return runs

    def get_summary(self, run_id: str) -> Dict[str, Any]:
        p = self.run_path(run_id) / "summary.json"
        if not p.exists():
            raise FileNotFoundError(f"Run {run_id} not found.")
        return json.loads(p.read_text(encoding="utf-8"))

    def export_geotiffs(self, run_id: str, dem_grid: DemGrid) -> Dict[str, str]:
        """
        Generates standard georeferenced GeoTIFFs for peak metrics.
        """
        run_dir = self.run_path(run_id)
        rows, cols = dem_grid.shape
        exported = {}

        tiffs_to_create = [
            ("max_depth.npy", "max_depth.tif", "Water Depth (m)", -9999.0),
            ("max_speed.npy", "max_speed.tif", "Water Flow Velocity (m/s)", -9999.0),
            ("arrival_time_hr.npy", "arrival_time_hr.tif", "Flood Arrival Time (hours)", -9999.0)
        ]

        for npy_name, tif_name, desc, nodata_val in tiffs_to_create:
            npy_path = run_dir / npy_name
            if not npy_path.exists():
                continue

            arr = np.load(npy_path).astype(np.float32)
            # Mask inactive or no-data cells
            arr[~dem_grid.valid] = nodata_val

            tif_path = run_dir / tif_name
            with rasterio.open(
                tif_path,
                "w",
                driver="GTiff",
                height=rows,
                width=cols,
                count=1,
                dtype="float32",
                crs=dem_grid.crs,
                transform=dem_grid.transform,
                nodata=nodata_val,
                compress="deflate"
            ) as dst:
                dst.write(arr, 1)
                dst.set_band_description(1, desc)

            exported[tif_name] = str(tif_path)

        return exported

    def load_frame(self, run_id: str, frame_index: int) -> Dict[str, np.ndarray]:
        """
        Loads a specific time-slice frame (depth, u, v).
        """
        run_dir = self.run_path(run_id)
        frame_file = run_dir / "frames" / f"frame_{frame_index:04d}.npz"
        if not frame_file.exists():
            raise FileNotFoundError(f"Frame {frame_index} not found for run {run_id}")

        data = np.load(frame_file)
        return {
            "depth": data["depth"],
            "u": data["u"],
            "v": data["v"]
        }

    def get_frame_count(self, run_id: str) -> int:
        run_dir = self.run_path(run_id)
        frames_dir = run_dir / "frames"
        if not frames_dir.exists():
            return 0
        return len(list(frames_dir.glob("frame_*.npz")))

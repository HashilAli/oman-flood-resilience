"""
Background Job Execution & Lifecycle Manager.

Executes numerical hydraulic simulations asynchronously in background worker threads/processes,
ensuring that long-running calculations never freeze the FastAPI event loop or browser requests.
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional
import numpy as np

from ..geodata.dem import DemGrid
from ..hydraulics.base import RainfallScenario, SimulationConfig, SimulationProgress
from ..hydraulics.local_inertial import LocalInertialEngine
from ..hydraulics.losses import LossModel, SCSCurveNumberModel, ConstantLossModel, ZeroLossModel
from ..results.layers import LayerRenderer
from ..results.store import RunStore

logger = logging.getLogger("floodsim.jobs")


class JobManager:
    """
    Coordinates active and historical simulation tasks.
    """

    def __init__(self, data_root: Path, max_workers: int = 2):
        self.data_root = data_root
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.cancel_flags: Dict[str, threading.Event] = {}
        self.lock = threading.Lock()
        self.run_store = RunStore(data_root / "runs")

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Retrieves current progress from memory or from disk.
        """
        with self.lock:
            if job_id in self.active_jobs:
                return self.active_jobs[job_id]

        # Check disk if completed or restarted
        prog_file = self.run_store.run_path(job_id) / "progress.json"
        if prog_file.exists():
            try:
                return json.loads(prog_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        summary_file = self.run_store.run_path(job_id) / "summary.json"
        if summary_file.exists():
            return {
                "job_id": job_id,
                "status": "completed",
                "percentage": 100.0,
                "message": "Simulation finished successfully."
            }

        return {"job_id": job_id, "status": "not_found", "message": "Job ID does not exist."}

    def cancel_job(self, job_id: str) -> bool:
        with self.lock:
            if job_id in self.cancel_flags:
                self.cancel_flags[job_id].set()
                if job_id in self.active_jobs:
                    self.active_jobs[job_id]["status"] = "cancelling"
                    self.active_jobs[job_id]["message"] = "Cancellation requested."
                return True
        return False

    def submit_simulation(
        self,
        job_id: str,
        domain_id: str,
        dem_grid: DemGrid,
        rainfall: RainfallScenario,
        config: SimulationConfig,
        loss_model: LossModel,
        extra_meta: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Enqueues a new background simulation.
        """
        cancel_event = threading.Event()
        with self.lock:
            self.cancel_flags[job_id] = cancel_event
            self.active_jobs[job_id] = {
                "job_id": job_id,
                "domain_id": domain_id,
                "status": "queued",
                "percentage": 0.0,
                "current_time_seconds": 0.0,
                "total_time_seconds": config.total_time_seconds,
                "message": "Job queued for execution.",
                "created_at": time.time(),
            }

        self.executor.submit(
            self._execute_simulation_worker,
            job_id=job_id,
            domain_id=domain_id,
            dem_grid=dem_grid,
            rainfall=rainfall,
            config=config,
            loss_model=loss_model,
            cancel_event=cancel_event,
            extra_meta=extra_meta or {}
        )
        return job_id

    def _execute_simulation_worker(
        self,
        job_id: str,
        domain_id: str,
        dem_grid: DemGrid,
        rainfall: RainfallScenario,
        config: SimulationConfig,
        loss_model: LossModel,
        cancel_event: threading.Event,
        extra_meta: Dict[str, Any]
    ) -> None:
        run_dir = self.run_store.run_path(job_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        progress_file = run_dir / "progress.json"

        def on_progress(p: SimulationProgress):
            prog_data = asdict(p)
            prog_data["job_id"] = job_id
            prog_data["domain_id"] = domain_id
            with self.lock:
                if job_id in self.active_jobs:
                    self.active_jobs[job_id].update(prog_data)
            try:
                progress_file.write_text(json.dumps(prog_data), encoding="utf-8")
            except Exception:
                pass

        try:
            with self.lock:
                self.active_jobs[job_id]["status"] = "running"
                self.active_jobs[job_id]["message"] = "Starting 2D hydrodynamic solver..."

            engine = LocalInertialEngine()
            summary = engine.run(
                dem=dem_grid.elevation,
                cell_size_m=dem_grid.cell_size,
                rainfall=rainfall,
                config=config,
                loss_model=loss_model,
                output_dir=run_dir,
                progress_callback=on_progress,
                cancel_check=lambda: cancel_event.is_set(),
            )

            if summary.get("status") == "cancelled":
                with self.lock:
                    self.active_jobs[job_id]["status"] = "cancelled"
                    self.active_jobs[job_id]["message"] = "Simulation was cancelled."
                return

            # Append domain_id and extra metadata to summary
            summary["domain_id"] = domain_id
            summary["extra_meta"] = extra_meta
            summary["dem_meta"] = {
                "crs": dem_grid.crs,
                "bbox_lonlat": list(dem_grid.bbox_lonlat),
                "rows": dem_grid.shape[0],
                "cols": dem_grid.shape[1],
                "cell_size_m": dem_grid.cell_size
            }
            (run_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            # Generate GeoTIFF exports
            self.run_store.export_geotiffs(job_id, dem_grid)

            # Pre-render key overlays: peak depth PNG and peak hazard PNG
            max_depth_arr = np.load(run_dir / "max_depth.npy")
            max_speed_arr = np.load(run_dir / "max_speed.npy")

            depth_rgba = LayerRenderer.depth_to_rgba(max_depth_arr)
            LayerRenderer.export_layer_png(depth_rgba, dem_grid, run_dir / "layers" / "max_depth.png")

            hazard_rgba = LayerRenderer.hazard_to_rgba(max_depth_arr, max_speed_arr)
            LayerRenderer.export_layer_png(hazard_rgba, dem_grid, run_dir / "layers" / "hazard.png")

            with self.lock:
                self.active_jobs[job_id]["status"] = "completed"
                self.active_jobs[job_id]["percentage"] = 100.0
                self.active_jobs[job_id]["message"] = "Simulation completed successfully."

        except Exception as exc:
            logger.exception("Simulation job %s encountered an error", job_id)
            with self.lock:
                self.active_jobs[job_id]["status"] = "failed"
                self.active_jobs[job_id]["message"] = f"Simulation failed: {str(exc)}"
            try:
                progress_file.write_text(json.dumps({
                    "job_id": job_id, "status": "failed", "error": str(exc)
                }), encoding="utf-8")
            except Exception:
                pass
        finally:
            with self.lock:
                self.cancel_flags.pop(job_id, None)

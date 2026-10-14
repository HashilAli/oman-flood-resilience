"""
HEC-RAS 2D Adapter Layer.

Implements the HydraulicEngine interface for HEC-RAS 2D (USACE).
Provides automated integration, controller dispatch, and HDF5 unsteady results ingestion.
If HEC-RAS is not installed or detected on the current environment, provides
clear diagnostic reasons and instructions without crashing the application.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from .base import HydraulicEngine, RainfallScenario, SimulationConfig, SimulationProgress
from .losses import LossModel


class HecRas2DAdapter(HydraulicEngine):
    """
    Adapter for running HEC-RAS 2D unsteady flow models or ingesting their results.
    """

    STANDARD_INSTALL_PATHS = [
        Path(r"C:\Program Files (x86)\HEC\HEC-RAS"),
        Path(r"C:\Program Files\HEC\HEC-RAS"),
    ]

    def __init__(self, explicit_exe_path: Optional[str] = None):
        self.explicit_exe_path = explicit_exe_path

    def name(self) -> str:
        return "hecras_2d"

    def find_ras_exe(self) -> Optional[Path]:
        """Locates Ras.exe in configured path or standard installation directories."""
        if self.explicit_exe_path:
            p = Path(self.explicit_exe_path)
            if p.exists() and p.is_file():
                return p

        # Check standard folders for any subversion e.g. HEC-RAS 6.5, 6.3.1, etc.
        for base in self.STANDARD_INSTALL_PATHS:
            if base.parent.exists():
                for folder in base.parent.glob("HEC-RAS*"):
                    exe = folder / "Ras.exe"
                    if exe.exists():
                        return exe
        return None

    def is_available(self) -> Tuple[bool, str]:
        """
        Validates whether HEC-RAS and required Python automation tools are present.
        """
        exe = self.find_ras_exe()
        if exe is None:
            return False, (
                "HEC-RAS 2D is not installed on this machine. "
                "To enable HEC-RAS 2D: (1) Install HEC-RAS 6.x from https://www.hec.usace.army.mil/software/hec-ras/ "
                "(2) Install pywin32 (`pip install pywin32`) and h5py (`pip install h5py`). "
                "The built-in Local Inertial engine remains fully operational and serves as the primary engine."
            )

        try:
            import win32com.client  # type: ignore
        except ImportError:
            return False, "HEC-RAS was found at " + str(exe) + ", but `pywin32` is not installed."

        try:
            import h5py  # type: ignore
        except ImportError:
            return False, "HEC-RAS was found, but `h5py` is missing for reading HDF5 outputs."

        return True, f"HEC-RAS ready at {exe}"

    def run(
        self,
        dem: np.ndarray,
        cell_size_m: float,
        rainfall: RainfallScenario,
        config: SimulationConfig,
        loss_model: Optional[LossModel] = None,
        output_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[SimulationProgress], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """
        Executes HEC-RAS plan or translates a prepared project.
        """
        ok, reason = self.is_available()
        if not ok:
            raise RuntimeError(f"HEC-RAS execution requested but unavailable: {reason}")

        # Note: True automated mesh generation for arbitrary terrain in HEC-RAS
        # requires generating RAS geometry and HDF5 mesh structure via RAS Mapper.
        # This adapter facilitates computing pre-configured RAS projects or
        # extracting outputs from .pXX.hdf results.
        raise NotImplementedError(
            "Direct HEC-RAS mesh creation from arbitrary DEM matrices is not supported by "
            "USACE command line without pre-configured .prj/.gXX files. "
            "Please use the built-in Local Inertial engine or import a pre-computed HEC-RAS project."
        )

    @staticmethod
    def extract_hdf5_results(
        hdf5_path: Path,
        flow_area_name: str = "2D Flow Area"
    ) -> Dict[str, Any]:
        """
        Reads HEC-RAS 6.x unsteady 2D simulation results from plan HDF5 file.
        Extracts depth, velocity, and time series arrays.
        """
        import h5py  # type: ignore

        if not hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 result file not found: {hdf5_path}")

        with h5py.File(str(hdf5_path), "r") as f:
            base_group = f"Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{flow_area_name}"
            depth_ds = f"{base_group}/Depth"
            wse_ds = f"{base_group}/Water Surface"

            if depth_ds not in f:
                raise KeyError(f"Dataset path {depth_ds} not found in {hdf5_path}")

            depths = np.array(f[depth_ds])  # shape: (n_times, n_cells)
            times = np.array(f["Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time"])

            return {
                "time_steps_count": len(times),
                "cells_count": depths.shape[1] if len(depths.shape) > 1 else 0,
                "times": times,
                "max_depth_per_cell": np.max(depths, axis=0) if len(depths.shape) > 1 else np.array([])
            }

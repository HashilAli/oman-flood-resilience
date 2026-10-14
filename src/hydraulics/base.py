"""
Base abstractions, interfaces, and data types for hydraulic simulation engines.

Enforces strict separation between the physical numerical solver,
the scenario specifications, boundary conditions, and output storage.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np


class BoundaryType(str, Enum):
    CLOSED = "closed"          # Solid wall (no flux out or in)
    FREE_OUTFLOW = "free"      # Critical / normal depth outflow at grid edges
    SEA_SINK = "sea_sink"      # Cells at or below sea level act as infinite sinks


@dataclass
class RainfallScenario:
    """
    Time-varying rainfall specification (hyetograph).
    """
    duration_hours: float
    total_depth_mm: float
    distribution: str = "scs_type2"   # "scs_type2", "uniform", or "custom"
    time_series_hours: Optional[List[float]] = None      # for custom distribution
    intensity_mm_per_hr: Optional[List[float]] = None    # for custom distribution

    def get_cumulative_hyetograph(self, times_seconds: Sequence[float]) -> np.ndarray:
        """
        Returns cumulative precipitation in millimetres at given elapsed times (seconds).
        """
        times_hr = np.array(times_seconds) / 3600.0
        tot = self.total_depth_mm
        dur = self.duration_hours

        if self.distribution == "uniform":
            frac = np.clip(times_hr / dur, 0.0, 1.0)
            return frac * tot

        elif self.distribution == "scs_type2":
            # SCS 24-hr Type II normalized cumulative distribution polynomial approximation
            # Normalized t_norm = t / dur in [0, 1]
            t_norm = np.clip(times_hr / dur, 0.0, 1.0)
            
            # Standard SCS Type II dimensionless S-curve (inflection at t/dur = 0.5)
            # Piecewise or smooth sigmoidal formulation
            frac = np.where(
                t_norm <= 0.5,
                0.5 * (2.0 * t_norm) ** 3.5,
                1.0 - 0.5 * (2.0 * (1.0 - t_norm)) ** 3.5
            )
            # Rescale slightly so fraction reaches exactly 1.0 at dur
            frac = np.clip(frac, 0.0, 1.0)
            return frac * tot

        elif self.distribution == "custom" and self.time_series_hours and self.intensity_mm_per_hr:
            ts = np.array(self.time_series_hours)
            intens = np.array(self.intensity_mm_per_hr)
            # Integrate intensities
            cum_vals = [0.0]
            for i in range(len(ts) - 1):
                dt_h = ts[i + 1] - ts[i]
                cum_vals.append(cum_vals[-1] + intens[i] * dt_h)
            return np.interp(times_hr, ts, cum_vals, left=0.0, right=cum_vals[-1])
        else:
            frac = np.clip(times_hr / dur, 0.0, 1.0)
            return frac * tot


@dataclass
class SimulationConfig:
    """
    Numerical and physical simulation parameters.
    """
    total_time_seconds: float = 3600.0 * 6.0    # 6 hours default
    output_interval_seconds: float = 900.0      # Save snapshot every 15 minutes
    manning_n: float = 0.040                    # Default Manning roughness coefficient
    manning_grid: Optional[np.ndarray] = None   # Optional spatially distributed Manning n
    boundary_type: BoundaryType = BoundaryType.FREE_OUTFLOW
    enable_sea_sink: bool = True
    sea_level_threshold_m: float = 0.2
    cfl_factor: float = 0.70                    # Courant-Friedrichs-Lewy stability factor
    max_dt_seconds: float = 10.0                # Maximum adaptive time step
    min_dt_seconds: float = 0.01                # Minimum allowed time step
    min_depth_wet_m: float = 0.005              # Minimum water depth to be considered wet (5 mm)


@dataclass
class SimulationProgress:
    current_time_seconds: float
    total_time_seconds: float
    percentage: float
    current_dt: float
    current_max_depth_m: float
    current_water_volume_m3: float
    mass_balance_error_pct: float
    status: str = "running"                     # "running", "completed", "failed", "cancelled"
    message: str = ""


class HydraulicEngine(ABC):
    """
    Abstract interface for 2D hydraulic simulation engines.
    Allows transparent swapping between Local Inertial (Python/NumPy) and HEC-RAS.
    """

    @abstractmethod
    def name(self) -> str:
        """Engine identifier (e.g. 'local_inertial', 'hec_ras_2d')."""
        pass

    @abstractmethod
    def is_available(self) -> Tuple[bool, str]:
        """Check whether dependencies/tools required by this engine are available."""
        pass

    @abstractmethod
    def run(
        self,
        dem: np.ndarray,
        cell_size_m: float,
        rainfall: RainfallScenario,
        config: SimulationConfig,
        loss_model: Any,
        output_dir: Path,
        progress_callback: Optional[Callable[[SimulationProgress], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """
        Execute simulation and write time-sliced outputs to output_dir.
        """
        pass

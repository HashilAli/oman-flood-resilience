"""
Hydrological Loss & Infiltration Models.

Translates gross hyetograph precipitation into surface runoff excess.
Implements:
1. Cumulative SCS Runoff Curve Number (CN) Method (USDA-SCS NEH-4).
2. Constant Infiltration Rate (Initial and Continuing Losses).
3. Zero Loss (impermeable baseline / theoretical tests).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np


class LossModel(ABC):
    @abstractmethod
    def compute_incremental_excess(
        self,
        cum_rain_prev_mm: float,
        cum_rain_curr_mm: float,
        dt_seconds: float
    ) -> Union[float, np.ndarray]:
        """
        Calculates incremental effective runoff depth in millimetres
        over the time step between previous cumulative rain and current cumulative rain.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal states before a new run."""
        pass


@dataclass
class ZeroLossModel(LossModel):
    """Impermeable domain (100% runoff, useful for verification and conservative safety limits)."""

    def compute_incremental_excess(
        self, cum_rain_prev_mm: float, cum_rain_curr_mm: float, dt_seconds: float
    ) -> float:
        return max(0.0, cum_rain_curr_mm - cum_rain_prev_mm)

    def reset(self) -> None:
        pass


@dataclass
class SCSCurveNumberModel(LossModel):
    """
    Cumulative SCS Runoff Curve Number model (USDA NRCS).
    S = 25400 / CN - 254 (Max potential retention in mm)
    Ia = initial_abstraction_ratio * S (default 0.2, or 0.05 for arid catchments)
    Pe(t) = (P(t) - Ia)^2 / (P(t) - Ia + S) for P(t) > Ia, else 0
    ΔPe = Pe(t) - Pe(t-1)
    """
    curve_number: float = 85.0
    initial_abstraction_ratio: float = 0.20
    depression_storage_mm: float = 0.0

    def __post_init__(self):
        cn = np.clip(self.curve_number, 30.0, 99.0)
        self.S = (25400.0 / cn) - 254.0
        self.Ia = self.initial_abstraction_ratio * self.S + self.depression_storage_mm
        self._prev_cum_excess = 0.0

    def compute_cumulative_excess(self, cum_rain_mm: float) -> float:
        if cum_rain_mm <= self.Ia:
            return 0.0
        P_Ia = cum_rain_mm - self.Ia
        return float((P_Ia ** 2) / (P_Ia + self.S))

    def compute_incremental_excess(
        self, cum_rain_prev_mm: float, cum_rain_curr_mm: float, dt_seconds: float
    ) -> float:
        pe_curr = self.compute_cumulative_excess(cum_rain_curr_mm)
        pe_prev = self.compute_cumulative_excess(cum_rain_prev_mm)
        incremental = max(0.0, pe_curr - pe_prev)
        return incremental

    def reset(self) -> None:
        self._prev_cum_excess = 0.0


@dataclass
class ConstantLossModel(LossModel):
    """
    Initial Loss / Continuing Loss (IL/CL) model widely applied in hydrology.
    - Initial Loss (mm): All rainfall up to IL is absorbed.
    - Continuing Loss (mm/h): Ongoing constant infiltration capacity.
    """
    initial_loss_mm: float = 15.0
    continuing_loss_mm_per_hr: float = 3.5

    def __post_init__(self):
        self._accumulated_loss = 0.0

    def compute_incremental_excess(
        self, cum_rain_prev_mm: float, cum_rain_curr_mm: float, dt_seconds: float
    ) -> float:
        rain_step = max(0.0, cum_rain_curr_mm - cum_rain_prev_mm)
        if rain_step <= 0.0:
            return 0.0

        # Satisfy initial loss first
        if self._accumulated_loss < self.initial_loss_mm:
            remaining_il = self.initial_loss_mm - self._accumulated_loss
            loss_taken = min(rain_step, remaining_il)
            self._accumulated_loss += loss_taken
            rain_step -= loss_taken

        if rain_step <= 0.0:
            return 0.0

        # Continuing loss
        cl_rate_mm_per_sec = self.continuing_loss_mm_per_hr / 3600.0
        max_cl = cl_rate_mm_per_sec * dt_seconds
        cl_taken = min(rain_step, max_cl)
        excess = rain_step - cl_taken
        return max(0.0, excess)

    def reset(self) -> None:
        self._accumulated_loss = 0.0

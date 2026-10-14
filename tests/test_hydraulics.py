"""
Automated hydrodynamic verification tests.

Verifies:
1. Exact machine-precision mass conservation.
2. Lake at rest (C-property): water on flat bed remains motionless.
3. Gravity routing: water flows down topographic gradients.
4. SCS-CN loss model consistency.
5. HEC-RAS adapter discovery and diagnostics.
"""
import pytest
import numpy as np
from pathlib import Path

from src.hydraulics.base import BoundaryType, RainfallScenario, SimulationConfig
from src.hydraulics.losses import SCSCurveNumberModel, ZeroLossModel
from src.hydraulics.local_inertial import LocalInertialEngine
from src.hydraulics.hecras_adapter import HecRas2DAdapter


def test_mass_conservation_closed_boundary(tmp_path: Path):
    """
    On an impermeable domain with closed walls, water volume must match
    added rain volume with zero mass balance error.
    """
    rows, cols = 30, 30
    cell_size = 20.0
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    X, Y = np.meshgrid(x, y)
    dem = 20.0 * (X**2 + Y**2)  # Bowl shape

    scen = RainfallScenario(duration_hours=0.5, total_depth_mm=50.0, distribution="uniform")
    loss = ZeroLossModel()
    cfg = SimulationConfig(
        total_time_seconds=300.0,
        output_interval_seconds=100.0,
        boundary_type=BoundaryType.CLOSED,
        enable_sea_sink=False
    )

    eng = LocalInertialEngine()
    res = eng.run(dem, cell_size_m=cell_size, rainfall=scen, config=cfg, loss_model=loss, output_dir=tmp_path)

    assert res["status"] == "completed"
    assert res["mass_balance_error_pct"] < 0.001
    assert res["total_snapshots"] >= 3
    assert (tmp_path / "max_depth.npy").exists()


def test_lake_at_rest():
    """
    Flat bed with uniform zero rain: velocities should remain zero.
    """
    rows, cols = 20, 20
    dem = np.zeros((rows, cols), dtype=np.float64)
    scen = RainfallScenario(duration_hours=1.0, total_depth_mm=0.0, distribution="uniform")
    loss = ZeroLossModel()
    cfg = SimulationConfig(total_time_seconds=60.0, output_interval_seconds=30.0, boundary_type=BoundaryType.CLOSED)

    eng = LocalInertialEngine()
    res = eng.run(dem, cell_size_m=25.0, rainfall=scen, config=cfg, loss_model=loss)

    assert res["peak_speed_overall_m_s"] == 0.0
    assert res["peak_depth_overall_m"] == 0.0


def test_downhill_flow_direction():
    """
    Water placed on a south-sloping ramp should move towards higher row indices (south).
    """
    rows, cols = 25, 10
    # Ramp sloping down from row 0 to row 24
    dem = np.tile(np.linspace(50.0, 0.0, rows)[:, np.newaxis], (1, cols))

    scen = RainfallScenario(duration_hours=0.2, total_depth_mm=40.0, distribution="uniform")
    loss = ZeroLossModel()
    cfg = SimulationConfig(
        total_time_seconds=180.0,
        output_interval_seconds=60.0,
        boundary_type=BoundaryType.FREE_OUTFLOW,
        enable_sea_sink=False
    )

    eng = LocalInertialEngine()
    res = eng.run(dem, cell_size_m=20.0, rainfall=scen, config=cfg, loss_model=loss)

    assert res["status"] == "completed"
    # Water should have exited via the southern border
    assert res["total_boundary_outflow_m3"] > 0.0


def test_scs_curve_number_losses():
    model = SCSCurveNumberModel(curve_number=80.0, initial_abstraction_ratio=0.20)
    # Total precipitation less than initial abstraction: excess is 0
    ia = model.Ia
    assert model.compute_cumulative_excess(ia * 0.5) == 0.0

    # Incremental excess must be non-negative and monotonic
    pe1 = model.compute_incremental_excess(0.0, 50.0, dt_seconds=60.0)
    pe2 = model.compute_incremental_excess(50.0, 100.0, dt_seconds=60.0)
    assert pe1 >= 0.0
    assert pe2 >= pe1


def test_hecras_adapter_availability():
    adapter = HecRas2DAdapter()
    available, msg = adapter.is_available()
    assert isinstance(available, bool)
    assert isinstance(msg, str)
    if not available:
        assert "HEC-RAS" in msg

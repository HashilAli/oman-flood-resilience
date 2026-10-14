"""
FastAPI REST Endpoints Integration Tests.
"""
import time
import pytest
from fastapi.testclient import TestClient

from src.api.server import app

client = TestClient(app)


def test_api_sources():
    resp = client.get("/api/sources")
    assert resp.status_code == 200
    data = resp.json()
    assert "sources" in data
    assert "copernicus_glo30" in data["sources"]
    assert "presets" in data
    assert "wadi_adai" in data["presets"]


def test_api_presets():
    resp = client.get("/api/presets")
    assert resp.status_code == 200
    presets = resp.json()
    assert len(presets) >= 4
    ids = [p["id"] for p in presets]
    assert "wadi_adai" in ids
    assert "wadi_shab" in ids


def test_domain_prepare_and_simulation_run():
    # 1. Prepare domain for Wadi Adai (preset)
    prep_resp = client.post("/api/domains/prepare", json={"preset_id": "wadi_adai"})
    assert prep_resp.status_code == 200
    domain_data = prep_resp.json()
    assert domain_data["domain_id"] == "wadi_adai"
    assert "bounds_wgs84" in domain_data

    # 2. Run small fast test simulation (0.5 hour, 15 min output)
    sim_payload = {
        "domain_id": "wadi_adai",
        "rainfall_total_mm": 50.0,
        "rainfall_duration_hours": 0.5,
        "rainfall_distribution": "uniform",
        "loss_model_type": "scs_cn",
        "curve_number": 88.0,
        "manning_n": 0.040,
        "simulation_hours": 0.5,
        "output_interval_minutes": 15.0
    }
    run_resp = client.post("/api/simulations/run", json=sim_payload)
    assert run_resp.status_code == 200
    job_info = run_resp.json()
    job_id = job_info["job_id"]
    assert job_info["status"] in ("queued", "running")

    # 3. Poll job status until completed (timeout 30 seconds)
    t0 = time.time()
    completed = False
    while time.time() - t0 < 50:
        stat_resp = client.get(f"/api/jobs/{job_id}")
        assert stat_resp.status_code == 200
        st = stat_resp.json()
        if st.get("status") == "completed":
            completed = True
            break
        time.sleep(1)

    assert completed is True

    # 4. Get run summary
    sum_resp = client.get(f"/api/runs/{job_id}/summary")
    assert sum_resp.status_code == 200
    summary = sum_resp.json()
    assert summary["status"] == "completed"
    assert summary["mass_balance_error_pct"] < 0.01
    assert "snapshots" in summary
    assert len(summary["snapshots"]) >= 2

    # 5. Query Point hazard
    center_lon = domain_data["center"][0]
    center_lat = domain_data["center"][1]
    pt_resp = client.get(f"/api/runs/{job_id}/point?lon={center_lon}&lat={center_lat}")
    assert pt_resp.status_code == 200
    pt_data = pt_resp.json()
    assert "elevation_m" in pt_data
    assert "peak_depth_m" in pt_data
    assert "hydrograph" in pt_data

    # 6. Check georeferenced layer PNG endpoint
    png_resp = client.get(f"/api/runs/{job_id}/layers/max_depth.png")
    assert png_resp.status_code == 200
    assert png_resp.headers["content-type"] == "image/png"

"""
FastAPI Application Server for 3D Hydrological Simulation Platform.

Provides RESTful endpoints for:
- Geospatial domain acquisition (Copernicus DEM & OSM vectors)
- Non-blocking hydrodynamic simulation execution
- Progress monitoring and job cancellation
- Georeferenced image layer serving (EPSG:4326 PNGs for CesiumJS)
- Real-time GIS hazard analytics (point queries, road severance, building impact)
- Static file serving for the interactive CesiumJS web client
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import get_settings
from ..geodata.catalog import get_sources
from ..geodata.dem import DemGrid, fetch_copernicus_dem, prepare_dem_grid, validate_dem
from ..geodata.osm import fetch_osm_layer
from ..hydraulics.base import BoundaryType, RainfallScenario, SimulationConfig
from ..hydraulics.losses import ConstantLossModel, SCSCurveNumberModel, ZeroLossModel
from ..results.analysis import FloodAnalytics
from ..results.layers import LayerRenderer
from ..results.store import RunStore
from .jobs import JobManager

# ---------------------------------------------------------------------------
# Setup & App Configuration
# ---------------------------------------------------------------------------
settings = get_settings()
app = FastAPI(
    title="منصة راصد عُمان: المحاكاة الهيدروليكية ثلاثية الأبعاد",
    description="3D Hydrological & Flash Flood Simulation Platform with Real Terrain for the Sultanate of Oman",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

job_manager = JobManager(settings.data_dir, max_workers=settings.workers)
run_store = RunStore(settings.runs_dir)

# Oman Catchment Preset Catalog
OMAN_PRESETS = {
    "wadi_adai": {
        "id": "wadi_adai",
        "name_ar": "حوض وادي عدي - محافظة مسقط",
        "name_en": "Wadi Adai, Muscat",
        "governorate": "محافظة مسقط",
        "bbox": [58.50, 23.54, 58.58, 23.62],
        "default_rainfall_mm": 120.0,
        "default_cn": 86.0,
        "description": "حوض جبلي حضري ذو استجابة سريعة جداً للسيول، يخترق قلب العاصمة مسقط ويصل بين العامرات والقرم."
    },
    "wadi_shab": {
        "id": "wadi_shab",
        "name_ar": "حوض وادي شاب وقلهات - محافظة جنوب الشرقية",
        "name_en": "Wadi Shab & Tiwi, South Al Sharqiyah",
        "governorate": "محافظة جنوب الشرقية",
        "bbox": [59.20, 22.80, 59.28, 22.88],
        "default_rainfall_mm": 150.0,
        "default_cn": 88.0,
        "description": "خوانق صخرية عميقة شديدة الانحدار تعبر الطريق الاستراتيجي قريات - صور وتصب في بحر عُمان."
    },
    "wadi_ahan": {
        "id": "wadi_ahan",
        "name_ar": "حوض وادي عاهن وصحم - محافظة شمال الباطنة",
        "name_en": "Wadi Ahan & Saham, North Al Batinah",
        "governorate": "محافظة شمال الباطنة",
        "bbox": [56.84, 23.95, 56.92, 24.03],
        "default_rainfall_mm": 110.0,
        "default_cn": 82.0,
        "description": "مروحة فيضية وسهول ممتدة تتأثر بالسيول المنحدرة من جبال الحجر الغربي نحو ساحل الباطنة."
    },
    "wadi_darbat": {
        "id": "wadi_darbat",
        "name_ar": "حوض وادي دربات وطاقة - محافظة ظفار",
        "name_en": "Wadi Darbat & Taqah, Dhofar",
        "governorate": "محافظة ظفار",
        "bbox": [54.40, 17.05, 54.48, 17.13],
        "default_rainfall_mm": 180.0,
        "default_cn": 80.0,
        "description": "حوض كارستي جيري يتأثر بأنواء بحر العرب المدارية الشديدة وشلالات دربات التاريخية."
    }
}


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class DomainPrepareRequest(BaseModel):
    preset_id: Optional[str] = None
    bbox: Optional[List[float]] = None  # [west, south, east, north]
    domain_id: Optional[str] = None
    resolution_m: float = Field(default=30.0, ge=10.0, le=90.0)


class SimulationRequest(BaseModel):
    domain_id: str
    rainfall_total_mm: float = Field(default=100.0, ge=5.0, le=500.0)
    rainfall_duration_hours: float = Field(default=6.0, ge=0.2, le=48.0)
    rainfall_distribution: str = Field(default="scs_type2")  # "scs_type2", "uniform"
    loss_model_type: str = Field(default="scs_cn")            # "scs_cn", "constant", "zero"
    curve_number: float = Field(default=85.0, ge=30.0, le=98.0)
    manning_n: float = Field(default=0.040, ge=0.015, le=0.15)
    simulation_hours: float = Field(default=6.0, ge=0.5, le=48.0)
    output_interval_minutes: float = Field(default=15.0, ge=2.0, le=120.0)


# ---------------------------------------------------------------------------
# API Endpoints: Metadata & Catalog
# ---------------------------------------------------------------------------
@app.get("/api/sources", tags=["Metadata"])
def get_data_sources():
    """Returns provenance, resolution, accuracy, and licenses of all geospatial data sources."""
    return {
        "sources": get_sources(),
        "presets": OMAN_PRESETS,
        "cesium_ion_available": bool(settings.cesium_ion_token),
        "cesium_ion_token": settings.cesium_ion_token or ""
    }


@app.get("/api/presets", tags=["Metadata"])
def list_presets():
    """Returns supported Omani catchment presets."""
    return list(OMAN_PRESETS.values())


# ---------------------------------------------------------------------------
# API Endpoints: Domains Management
# ---------------------------------------------------------------------------
def _domain_dir(domain_id: str) -> Path:
    return settings.domains_dir / domain_id


@app.get("/api/domains", tags=["Domains"])
def list_domains():
    """Lists prepared DEM domains stored on server."""
    domains = []
    for d in settings.domains_dir.iterdir():
        if d.is_dir() and (d / "dem.tif").exists():
            meta_path = d / "dem_meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            val_path = d / "validation.json"
            val = json.loads(val_path.read_text(encoding="utf-8")) if val_path.exists() else {}
            domains.append({
                "domain_id": d.name,
                "bbox_lonlat": meta.get("bbox_lonlat"),
                "crs": meta.get("crs"),
                "area_km2": val.get("area_km2"),
                "rows": val.get("rows"),
                "cols": val.get("cols"),
                "elevation_range": [val.get("elevation_min_m"), val.get("elevation_max_m")],
            })
    return domains


@app.post("/api/domains/prepare", tags=["Domains"])
def prepare_domain(req: DomainPrepareRequest):
    """
    Downloads real Copernicus GLO-30 DEM, reprojects to UTM, validates terrain,
    and caches the domain locally.
    """
    if req.preset_id:
        if req.preset_id not in OMAN_PRESETS:
            raise HTTPException(400, f"Preset '{req.preset_id}' not found.")
        preset = OMAN_PRESETS[req.preset_id]
        bbox = preset["bbox"]
        domain_id = req.domain_id or req.preset_id
    elif req.bbox:
        bbox = req.bbox
        domain_id = req.domain_id or f"custom_{int(time.time())}"
    else:
        raise HTTPException(400, "Either 'preset_id' or 'bbox' must be provided.")

    target_dir = _domain_dir(domain_id)

    # If domain already exists and is valid, return immediately
    if (target_dir / "dem.tif").exists() and (target_dir / "validation.json").exists():
        val = json.loads((target_dir / "validation.json").read_text(encoding="utf-8"))
        grid = DemGrid.load(target_dir)
        return {
            "status": "ready",
            "message": "Domain already prepared and cached.",
            "domain_id": domain_id,
            "validation": val,
            "bounds_wgs84": grid.bounds_lonlat(),
            "center": [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]
        }

    try:
        raw_dem_path = fetch_copernicus_dem(
            bbox=bbox,
            cache_dir=settings.cache_dir / "dem",
            base_url=settings.copernicus_dem_base_url
        )
        grid = prepare_dem_grid(
            src_path=raw_dem_path,
            bbox=bbox,
            resolution_m=req.resolution_m,
            max_cells=settings.max_cells
        )
        grid.save(target_dir)

        val_report = validate_dem(grid)
        (target_dir / "validation.json").write_text(
            json.dumps(val_report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        return {
            "status": "ready",
            "message": "DEM fetched and metric grid prepared successfully.",
            "domain_id": domain_id,
            "validation": val_report,
            "bounds_wgs84": grid.bounds_lonlat(),
            "center": [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]
        }
    except Exception as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Failed to prepare domain: {str(exc)}")


@app.get("/api/domains/{domain_id}/osm/{layer}", tags=["Domains"])
def get_osm_vector_layer(domain_id: str, layer: str):
    """
    Returns real OpenStreetMap vector layer (roads, buildings, or waterways) for domain.
    """
    d_dir = _domain_dir(domain_id)
    if not d_dir.exists():
        raise HTTPException(404, f"Domain '{domain_id}' not found.")

    grid = DemGrid.load(d_dir)
    try:
        fc = fetch_osm_layer(
            layer=layer,
            bbox=grid.bbox_lonlat,
            cache_dir=settings.cache_dir / "osm",
            overpass_url=settings.overpass_url,
            user_agent=settings.user_agent
        )
        return fc
    except Exception as exc:
        raise HTTPException(502, f"Failed to fetch OSM layer '{layer}': {str(exc)}")


# ---------------------------------------------------------------------------
# API Endpoints: Simulation Execution & Job Management
# ---------------------------------------------------------------------------
@app.post("/api/simulations/run", tags=["Simulations"])
def start_simulation(req: SimulationRequest):
    """
    Initiates a 2D hydrodynamic flood simulation job in the background.
    """
    d_dir = _domain_dir(req.domain_id)
    if not (d_dir / "dem.tif").exists():
        raise HTTPException(404, f"Domain '{req.domain_id}' does not exist. Prepare it first.")

    grid = DemGrid.load(d_dir)

    # 1. Setup rainfall scenario
    rainfall = RainfallScenario(
        duration_hours=req.rainfall_duration_hours,
        total_depth_mm=req.rainfall_total_mm,
        distribution=req.rainfall_distribution
    )

    # 2. Setup loss model
    if req.loss_model_type == "scs_cn":
        loss_model = SCSCurveNumberModel(curve_number=req.curve_number)
    elif req.loss_model_type == "constant":
        loss_model = ConstantLossModel()
    else:
        loss_model = ZeroLossModel()

    # 3. Setup simulation config
    config = SimulationConfig(
        total_time_seconds=req.simulation_hours * 3600.0,
        output_interval_seconds=req.output_interval_minutes * 60.0,
        manning_n=req.manning_n,
        boundary_type=BoundaryType.FREE_OUTFLOW,
        enable_sea_sink=True
    )

    job_id = f"run_{req.domain_id}_{int(time.time())}"
    job_manager.submit_simulation(
        job_id=job_id,
        domain_id=req.domain_id,
        dem_grid=grid,
        rainfall=rainfall,
        config=config,
        loss_model=loss_model,
        extra_meta=req.model_dump()
    )

    return {
        "job_id": job_id,
        "domain_id": req.domain_id,
        "status": "queued",
        "message": "Simulation launched in background."
    }


@app.get("/api/jobs/{job_id}", tags=["Simulations"])
def get_job_progress(job_id: str):
    """Polls real-time progress of a simulation job."""
    return job_manager.get_job_status(job_id)


@app.post("/api/jobs/{job_id}/cancel", tags=["Simulations"])
def cancel_job(job_id: str):
    """Cancels a running background simulation."""
    success = job_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(400, "Job cannot be cancelled (either not active or finished).")
    return {"job_id": job_id, "status": "cancelling"}


# ---------------------------------------------------------------------------
# API Endpoints: Simulation Results & Visual Layers
# ---------------------------------------------------------------------------
@app.get("/api/runs", tags=["Results"])
def list_runs():
    """Lists all historical simulation runs."""
    return run_store.list_runs()


@app.get("/api/runs/{run_id}/summary", tags=["Results"])
def get_run_summary(run_id: str):
    """Returns complete metadata, KPIs, and snapshot catalog for a run."""
    try:
        return run_store.get_summary(run_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Run '{run_id}' not found.")


@app.get("/api/runs/{run_id}/layers/depth/{frame_index}.png", tags=["Results"])
def get_depth_frame_png(run_id: str, frame_index: int):
    """
    Generates and returns an on-the-fly or cached georeferenced transparent PNG
    for water depth at frame index i.
    """
    summary = run_store.get_summary(run_id)
    domain_id = summary.get("domain_id")
    grid = DemGrid.load(_domain_dir(domain_id))

    run_dir = run_store.run_path(run_id)
    cached_png = run_dir / "layers" / f"depth_frame_{frame_index:04d}.png"
    if cached_png.exists():
        return FileResponse(cached_png, media_type="image/png")

    try:
        frame_data = run_store.load_frame(run_id, frame_index)
    except FileNotFoundError:
        raise HTTPException(404, f"Frame {frame_index} not found for run {run_id}.")

    rgba = LayerRenderer.depth_to_rgba(frame_data["depth"])
    LayerRenderer.export_layer_png(rgba, grid, cached_png)

    return FileResponse(cached_png, media_type="image/png")


@app.get("/api/runs/{run_id}/layers/max_depth.png", tags=["Results"])
def get_max_depth_png(run_id: str):
    """Returns georeferenced PNG for peak water depth."""
    summary = run_store.get_summary(run_id)
    domain_id = summary.get("domain_id")
    grid = DemGrid.load(_domain_dir(domain_id))

    png_path = run_store.run_path(run_id) / "layers" / "max_depth.png"
    if not png_path.exists():
        arr = np.load(run_store.run_path(run_id) / "max_depth.npy")
        rgba = LayerRenderer.depth_to_rgba(arr)
        LayerRenderer.export_layer_png(rgba, grid, png_path)

    return FileResponse(png_path, media_type="image/png")


@app.get("/api/runs/{run_id}/layers/hazard.png", tags=["Results"])
def get_hazard_png(run_id: str):
    """Returns georeferenced PNG for hydraulic hazard (depth x velocity)."""
    summary = run_store.get_summary(run_id)
    domain_id = summary.get("domain_id")
    grid = DemGrid.load(_domain_dir(domain_id))

    png_path = run_store.run_path(run_id) / "layers" / "hazard.png"
    if not png_path.exists():
        d_arr = np.load(run_store.run_path(run_id) / "max_depth.npy")
        v_arr = np.load(run_store.run_path(run_id) / "max_speed.npy")
        rgba = LayerRenderer.hazard_to_rgba(d_arr, v_arr)
        LayerRenderer.export_layer_png(rgba, grid, png_path)

    return FileResponse(png_path, media_type="image/png")


@app.get("/api/runs/{run_id}/vectors/{frame_index}", tags=["Results"])
def get_velocity_vectors(run_id: str, frame_index: int):
    """Returns flow velocity vectors as GeoJSON points with speed and direction angles."""
    summary = run_store.get_summary(run_id)
    grid = DemGrid.load(_domain_dir(summary.get("domain_id")))
    frame_data = run_store.load_frame(run_id, frame_index)

    return LayerRenderer.extract_flow_vectors(
        depth=frame_data["depth"],
        u=frame_data["u"],
        v=frame_data["v"],
        dem_grid=grid,
        step=8
    )


# ---------------------------------------------------------------------------
# API Endpoints: GIS Hazard & Point Analytics
# ---------------------------------------------------------------------------
@app.get("/api/runs/{run_id}/point", tags=["Analytics"])
def query_point_hazard(
    run_id: str,
    lon: float = Query(..., ge=-180, le=180),
    lat: float = Query(..., ge=-90, le=90)
):
    """
    Inspects elevation, flood depth, arrival time, velocity, and time-series hydrograph
    at clicked coordinates.
    """
    summary = run_store.get_summary(run_id)
    grid = DemGrid.load(_domain_dir(summary.get("domain_id")))
    analytics = FloodAnalytics(grid, run_store, run_id)
    return analytics.inspect_point(lon, lat)


@app.get("/api/runs/{run_id}/assess/roads", tags=["Analytics"])
def assess_roads(run_id: str):
    """Evaluates road severance by overlaying maximum flood depth on OSM roads."""
    summary = run_store.get_summary(run_id)
    domain_id = summary.get("domain_id")
    grid = DemGrid.load(_domain_dir(domain_id))

    roads_geojson = fetch_osm_layer(
        "roads", grid.bbox_lonlat, settings.cache_dir / "osm",
        settings.overpass_url, settings.user_agent
    )
    analytics = FloodAnalytics(grid, run_store, run_id)
    return analytics.assess_osm_roads(roads_geojson)


@app.get("/api/runs/{run_id}/assess/buildings", tags=["Analytics"])
def assess_buildings(run_id: str):
    """Evaluates threatened buildings by overlaying maximum flood depth on OSM buildings."""
    summary = run_store.get_summary(run_id)
    domain_id = summary.get("domain_id")
    grid = DemGrid.load(_domain_dir(domain_id))

    bldgs_geojson = fetch_osm_layer(
        "buildings", grid.bbox_lonlat, settings.cache_dir / "osm",
        settings.overpass_url, settings.user_agent
    )
    analytics = FloodAnalytics(grid, run_store, run_id)
    return analytics.assess_osm_buildings(bldgs_geojson)


@app.get("/api/runs/compare", tags=["Analytics"])
def compare_runs(run_a: str = Query(...), run_b: str = Query(...)):
    """Compares inundated area and peak depths between two scenario runs."""
    return FloodAnalytics.compare_scenarios(run_store, run_a, run_b)


@app.get("/api/runs/{run_id}/download/{file_name}", tags=["Exports"])
def download_run_file(run_id: str, file_name: str):
    """Downloads GeoTIFF raster exports or JSON summaries."""
    p = run_store.run_path(run_id) / file_name
    if not p.exists():
        raise HTTPException(404, f"File '{file_name}' not found for run {run_id}.")
    return FileResponse(p, filename=file_name)


# ---------------------------------------------------------------------------
# Serve 3D Web Application Front-End (Static Files)
# ---------------------------------------------------------------------------
WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

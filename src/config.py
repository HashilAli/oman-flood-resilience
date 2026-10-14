"""
Settings loader.

Reads ``config/settings.toml`` if present, otherwise ``config/settings.example.toml``.
Environment variables override selected keys. Secrets are only read from the
environment and are never stored in the repository.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    default_resolution_m: float = 30.0
    max_cells: int = 250_000
    default_engine: str = "local_inertial"
    workers: int = 1
    copernicus_dem_base_url: str = "https://copernicus-dem-30m.s3.amazonaws.com"
    overpass_url: str = "https://overpass-api.de/api/interpreter"
    nominatim_url: str = "https://nominatim.openstreetmap.org/search"
    user_agent: str = "AI4SGs-FloodSim/2.0 (research prototype)"
    ras_exe: str = ""
    # Secret — environment only (optional; enables Cesium World Terrain in the viewer)
    cesium_ion_token: Optional[str] = field(default=None, repr=False)

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def domains_dir(self) -> Path:
        return self.data_dir / "domains"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"


def _load_toml() -> dict:
    for name in ("settings.toml", "settings.example.toml"):
        path = CONFIG_DIR / name
        if path.exists():
            with open(path, "rb") as fh:
                return tomllib.load(fh)
    return {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    cfg = _load_toml()
    paths = cfg.get("paths", {})
    dom = cfg.get("domain", {})
    eng = cfg.get("engine", {})
    svc = cfg.get("services", {})
    ras = cfg.get("hecras", {})

    data_dir = Path(os.environ.get("FLOODSIM_DATA_DIR", paths.get("data_dir", "data")))
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    s = Settings(
        data_dir=data_dir,
        default_resolution_m=float(dom.get("default_resolution_m", 30.0)),
        max_cells=int(os.environ.get("FLOODSIM_MAX_CELLS", dom.get("max_cells", 250_000))),
        default_engine=str(eng.get("default", "local_inertial")),
        workers=int(os.environ.get("FLOODSIM_WORKERS", eng.get("workers", 1))),
        copernicus_dem_base_url=svc.get("copernicus_dem_base_url", Settings.copernicus_dem_base_url),
        overpass_url=svc.get("overpass_url", Settings.overpass_url),
        nominatim_url=svc.get("nominatim_url", Settings.nominatim_url),
        user_agent=svc.get("user_agent", Settings.user_agent),
        ras_exe=os.environ.get("HECRAS_EXE", ras.get("ras_exe", "")),
        cesium_ion_token=os.environ.get("CESIUM_ION_TOKEN") or None,
    )
    for d in (s.cache_dir, s.domains_dir, s.runs_dir):
        d.mkdir(parents=True, exist_ok=True)
    return s

"""
Metadata catalogue for geospatial data sources.

Every layer shown in the viewer references one of these entries so the user
can inspect provenance, resolution, accuracy, date and licence. Only facts
documented by the providers are listed; unknown values are stated as such.
"""
from __future__ import annotations

from typing import Any, Dict

SOURCES: Dict[str, Dict[str, Any]] = {
    "copernicus_glo30": {
        "name": "Copernicus DEM GLO-30",
        "provider": "ESA / European Union (Copernicus), produced by Airbus from TanDEM-X",
        "access": "AWS Open Data: s3://copernicus-dem-30m (Cloud Optimized GeoTIFF, no account needed)",
        "url": "https://registry.opendata.aws/copernicus-dem/",
        "type": "DSM (surface model: includes buildings and vegetation canopy)",
        "horizontal_resolution": "1 arc-second (~30 m) below 50° latitude",
        "vertical_accuracy": "Specification: absolute vertical accuracy < 4 m (LE90) — Copernicus DEM Product Handbook",
        "horizontal_crs": "EPSG:4326 (WGS84)",
        "vertical_datum": "EGM2008 geoid heights",
        "acquisition_period": "TanDEM-X acquisitions 2011–2015",
        "license": "Free for any use with attribution: © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved",
        "caveats": [
            "DSM: roofs and trees raise the surface; narrow wadi channels (< ~60 m) are smoothed at 30 m.",
            "Not a substitute for LiDAR/survey-grade terrain in urban hydraulics.",
        ],
    },
    "osm": {
        "name": "OpenStreetMap (roads, buildings, waterways)",
        "provider": "OpenStreetMap contributors",
        "access": "Overpass API (public instance, rate-limited)",
        "url": "https://www.openstreetmap.org/copyright",
        "type": "Volunteered vector data",
        "horizontal_resolution": "Variable (digitised features)",
        "vertical_accuracy": "n/a",
        "horizontal_crs": "EPSG:4326",
        "acquisition_period": "Live database; completeness varies by area and is NOT guaranteed",
        "license": "ODbL 1.0 — © OpenStreetMap contributors",
        "caveats": [
            "Building coverage in Oman is incomplete in many areas; absence of a building in OSM does not mean absence on the ground.",
            "OSM 'waterway' lines are mapped channels, not computed drainage.",
        ],
    },
    "user_rainfall": {
        "name": "User-defined rainfall scenario",
        "provider": "User input",
        "access": "Scenario panel / API",
        "type": "Design or hypothetical hyetograph, spatially uniform",
        "horizontal_resolution": "Uniform over the study area",
        "vertical_accuracy": "n/a",
        "acquisition_period": "n/a",
        "license": "n/a",
        "caveats": [
            "No observed or forecast rainfall is ingested yet. Results are scenario outcomes, not forecasts.",
        ],
    },
    "local_inertial_engine": {
        "name": "Local inertial 2D shallow-water engine (built-in)",
        "provider": "This project (src/hydraulics/local_inertial.py)",
        "type": "Numerical model output",
        "reference": "Bates, Horritt & Fewtrell (2010), J. Hydrology 387:33-45",
        "caveats": [
            "Uncalibrated. Not validated against observed floods in Oman.",
            "Neglects convective acceleration; less reliable for supercritical flow in steep gorges.",
        ],
    },
}


def get_sources() -> Dict[str, Dict[str, Any]]:
    return SOURCES

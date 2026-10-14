"""
OpenStreetMap vector layers (roads, buildings, waterways) via the Overpass API.

Data: © OpenStreetMap contributors, ODbL 1.0. Completeness in Oman varies and
is not guaranteed — the API response metadata says so explicitly.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import requests

from .dem import validate_bbox

LAYER_QUERIES = {
    "roads": 'way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|track)$"]',
    "buildings": 'way["building"]',
    "waterways": 'way["waterway"~"^(river|stream|wadi|canal|drain|ditch)$"]',
}
# Hard cap to keep browser payloads bounded
MAX_FEATURES = {"roads": 20000, "buildings": 40000, "waterways": 10000}


class OsmError(RuntimeError):
    pass


def _overpass_query(layer: str, bbox) -> str:
    w, s, e, n = bbox
    return f"[out:json][timeout:90];({LAYER_QUERIES[layer]}({s},{w},{n},{e}););out tags geom;"


def _to_feature(el: Dict[str, Any], layer: str) -> Dict[str, Any] | None:
    geom = el.get("geometry")
    if not geom or len(geom) < 2:
        return None
    coords = [[round(p["lon"], 7), round(p["lat"], 7)] for p in geom]
    tags = el.get("tags", {})
    if layer == "buildings":
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        if len(coords) < 4:
            return None
        geometry = {"type": "Polygon", "coordinates": [coords]}
    else:
        geometry = {"type": "LineString", "coordinates": coords}
    props = {
        "osm_id": el.get("id"),
        "name": tags.get("name:ar") or tags.get("name") or tags.get("name:en") or "",
        "kind": tags.get("highway") or tags.get("waterway") or tags.get("building") or "",
    }
    if layer == "buildings":
        props["levels"] = tags.get("building:levels")
        props["amenity"] = tags.get("amenity")
    return {"type": "Feature", "geometry": geometry, "properties": props}


def fetch_osm_layer(layer: str, bbox: Sequence[float], cache_dir: Path, overpass_url: str,
                    user_agent: str, timeout_s: int = 120) -> Dict[str, Any]:
    """Return a GeoJSON FeatureCollection (EPSG:4326) for ``layer``; cached on disk."""
    if layer not in LAYER_QUERIES:
        raise OsmError(f"Unknown layer '{layer}'. Choose from {list(LAYER_QUERIES)}")
    bbox = validate_bbox(bbox)
    key = hashlib.sha1((layer + ",".join(f"{v:.5f}" for v in bbox)).encode()).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"osm_{layer}_{key}.geojson"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    last_err = None
    for attempt in range(3):
        try:
            resp = requests.post(overpass_url, data={"data": _overpass_query(layer, bbox)},
                                 headers={"User-Agent": user_agent}, timeout=timeout_s)
            if resp.status_code in (429, 504):
                last_err = f"Overpass HTTP {resp.status_code}"
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
            elements = resp.json().get("elements", [])
            break
        except (requests.RequestException, ValueError) as exc:
            last_err = str(exc)
            time.sleep(3 * (attempt + 1))
    else:
        raise OsmError(f"Overpass request failed for {layer}: {last_err}")

    feats: List[Dict[str, Any]] = []
    for el in elements:
        f = _to_feature(el, layer)
        if f:
            feats.append(f)
    truncated = len(feats) > MAX_FEATURES[layer]
    feats = feats[: MAX_FEATURES[layer]]
    fc = {
        "type": "FeatureCollection",
        "features": feats,
        "metadata": {
            "layer": layer,
            "source_id": "osm",
            "bbox": list(bbox),
            "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "feature_count": len(feats),
            "truncated": truncated,
            "attribution": "© OpenStreetMap contributors (ODbL)",
            "completeness_note": "OSM completeness varies; missing features are not evidence of absence.",
        },
    }
    path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    return fc

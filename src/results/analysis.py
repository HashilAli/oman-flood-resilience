"""
Hydrological Hazard & Infrastructure Vulnerability Analytics.

Performs GIS overlay analysis between hydrodynamic flood outputs (depth, velocity, arrival time)
and real OpenStreetMap spatial features (roads, buildings, critical infrastructure).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from shapely.geometry import LineString, Point, Polygon, shape

from ..geodata.dem import DemGrid
from .store import RunStore


class FloodAnalytics:
    """
    Comprehensive flood risk, infrastructure impact, and point query engine.
    """

    def __init__(self, dem_grid: DemGrid, run_store: RunStore, run_id: str):
        self.dem_grid = dem_grid
        self.run_store = run_store
        self.run_id = run_id
        self.summary = run_store.get_summary(run_id)

        run_dir = run_store.run_path(run_id)
        self.max_depth = np.load(run_dir / "max_depth.npy")
        self.max_speed = np.load(run_dir / "max_speed.npy")
        self.arrival_time = np.load(run_dir / "arrival_time_hr.npy")

    def inspect_point(self, lon: float, lat: float) -> Dict[str, Any]:
        """
        Samples elevation, peak depth, velocity, and time-series hydrograph at (lon, lat).
        """
        r, c = self.dem_grid.lonlat_to_rowcol(lon, lat)
        if not self.dem_grid.contains(r, c):
            return {"error": "Point outside study area bounds."}

        elevation = float(self.dem_grid.elevation[r, c])
        peak_d = float(self.max_depth[r, c])
        peak_v = float(self.max_speed[r, c])
        arr_t = float(self.arrival_time[r, c])

        # Extract depth time-series across all saved snapshots
        hydrograph = []
        n_frames = self.run_store.get_frame_count(self.run_id)
        snapshots = self.summary.get("snapshots", [])

        for idx in range(min(n_frames, len(snapshots))):
            frame = self.run_store.load_frame(self.run_id, idx)
            d_val = float(frame["depth"][r, c])
            t_hr = snapshots[idx]["time_hours"]
            hydrograph.append({
                "time_hours": t_hr,
                "depth_m": round(d_val, 3)
            })

        # Calculate hazard rating
        hazard_level = "Dry / Safe"
        if peak_d >= 1.5 or (peak_d * peak_v >= 1.2):
            hazard_level = "Extreme Danger"
        elif peak_d >= 0.8 or (peak_d * peak_v >= 0.6):
            hazard_level = "High Hazard"
        elif peak_d >= 0.3:
            hazard_level = "Moderate Hazard"
        elif peak_d >= 0.05:
            hazard_level = "Low Warning"

        return {
            "coordinates": [round(lon, 6), round(lat, 6)],
            "grid_row": r,
            "grid_col": c,
            "elevation_m": round(elevation, 2),
            "peak_depth_m": round(peak_d, 2),
            "peak_speed_m_s": round(peak_v, 2),
            "arrival_time_hours": round(arr_t, 2) if arr_t >= 0 else None,
            "hazard_level": hazard_level,
            "hydrograph": hydrograph
        }

    def assess_osm_roads(
        self,
        roads_geojson: Dict[str, Any],
        light_vehicle_cutoff_m: float = 0.30,
        heavy_vehicle_cutoff_m: float = 0.60
    ) -> Dict[str, Any]:
        """
        Intersects maximum flood depth with real OSM road vectors.
        """
        features = roads_geojson.get("features", [])
        assessed_roads = []

        total_length_km = 0.0
        severed_length_km = 0.0
        severed_count = 0

        for f in features:
            geom = f.get("geometry", {})
            props = f.get("properties", {})
            coords = geom.get("coordinates", [])

            if len(coords) < 2:
                continue

            # Sample depths along coordinate points
            depths = []
            for pt in coords:
                lon, lat = pt[0], pt[1]
                r, c = self.dem_grid.lonlat_to_rowcol(lon, lat)
                if self.dem_grid.contains(r, c):
                    depths.append(float(self.max_depth[r, c]))

            if not depths:
                continue

            max_d = max(depths)
            avg_d = sum(depths) / len(depths)

            # Compute approximate length in km
            length_km = 0.0
            for i in range(len(coords) - 1):
                p1, p2 = coords[i], coords[i + 1]
                dx = (p2[0] - p1[0]) * 111.32 * math.cos(math.radians((p1[1] + p2[1]) / 2.0))
                dy = (p2[1] - p1[1]) * 110.57
                length_km += math.sqrt(dx**2 + dy**2)

            total_length_km += length_km

            is_severed = max_d >= heavy_vehicle_cutoff_m
            is_impassable_cars = max_d >= light_vehicle_cutoff_m

            if is_severed:
                status = "Severed (مقطوع كلياً)"
                status_color = "#ef4444"
                severed_length_km += length_km
                severed_count += 1
            elif is_impassable_cars:
                status = "Impassable for Cars (خطر للمركبات الخفيفة)"
                status_color = "#f59e0b"
                severed_length_km += (length_km * 0.5)
            else:
                status = "Passable (سالك)"
                status_color = "#10b981"

            assessed_roads.append({
                "osm_id": props.get("osm_id"),
                "name": props.get("name") or "طريق غير مسمى",
                "kind": props.get("kind", "road"),
                "length_km": round(length_km, 2),
                "max_depth_m": round(max_d, 2),
                "avg_depth_m": round(avg_d, 2),
                "status": status,
                "status_color": status_color,
                "is_severed": is_severed
            })

        severance_rate_pct = (severed_length_km / total_length_km * 100.0) if total_length_km > 0 else 0.0

        return {
            "total_roads_assessed": len(assessed_roads),
            "total_length_km": round(total_length_km, 2),
            "severed_length_km": round(severed_length_km, 2),
            "severance_rate_pct": round(severance_rate_pct, 1),
            "severed_count": severed_count,
            "roads": sorted(assessed_roads, key=lambda r: -r["max_depth_m"])[:200]
        }

    def assess_osm_buildings(
        self,
        buildings_geojson: Dict[str, Any],
        flood_threshold_m: float = 0.15
    ) -> Dict[str, Any]:
        """
        Intersects maximum flood depth with OSM building footprints.
        """
        features = buildings_geojson.get("features", [])
        total_buildings = len(features)
        flooded_buildings = []

        for f in features:
            geom = f.get("geometry", {})
            props = f.get("properties", {})
            coords = geom.get("coordinates", [])

            if not coords or len(coords[0]) < 3:
                continue

            # Centroid approximation
            ring = coords[0]
            lons = [p[0] for p in ring]
            lats = [p[1] for p in ring]
            c_lon = sum(lons) / len(lons)
            c_lat = sum(lats) / len(lats)

            r, c = self.dem_grid.lonlat_to_rowcol(c_lon, c_lat)
            if self.dem_grid.contains(r, c):
                d = float(self.max_depth[r, c])
                if d >= flood_threshold_m:
                    elev = float(self.dem_grid.elevation[r, c])
                    flooded_buildings.append({
                        "osm_id": props.get("osm_id"),
                        "name": props.get("name") or "مبنى سكني/تجاري",
                        "amenity": props.get("amenity"),
                        "coordinates": [round(c_lon, 6), round(c_lat, 6)],
                        "depth_m": round(d, 2),
                        "elevation_m": round(elev, 1),
                        "risk_level": "Severe" if d >= 0.8 else ("Moderate" if d >= 0.3 else "Low")
                    })

        return {
            "total_buildings_in_domain": total_buildings,
            "flooded_buildings_count": len(flooded_buildings),
            "flooded_pct": round((len(flooded_buildings) / total_buildings * 100.0), 1) if total_buildings > 0 else 0.0,
            "sample_flooded_buildings": sorted(flooded_buildings, key=lambda b: -b["depth_m"])[:100]
        }

    @staticmethod
    def compare_scenarios(
        run_store: RunStore,
        run_id_a: str,
        run_id_b: str
    ) -> Dict[str, Any]:
        """
        Computes comparative metrics between two scenario runs.
        """
        sum_a = run_store.get_summary(run_id_a)
        sum_b = run_store.get_summary(run_id_b)

        path_a = run_store.run_path(run_id_a)
        path_b = run_store.run_path(run_id_b)

        depth_a = np.load(path_a / "max_depth.npy")
        depth_b = np.load(path_b / "max_depth.npy")

        # Compare inundated cell counts (>= 0.05m)
        wet_a = depth_a >= 0.05
        wet_b = depth_b >= 0.05

        cell_area_km2 = (sum_a.get("cell_size_m", 30.0) ** 2) / 1e6
        area_a = float(np.sum(wet_a) * cell_area_km2)
        area_b = float(np.sum(wet_b) * cell_area_km2)

        depth_diff = depth_b - depth_a

        return {
            "run_a": {"id": run_id_a, "peak_depth_m": sum_a.get("peak_depth_overall_m", 0), "flooded_area_km2": round(area_a, 3)},
            "run_b": {"id": run_id_b, "peak_depth_m": sum_b.get("peak_depth_overall_m", 0), "flooded_area_km2": round(area_b, 3)},
            "delta_area_km2": round(area_b - area_a, 3),
            "delta_peak_depth_m": round(sum_b.get("peak_depth_overall_m", 0) - sum_a.get("peak_depth_overall_m", 0), 2),
            "newly_flooded_area_km2": round(float(np.sum(wet_b & ~wet_a) * cell_area_km2), 3),
            "receded_area_km2": round(float(np.sum(wet_a & ~wet_b) * cell_area_km2), 3)
        }

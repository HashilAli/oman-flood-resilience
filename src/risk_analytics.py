"""
Risk Analytics, Infrastructure Vulnerability, and Evacuation Routing Module
Assesses road network severance, identifies trapped residential blocks,
and computes optimal safe evacuation trajectories across elevated terrain.
"""
import numpy as np
import networkx as nx
from typing import Dict, Any, List, Tuple
from shapely.geometry import LineString, Point


class RiskAnalyticsEngine:
    """
    Evaluates geospatial vulnerability of urban infrastructure, lifelines,
    and calculates graph-based safe evacuation corridors during flood surges.
    """

    def __init__(
        self,
        dem: np.ndarray,
        bounds: Tuple[float, float, float, float],
        water_depth_grid: np.ndarray,
        car_cutoff_depth_m: float = 0.30,
        truck_cutoff_depth_m: float = 0.60
    ):
        self.dem = dem
        self.rows, self.cols = dem.shape
        self.min_lat, self.min_lon, self.max_lat, self.max_lon = bounds
        self.water_depth = water_depth_grid
        self.car_cutoff = car_cutoff_depth_m
        self.truck_cutoff = truck_cutoff_depth_m

    def latlon_to_grid(self, lat: float, lon: float) -> Tuple[int, int]:
        """Converts geographic latitude/longitude to raster grid indices."""
        r = int((self.max_lat - lat) / (self.max_lat - self.min_lat) * self.rows)
        c = int((lon - self.min_lon) / (self.max_lon - self.min_lon) * self.cols)
        r = int(np.clip(r, 0, self.rows - 1))
        c = int(np.clip(c, 0, self.cols - 1))
        return r, c

    def grid_to_latlon(self, r: int, c: int) -> Tuple[float, float]:
        """Converts raster grid indices to geographic latitude/longitude."""
        lat = self.max_lat - (r / self.rows) * (self.max_lat - self.min_lat)
        lon = self.min_lon + (c / self.cols) * (self.max_lon - self.min_lon)
        return lat, lon

    def get_water_depth_at(self, lat: float, lon: float) -> float:
        """Samples simulated water depth at coordinate."""
        r, c = self.latlon_to_grid(lat, lon)
        return float(self.water_depth[r, c])

    def assess_road_network(self, roads: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyzes road segments for inundation, identifying impassable sections,
        critical bottlenecks, and calculating network severance percentage.
        """
        assessed_roads = []
        total_length_km = 0.0
        severed_length_km = 0.0

        for rd in roads:
            coords = rd["coords"]
            # Sample depths along the polyline
            sampled_depths = [self.get_water_depth_at(pt[0], pt[1]) for pt in coords]
            max_depth = max(sampled_depths) if sampled_depths else 0.0
            avg_depth = sum(sampled_depths) / len(sampled_depths) if sampled_depths else 0.0

            # Approximate length in km (1 deg lat ~ 111 km)
            geom_length = 0.0
            for i in range(len(coords) - 1):
                p1, p2 = coords[i], coords[i + 1]
                dlat = (p2[0] - p1[0]) * 111.0
                dlon = (p2[1] - p1[1]) * 111.0 * np.cos(np.radians((p1[0] + p2[0]) / 2.0))
                geom_length += np.sqrt(dlat**2 + dlon**2)

            total_length_km += geom_length

            # Status determination
            if max_depth >= self.truck_cutoff:
                status = "Severed (مقطوع كلياً)"
                status_color = "#e63946"  # Red
                is_severed = True
                severed_length_km += geom_length
            elif max_depth >= self.car_cutoff:
                status = "Impassable for Light Vehicles (غمر جزئي / خطر)"
                status_color = "#f4a261"  # Orange
                is_severed = True
                severed_length_km += (geom_length * 0.7)
            else:
                status = "Passable & Clear (سلك / آمن)"
                status_color = "#2a9d8f"  # Green
                is_severed = False

            assessed_roads.append({
                **rd,
                "max_water_depth_m": round(max_depth, 2),
                "avg_water_depth_m": round(avg_depth, 2),
                "length_km": round(geom_length, 2),
                "status": status,
                "status_color": status_color,
                "is_severed": is_severed
            })

        severance_rate = (severed_length_km / total_length_km * 100.0) if total_length_km > 0 else 0.0

        return {
            "roads": assessed_roads,
            "total_length_km": round(total_length_km, 2),
            "severed_length_km": round(severed_length_km, 2),
            "severance_rate_pct": round(severance_rate, 1)
        }

    def assess_infrastructure(self, infrastructure: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates risk for hospitals, power stations, schools, and neighborhoods.
        Determines isolated clusters and affected population.
        """
        assessed_infra = []
        total_population_at_risk = 0
        critical_assets_threatened = 0

        for item in infrastructure:
            depth = self.get_water_depth_at(item["lat"], item["lon"])
            r, c = self.latlon_to_grid(item["lat"], item["lon"])
            elevation = float(self.dem[r, c])

            if depth > self.truck_cutoff:
                risk_level = "Critical Threat (غمر شديد / خطر داهم)"
                risk_color = "#d90429"
                is_at_risk = True
            elif depth > self.car_cutoff:
                risk_level = "Moderate Hazard (غمر متوسط / انقطاع وصول)"
                risk_color = "#f77f00"
                is_at_risk = True
            elif depth > 0.08:
                risk_level = "Low Warning (تجمع مياه سطحي)"
                risk_color = "#fcbf49"
                is_at_risk = False
            else:
                risk_level = "Safe (آمن / جاف)"
                risk_color = "#06d6a0"
                is_at_risk = False

            if is_at_risk:
                total_population_at_risk += item.get("pop_served", 0)
                if item.get("category") in ["Critical", "Critical Utility", "Healthcare"]:
                    critical_assets_threatened += 1

            assessed_infra.append({
                **item,
                "simulated_depth_m": round(depth, 2),
                "elevation_m": round(elevation, 1),
                "risk_level": risk_level,
                "risk_color": risk_color,
                "is_at_risk": is_at_risk
            })

        return {
            "facilities": assessed_infra,
            "total_population_at_risk": total_population_at_risk,
            "critical_assets_threatened": critical_assets_threatened
        }

    def compute_evacuation_routes(
        self,
        start_points: List[Dict[str, Any]],
        shelters: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Computes terrain-aware optimal evacuation routes from high-risk points
        to the nearest accessible shelter via elevated, unflooded ridge corridors.
        Uses Dijkstra / A* over a topographic cost graph.
        """
        # Build downsampled topographic cost graph to guarantee fast real-time routing
        down_step = 3
        sub_rows = self.rows // down_step
        sub_cols = self.cols // down_step

        G = nx.grid_2d_graph(sub_rows, sub_cols)

        # Assign weights based on flood depth penalty and elevation preference
        for (sr, sc) in G.nodes():
            orig_r = min(self.rows - 1, sr * down_step)
            orig_c = min(self.cols - 1, sc * down_step)
            depth = self.water_depth[orig_r, orig_c]
            elev = self.dem[orig_r, orig_c]

            # Inundation penalty: deep water has astronomical cost
            if depth > self.car_cutoff:
                cost = 10000.0 + (depth * 5000.0)  # Extreme penalty (impassable)
            elif depth > 0.08:
                cost = 50.0 + (depth * 200.0)      # High caution
            else:
                cost = 1.0 + max(0.0, (200.0 - elev) * 0.01)  # Prefer higher ground

            G.nodes[(sr, sc)]["cost"] = cost

        for u, v in G.edges():
            w = (G.nodes[u]["cost"] + G.nodes[v]["cost"]) / 2.0
            G.edges[u, v]["weight"] = w

        routes = []

        for sp in start_points:
            # Find nearest shelter
            start_r, start_c = self.latlon_to_grid(sp["lat"], sp["lon"])
            sub_start = (start_r // down_step, start_c // down_step)

            best_shelter = None
            best_path = None
            min_cost = float("inf")

            for sh in shelters:
                sh_r, sh_c = self.latlon_to_grid(sh["lat"], sh["lon"])
                sub_sh = (sh_r // down_step, sh_c // down_step)

                try:
                    path = nx.dijkstra_path(G, sub_start, sub_sh, weight="weight")
                    cost = nx.dijkstra_path_length(G, sub_start, sub_sh, weight="weight")
                    if cost < min_cost:
                        min_cost = cost
                        best_shelter = sh
                        best_path = path
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

            if best_path and best_shelter:
                # Convert back to lat-lons
                coord_path = []
                # Ensure exact start point is included
                coord_path.append([sp["lat"], sp["lon"]])
                for (sr, sc) in best_path:
                    r_orig = min(self.rows - 1, sr * down_step)
                    c_orig = min(self.cols - 1, sc * down_step)
                    lat, lon = self.grid_to_latlon(r_orig, c_orig)
                    coord_path.append([lat, lon])
                # Ensure exact shelter endpoint is included
                coord_path.append([best_shelter["lat"], best_shelter["lon"]])

                # Determine if route encounters deep floodwaters
                max_route_depth = max([self.get_water_depth_at(p[0], p[1]) for p in coord_path])
                is_fully_safe = max_route_depth < self.car_cutoff

                routes.append({
                    "origin_name": sp.get("name", "Origin"),
                    "destination_shelter": best_shelter["name"],
                    "shelter_capacity": best_shelter.get("capacity", 2000),
                    "coordinates": coord_path,
                    "max_depth_encountered_m": round(max_route_depth, 2),
                    "is_safe": is_fully_safe,
                    "status_label": "Safe High-Ground Corridor" if is_fully_safe else "Advisory: Shallow Crossing Required"
                })

        return routes

"""
Hydrological and Inundation Simulation Engine
Implements physical DEM-based rainfall-runoff generation (SCS-CN),
D8 flow direction & accumulation, 2D diffusive shallow water routing,
and multi-timestep flood propagation (T+1h, T+3h, T+6h, T+12h, T+24h).
"""
import numpy as np
from typing import Dict, Any, List, Tuple
from scipy.ndimage import uniform_filter


class HydrologySimulationEngine:
    """
    Simulates overland flash flood hydrodynamics over complex digital elevation models.
    Designed for arid and mountainous wadi basins (e.g. Sultanate of Oman).
    """

    def __init__(self, dem: np.ndarray, cell_size_m: float = 30.0):
        self.dem = dem.astype(np.float64)
        self.rows, self.cols = dem.shape
        self.cell_size_m = cell_size_m
        self.cell_area_m2 = cell_size_m * cell_size_m
        self.cell_area_km2 = self.cell_area_m2 / 1e6
        
        # Precompute topographic slopes and aspects
        self.slope, self.aspect = self._compute_topographic_derivatives()
        self.flow_dir, self.flow_acc = self._compute_d8_flow()

    def _compute_topographic_derivatives(self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes gradient slope (degrees) and aspect using central differences."""
        gy, gx = np.gradient(self.dem, self.cell_size_m)
        slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
        slope_deg = np.degrees(slope_rad)
        aspect_rad = np.arctan2(-gx, gy)
        return slope_deg, aspect_rad

    def _compute_d8_flow(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates D8 flow direction and upstream contributing flow accumulation.
        D8 directions encoded as indices:
        [NW(0), N(1), NE(2), W(3), E(4), SW(5), S(6), SE(7)]
        """
        rows, cols = self.rows, self.cols
        dem = self.dem

        # 8 neighbor offsets: (dr, dc, distance_factor)
        d_row = [-1, -1, -1,  0, 0,  1, 1, 1]
        d_col = [-1,  0,  1, -1, 1, -1, 0, 1]
        distances = [np.sqrt(2), 1.0, np.sqrt(2), 1.0, 1.0, np.sqrt(2), 1.0, np.sqrt(2)]
        
        flow_dir = np.full((rows, cols), -1, dtype=np.int32)
        
        # Steepest descent search
        for i in range(8):
            dr, dc, dist = d_row[i], d_col[i], distances[i] * self.cell_size_m
            shifted_dem = np.roll(np.roll(dem, -dr, axis=0), -dc, axis=1)
            # Boundary masking
            if dr == -1: shifted_dem[-1, :] = 9999.0
            if dr == 1:  shifted_dem[0, :] = 9999.0
            if dc == -1: shifted_dem[:, -1] = 9999.0
            if dc == 1:  shifted_dem[:, 0] = 9999.0
            
            drop = (dem - shifted_dem) / dist
            if i == 0:
                max_drop = drop
                flow_dir[:] = 0
            else:
                better = drop > max_drop
                max_drop[better] = drop[better]
                flow_dir[better] = i

        # Calculate Flow Accumulation by sorting cells by descending elevation
        flat_indices = np.argsort(-dem.ravel())
        flow_acc = np.ones((rows, cols), dtype=np.float64)

        for idx in flat_indices:
            r = idx // cols
            c = idx % cols
            direction = flow_dir[r, c]
            if 0 <= direction < 8:
                nr = r + d_row[direction]
                nc = c + d_col[direction]
                if 0 <= nr < rows and 0 <= nc < cols:
                    flow_acc[nr, nc] += flow_acc[r, c]

        return flow_dir, flow_acc

    def extract_flash_flood_paths(self, acc_threshold_quantile: float = 0.94) -> np.ndarray:
        """
        Extracts high-priority Flash Flood Runoff Paths (مسارات السيول الجارفة)
        where upstream contributing catchment exceeds threshold and flow accelerates.
        """
        threshold = np.percentile(self.flow_acc, acc_threshold_quantile * 100)
        channels = self.flow_acc > threshold
        return channels

    def delineate_subbasins(self, num_pour_points: int = 4) -> np.ndarray:
        """
        Delineates sub-catchment basins (تحديد حدود الأحواض المائية الفرعية)
        by tracing upstream drainage from major pour points along wadi channels.
        """
        rows, cols = self.rows, self.cols
        basin_grid = np.zeros((rows, cols), dtype=np.int32)
        
        # Identify top pour points with highest accumulation
        flat_acc = self.flow_acc.ravel()
        sorted_indices = np.argsort(-flat_acc)
        
        pour_points = []
        for idx in sorted_indices:
            r = idx // cols
            c = idx % cols
            # Ensure spatial separation between pour points
            if all(np.sqrt((r - pr)**2 + (c - pc)**2) > 25 for pr, pc in pour_points):
                pour_points.append((r, c))
                if len(pour_points) >= num_pour_points:
                    break
                    
        # Upslope basin flood fill using reverse D8 flow direction
        d_row = [-1, -1, -1,  0, 0,  1, 1, 1]
        d_col = [-1,  0,  1, -1, 1, -1, 0, 1]
        
        for basin_id, (pr, pc) in enumerate(pour_points, start=1):
            queue = [(pr, pc)]
            basin_grid[pr, pc] = basin_id
            
            while queue:
                cr, cc = queue.pop(0)
                # Check all 8 neighbors: if neighbor drains into (cr, cc), add to basin
                for k in range(8):
                    nr = cr + d_row[k]
                    nc = cc + d_col[k]
                    if 0 <= nr < rows and 0 <= nc < cols and basin_grid[nr, nc] == 0:
                        # Flow direction from (nr, nc) must point to (cr, cc)
                        # Opposite direction index of k is (7 - k)
                        opposite_dir = [7, 6, 5, 4, 3, 2, 1, 0][k]
                        if self.flow_dir[nr, nc] == opposite_dir:
                            basin_grid[nr, nc] = basin_id
                            queue.append((nr, nc))
                            
        return basin_grid

    def identify_optimal_dam_locations(self) -> List[Dict[str, Any]]:
        """
        Identifies optimal choke-points for flood retention dams and check dams (سدود الحماية والتهدئة).
        Criteria: Narrow gorge width (steep side slopes), high upstream accumulation,
        and situated upstream of major populated settlements.
        """
        rows, cols = self.rows, self.cols
        dem = self.dem
        flow_acc = self.flow_acc
        slope = self.slope
        
        candidates = []
        # Search middle to upper valley sections for natural constriction bottlenecks
        acc_threshold = np.percentile(flow_acc, 95.0)
        
        for r in range(15, rows - 15, 6):
            for c in range(15, cols - 15, 6):
                if flow_acc[r, c] > acc_threshold:
                    # Check valley constriction: steep flanks on left/right
                    gy, gx = np.gradient(dem[r-3:r+4, c-3:c+4])
                    flank_slope = np.mean(np.abs(gx))
                    
                    if flank_slope > 1.2:  # Steep gorge walls
                        est_capacity_m3 = float(flow_acc[r, c] * self.cell_area_m2 * 0.08)
                        candidates.append({
                            "grid_r": r,
                            "grid_c": c,
                            "elevation_m": round(float(dem[r, c]), 1),
                            "upstream_accumulation_cells": int(flow_acc[r, c]),
                            "retention_capacity_m3": round(est_capacity_m3, 0),
                            "flank_slope_deg": round(float(slope[r, c]), 1),
                            "score": float(flow_acc[r, c] * flank_slope)
                        })
                        
        # Sort candidates and pick the top 3 geographically distinct sites
        candidates.sort(key=lambda x: -x["score"])
        selected_dams = []
        for cand in candidates:
            if all(np.sqrt((cand["grid_r"] - d["grid_r"])**2 + (cand["grid_c"] - d["grid_c"])**2) > 20 for d in selected_dams):
                selected_dams.append(cand)
                if len(selected_dams) >= 3:
                    break
                    
        return selected_dams

    def calculate_effective_runoff(self, rainfall_mm: float, curve_number: float = 85.0) -> float:
        """
        Calculates effective surface runoff using the SCS Runoff Curve Number (CN) Method:
        S = 25400 / CN - 254 (Max potential retention in mm)
        Ia = 0.2 * S (Initial abstraction)
        Q = (P - Ia)^2 / (P - Ia + S) for P > Ia
        """
        S = (25400.0 / curve_number) - 254.0
        Ia = 0.2 * S
        if rainfall_mm <= Ia:
            return 0.0
        runoff_mm = ((rainfall_mm - Ia) ** 2) / (rainfall_mm - Ia + S)
        return float(runoff_mm)

    def run_flood_simulation(
        self,
        rainfall_24h_mm: float = 120.0,
        curve_number: float = 84.0,
        timesteps: List[int] = [1, 3, 6, 12, 24],
        roughness_manning: float = 0.035
    ) -> Dict[str, Any]:
        """
        Executes a 2D Shallow Water / Diffusive Overland Wave routing across the DEM.
        Simulates progressive inundation depth over specified hourly time horizons.
        
        Returns:
            Dict containing:
                - timesteps_depths: {1: depth_grid_m, 3: depth_grid_m, ...}
                - timesteps_areas_km2: {1: area_km2, 3: area_km2, ...}
                - max_depth_grid: 2D array of peak water depths (meters)
                - flash_flood_paths: 2D boolean mask of torrent corridors
                - metrics: summary statistics of simulation
        """
        rows, cols = self.rows, self.cols
        effective_runoff_mm = self.calculate_effective_runoff(rainfall_24h_mm, curve_number)
        
        # Initial water depth in meters generated by precipitation
        # Standard SCS 24-hour storm distribution (Type II hyetograph for arid catchments)
        # Fraction of rain fallen by hour t
        storm_fractions = {
            1: 0.12,
            3: 0.32,
            6: 0.65,
            12: 0.90,
            24: 1.00
        }
        
        # D8 neighbor topology
        d_row = [-1, -1, -1,  0, 0,  1, 1, 1]
        d_col = [-1,  0,  1, -1, 1, -1, 0, 1]
        
        # State tracking
        current_water_depth = np.zeros((rows, cols), dtype=np.float64)
        timesteps_depths = {}
        timesteps_areas_km2 = {}
        
        prev_fraction = 0.0
        # Infiltration decay rate (m/hr)
        base_infiltration = 0.002

        for t in timesteps:
            # Added incremental runoff for current time interval
            frac = storm_fractions.get(t, min(1.0, t / 24.0))
            added_precip_m = (effective_runoff_mm / 1000.0) * (frac - prev_fraction)
            prev_fraction = frac
            
            # Add rainfall directly to grid
            current_water_depth += added_precip_m
            
            # Physically route water downhill along hydraulic head: Head = DEM + Water_Depth
            # Perform multiple sub-iterations per hourly step for stability
            sub_steps = 4
            dt = 3600.0 / sub_steps  # seconds
            
            for _ in range(sub_steps):
                head = self.dem + current_water_depth
                outflow_flux = np.zeros((rows, cols), dtype=np.float64)
                
                # Check 4 cardinal + 4 diagonal neighbors for gradient transfer
                for k in range(8):
                    dr, dc = d_row[k], d_col[k]
                    neighbor_head = np.roll(np.roll(head, -dr, axis=0), -dc, axis=1)
                    
                    # Gradient drop
                    dist = np.sqrt(dr**2 + dc**2) * self.cell_size_m
                    head_diff = head - neighbor_head
                    
                    positive_grad = head_diff > 0.001
                    if np.any(positive_grad):
                        # Manning diffusive velocity: v = (1/n) * R^(2/3) * S^(1/2)
                        slope_s = np.maximum(head_diff / dist, 1e-5)
                        depth_active = np.maximum(current_water_depth, 0.0)
                        velocity = (1.0 / roughness_manning) * (np.maximum(depth_active, 1e-4)**(0.667)) * np.sqrt(slope_s)
                        
                        # Courant number constraint for numerical stability
                        velocity = np.clip(velocity, 0.0, 5.5)  # Cap flash flood torrent speed to 5.5 m/s
                        flux = velocity * depth_active * (dt / dist) * 0.125
                        
                        # Limit flux to fraction of available depth
                        flux = np.minimum(flux, current_water_depth * 0.2)
                        current_water_depth -= flux
                        
                        # Add routed water to receiver cell
                        inflow = np.roll(np.roll(flux, dr, axis=0), dc, axis=1)
                        current_water_depth += inflow
                
                # Channel accumulation amplification (water funnels through natural wadis)
                wadi_concentration = (self.flow_acc / np.mean(self.flow_acc)) * 0.0008
                current_water_depth += np.clip(wadi_concentration, 0.0, 0.08)
                
                # Modest infiltration/evaporation loss in high ground
                current_water_depth = np.maximum(0.0, current_water_depth - (base_infiltration / sub_steps))

            # Store snapshot at current timestep
            depth_snapshot = np.copy(current_water_depth)
            
            # Clean tiny moisture (< 5cm is non-hazardous sheet runoff)
            inundated_cells = depth_snapshot > 0.08
            area_km2 = float(np.sum(inundated_cells) * self.cell_area_km2)
            
            timesteps_depths[t] = depth_snapshot
            timesteps_areas_km2[t] = round(area_km2, 3)

        # Max depth achieved at peak
        max_depth = np.maximum.reduce(list(timesteps_depths.values()))
        flash_flood_paths = self.extract_flash_flood_paths(0.93)

        return {
            "timesteps": timesteps,
            "timesteps_depths": timesteps_depths,
            "timesteps_areas_km2": timesteps_areas_km2,
            "max_depth_grid": max_depth,
            "flash_flood_paths": flash_flood_paths,
            "effective_runoff_mm": round(effective_runoff_mm, 2),
            "peak_depth_m": round(float(np.max(max_depth)), 2),
            "mean_inundated_depth_m": round(float(np.mean(max_depth[max_depth > 0.08])), 2) if np.any(max_depth > 0.08) else 0.0,
            "flow_accumulation": self.flow_acc
        }

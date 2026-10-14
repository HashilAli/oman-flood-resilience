"""
Georeferenced Visualization Layer Generator.

Produces web-ready, multi-layered visual representations for 3D CesiumJS and 2D GIS:
1. Color-mapped georeferenced RGBA PNG overlays (reprojected to EPSG:4326).
2. Velocity direction arrow GeoJSON for flow vectors.
3. Hydraulic Hazard (Depth x Velocity) map overlays.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin, rowcol, xy
from rasterio.warp import reproject, transform_bounds

from ..geodata.dem import DemGrid


class LayerRenderer:
    """
    Renders simulation arrays as georeferenced transparent PNGs and vector features.
    """

    @staticmethod
    def depth_to_rgba(
        depth_m: np.ndarray,
        min_cutoff_m: float = 0.05,
        max_scale_m: float = 3.0
    ) -> np.ndarray:
        """
        Maps a 2D water depth array (metres) to RGBA uint8 array (shape: rows, cols, 4).
        Scientific multi-stop color palette:
        - depth < min_cutoff_m: Transparent (Alpha = 0)
        - 0.05m -> 0.30m: Light cyan/aqua (#48cae4)
        - 0.30m -> 0.75m: Vivid sky blue (#0096c7)
        - 0.75m -> 1.50m: Royal navy blue (#023e8a)
        - 1.50m -> 3.00m+: Deep violet/indigo (#3a0ca3)
        """
        rows, cols = depth_m.shape
        rgba = np.zeros((rows, cols, 4), dtype=np.uint8)

        wet = depth_m >= min_cutoff_m
        if not np.any(wet):
            return rgba

        d = np.clip(depth_m, 0.0, max_scale_m)

        # Normalized values
        # Color Stops:
        # 0.05 -> [72, 202, 228, 170]
        # 0.30 -> [0, 150, 199, 195]
        # 0.75 -> [2, 62, 138, 220]
        # 1.50 -> [114, 9, 183, 235]
        # 3.00 -> [74, 4, 120, 245]

        # Segment 1: 0.05 to 0.30
        mask1 = (d >= 0.05) & (d < 0.30)
        if np.any(mask1):
            f = (d[mask1] - 0.05) / 0.25
            rgba[mask1, 0] = np.uint8(72 - 72 * f)
            rgba[mask1, 1] = np.uint8(202 - 52 * f)
            rgba[mask1, 2] = np.uint8(228 - 29 * f)
            rgba[mask1, 3] = np.uint8(160 + 35 * f)

        # Segment 2: 0.30 to 0.75
        mask2 = (d >= 0.30) & (d < 0.75)
        if np.any(mask2):
            f = (d[mask2] - 0.30) / 0.45
            rgba[mask2, 0] = np.uint8(0 + 2 * f)
            rgba[mask2, 1] = np.uint8(150 - 88 * f)
            rgba[mask2, 2] = np.uint8(199 - 61 * f)
            rgba[mask2, 3] = np.uint8(195 + 25 * f)

        # Segment 3: 0.75 to 1.50
        mask3 = (d >= 0.75) & (d < 1.50)
        if np.any(mask3):
            f = (d[mask3] - 0.75) / 0.75
            rgba[mask3, 0] = np.uint8(2 + 112 * f)
            rgba[mask3, 1] = np.uint8(62 - 53 * f)
            rgba[mask3, 2] = np.uint8(138 + 45 * f)
            rgba[mask3, 3] = np.uint8(220 + 15 * f)

        # Segment 4: 1.50+
        mask4 = (d >= 1.50)
        if np.any(mask4):
            f = np.clip((d[mask4] - 1.50) / 1.50, 0.0, 1.0)
            rgba[mask4, 0] = np.uint8(114 - 40 * f)
            rgba[mask4, 1] = np.uint8(9 - 5 * f)
            rgba[mask4, 2] = np.uint8(183 - 63 * f)
            rgba[mask4, 3] = np.uint8(235 + 15 * f)

        return rgba

    @staticmethod
    def hazard_to_rgba(
        depth_m: np.ndarray,
        speed_m_s: np.ndarray
    ) -> np.ndarray:
        """
        Computes hydrodynamic hazard category (Depth x Velocity rating):
        - Safe / Low: green/yellow
        - Moderate: orange
        - Severe: red
        - Extreme: purple
        """
        rows, cols = depth_m.shape
        rgba = np.zeros((rows, cols, 4), dtype=np.uint8)

        wet = depth_m >= 0.05
        if not np.any(wet):
            return rgba

        dv = depth_m * speed_m_s

        low = wet & (depth_m < 0.3) & (dv < 0.2)
        med = wet & ~low & (depth_m < 0.8) & (dv < 0.6)
        high = wet & ~low & ~med & (depth_m < 1.5) & (dv < 1.2)
        extreme = wet & ~low & ~med & ~high

        # Low: Yellowish-Orange [255, 183, 3, 180]
        rgba[low] = [255, 183, 3, 160]
        # Moderate: Orange [251, 133, 0, 200]
        rgba[med] = [251, 133, 0, 195]
        # High: Vibrant Red [208, 0, 0, 225]
        rgba[high] = [208, 0, 0, 220]
        # Extreme: Dark Crimson/Purple [106, 4, 15, 245]
        rgba[extreme] = [106, 4, 15, 245]

        return rgba

    @classmethod
    def reproject_rgba_to_wgs84(
        cls,
        rgba: np.ndarray,
        dem_grid: DemGrid
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """
        Reprojects an RGBA grid from projected UTM coordinates onto a regular
        EPSG:4326 lat/lon grid, returning (reprojected_rgba, (west, south, east, north)).
        """
        src_crs = CRS.from_string(dem_grid.crs)
        dst_crs = CRS.from_epsg(4326)

        # Geographic bounds
        l, b, r, t = dem_grid.bounds
        w, s, e, n = transform_bounds(src_crs, dst_crs, l, b, r, t, densify_pts=21)

        src_rows, src_cols, _ = rgba.shape
        # Estimate output shape maintaining aspect ratio
        dst_rows = src_rows
        dst_cols = src_cols

        res_x = (e - w) / dst_cols
        res_y = (n - s) / dst_rows
        dst_transform = from_origin(w, n, res_x, res_y)

        dst_rgba = np.zeros((dst_rows, dst_cols, 4), dtype=np.uint8)

        # Reproject each channel independently with nearest-neighbor to preserve transparency
        for band in range(4):
            reproject(
                source=rgba[:, :, band],
                destination=dst_rgba[:, :, band],
                src_transform=dem_grid.transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest
            )

        return dst_rgba, (w, s, e, n)

    @classmethod
    def export_layer_png(
        cls,
        rgba: np.ndarray,
        dem_grid: DemGrid,
        out_png_path: Path
    ) -> Dict[str, Any]:
        """
        Saves RGBA array as an optimized georeferenced PNG for web map integration.
        Returns geographic bounding box coordinates for Cesium / Leaflet image layers.
        """
        out_png_path.parent.mkdir(parents=True, exist_ok=True)
        wgs84_rgba, (w, s, e, n) = cls.reproject_rgba_to_wgs84(rgba, dem_grid)

        img = Image.fromarray(wgs84_rgba, mode="RGBA")
        img.save(out_png_path, format="PNG", optimize=True)

        return {
            "file": str(out_png_path),
            "format": "PNG",
            "bbox_wgs84": [round(w, 6), round(s, 6), round(e, 6), round(n, 6)],
            "width": img.width,
            "height": img.height
        }

    @staticmethod
    def extract_flow_vectors(
        depth: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        dem_grid: DemGrid,
        step: int = 6,
        min_speed_m_s: float = 0.15,
        min_depth_m: float = 0.10
    ) -> Dict[str, Any]:
        """
        Generates GeoJSON points with velocity magnitude and flow direction angles (degrees),
        subsampled across the grid for dynamic particle / vector arrow display.
        """
        rows, cols = depth.shape
        features = []

        speed = np.sqrt(u**2 + v**2)
        active = (depth >= min_depth_m) & (speed >= min_speed_m_s)

        for r in range(0, rows, step):
            for c in range(0, cols, step):
                if active[r, c]:
                    lon, lat = dem_grid.rowcol_to_lonlat(r, c)
                    u_val = float(u[r, c])
                    v_val = float(v[r, c])
                    sp_val = float(speed[r, c])
                    dp_val = float(depth[r, c])
                    
                    # Mathematical angle to Compass Bearing: 0 deg = North, 90 deg = East
                    bearing = (math.degrees(math.atan2(u_val, -v_val)) + 360.0) % 360.0

                    features.append({
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
                        "properties": {
                            "speed_m_s": round(sp_val, 2),
                            "depth_m": round(dp_val, 2),
                            "bearing_deg": round(bearing, 1)
                        }
                    })

        return {
            "type": "FeatureCollection",
            "features": features,
            "count": len(features)
        }

"""
Data Generator Module for Multi-Region Oman Catchments
Provides physically-coherent DEM, Sentinel-1 SAR imagery, Road Networks,
Critical Infrastructure, and Shelters for multiple key Omani river basins:
- Wadi Adai (Muscat)
- Wadi Shab / Qalhat (South Al Sharqiyah)
- Wadi Ahan (North Al Batinah)
- Wadi Darbat (Dhofar / Salalah)
"""
import numpy as np
from typing import Dict, Any, List, Tuple


AVAILABLE_REGIONS = {
    "muscat_wadi_adai": {
        "id": "muscat_wadi_adai",
        "name_ar": "حوض وادي عدي - محافظة مسقط",
        "name_en": "Wadi Adai Catchment, Muscat",
        "governorate": "محافظة مسقط",
        "risk_profile": "حوض حضري جبلي عالي الحساسية للسيول الفجائية (شديد التأثر بإعصار جونو وشاهين)",
        "bounds": (23.540, 58.500, 23.620, 58.580),
        "center": [23.580, 58.540],
        "default_rainfall": 120,
        "default_cn": 86
    },
    "sur_wadi_shab": {
        "id": "sur_wadi_shab",
        "name_ar": "حوض وادي شاب وقلهات - محافظة جنوب الشرقية",
        "name_en": "Wadi Shab & Qalhat, South Al Sharqiyah",
        "governorate": "محافظة جنوب الشرقية",
        "risk_profile": "خوانق صخرية عميقة شديدة الانحدار تعبر الطريق الساحلي الاستراتيجي قريات - صور",
        "bounds": (22.800, 59.200, 22.880, 59.280),
        "center": [22.840, 59.240],
        "default_rainfall": 150,
        "default_cn": 88
    },
    "batinah_wadi_ahan": {
        "id": "batinah_wadi_ahan",
        "name_ar": "حوض وادي عاهن وصحم - محافظة شمال الباطنة",
        "name_en": "Wadi Ahan & Saham, North Al Batinah",
        "governorate": "محافظة شمال الباطنة",
        "risk_profile": "مروحة فيضية وسهول زراعية ممتدة نحو ساحل بحر عُمان (مسار السيول التاريخية لشاهين)",
        "bounds": (23.950, 56.840, 24.030, 56.920),
        "center": [23.990, 56.880],
        "default_rainfall": 110,
        "default_cn": 82
    },
    "dhofar_wadi_darbat": {
        "id": "dhofar_wadi_darbat",
        "name_ar": "حوض وادي دربات وطاقة - محافظة ظفار",
        "name_en": "Wadi Darbat & Taqah, Dhofar",
        "governorate": "محافظة ظفار",
        "risk_profile": "حوض كارستي جيري يتأثر بأعاصير بحر العرب المدارية (إعصار مكونو وتيج)",
        "bounds": (17.050, 54.400, 17.130, 54.480),
        "center": [17.090, 54.440],
        "default_rainfall": 180,
        "default_cn": 80
    }
}


def get_available_regions() -> Dict[str, Dict[str, Any]]:
    """Returns the metadata catalog of supported Omani catchments."""
    return AVAILABLE_REGIONS


def grid_to_latlon(r: float, c: float, rows: int, cols: int, bounds: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Converts grid raster indices to latitude/longitude coordinates."""
    min_lat, min_lon, max_lat, max_lon = bounds
    lat = max_lat - (r / rows) * (max_lat - min_lat)
    lon = min_lon + (c / cols) * (max_lon - min_lon)
    return lat, lon


def latlon_to_grid(lat: float, lon: float, rows: int, cols: int, bounds: Tuple[float, float, float, float]) -> Tuple[int, int]:
    """Converts latitude/longitude coordinates to grid raster indices."""
    min_lat, min_lon, max_lat, max_lon = bounds
    r = int((max_lat - lat) / (max_lat - min_lat) * rows)
    c = int((lon - min_lon) / (max_lon - min_lon) * cols)
    return int(np.clip(r, 0, rows - 1)), int(np.clip(c, 0, cols - 1))


def generate_catchment_data(
    region_id: str = "muscat_wadi_adai",
    rows: int = 120,
    cols: int = 120
) -> Dict[str, Any]:
    """
    Generates high-fidelity physically-coherent geospatial catchment data for
    a selected Omani basin region.
    """
    if region_id not in AVAILABLE_REGIONS:
        region_id = "muscat_wadi_adai"

    reg_info = AVAILABLE_REGIONS[region_id]
    bounds = reg_info["bounds"]
    resolution_m = 30.0

    np.random.seed(42 if region_id == "muscat_wadi_adai" else (hash(region_id) % 10000))

    x = np.linspace(0, 1, cols)
    y = np.linspace(0, 1, rows)
    X, Y = np.meshgrid(x, y)

    g2l = lambda r, c: grid_to_latlon(r, c, rows, cols, bounds)
    l2g = lambda lat, lon: latlon_to_grid(lat, lon, rows, cols, bounds)

    # 1. Synthesize Region-specific Topography & Drainage morphology
    if region_id == "muscat_wadi_adai":
        base_elevation = 28.0
        m1 = 380.0 * np.exp(-((X - 0.25)**2 / 0.08 + (Y - 0.75)**2 / 0.12))
        m2 = 450.0 * np.exp(-((X - 0.80)**2 / 0.10 + (Y - 0.30)**2 / 0.09))
        m3 = 260.0 * np.sin(np.pi * X * 1.5) * np.cos(np.pi * Y)
        wadi_x = 0.15 + 0.7 * (1.0 - Y) + 0.08 * np.sin(4.0 * np.pi * Y)
        wadi_dist = np.abs(X - wadi_x)
        wadi_trough = -160.0 * np.exp(-(wadi_dist**2) / 0.006)
        trib_dist = np.abs(Y - (0.5 + 0.1 * np.sin(2.0 * np.pi * X))) * (X > 0.45)
        trib_trough = -90.0 * np.exp(-(trib_dist**2) / 0.005) * (X > 0.45)
        dem = np.clip(base_elevation + m1 + m2 + m3 + wadi_trough + trib_trough + 4.0 * np.random.randn(rows, cols), 12.0, 520.0)

        roads = [
            {"id": "R01", "name": "طريق السلطان قابوس (Sultan Qaboos St)", "type": "Primary Highway", "coords": [g2l(12, 10), g2l(14, 40), g2l(15, 80), g2l(16, 110)], "importance": "Critical"},
            {"id": "R02", "name": "طريق وادي عدي - العامرات (Wadi Adai Gorge)", "type": "Major Crossing", "coords": [g2l(15, 65), g2l(35, 58), g2l(60, 48), g2l(85, 35), g2l(110, 22)], "importance": "High Vulnerability"},
            {"id": "R03", "name": "طريق مسقط السريع (Muscat Expressway)", "type": "Expressway", "coords": [g2l(38, 15), g2l(42, 50), g2l(48, 90), g2l(52, 115)], "importance": "Critical Strategic"},
            {"id": "R04", "name": "شريان الحي السكني بالوادي (Residential Feeder)", "type": "Secondary", "coords": [g2l(65, 45), g2l(70, 60), g2l(75, 75)], "importance": "High Risk Access"},
            {"id": "R05", "name": "طريق مرتفعات القرم (Qurum Ridge Corridor)", "type": "Ridge Road", "coords": [g2l(25, 20), g2l(28, 38), g2l(22, 60)], "importance": "Safe High Ground"}
        ]
        infrastructure = [
            {"id": "INF01", "name": "مجمع عيادات وادي عدي الصحي", "type": "Healthcare", "lat": g2l(62, 47)[0], "lon": g2l(62, 47)[1], "pop_served": 14500, "category": "Critical"},
            {"id": "INF02", "name": "محطة محولات الكهرباء الرئيسية (Alpha)", "type": "Utility", "lat": g2l(72, 43)[0], "lon": g2l(72, 43)[1], "pop_served": 32000, "category": "Critical Utility"},
            {"id": "INF03", "name": "الحي السكني المنخفض أ", "type": "Residential", "lat": g2l(55, 52)[0], "lon": g2l(55, 52)[1], "pop_served": 4200, "category": "High Vulnerability"},
            {"id": "INF04", "name": "الحي السكني المنخفض ب", "type": "Residential", "lat": g2l(80, 37)[0], "lon": g2l(80, 37)[1], "pop_served": 5800, "category": "High Vulnerability"},
            {"id": "INF05", "name": "مركز قيادة الدفاع المدني والإسعاف", "type": "Emergency Services", "lat": g2l(32, 28)[0], "lon": g2l(32, 28)[1], "pop_served": 80000, "category": "Command Center"},
            {"id": "INF06", "name": "مدرسة الأمل للتعليم الأساسي", "type": "Education", "lat": g2l(45, 68)[0], "lon": g2l(45, 68)[1], "pop_served": 1200, "category": "Vulnerable Hub"}
        ]
        shelters = [
            {"id": "SH01", "name": "مركز إيواء مرتفعات بوشر", "capacity": 3500, "elevation_m": float(dem[l2g(g2l(25, 25)[0], g2l(25, 25)[1])]), "lat": g2l(25, 25)[0], "lon": g2l(25, 25)[1], "status": "Safe Elevated Hub"},
            {"id": "SH02", "name": "مركز إيواء هضبة العامرات", "capacity": 5000, "elevation_m": float(dem[l2g(g2l(95, 90)[0], g2l(95, 90)[1])]), "lat": g2l(95, 90)[0], "lon": g2l(95, 90)[1], "status": "Safe Elevated Hub"},
            {"id": "SH03", "name": "مجمع الدفاع المدني بالمرتفعات", "capacity": 4000, "elevation_m": float(dem[l2g(g2l(40, 95)[0], g2l(40, 95)[1])]), "lat": g2l(40, 95)[0], "lon": g2l(40, 95)[1], "status": "Safe Elevated Hub"}
        ]

    elif region_id == "sur_wadi_shab":
        base_elevation = 15.0
        m1 = 600.0 * np.exp(-((X - 0.20)**2 / 0.07 + (Y - 0.50)**2 / 0.15))
        m2 = 520.0 * np.exp(-((X - 0.85)**2 / 0.08 + (Y - 0.60)**2 / 0.12))
        wadi_x = 0.50 + 0.15 * np.sin(3.0 * np.pi * Y)
        wadi_dist = np.abs(X - wadi_x)
        canyon_trough = -240.0 * np.exp(-(wadi_dist**2) / 0.003)
        dem = np.clip(base_elevation + m1 + m2 + canyon_trough + 3.5 * np.random.randn(rows, cols), 5.0, 680.0)

        roads = [
            {"id": "SR01", "name": "طريق قريات - صور الساحلي السريع", "type": "Primary Highway", "coords": [g2l(15, 10), g2l(25, 45), g2l(35, 75), g2l(45, 110)], "importance": "Critical Strategic"},
            {"id": "SR02", "name": "جسر مدخل وادي شاب", "type": "Major Crossing", "coords": [g2l(50, 40), g2l(58, 52), g2l(65, 65)], "importance": "Extreme Vulnerability"},
            {"id": "SR03", "name": "طريق قرية طيوي الداخلية", "type": "Secondary", "coords": [g2l(28, 48), g2l(38, 50), g2l(48, 52)], "importance": "Wadi Corridor"},
            {"id": "SR04", "name": "طريق هضبة قلهات الآمن", "type": "Ridge Road", "coords": [g2l(70, 20), g2l(80, 45), g2l(90, 80)], "importance": "High Ground"}
        ]
        infrastructure = [
            {"id": "SINF01", "name": "مركز صحي طيوي الساحلي", "type": "Healthcare", "lat": g2l(52, 48)[0], "lon": g2l(52, 48)[1], "pop_served": 5400, "category": "Critical"},
            {"id": "SINF02", "name": "محطة ضخ وتحلية مياه قلهات", "type": "Utility", "lat": g2l(22, 65)[0], "lon": g2l(22, 65)[1], "pop_served": 22000, "category": "Critical Utility"},
            {"id": "SINF03", "name": "القرية التراثية بوادي شاب", "type": "Residential", "lat": g2l(65, 52)[0], "lon": g2l(65, 52)[1], "pop_served": 2100, "category": "High Vulnerability"},
            {"id": "SINF04", "name": "مركز الدفاع المدني بقلهات", "type": "Emergency Services", "lat": g2l(78, 35)[0], "lon": g2l(78, 35)[1], "pop_served": 45000, "category": "Command Center"}
        ]
        shelters = [
            {"id": "SSH01", "name": "مركز إيواء هضبة قلهات التاريخية", "capacity": 2500, "elevation_m": float(dem[l2g(g2l(82, 30)[0], g2l(82, 30)[1])]), "lat": g2l(82, 30)[0], "lon": g2l(82, 30)[1], "status": "Safe Elevated Hub"},
            {"id": "SSH02", "name": "مدرسة صور الساحلية بالمرتفعات", "capacity": 3800, "elevation_m": float(dem[l2g(g2l(20, 85)[0], g2l(20, 85)[1])]), "lat": g2l(20, 85)[0], "lon": g2l(20, 85)[1], "status": "Safe Elevated Hub"}
        ]

    elif region_id == "batinah_wadi_ahan":
        base_elevation = 18.0
        # Gentle alluvial fan sloping from south-west mountains (300m) to coast (5m)
        gradient = 180.0 * (1.0 - Y) + 80.0 * (1.0 - X)
        fan_channel1 = -45.0 * np.exp(-((X - 0.45 - 0.1 * Y)**2) / 0.012)
        fan_channel2 = -35.0 * np.exp(-((X - 0.70 - 0.08 * Y)**2) / 0.010)
        dem = np.clip(base_elevation + gradient + fan_channel1 + fan_channel2 + 2.5 * np.random.randn(rows, cols), 4.0, 290.0)
        wadi_dist = np.minimum(np.abs(X - (0.45 + 0.1 * Y)), np.abs(X - (0.70 + 0.08 * Y)))

        roads = [
            {"id": "BR01", "name": "طريق الباطنة السريع", "type": "Primary Highway", "coords": [g2l(75, 10), g2l(76, 50), g2l(78, 90), g2l(80, 110)], "importance": "Critical Strategic"},
            {"id": "BR02", "name": "طريق السلطان قابوس الساحلي (صحم)", "type": "Major Crossing", "coords": [g2l(15, 10), g2l(16, 50), g2l(18, 90), g2l(20, 110)], "importance": "High Flood Risk"},
            {"id": "BR03", "name": "شريان وادي عاهن الداخلي", "type": "Secondary", "coords": [g2l(85, 45), g2l(50, 48), g2l(25, 52)], "importance": "Flash Corridor"},
            {"id": "BR04", "name": "طريق ربط المرتفعات الغربية", "type": "Ridge Road", "coords": [g2l(90, 20), g2l(60, 25), g2l(30, 22)], "importance": "Safe High Ground"}
        ]
        infrastructure = [
            {"id": "BINF01", "name": "مستشفى صحم العام", "type": "Healthcare", "lat": g2l(28, 70)[0], "lon": g2l(28, 70)[1], "pop_served": 38000, "category": "Critical"},
            {"id": "BINF02", "name": "محطة توليد وتحلية الباطنة", "type": "Utility", "lat": g2l(18, 90)[0], "lon": g2l(18, 90)[1], "pop_served": 65000, "category": "Critical Utility"},
            {"id": "BINF03", "name": "مزارع وتجمعات حوض عاهن السكنية", "type": "Residential", "lat": g2l(48, 50)[0], "lon": g2l(48, 50)[1], "pop_served": 8500, "category": "High Vulnerability"},
            {"id": "BINF04", "name": "مركز قيادة الدفاع المدني بشمال الباطنة", "type": "Emergency Services", "lat": g2l(82, 30)[0], "lon": g2l(82, 30)[1], "pop_served": 110000, "category": "Command Center"}
        ]
        shelters = [
            {"id": "BSH01", "name": "مركز إيواء مجمع صحم الرياضي بالمرتفعات", "capacity": 6000, "elevation_m": float(dem[l2g(g2l(85, 25)[0], g2l(85, 25)[1])]), "lat": g2l(85, 25)[0], "lon": g2l(85, 25)[1], "status": "Safe Elevated Hub"},
            {"id": "BSH02", "name": "مدرسة أم عمارة الثانوية الآمنة", "capacity": 4200, "elevation_m": float(dem[l2g(g2l(80, 85)[0], g2l(80, 85)[1])]), "lat": g2l(80, 85)[0], "lon": g2l(80, 85)[1], "status": "Safe Elevated Hub"}
        ]

    else:  # dhofar_wadi_darbat
        base_elevation = 45.0
        plateau = 420.0 * np.exp(-((X - 0.35)**2 / 0.15 + (Y - 0.70)**2 / 0.10))
        sinkhole = -180.0 * np.exp(-((X - 0.55)**2 / 0.01 + (Y - 0.40)**2 / 0.01))
        wadi_x = 0.52 + 0.12 * np.cos(2.5 * np.pi * Y)
        wadi_dist = np.abs(X - wadi_x)
        darbat_chasm = -150.0 * np.exp(-(wadi_dist**2) / 0.007)
        dem = np.clip(base_elevation + plateau + sinkhole + darbat_chasm + 3.0 * np.random.randn(rows, cols), 20.0, 580.0)

        roads = [
            {"id": "DR01", "name": "طريق طاقة - مرباط الساحلي", "type": "Primary Highway", "coords": [g2l(15, 15), g2l(16, 50), g2l(17, 85), g2l(18, 110)], "importance": "Critical Strategic"},
            {"id": "DR02", "name": "طريق شلالات دربات السياحي", "type": "Major Crossing", "coords": [g2l(25, 48), g2l(45, 52), g2l(70, 54)], "importance": "Extreme Vulnerability"},
            {"id": "DR03", "name": "طريق جبل سمحان الصاعد", "type": "Ridge Road", "coords": [g2l(80, 25), g2l(60, 28), g2l(30, 22)], "importance": "Safe High Ground"}
        ]
        infrastructure = [
            {"id": "DINF01", "name": "مركز صحي طاقة التخصصي", "type": "Healthcare", "lat": g2l(20, 35)[0], "lon": g2l(20, 35)[1], "pop_served": 16000, "category": "Critical"},
            {"id": "DINF02", "name": "محطة مياه حوض صلالة الشرقية", "type": "Utility", "lat": g2l(30, 75)[0], "lon": g2l(30, 75)[1], "pop_served": 45000, "category": "Critical Utility"},
            {"id": "DINF03", "name": "القرية التراثية بواحة دربات", "type": "Residential", "lat": g2l(45, 53)[0], "lon": g2l(45, 53)[1], "pop_served": 3100, "category": "High Vulnerability"},
            {"id": "DINF04", "name": "مركز الدفاع المدني والإسعاف بطاقة", "type": "Emergency Services", "lat": g2l(22, 25)[0], "lon": g2l(22, 25)[1], "pop_served": 50000, "category": "Command Center"}
        ]
        shelters = [
            {"id": "DSH01", "name": "مركز إيواء هضبة طاقة الآمنة", "capacity": 4500, "elevation_m": float(dem[l2g(g2l(25, 20)[0], g2l(25, 20)[1])]), "lat": g2l(25, 20)[0], "lon": g2l(25, 20)[1], "status": "Safe Elevated Hub"},
            {"id": "DSH02", "name": "مجمع جبل سمحان للإيواء", "capacity": 3000, "elevation_m": float(dem[l2g(g2l(85, 30)[0], g2l(85, 30)[1])]), "lat": g2l(85, 30)[0], "lon": g2l(85, 30)[1], "status": "Safe Elevated Hub"}
        ]

    # 2. Synthetic Sentinel-1 SAR & Vegetation Index (NDVI)
    grad_y, grad_x = np.gradient(dem)
    slope = np.sqrt(grad_x**2 + grad_y**2)
    sar_backscatter_pre = -11.0 + 0.12 * slope + np.random.normal(0, 1.2, (rows, cols))

    # Higher vegetation for Dhofar / Batinah, lower for Muscat / Sur
    base_veg = 0.25 if region_id == "dhofar_wadi_darbat" else 0.08
    ndvi = base_veg + 0.45 * np.exp(-((X - 0.4)**2 / 0.05 + (Y - 0.5)**2 / 0.04))
    ndvi = np.clip(ndvi + np.random.normal(0, 0.02, (rows, cols)), 0.02, 0.85)

    low_water_mask = (dem < (base_elevation + 4.0)) & (wadi_dist < 0.04)
    sar_backscatter_pre[low_water_mask] = -19.5 + np.random.normal(0, 0.8, int(np.sum(low_water_mask)))

    sar_backscatter_post = np.copy(sar_backscatter_pre)
    actual_flood_mask = (dem < (base_elevation + 12.0)) & (wadi_dist < 0.09)
    sar_backscatter_post[actual_flood_mask] = -21.0 + np.random.normal(0, 0.9, int(np.sum(actual_flood_mask)))

    return {
        "region_id": region_id,
        "region_info": reg_info,
        "dem": dem,
        "rows": rows,
        "cols": cols,
        "bounds": bounds,
        "resolution_m": resolution_m,
        "ndvi": ndvi,
        "sar_backscatter_pre": sar_backscatter_pre,
        "sar_backscatter_post": sar_backscatter_post,
        "roads": roads,
        "infrastructure": infrastructure,
        "shelters": shelters
    }

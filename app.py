"""
========================================================================================
AI4SDGs 2026 - Water Mapping for Flood Resilience MVP
Sultanate of Oman | Oman Lens x Star.Vision Aerospace Hackathon Track
========================================================================================
Interactive Geospatial AI & Hydrological Inundation Simulation Platform
Multi-Region Edition (Muscat, Sur/Sharqiyah, Al Batinah, Dhofar)
"""
import streamlit as st
import numpy as np
import time
import json
from streamlit_folium import st_folium

from src.data_generator import (
    generate_catchment_data,
    get_available_regions,
    AVAILABLE_REGIONS
)
from src.sar_processor import Sentinel1SARProcessor
from src.hydrology_engine import HydrologySimulationEngine
from src.risk_analytics import RiskAnalyticsEngine
from src.visualization import (
    create_resilience_map,
    generate_hydrograph_figure,
    create_animated_motion_graph
)

# -----------------------------------------------------------------------------
# Streamlit Page Config & High-Tech Futuristic Command Center Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="منظومة راصد عُمان الهيدرولوجية | AI4SDGs 2026",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ultra-Premium CSS Theme (Glassmorphism, Neon Accents, RTL Typography)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap');
    
    :root {
        --bg-main: #0a0f1d;
        --card-bg: rgba(16, 24, 44, 0.85);
        --accent-cyan: #00f0ff;
        --accent-blue: #3b82f6;
        --accent-emerald: #10b981;
        --accent-amber: #f59e0b;
        --accent-rose: #ef4444;
        --border-glass: rgba(255, 255, 255, 0.08);
    }

    html, body, [class*="css"] {
        font-family: 'Cairo', -apple-system, sans-serif;
    }

    /* Command Center Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(10, 20, 38, 0.95) 0%, rgba(15, 32, 67, 0.9) 50%, rgba(13, 40, 60, 0.85) 100%);
        border: 1px solid rgba(0, 240, 255, 0.25);
        border-right: 6px solid #00f0ff;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), 0 0 25px rgba(0, 240, 255, 0.1);
        backdrop-filter: blur(12px);
    }

    .pulse-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #10b981;
        border-radius: 50%;
        margin-left: 8px;
        box-shadow: 0 0 12px #10b981;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1.1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    .badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        font-size: 11px;
        font-weight: 700;
        border-radius: 9999px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid var(--border-glass);
        color: #94a3b8;
        margin-left: 6px;
        margin-top: 4px;
    }
    .badge-cyan { border-color: rgba(0, 240, 255, 0.4); color: #00f0ff; background: rgba(0, 240, 255, 0.08); }
    .badge-gold { border-color: rgba(245, 158, 11, 0.4); color: #fbbf24; background: rgba(245, 158, 11, 0.08); }
    .badge-emerald { border-color: rgba(16, 185, 129, 0.4); color: #34d399; background: rgba(16, 185, 129, 0.08); }

    /* Futuristic KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 14px;
        margin-bottom: 20px;
    }

    .kpi-box {
        background: linear-gradient(180deg, rgba(20, 30, 55, 0.75) 0%, rgba(13, 20, 38, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 16px 18px;
        text-align: right;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-box:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 240, 255, 0.3);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
    }
    .kpi-box::after {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 4px;
        height: 100%;
        background: #00f0ff;
    }
    .kpi-box.danger::after { background: #ef4444; }
    .kpi-box.warning::after { background: #f59e0b; }
    .kpi-box.success::after { background: #10b981; }

    .kpi-label {
        font-size: 12px;
        font-weight: 600;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .kpi-number {
        font-family: 'JetBrains Mono', 'Cairo', sans-serif;
        font-size: 26px;
        font-weight: 900;
        color: #f8fafc;
        line-height: 1.2;
    }
    .kpi-detail {
        font-size: 11px;
        color: #64748b;
        margin-top: 4px;
    }

    /* Region Selector Card */
    .region-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(0, 240, 255, 0.2);
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 18px;
        text-align: right;
    }
    .region-title {
        font-size: 14px;
        font-weight: 700;
        color: #00f0ff;
        margin-bottom: 4px;
    }
    .region-desc {
        font-size: 11px;
        color: #94a3b8;
        line-height: 1.5;
    }

    /* Action Alert Banner */
    .alert-card {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 10px;
        padding: 12px 16px;
        color: #fca5a5;
        font-size: 13px;
        text-align: right;
        margin-top: 10px;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Multi-Region Data Loader & Hydrology Engine
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_region_dataset(region_key: str):
    """Loads and caches catchment terrain, baseline SAR, and infrastructure."""
    data = generate_catchment_data(region_id=region_key, rows=120, cols=120)
    sar_proc = Sentinel1SARProcessor(pixel_size_m=data["resolution_m"])
    sar_change = sar_proc.detect_flood_changes(
        data["sar_backscatter_pre"],
        data["sar_backscatter_post"],
        fixed_threshold=-16.5
    )
    return data, sar_change


# -----------------------------------------------------------------------------
# Sidebar: Location Switcher & Scenario Configurations
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📍 اختيار حوض الوادي (Location)")
    
    region_options = get_available_regions()
    region_keys = list(region_options.keys())
    
    selected_region_key = st.selectbox(
        "اختر المنطقة الجغرافية المستهدفة:",
        options=region_keys,
        format_func=lambda k: region_options[k]["name_ar"],
        index=0,
        help="تتيح المنظومة التبديل اللحظي بين مختلف الأحواض المائية التضاريسية بسلطنة عُمان"
    )
    
    current_reg = region_options[selected_region_key]
    
    st.markdown(f"""
    <div class="region-card">
        <div class="region-title">🗺️ {current_reg['governorate']}</div>
        <div class="region-desc">{current_reg['risk_profile']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🌀 سيناريوهات الأعاصير المدارية")
    
    # Preset scenarios for realistic Omani cyclones
    preset_choice = st.selectbox(
        "سيناريو العاصفة المسبق (Preset Scenario):",
        options=[
            "إعصار شاهين التاريخي (150 ملم)",
            "إعصار جونو الاستثنائي (200 ملم)",
            "إعصار مكونو / تيج المداري (180 ملم)",
            "منخفض الوبل / المطير 2024 (110 ملم)",
            "سيناريو مخصص (Custom)"
        ],
        index=0
    )

    preset_values = {
        "إعصار شاهين التاريخي (150 ملم)": 150,
        "إعصار جونو الاستثنائي (200 ملم)": 200,
        "إعصار مكونو / تيج المداري (180 ملم)": 180,
        "منخفض الوبل / المطير 2024 (110 ملم)": 110,
        "سيناريو مخصص (Custom)": current_reg["default_rainfall"]
    }
    
    default_precip = preset_values[preset_choice]

    rainfall_input = st.slider(
        "🌧️ معدل هطول الأمطار التراكمي (خلال 24 ساعة):",
        min_value=30,
        max_value=250,
        value=default_precip,
        step=10,
        help="المعدل التراكمي لهطول الأمطار المقاس بالملم على كامل مساحة الحوض"
    )

    curve_number = st.slider(
        "⛰️ معامل نفاذية التربة (SCS Curve Number):",
        min_value=70,
        max_value=95,
        value=current_reg["default_cn"],
        step=1,
        help="معامل نفاذية تربة الحجر والأودية الصخرية (قيم أعلى = جريان سطحي أسرع)"
    )

    st.markdown("---")
    st.markdown("### ⏱️ المحاكاة الحركية للتدفق (Motion Animation)")
    
    time_options = [1, 3, 6, 12, 24]
    
    if "sim_time_index" not in st.session_state:
        st.session_state.sim_time_index = 2  # default T+6h
    if "is_animating" not in st.session_state:
        st.session_state.is_animating = False

    col_play, col_pause, col_reset = st.columns([1, 1, 1])
    with col_play:
        if st.button("▶️ تشغيل", use_container_width=True, help="تشغيل الحركة التلقائية للفيضان"):
            st.session_state.is_animating = True
    with col_pause:
        if st.button("⏸️ إيقاف", use_container_width=True, help="إيقاف الحركة عند المرحلة الحالية"):
            st.session_state.is_animating = False
    with col_reset:
        if st.button("🔄 إعادة", use_container_width=True, help="إعادة المحاكاة للبداية T+1h"):
            st.session_state.sim_time_index = 0
            st.session_state.is_animating = False

    # Slider for manual control
    selected_time = st.select_slider(
        "الأفق الزمني للتوقع:",
        options=time_options,
        value=time_options[st.session_state.sim_time_index],
        format_func=lambda x: f"T+{x}h",
        key="time_slider_select"
    )
    # Sync index if slider dragged manually
    st.session_state.sim_time_index = time_options.index(selected_time)

    st.markdown("---")
    st.markdown("### 🗺️ طبقات التحليل الجغرافي")
    show_sar = st.checkbox("🛰️ رصد الأقمار الصناعية (Sentinel-1 SAR)", value=True)
    show_sim = st.checkbox("🌊 محاكاة انتشار الغمر التوقعي", value=True)
    show_torrents = st.checkbox("⚡ مسارات السيول الجارفة (Flash Paths)", value=True)
    show_evac = st.checkbox("🟢 مسارات الإخلاء الآمنة المقترحة", value=True)
    show_dams = st.checkbox("🏛️ سدود التهدئة والحماية المقترحة", value=True)
    show_subbasins = st.checkbox("🏞️ حدود الأحواض المائية الفرعية", value=False)
    show_ndvi = st.checkbox("🌿 مؤشر الغطاء النباتي والنفاذية (NDVI)", value=False)

    base_tile_choice = st.selectbox(
        "خريطة الأساس:",
        options=["OpenStreetMap", "CartoDB positron", "CartoDB dark_matter"],
        index=0
    )


# -----------------------------------------------------------------------------
# Load Active Region Data & Initialize Engines
# -----------------------------------------------------------------------------
data_env, sar_results = load_region_dataset(selected_region_key)
dem = data_env["dem"]
bounds = data_env["bounds"]

hydro_engine = HydrologySimulationEngine(dem, cell_size_m=data_env["resolution_m"])

simulation_results = hydro_engine.run_flood_simulation(
    rainfall_24h_mm=rainfall_input,
    curve_number=curve_number,
    timesteps=[1, 3, 6, 12, 24]
)

current_depth_grid = simulation_results["timesteps_depths"][selected_time]
current_flood_area_km2 = simulation_results["timesteps_areas_km2"][selected_time]
flash_flood_paths_mask = simulation_results["flash_flood_paths"]

# Sub-basins & Optimal Retention Dams
subbasins_grid = hydro_engine.delineate_subbasins(num_pour_points=4)
proposed_dams = hydro_engine.identify_optimal_dam_locations()

# Risk Analytics Engine
risk_engine = RiskAnalyticsEngine(
    dem=dem,
    bounds=bounds,
    water_depth_grid=current_depth_grid,
    car_cutoff_depth_m=0.30,
    truck_cutoff_depth_m=0.60
)

road_assessment = risk_engine.assess_road_network(data_env["roads"])
infra_assessment = risk_engine.assess_infrastructure(data_env["infrastructure"])

# Safe Evacuation Routing
at_risk_facilities = [f for f in infra_assessment["facilities"] if f["is_at_risk"]]
if not at_risk_facilities:
    at_risk_facilities = [data_env["infrastructure"][-1]]

evacuation_routes = risk_engine.compute_evacuation_routes(
    start_points=at_risk_facilities,
    shelters=data_env["shelters"]
)


# -----------------------------------------------------------------------------
# Top Hero Banner (Operational Command Header)
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
        <div style="text-align: right;">
            <div style="display: flex; align-items: center; margin-bottom: 6px;">
                <span class="pulse-dot"></span>
                <span style="font-size: 13px; font-weight: 700; color: #10b981;">بث راداري مداري مباشر | Sentinel-1 SAR Live Sync</span>
            </div>
            <h1 style="margin: 0; font-size: 26px; font-weight: 900; color: #ffffff;">
                🌊 منصة راصد: محاكاة فيضانات الأودية وتعزيز الصمود المناخي
            </h1>
            <p style="margin: 6px 0 0 0; font-size: 14px; color: #cbd5e1;">
                المنطقة النشطة: <b style="color: #00f0ff;">{current_reg['name_ar']}</b> | 
                مسار رسم خرائط المياه - مسابقة <b>AI4SDGs 2026</b> بشراكة <b>عدسة عُمان (Oman Lens)</b> و <b>Star.Vision Aerospace</b>
            </p>
        </div>
        <div style="margin-top: 10px; text-align: left;">
            <span class="badge badge-cyan">🛰️ SAR Edge-AI</span>
            <span class="badge badge-emerald">SDG 11: مدن مستدامة</span>
            <span class="badge badge-gold">SDG 13: العمل المناخي</span>
            <span class="badge" style="color: #e2e8f0;">رؤية عُمان 2040</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Real-Time Operational KPIs Bar
# -----------------------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-label">مساحة الغمر (T+{selected_time}h)</div>
        <div class="kpi-number" style="color: #00f0ff;">{current_flood_area_km2} <span style="font-size: 14px;">كم²</span></div>
        <div class="kpi-detail">الرصد الراداري: {sar_results['total_water_area_km2']} كم²</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-box danger">
        <div class="kpi-label">السكان في دائرة الخطر</div>
        <div class="kpi-number" style="color: #ef4444;">{infra_assessment['total_population_at_risk']:,}</div>
        <div class="kpi-detail">ضمن نطاق الغمر > 0.3م</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-box warning">
        <div class="kpi-label">نسبة انقطاع الطرق</div>
        <div class="kpi-number" style="color: #f59e0b;">{road_assessment['severance_rate_pct']}%</div>
        <div class="kpi-detail">المقطوع: {road_assessment['severed_length_km']} من {road_assessment['total_length_km']} كم</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-label">أقصى عمق متوقع</div>
        <div class="kpi-number" style="color: #fbbf24;">{simulation_results['peak_depth_m']} <span style="font-size: 14px;">متر</span></div>
        <div class="kpi-detail">الجريان الفعال: {simulation_results['effective_runoff_mm']} ملم</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="kpi-box success">
        <div class="kpi-label">مسارات الإخلاء الآمنة</div>
        <div class="kpi-number" style="color: #10b981;">{len(evacuation_routes)} <span style="font-size: 14px;">مسار</span></div>
        <div class="kpi-detail">نحو {len(data_env['shelters'])} مراكز إيواء مرتفعة</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Main Visualization: Dual Map & Hydrograph
# -----------------------------------------------------------------------------
map_col, chart_col = st.columns([7, 3])

with map_col:
    st.markdown(f"#### 🗺️ الخريطة التفاعلية: الرصد الفعلي بالأقمار الصناعية مقابل المحاكاة التوقعية (T+{selected_time}h)")
    
    m = create_resilience_map(
        bounds=bounds,
        sar_detected_water_mask=sar_results["mask_post"] if show_sar else None,
        simulated_depth_grid=current_depth_grid if show_sim else None,
        flash_flood_paths_mask=flash_flood_paths_mask if show_torrents else None,
        assessed_roads=road_assessment["roads"],
        assessed_infra=infra_assessment["facilities"],
        shelters=data_env["shelters"],
        evacuation_routes=evacuation_routes if show_evac else None,
        subbasins_grid=subbasins_grid if show_subbasins else None,
        proposed_dams=proposed_dams if show_dams else None,
        ndvi_grid=data_env.get("ndvi") if show_ndvi else None,
        show_sar=show_sar,
        show_sim=show_sim,
        show_torrents=show_torrents,
        show_evac=show_evac,
        show_subbasins=show_subbasins,
        show_dams=show_dams,
        show_ndvi=show_ndvi,
        base_tile=base_tile_choice
    )

    st_folium(m, width="100%", height=560, returned_objects=[])

with chart_col:
    st.markdown("#### 📈 الرسم البياني الحركي للتدفق (Graph Motion)")
    motion_fig = create_animated_motion_graph(
        rainfall_scenario_mm=rainfall_input,
        timesteps=[1, 3, 6, 12, 24],
        flooded_areas_km2=simulation_results["timesteps_areas_km2"],
        peak_depth_m=simulation_results["peak_depth_m"],
        current_time=selected_time
    )
    st.plotly_chart(motion_fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("#### 🚨 الإنذار المبكر والتوجيه الميداني")
    if road_assessment["severance_rate_pct"] > 35:
        severed_road_names = [r['name'] for r in road_assessment['roads'] if r.get('is_severed')]
        first_severed = severed_road_names[0] if severed_road_names else "المحاور المنخفضة"
        st.markdown(f"""
        <div class="alert-card">
            <b>⚠️ تنبيه عملياتي عالي الخطورة:</b><br>
            رصد انقطاع حرج في <b>{first_severed}</b> في {current_reg['name_ar']}. 
            يوصى بإغلاق فوري لمداخل الوادي وتفعيل مسارات الإخلاء عبر المرتفعات الآمنة.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 10px; padding: 12px; font-size: 13px; text-align: right; color: #a7f3d0;">
            <b>✅ الوضع مستقر هيدرولوجياً:</b> التدفق محصور داخل المجاري الرئيسية لوادي {current_reg['name_ar']}. الطرق الاستراتيجية سالكة.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    report_dict = {
        "report_id": f"OMAN-FL-REC-{int(time.time())}",
        "region_id": selected_region_key,
        "region_name": current_reg["name_ar"],
        "scenario_rainfall_24h_mm": rainfall_input,
        "selected_horizon": f"T+{selected_time}h",
        "inundated_area_km2": current_flood_area_km2,
        "severance_rate_pct": road_assessment["severance_rate_pct"],
        "population_at_risk": infra_assessment["total_population_at_risk"],
        "proposed_dams_count": len(proposed_dams)
    }
    st.download_button(
        label="📥 تصدير تقرير العمليات الرقمي (JSON)",
        data=json.dumps(report_dict, indent=2, ensure_ascii=False),
        file_name=f"report_{selected_region_key}_T{selected_time}h.json",
        mime="application/json",
        use_container_width=True
    )


# -----------------------------------------------------------------------------
# Detailed Analytics Tabs
# -----------------------------------------------------------------------------
st.markdown("---")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🛣️ تقييم قطاعات الطرق والمعابر",
    "🏥 المنشآت الحيوية والأحياء",
    "🛡️ مسارات الإخلاء ومراكز الإيواء",
    "🏛️ سدود التهدئة والحلول الهندسية",
    "📊 مقارنة المناطق والأحواض العُمانية"
])

with tab1:
    road_data = []
    for r in road_assessment["roads"]:
        road_data.append({
            "اسم الطريق": r["name"],
            "التصنيف": r["type"],
            "أقصى عمق مياه (م)": r["max_water_depth_m"],
            "الطول (كم)": r["length_km"],
            "الحالة التشغيلية": r["status"]
        })
    st.dataframe(road_data, use_container_width=True)

with tab2:
    infra_data = []
    for item in infra_assessment["facilities"]:
        infra_data.append({
            "المنشأة / الحي": item["name"],
            "التصنيف": item["type"],
            "السكان المخدومون": f"{item['pop_served']:,}",
            "الارتفاع عن البحر (م)": item["elevation_m"],
            "عمق الغمر التوقعي (م)": item["simulated_depth_m"],
            "مستوى التهديد": item["risk_level"]
        })
    st.dataframe(infra_data, use_container_width=True)

with tab3:
    evac_data = []
    for ev in evacuation_routes:
        evac_data.append({
            "نقطة الانطلاق": ev["origin_name"],
            "مركز الإيواء المستهدف": ev["destination_shelter"],
            "السعة الاستيعابية": f"{ev['shelter_capacity']:,} فرد",
            "أقصى منسوب مياه بالمسار": f"{ev['max_depth_encountered_m']} م",
            "تقييم الأمان": ev["status_label"]
        })
    st.dataframe(evac_data, use_container_width=True)

with tab4:
    st.markdown("#### 🏛️ الخوانق الجبلية المقترحة لإنشاء سدود التهدئة والحماية")
    st.caption("وفقاً لدراسة منظمة المجتمع العلمي العربي (ArSCO)، تم تحديد أضيق الخوانق ذات مساحات التجمع الكبرى لإنشاء سدود إعاقة وتهدئة:")
    dam_data = []
    for i, d in enumerate(proposed_dams, 1):
        dam_data.append({
            "الموقع المقترح": f"سد تهدئة {current_reg['name_ar']} #{i}",
            "الارتفاع عن البحر": f"{d['elevation_m']} م",
            "السعة التخزينية التقديرية": f"{d['retention_capacity_m3']:,.0f} م³",
            "انحدار حواف المضيق": f"{d['flank_slope_deg']}°",
            "الأثر الميداني": "حجز وتأخير ذروة الموجة الفيضية وحماية المستوطنات ومحاور الطرق بالداونستريم"
        })
    st.dataframe(dam_data, use_container_width=True)

with tab5:
    st.markdown("#### 📊 مقارنة التهديدات الهيدرولوجية بين الأحواض العُمانية الرئيسية")
    st.caption("مقارنة الخصائص الطوبوغرافية وأنماط التصريف للأحواض المشمولة بالمنظومة:")
    reg_comparison = []
    for r_k, r_v in region_options.items():
        reg_comparison.append({
            "الحوض المائي": r_v["name_ar"],
            "المحافظة": r_v["governorate"],
            "طبيعة الخطر الهيدرولوجي": r_v["risk_profile"],
            "معدل الأمطار المعياري": f"{r_v['default_rainfall']} ملم / 24س",
            "معامل النفاذية CN": r_v["default_cn"]
        })
    st.dataframe(reg_comparison, use_container_width=True)


# -----------------------------------------------------------------------------
# Innovation Footer
# -----------------------------------------------------------------------------
st.markdown(f"""
<div style="background-color: #0b1120; border-radius: 12px; padding: 18px; margin-top: 35px; text-align: center; border: 1px solid rgba(255, 255, 255, 0.08);">
    <p style="margin: 0; font-size: 13px; color: #94a3b8;">
        مشروع مقدم لمسابقة <b>AI4SDGs 2026</b> تحت إشراف <b>عدسة عُمان (Oman Lens)</b> و <b>Star.Vision Aerospace</b> | 
        يدعم مستهدفات <b>رؤية عُمان 2040</b> في الإدارة الذكية للأزمات الطبيعية وحماية البنية الأساسية للمدن المستدامة.
    </p>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Automatic Simulation Motion Player Loop
# -----------------------------------------------------------------------------
if st.session_state.get("is_animating", False):
    time.sleep(0.9)
    if st.session_state.sim_time_index < len(time_options) - 1:
        st.session_state.sim_time_index += 1
        st.rerun()
    else:
        st.session_state.is_animating = False
        st.rerun()

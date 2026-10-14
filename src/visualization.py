"""
Geospatial Visualization Module
Builds rich, multi-layered interactive Folium maps, dual satellite/simulation
overlays, and dynamic hydrological hydrographs for decision-makers.
"""
import folium
from folium import plugins
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple


def create_resilience_map(
    bounds: Tuple[float, float, float, float],
    sar_detected_water_mask: np.ndarray,
    simulated_depth_grid: np.ndarray,
    flash_flood_paths_mask: np.ndarray,
    assessed_roads: List[Dict[str, Any]],
    assessed_infra: List[Dict[str, Any]],
    shelters: List[Dict[str, Any]],
    evacuation_routes: List[Dict[str, Any]] = None,
    subbasins_grid: np.ndarray = None,
    proposed_dams: List[Dict[str, Any]] = None,
    ndvi_grid: np.ndarray = None,
    show_sar: bool = True,
    show_sim: bool = True,
    show_torrents: bool = True,
    show_evac: bool = True,
    show_subbasins: bool = False,
    show_dams: bool = True,
    show_ndvi: bool = False,
    base_tile: str = "OpenStreetMap"
) -> folium.Map:
    """
    Constructs an interactive emergency command map with multi-layer overlays:
    - Sentinel-1 SAR Observed Flood
    - DEM-based Simulated Inundation Depth
    - Flash Flood Torrent Corridors
    - Vulnerable Roads & Severed Arteries
    - Critical Facilities & Evacuation Shelters
    - Optimal High-Ground Evacuation Corridors
    """
    min_lat, min_lon, max_lat, max_lon = bounds
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles=base_tile,
        control_scale=True
    )

    # 1. Overlay: Sentinel-1 SAR Observed Flood (Cyan / Solid Blue)
    if show_sar and sar_detected_water_mask is not None:
        sar_group = folium.FeatureGroup(name="🛰️ رصد الأقمار الصناعية (Sentinel-1 SAR)", show=True)
        # Create an RGBA overlay
        rows, cols = sar_detected_water_mask.shape
        sar_rgba = np.zeros((rows, cols, 4), dtype=np.uint8)
        # Cyan color for radar specular water: R=0, G=180, B=216, A=180
        sar_rgba[sar_detected_water_mask] = [0, 180, 216, 190]

        sar_layer = folium.raster_layers.ImageOverlay(
            image=sar_rgba,
            bounds=[[min_lat, min_lon], [max_lat, max_lon]],
            opacity=0.75,
            name="Sentinel-1 SAR Detection",
            interactive=True,
            cross_origin=False,
            zindex=2
        )
        sar_layer.add_to(sar_group)
        sar_group.add_to(m)

    # 2. Overlay: Simulated Flood Inundation Depth (Gradient from Blue to Deep Navy)
    if show_sim and simulated_depth_grid is not None:
        sim_group = folium.FeatureGroup(name="🌊 محاكاة تدفق الفيضان التوقعي (Hydrological Simulation)", show=True)
        rows, cols = simulated_depth_grid.shape
        sim_rgba = np.zeros((rows, cols, 4), dtype=np.uint8)

        # Depth gradient
        # 0.08m - 0.35m: Light Blue
        # 0.35m - 0.80m: Royal Blue
        # > 0.80m: Deep Midnight Blue
        shallow = (simulated_depth_grid >= 0.08) & (simulated_depth_grid < 0.35)
        moderate = (simulated_depth_grid >= 0.35) & (simulated_depth_grid < 0.80)
        deep = simulated_depth_grid >= 0.80

        sim_rgba[shallow] = [67, 97, 238, 160]    # #4361ee
        sim_rgba[moderate] = [58, 12, 163, 190]   # #3a0ca3
        sim_rgba[deep] = [114, 9, 183, 220]      # High momentum deep flood

        sim_layer = folium.raster_layers.ImageOverlay(
            image=sim_rgba,
            bounds=[[min_lat, min_lon], [max_lat, max_lon]],
            opacity=0.75,
            name="Simulated Inundation Depth",
            interactive=True,
            cross_origin=False,
            zindex=3
        )
        sim_layer.add_to(sim_group)
        sim_group.add_to(m)

    # 3. Overlay: Flash Flood Torrent Paths (مسارات السيول الجارفة)
    if show_torrents and flash_flood_paths_mask is not None:
        torrent_group = folium.FeatureGroup(name="⚡ مسارات السيول الجارفة (Flash Flood Paths)", show=True)
        rows, cols = flash_flood_paths_mask.shape
        torrent_rgba = np.zeros((rows, cols, 4), dtype=np.uint8)
        # Neon Orange/Red torrent indicator
        torrent_rgba[flash_flood_paths_mask] = [255, 75, 75, 230]

        torrent_layer = folium.raster_layers.ImageOverlay(
            image=torrent_rgba,
            bounds=[[min_lat, min_lon], [max_lat, max_lon]],
            opacity=0.85,
            name="Flash Flood Paths",
            interactive=True,
            cross_origin=False,
            zindex=4
        )
        torrent_layer.add_to(torrent_group)
        torrent_group.add_to(m)

    # 3b. Overlay: Sub-catchment Basins (حدود الأحواض المائية الفرعية)
    if show_subbasins and subbasins_grid is not None:
        basin_group = folium.FeatureGroup(name="🏞️ الأحواض المائية الفرعية (Sub-basins)", show=False)
        rows, cols = subbasins_grid.shape
        basin_rgba = np.zeros((rows, cols, 4), dtype=np.uint8)
        # Unique pastel tints for each basin
        palette = [
            [255, 183, 3, 110],    # Basin 1 Gold
            [142, 202, 230, 110],  # Basin 2 Cyan
            [247, 37, 133, 110],   # Basin 3 Magenta
            [114, 9, 183, 110]     # Basin 4 Purple
        ]
        for b_id in range(1, 5):
            mask = subbasins_grid == b_id
            if np.any(mask):
                basin_rgba[mask] = palette[(b_id - 1) % len(palette)]
                
        basin_layer = folium.raster_layers.ImageOverlay(
            image=basin_rgba,
            bounds=[[min_lat, min_lon], [max_lat, max_lon]],
            opacity=0.60,
            name="Sub-basins",
            interactive=True,
            cross_origin=False,
            zindex=1
        )
        basin_layer.add_to(basin_group)
        basin_group.add_to(m)

    # 3c. Overlay: NDVI Vegetation & Infiltration Map (الغطاء النباتي والنفاذية)
    if show_ndvi and ndvi_grid is not None:
        ndvi_group = folium.FeatureGroup(name="🌿 الغطاء النباتي والنفاذية (NDVI Index)", show=False)
        rows, cols = ndvi_grid.shape
        ndvi_rgba = np.zeros((rows, cols, 4), dtype=np.uint8)
        # Green tint scaling with NDVI
        veg_mask = ndvi_grid > 0.25
        ndvi_rgba[veg_mask, 1] = np.clip((ndvi_grid[veg_mask] * 255).astype(np.uint8), 80, 240)
        ndvi_rgba[veg_mask, 0] = 30
        ndvi_rgba[veg_mask, 2] = 40
        ndvi_rgba[veg_mask, 3] = 160
        
        ndvi_layer = folium.raster_layers.ImageOverlay(
            image=ndvi_rgba,
            bounds=[[min_lat, min_lon], [max_lat, max_lon]],
            opacity=0.65,
            name="NDVI Vegetation",
            interactive=True,
            cross_origin=False,
            zindex=2
        )
        ndvi_layer.add_to(ndvi_group)
        ndvi_group.add_to(m)

    # 3d. Proposed Retention Dams & Check Dams (سدود الحماية المقترحة)
    if show_dams and proposed_dams:
        dam_group = folium.FeatureGroup(name="🏛️ مواقع سدود التهدئة المقترحة (Proposed Retention Dams)", show=True)
        for i, dam in enumerate(proposed_dams, 1):
            dam_lat = max_lat - (dam["grid_r"] / dem_shape[0] if 'dem_shape' in locals() else rows) * (max_lat - min_lat)
            dam_lon = min_lon + (dam["grid_c"] / dem_shape[1] if 'dem_shape' in locals() else cols) * (max_lon - min_lon)
            
            dam_html = f"""
            <div style="font-family: sans-serif; direction: rtl; text-align: right; min-width: 190px;">
                <h4 style="margin: 0; color: #e63946;">🏛️ سد حماية مقترح #{i}</h4>
                <p style="margin: 4px 0;"><b>الارتفاع:</b> {dam['elevation_m']} م</p>
                <p style="margin: 4px 0;"><b>السعة التخزينية التقديرية:</b> {dam['retention_capacity_m3']:,.0f} م³</p>
                <p style="margin: 4px 0;"><b>انحدار جانبي المضيق:</b> {dam['flank_slope_deg']}°</p>
                <p style="margin: 4px 0; color: #2a9d8f;"><b>الهدف:</b> اعتراض ذروة السيل قبل دخول النسيج العمراني</p>
            </div>
            """
            folium.Marker(
                location=[dam_lat, dam_lon],
                icon=folium.Icon(color="purple", icon="tint", prefix="fa"),
                tooltip=f"موقع سد حماية مقترح #{i} ({dam['retention_capacity_m3']:,.0f} م³)",
                popup=folium.Popup(dam_html, max_width=320)
            ).add_to(dam_group)
        dam_group.add_to(m)

    # 4. Roads Network with Severance Status
    road_group = folium.FeatureGroup(name="🛣️ شبكة الطرق والجسور (Road Network)", show=True)
    for rd in assessed_roads:
        tooltip_html = f"""
        <div style="font-family: sans-serif; min-width: 170px; direction: rtl; text-align: right;">
            <b>{rd['name']}</b><br>
            <b>النوع:</b> {rd['type']}<br>
            <b>الحالة:</b> <span style="color: {rd['status_color']}; font-weight: bold;">{rd['status']}</span><br>
            <b>أقصى منسوب مياه:</b> {rd['max_water_depth_m']} م<br>
            <b>الطول:</b> {rd['length_km']} كم
        </div>
        """
        line_weight = 5 if rd.get("is_severed") else 4
        folium.PolyLine(
            locations=rd["coords"],
            color=rd["status_color"],
            weight=line_weight,
            opacity=0.9,
            tooltip=folium.Tooltip(tooltip_html),
            popup=folium.Popup(tooltip_html, max_width=300)
        ).add_to(road_group)
    road_group.add_to(m)

    # 5. Evacuation Safe Corridors
    if show_evac and evacuation_routes:
        evac_group = folium.FeatureGroup(name="🟢 مسارات الإخلاء الآمنة (Safe Evacuation Routes)", show=True)
        for route in evacuation_routes:
            evac_tooltip = f"""
            <div style="font-family: sans-serif; direction: rtl; text-align: right;">
                <b>مسار إخلاء آمن</b><br>
                <b>نقطة الانطلاق:</b> {route['origin_name']}<br>
                <b>المقصد:</b> {route['destination_shelter']}<br>
                <b>أقصى منسوب مياه بالمسار:</b> {route['max_depth_encountered_m']} م<br>
                <b>التقييم:</b> <span style="color: #06d6a0;">{route['status_label']}</span>
            </div>
            """
            folium.PolyLine(
                locations=route["coordinates"],
                color="#06d6a0",
                weight=5,
                dash_array="8, 6",
                opacity=0.95,
                tooltip=folium.Tooltip(evac_tooltip)
            ).add_to(evac_group)
        evac_group.add_to(m)

    # 6. Critical Infrastructure & Shelters
    infra_group = folium.FeatureGroup(name="🏥 المنشآت الحيوية ومراكز الإيواء (Critical Assets & Shelters)", show=True)
    
    # Shelters
    for sh in shelters:
        shelter_html = f"""
        <div style="font-family: sans-serif; direction: rtl; text-align: right; min-width: 180px;">
            <h4 style="margin: 0; color: #2a9d8f;">🛡️ {sh['name']}</h4>
            <p style="margin: 4px 0;"><b>السعة الاستيعابية:</b> {sh['capacity']} فرد</p>
            <p style="margin: 4px 0;"><b>الارتفاع عن البحر:</b> {sh['elevation_m']} م (منطقة آمنة مرتفعة)</p>
            <p style="margin: 4px 0; color: #2a9d8f;"><b>الحالة:</b> {sh['status']}</p>
        </div>
        """
        folium.Marker(
            location=[sh["lat"], sh["lon"]],
            icon=folium.Icon(color="green", icon="shield", prefix="fa"),
            tooltip=f"مركز إيواء آمن: {sh['name']}",
            popup=folium.Popup(shelter_html, max_width=300)
        ).add_to(infra_group)

    # Critical Facilities
    for item in assessed_infra:
        icon_name = "plus" if item["type"] == "Healthcare" else ("bolt" if item["type"] == "Utility" else "home")
        icon_color = "red" if item["is_at_risk"] else "blue"
        
        infra_html = f"""
        <div style="font-family: sans-serif; direction: rtl; text-align: right; min-width: 180px;">
            <h4 style="margin: 0; color: {item['risk_color']};">{item['name']}</h4>
            <p style="margin: 4px 0;"><b>النوع:</b> {item['type']}</p>
            <p style="margin: 4px 0;"><b>السكان المخدومون:</b> {item['pop_served']:,} نسمة</p>
            <p style="margin: 4px 0;"><b>منسوب الغمر التوقعي:</b> {item['simulated_depth_m']} م</p>
            <p style="margin: 4px 0;"><b>مستوى الخطر:</b> <span style="color: {item['risk_color']}; font-weight: bold;">{item['risk_level']}</span></p>
        </div>
        """
        folium.Marker(
            location=[item["lat"], item["lon"]],
            icon=folium.Icon(color=icon_color, icon=icon_name, prefix="fa"),
            tooltip=f"{item['name']} ({item['risk_level']})",
            popup=folium.Popup(infra_html, max_width=300)
        ).add_to(infra_group)

    infra_group.add_to(m)

    # Layer Control
    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    plugins.Fullscreen(position="topleft").add_to(m)
    plugins.MeasureControl(position="bottomleft").add_to(m)

    return m


def generate_hydrograph_figure(
    rainfall_scenario_mm: float,
    timesteps: List[int],
    flooded_areas_km2: Dict[int, float],
    peak_depth_m: float
) -> plt.Figure:
    """
    Renders an emergency dispatch hydrograph chart showing temporal storm progression
    and expanding inundation footprint over time.
    """
    fig, ax1 = plt.subplots(figsize=(7, 3.2), dpi=120)
    fig.patch.set_facecolor('#0e1117')
    ax1.set_facecolor('#161b22')

    # Rainfall Hyetograph (Bars)
    times = np.array(timesteps)
    # Proportional rainfall per interval
    cum_fractions = [0.12, 0.32, 0.65, 0.90, 1.00]
    cum_rain = np.array(cum_fractions) * rainfall_scenario_mm
    
    ax2 = ax1.twinx()
    ax2.set_facecolor('#161b22')

    bars = ax2.bar(
        times - 0.4,
        cum_rain,
        width=1.2,
        color='#48cae4',
        alpha=0.35,
        label='تراكم الأمطار (Cumulative Rain mm)'
    )
    ax2.set_ylabel('الأمطار التراكمية (ملم)', color='#48cae4', fontsize=9)
    ax2.tick_params(axis='y', labelcolor='#48cae4', labelsize=8)
    ax2.set_ylim(0, max(rainfall_scenario_mm * 1.3, 10))

    # Inundated Area (Line)
    areas = [flooded_areas_km2[t] for t in times]
    line = ax1.plot(
        times,
        areas,
        color='#e63946',
        linewidth=2.5,
        marker='o',
        markersize=6,
        label='رقعة الغمر التراكمية (Flooded Area km²)'
    )
    ax1.set_xlabel('الوقت المنقضي من العاصفة (ساعات)', color='#c9d1d9', fontsize=9)
    ax1.set_ylabel('مساحة الغمر (كم²)', color='#e63946', fontsize=9)
    ax1.tick_params(axis='x', colors='#c9d1d9', labelsize=8)
    ax1.tick_params(axis='y', labelcolor='#e63946', labelsize=8)
    ax1.set_xticks(times)
    ax1.set_xticklabels([f"T+{t}h" for t in times])
    ax1.grid(True, linestyle='--', alpha=0.2, color='#8b949e')

    # Title & styling
    ax1.set_title(
        f"منحنى التدفق الهيدرولوجي وسيناريو الغمر ({rainfall_scenario_mm} ملم / 24 ساعة)",
        color='#f0f6fc',
        fontsize=10,
        pad=10
    )

    for spine in ax1.spines.values():
        spine.set_color('#30363d')
    for spine in ax2.spines.values():
        spine.set_color('#30363d')

    fig.tight_layout()
    return fig


def create_animated_motion_graph(
    rainfall_scenario_mm: float,
    timesteps: List[int],
    flooded_areas_km2: Dict[int, float],
    peak_depth_m: float,
    current_time: int = 6
):
    """
    Constructs a rich Plotly Motion Graph featuring:
    - Frame-by-frame temporal animation of flood propagation
    - Cumulative rainfall hyetograph vs expanding inundation footprint
    - Interactive play/pause motion controls and smooth cubic transitions
    """
    import plotly.graph_objects as go

    times = list(timesteps)
    areas = [flooded_areas_km2[t] for t in times]
    cum_fractions = [0.12, 0.32, 0.65, 0.90, 1.00]
    cum_rain = [round(frac * rainfall_scenario_mm, 1) for frac in cum_fractions]
    
    # Discharge proxy (m3/s) scaling with area & peak depth
    discharge_rates = [round(a * peak_depth_m * 12.5, 1) for a in areas]

    # Base initial figure (at current_time or start)
    fig = go.Figure()

    # Trace 1: Cumulative Rainfall Bars
    fig.add_trace(go.Bar(
        x=[f"T+{t}h" for t in times],
        y=cum_rain,
        name="تراكم الأمطار (ملم)",
        marker=dict(
            color='rgba(59, 130, 246, 0.35)',
            line=dict(color='#3b82f6', width=1.5)
        ),
        yaxis="y2",
        hoverinfo="y+name"
    ))

    # Trace 2: Flooded Area Curve (Neon Cyan Glowing Line + Area fill)
    fig.add_trace(go.Scatter(
        x=[f"T+{t}h" for t in times],
        y=areas,
        name="مساحة الغمر (كم²)",
        mode="lines+markers",
        line=dict(color="#00f0ff", width=3, shape="spline"),
        marker=dict(size=8, color="#00f0ff", symbol="circle"),
        fill="tozeroy",
        fillcolor="rgba(0, 240, 255, 0.12)",
        yaxis="y"
    ))

    # Trace 3: Active Time-step Tracker Marker (Motion Pulse)
    active_idx = times.index(current_time) if current_time in times else 0
    fig.add_trace(go.Scatter(
        x=[f"T+{current_time}h"],
        y=[areas[active_idx]],
        name="المرحلة الحالية (Active)",
        mode="markers+text",
        text=[f"📍 T+{current_time}h<br>{areas[active_idx]} كم²"],
        textposition="top center",
        textfont=dict(color="#ffffff", size=11, family="Cairo"),
        marker=dict(
            size=18,
            color="#ef4444",
            symbol="diamond",
            line=dict(color="#ffffff", width=2)
        ),
        yaxis="y"
    ))

    # Build Animation Frames for Graph Motion
    frames = []
    for i, t in enumerate(times):
        # Progressively show data up to timestep t
        sub_x = [f"T+{times[k]}h" for k in range(i + 1)]
        sub_y_area = [areas[k] for k in range(i + 1)]
        sub_y_rain = [cum_rain[k] for k in range(i + 1)]

        frame = go.Frame(
            name=f"T+{t}h",
            data=[
                go.Bar(
                    x=[f"T+{times[k]}h" for k in range(len(times))],
                    y=[cum_rain[k] if k <= i else 0 for k in range(len(times))],
                    marker=dict(color='rgba(59, 130, 246, 0.35)', line=dict(color='#3b82f6', width=1.5)),
                    yaxis="y2"
                ),
                go.Scatter(
                    x=sub_x,
                    y=sub_y_area,
                    mode="lines+markers",
                    line=dict(color="#00f0ff", width=3.5, shape="spline"),
                    marker=dict(size=9, color="#00f0ff"),
                    fill="tozeroy",
                    fillcolor="rgba(0, 240, 255, 0.15)",
                    yaxis="y"
                ),
                go.Scatter(
                    x=[f"T+{t}h"],
                    y=[areas[i]],
                    mode="markers+text",
                    text=[f"📍 T+{t}h: {areas[i]} كم²"],
                    textposition="top center",
                    textfont=dict(color="#ffffff", size=11, family="Cairo"),
                    marker=dict(size=18, color="#ef4444", symbol="diamond", line=dict(color="#ffffff", width=2)),
                    yaxis="y"
                )
            ]
        )
        frames.append(frame)

    fig.frames = frames

    # Motion Animation Controls (Play/Pause & Timeline Scrubbing)
    updatemenus = [
        dict(
            type="buttons",
            direction="left",
            showactive=True,
            x=0.02,
            y=1.18,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="rgba(0, 240, 255, 0.3)",
            font=dict(color="#00f0ff", size=11, family="Cairo"),
            buttons=[
                dict(
                    label="▶️ تشغيل الحركة (Play)",
                    method="animate",
                    args=[None, dict(frame=dict(duration=800, redraw=True), fromcurrent=True, transition=dict(duration=400, easing="cubic-in-out"))]
                ),
                dict(
                    label="⏸️ إيقاف (Pause)",
                    method="animate",
                    args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))]
                )
            ]
        )
    ]

    sliders = [
        dict(
            steps=[
                dict(
                    method="animate",
                    args=[[f"T+{t}h"], dict(mode="immediate", frame=dict(duration=400, redraw=True), transition=dict(duration=200))],
                    label=f"T+{t}h"
                ) for t in times
            ],
            transition=dict(duration=200),
            x=0.02,
            y=-0.08,
            currentvalue=dict(font=dict(size=11, color="#94a3b8"), prefix="الأفق الزمني: ", visible=True, xanchor="right"),
            len=0.96,
            bgcolor="rgba(15, 23, 42, 0.6)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            font=dict(color="#ffffff", size=10)
        )
    ]

    max_rain = max(rainfall_scenario_mm * 1.25, 20)
    max_area = max(max(areas) * 1.35, 1.0)

    fig.update_layout(
        updatemenus=updatemenus,
        sliders=sliders,
        title=dict(
            text="📈 محاكاة التدفق الزمني التفاعلية (Graph Motion Simulation)",
            font=dict(family="Cairo", size=13, color="#f8fafc"),
            x=0.95,
            xanchor="right"
        ),
        paper_bgcolor="#0b1120",
        plot_bgcolor="#0f172a",
        margin=dict(l=45, r=45, t=65, b=55),
        height=360,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=0.98,
            font=dict(size=10, color="#94a3b8", family="Cairo")
        ),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            tickfont=dict(color="#cbd5e1", size=10),
            showline=True,
            linecolor="rgba(255, 255, 255, 0.1)"
        ),
        yaxis=dict(
            title=dict(text="مساحة الغمر (كم²)", font=dict(color="#00f0ff", size=11, family="Cairo")),
            tickfont=dict(color="#00f0ff", size=10),
            range=[0, max_area],
            gridcolor="rgba(255, 255, 255, 0.06)",
            side="right"
        ),
        yaxis2=dict(
            title=dict(text="الأمطار التراكمية (ملم)", font=dict(color="#3b82f6", size=11, family="Cairo")),
            tickfont=dict(color="#3b82f6", size=10),
            range=[0, max_rain],
            overlaying="y",
            side="left",
            showgrid=False
        )
    )

    return fig

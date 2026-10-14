/**
 * منظومة راصد عُمان: المحاكاة الهيدروليكية ثلاثية الأبعاد (CesiumJS 3D Client)
 * Connects to FastAPI backend, controls 3D Cesium terrain, handles time-stepped
 * georeferenced flood layers, point queries, and GIS vulnerability overlays.
 */

// Application State
const state = {
    viewer: null,
    currentDomainId: "wadi_adai",
    currentRunId: null,
    domainBounds: null,
    domainCenter: [58.54, 23.58],
    activeJobId: null,
    pollInterval: null,
    runSummary: null,
    currentFrameIndex: 0,
    isPlaying: false,
    playTimer: null,
    playSpeedMs: 900,
    // Cesium Layer References
    baseImageryLayer: null,
    floodImageryLayer: null,
    hazardImageryLayer: null,
    osmWaterwaysDataSource: null,
    osmRoadsDataSource: null,
    osmBuildingsDataSource: null,
    vectorDataSource: null,
    clickPinEntity: null,
    presets: {},
    cesiumIonToken: null
};

// -----------------------------------------------------------------------------
// 1. Initialize CesiumJS 3D Globe
// -----------------------------------------------------------------------------
function createBasemapProvider(type) {
    if (type === "osm") {
        return new Cesium.UrlTemplateImageryProvider({
            url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            subdomains: ["a", "b", "c"],
            maximumLevel: 19,
            credit: "© OpenStreetMap contributors"
        });
    } else if (type === "topo") {
        return new Cesium.UrlTemplateImageryProvider({
            url: "https://services.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
            maximumLevel: 19,
            credit: "Esri World Topographic"
        });
    } else if (type === "dark") {
        return new Cesium.UrlTemplateImageryProvider({
            url: "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
            credit: "CartoDB Dark Matter"
        });
    }
    // Default: ArcGIS World Imagery (Satellite) - Key Free, Global, Fast
    return new Cesium.UrlTemplateImageryProvider({
        url: "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        maximumLevel: 19,
        credit: "Esri, Maxar, Earthstar Geographics"
    });
}

async function initCesium() {
    let ionToken = null;
    try {
        const resp = await fetch("/api/sources");
        const data = await resp.json();
        state.presets = data.presets || {};
        if (data.cesium_ion_token) {
            ionToken = data.cesium_ion_token;
            state.cesiumIonToken = ionToken;
            Cesium.Ion.defaultAccessToken = ionToken;
        }
    } catch (e) {
        console.warn("Could not fetch presets catalog:", e);
    }

    // Default base satellite imagery layer
    const defaultBasemap = createBasemapProvider("satellite");
    const baseLayer = new Cesium.ImageryLayer(defaultBasemap);

    // Prepare terrain provider safely (use ellipsoid by default to avoid 401 Ion errors)
    let terrainProvider;
    if (ionToken && ionToken.length > 20) {
        try {
            terrainProvider = await Cesium.createWorldTerrainAsync({
                requestWaterMask: true,
                requestVertexNormals: true
            });
        } catch (e) {
            console.warn("Could not load WorldTerrain, fallback to ellipsoid:", e);
            terrainProvider = new Cesium.EllipsoidTerrainProvider();
        }
    } else {
        terrainProvider = new Cesium.EllipsoidTerrainProvider();
    }

    // Configure Cesium Viewer with explicit baseLayer to disable Cesium Ion default imagery lookups
    state.viewer = new Cesium.Viewer("cesiumContainer", {
        animation: false,
        timeline: false,
        baseLayerPicker: false,
        baseLayer: baseLayer,
        geocoder: false,
        homeButton: false,
        sceneModePicker: false,
        navigationHelpButton: false,
        infoBox: false,
        selectionIndicator: false,
        terrainProvider: terrainProvider
    });

    state.baseImageryLayer = baseLayer;
    state.viewer.scene.globe.depthTestAgainstTerrain = false;
    state.viewer.scene.globe.enableLighting = false;

    // Fly to initial Oman location (Wadi Adai, Muscat)
    state.viewer.camera.flyTo({
        destination: Cesium.Rectangle.fromDegrees(58.50, 23.53, 58.59, 23.62),
        duration: 2.0
    });

    // Setup Click Handler for Point Inspection
    const handler = new Cesium.ScreenSpaceEventHandler(state.viewer.scene.canvas);
    handler.setInputAction((click) => {
        handleMapClick(click.position);
    }, Cesium.ScreenSpaceEventType.LEFT_CLICK);

    // Setup UI event listeners
    setupEventListeners();

    // Prepare default domain
    prepareDomain("wadi_adai");
}

function setBasemap(type) {
    if (!state.viewer) return;
    const provider = createBasemapProvider(type);
    if (state.baseImageryLayer) {
        state.viewer.imageryLayers.remove(state.baseImageryLayer, true);
        state.baseImageryLayer = null;
    }
    // Add as bottom-most layer (index 0)
    state.baseImageryLayer = state.viewer.imageryLayers.addImageryProvider(provider, 0);
}

function flyToLocation(lon, lat, height = 15000, pitchDeg = -45) {
    state.viewer.camera.flyTo({
        destination: Cesium.Cartesian3.fromDegrees(lon, lat, height),
        orientation: {
            heading: Cesium.Math.toRadians(0),
            pitch: Cesium.Math.toRadians(pitchDeg),
            roll: 0.0
        },
        duration: 2.0
    });
}

// -----------------------------------------------------------------------------
// 2. Domain & Preset Management
// -----------------------------------------------------------------------------
async function prepareDomain(presetId = null, customBbox = null) {
    const statusEl = document.getElementById("domainStatus");
    statusEl.innerText = "جاري الاتصال وتجهيز تضاريس Copernicus DEM...";
    statusEl.style.color = "#00f0ff";

    const payload = {};
    if (presetId && presetId !== "custom") {
        payload.preset_id = presetId;
    } else if (customBbox) {
        payload.bbox = customBbox;
    } else {
        payload.preset_id = "wadi_adai";
    }

    try {
        const resp = await fetch("/api/domains/prepare", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || "Failed to prepare domain");
        }
        const data = await resp.json();
        state.currentDomainId = data.domain_id;
        state.domainBounds = data.bounds_wgs84;
        state.domainCenter = data.center;

        statusEl.innerText = `تم تجهيز التضاريس بنجاح (${data.validation.cells.toLocaleString()} خلية، دقة ${data.validation.cell_size_m}م)`;
        statusEl.style.color = "#10b981";

        // Zoom 3D camera to domain bounding rectangle
        const [w, s, e, n] = state.domainBounds;
        state.viewer.camera.flyTo({
            destination: Cesium.Rectangle.fromDegrees(w, s, e, n),
            duration: 2.0
        });

        // Fetch and render real OSM vector layers
        loadOsmVectors(data.domain_id);
    } catch (e) {
        statusEl.innerText = `خطأ: ${e.message}`;
        statusEl.style.color = "#ef4444";
    }
}

async function loadOsmVectors(domainId) {
    // 1. Waterways
    try {
        const respWadi = await fetch(`/api/domains/${domainId}/osm/waterways`);
        if (respWadi.ok) {
            const geojson = await respWadi.json();
            if (state.osmWaterwaysDataSource) state.viewer.dataSources.remove(state.osmWaterwaysDataSource);
            state.osmWaterwaysDataSource = await Cesium.GeoJsonDataSource.load(geojson, {
                stroke: Cesium.Color.fromCssColorString("#00b4d8").withAlpha(0.85),
                strokeWidth: 3,
                clampToGround: true
            });
            state.viewer.dataSources.add(state.osmWaterwaysDataSource);
            state.osmWaterwaysDataSource.show = document.getElementById("layerWaterways").checked;
        }
    } catch (e) { console.warn("Waterways load error:", e); }

    // 2. Roads
    try {
        const respRoads = await fetch(`/api/domains/${domainId}/osm/roads`);
        if (respRoads.ok) {
            const geojson = await respRoads.json();
            if (state.osmRoadsDataSource) state.viewer.dataSources.remove(state.osmRoadsDataSource);
            state.osmRoadsDataSource = await Cesium.GeoJsonDataSource.load(geojson, {
                stroke: Cesium.Color.fromCssColorString("#f59e0b").withAlpha(0.75),
                strokeWidth: 2,
                clampToGround: true
            });
            state.viewer.dataSources.add(state.osmRoadsDataSource);
            state.osmRoadsDataSource.show = document.getElementById("layerRoads").checked;
        }
    } catch (e) { console.warn("Roads load error:", e); }

    // 3. Buildings
    try {
        const respBldgs = await fetch(`/api/domains/${domainId}/osm/buildings`);
        if (respBldgs.ok) {
            const geojson = await respBldgs.json();
            if (state.osmBuildingsDataSource) state.viewer.dataSources.remove(state.osmBuildingsDataSource);
            state.osmBuildingsDataSource = await Cesium.GeoJsonDataSource.load(geojson, {
                fill: Cesium.Color.fromCssColorString("#64748b").withAlpha(0.6),
                stroke: Cesium.Color.fromCssColorString("#94a3b8"),
                strokeWidth: 1,
                clampToGround: true
            });
            state.viewer.dataSources.add(state.osmBuildingsDataSource);
            state.osmBuildingsDataSource.show = document.getElementById("layerBuildings").checked;
        }
    } catch (e) { console.warn("Buildings load error:", e); }
}

// -----------------------------------------------------------------------------
// 3. Simulation Job Execution & Monitoring
// -----------------------------------------------------------------------------
async function triggerSimulation() {
    const rain = parseFloat(document.getElementById("rainInput").value);
    const dur = parseFloat(document.getElementById("durationSelect").value);
    const dist = document.getElementById("distSelect").value;
    const cn = parseFloat(document.getElementById("cnInput").value);
    const manning = parseFloat(document.getElementById("manningInput").value);

    const payload = {
        domain_id: state.currentDomainId,
        rainfall_total_mm: rain,
        rainfall_duration_hours: dur,
        rainfall_distribution: dist,
        curve_number: cn,
        manning_n: manning,
        simulation_hours: dur,
        output_interval_minutes: 15.0
    };

    document.getElementById("btnRunSimulation").style.display = "none";
    document.getElementById("btnCancelSimulation").style.display = "block";
    document.getElementById("simProgressContainer").style.display = "block";
    document.getElementById("progressBarFill").style.width = "0%";
    document.getElementById("progressPctText").innerText = "0%";
    document.getElementById("progressStatusText").innerText = "بدء تشغيل المحاكاة...";

    try {
        const resp = await fetch("/api/simulations/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        if (!resp.ok) throw new Error("فشل إطلاق المحاكاة");
        const data = await resp.json();
        state.activeJobId = data.job_id;

        // Start polling progress
        state.pollInterval = setInterval(() => pollSimulationProgress(data.job_id), 1200);
    } catch (e) {
        alert("خطأ: " + e.message);
        resetSimButtons();
    }
}

async function pollSimulationProgress(jobId) {
    try {
        const resp = await fetch(`/api/jobs/${jobId}`);
        if (!resp.ok) return;
        const data = await resp.json();

        const pct = data.percentage || 0;
        document.getElementById("progressBarFill").style.width = `${pct}%`;
        document.getElementById("progressPctText").innerText = `${pct.toFixed(0)}%`;
        document.getElementById("progressStatusText").innerText = data.message || "جاري الحساب...";

        if (data.current_max_depth_m) {
            document.getElementById("progressDetailsText").innerText =
                `أقصى عمق حالي: ${data.current_max_depth_m}م | نسبة خطأ الكتلة: ${data.mass_balance_error_pct}%`;
        }

        if (data.status === "completed") {
            clearInterval(state.pollInterval);
            state.currentRunId = jobId;
            resetSimButtons();
            document.getElementById("simProgressContainer").style.display = "none";
            loadRunResults(jobId);
        } else if (data.status === "failed" || data.status === "cancelled") {
            clearInterval(state.pollInterval);
            resetSimButtons();
            alert(`توقفت المحاكاة: ${data.message || data.status}`);
        }
    } catch (e) {
        console.warn("Poll error:", e);
    }
}

function resetSimButtons() {
    document.getElementById("btnRunSimulation").style.display = "block";
    document.getElementById("btnCancelSimulation").style.display = "none";
}

async function cancelCurrentJob() {
    if (!state.activeJobId) return;
    await fetch(`/api/jobs/${state.activeJobId}/cancel`, { method: "POST" });
}

// -----------------------------------------------------------------------------
// 4. Results Visualization & Synchronous Time Slider
// -----------------------------------------------------------------------------
async function loadRunResults(runId) {
    const resp = await fetch(`/api/runs/${runId}/summary`);
    if (!resp.ok) return;
    state.runSummary = await resp.json();

    const snapshots = state.runSummary.snapshots || [];
    const slider = document.getElementById("timeSlider");
    slider.min = 0;
    slider.max = Math.max(0, snapshots.length - 1);
    slider.value = 0;
    state.currentFrameIndex = 0;

    // Render time ticks
    const ticksContainer = document.getElementById("timeTicksContainer");
    ticksContainer.innerHTML = "";
    if (snapshots.length > 0) {
        ticksContainer.innerHTML = `<span>T+0h</span><span>T+${(state.runSummary.simulated_duration_hours / 2).toFixed(1)}h</span><span>T+${state.runSummary.simulated_duration_hours}h</span>`;
    }

    // Display frame 0
    updateTimeFrame(0);

    // Assess roads and buildings impacts
    loadImpactAssessments(runId);
}

function updateTimeFrame(frameIndex) {
    if (!state.runSummary || !state.runSummary.snapshots) return;
    const snapshots = state.runSummary.snapshots;
    if (frameIndex >= snapshots.length) return;

    state.currentFrameIndex = frameIndex;
    const snap = snapshots[frameIndex];

    // Update Header Labels & KPIs
    document.getElementById("currentHorizonLabel").innerText = `T+${snap.time_hours.toFixed(2)}h`;
    document.getElementById("currentTimeFormatted").innerText = `${(snap.time_hours * 60).toFixed(0)} دقيقة من بداية العاصفة`;

    document.getElementById("kpiFloodedArea").innerText = `${snap.flooded_area_km2.toFixed(2)} كم²`;
    document.getElementById("kpiMaxDepth").innerText = `${snap.peak_depth_m.toFixed(2)} م`;

    // Swap Cesium Imagery Layer (Georeferenced PNG)
    const [w, s, e, n] = state.domainBounds;
    const rect = Cesium.Rectangle.fromDegrees(w, s, e, n);

    if (state.floodImageryLayer) {
        state.viewer.imageryLayers.remove(state.floodImageryLayer);
        state.floodImageryLayer = null;
    }

    if (document.getElementById("layerFlood").checked) {
        const imageUrl = `/api/runs/${state.currentRunId}/layers/depth/${frameIndex}.png?t=${Date.now()}`;
        const provider = new Cesium.SingleTileImageryProvider({
            url: imageUrl,
            rectangle: rect
        });
        state.floodImageryLayer = state.viewer.imageryLayers.addImageryProvider(provider);
        state.floodImageryLayer.alpha = 0.82;
    }

    // Update Vectors if enabled
    if (document.getElementById("layerVectors").checked) {
        updateFlowVectors(frameIndex);
    }
}

async function updateFlowVectors(frameIndex) {
    try {
        const resp = await fetch(`/api/runs/${state.currentRunId}/vectors/${frameIndex}`);
        if (!resp.ok) return;
        const geojson = await resp.json();

        if (state.vectorDataSource) state.viewer.dataSources.remove(state.vectorDataSource);

        state.vectorDataSource = new Cesium.CustomDataSource("vectors");
        geojson.features.forEach(f => {
            const [lon, lat] = f.geometry.coordinates;
            const sp = f.properties.speed_m_s;
            const bearing = f.properties.bearing_deg;

            state.vectorDataSource.entities.add({
                position: Cesium.Cartesian3.fromDegrees(lon, lat, 10),
                point: {
                    pixelSize: Math.min(10, Math.max(4, sp * 3)),
                    color: Cesium.Color.fromCssColorString("#00f0ff").withAlpha(0.85)
                }
            });
        });
        state.viewer.dataSources.add(state.vectorDataSource);
    } catch (e) {
        console.warn("Vectors error:", e);
    }
}

async function loadImpactAssessments(runId) {
    // Roads
    try {
        const respRoads = await fetch(`/api/runs/${runId}/assess/roads`);
        if (respRoads.ok) {
            const roadData = await respRoads.json();
            document.getElementById("kpiRoadSeverance").innerText = `${roadData.severance_rate_pct}%`;
        }
    } catch (e) {}

    // Buildings
    try {
        const respBldgs = await fetch(`/api/runs/${runId}/assess/buildings`);
        if (respBldgs.ok) {
            const bldgData = await respBldgs.json();
            document.getElementById("kpiBuildingsAffected").innerText = `${bldgData.flooded_buildings_count}`;
        }
    } catch (e) {}
}

// -----------------------------------------------------------------------------
// 5. Point Click Inspection (Elevation, Depth, Velocity, Hydrograph)
// -----------------------------------------------------------------------------
async function handleMapClick(windowPosition) {
    if (!state.currentRunId) return;

    const ray = state.viewer.camera.getPickRay(windowPosition);
    const cartesian = state.viewer.scene.globe.pick(ray, state.viewer.scene);
    if (!cartesian) return;

    const cartographic = Cesium.Cartographic.fromCartesian(cartesian);
    const lon = Cesium.Math.toDegrees(cartographic.longitude);
    const lat = Cesium.Math.toDegrees(cartographic.latitude);

    // Place visual pinpoint pin
    if (state.clickPinEntity) state.viewer.entities.remove(state.clickPinEntity);
    state.clickPinEntity = state.viewer.entities.add({
        position: cartesian,
        point: {
            pixelSize: 12,
            color: Cesium.Color.fromCssColorString("#00f0ff"),
            outlineColor: Cesium.Color.WHITE,
            outlineWidth: 2
        }
    });

    const card = document.getElementById("pointInspectorCard");
    card.style.display = "block";
    document.getElementById("inspCoords").innerText = `${lon.toFixed(4)}°, ${lat.toFixed(4)}°`;
    document.getElementById("inspElev").innerText = "جاري التحميل...";

    try {
        const resp = await fetch(`/api/runs/${state.currentRunId}/point?lon=${lon}&lat=${lat}`);
        if (!resp.ok) return;
        const data = await resp.json();

        document.getElementById("inspElev").innerText = `${data.elevation_m} م`;
        document.getElementById("inspDepth").innerText = `${data.peak_depth_m} م`;
        document.getElementById("inspSpeed").innerText = `${data.peak_speed_m_s} م/ث`;
        document.getElementById("inspArrival").innerText = data.arrival_time_hours !== null ? `${data.arrival_time_hours} س` : "لم يغمر";
        document.getElementById("inspHazard").innerText = data.hazard_level;

        renderPointHydrograph(data.hydrograph);
    } catch (e) {
        console.warn("Inspector error:", e);
    }
}

function renderPointHydrograph(hydrograph) {
    const canvas = document.getElementById("pointHydrographCanvas");
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    if (!hydrograph || hydrograph.length < 2) {
        ctx.fillStyle = "#64748b";
        ctx.font = "11px Cairo";
        ctx.fillText("لا يوجد غمر مائي في هذه النقطة", 20, h / 2);
        return;
    }

    const depths = hydrograph.map(p => p.depth_m);
    const maxD = Math.max(...depths, 0.5);

    // Draw grid line
    ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
    ctx.beginPath();
    ctx.moveTo(0, h - 1);
    ctx.lineTo(w, h - 1);
    ctx.stroke();

    // Plot Spline Area
    ctx.beginPath();
    hydrograph.forEach((pt, i) => {
        const x = (i / (hydrograph.length - 1)) * (w - 20) + 10;
        const y = h - (pt.depth_m / maxD) * (h - 15) - 5;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });

    ctx.strokeStyle = "#00f0ff";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Fill gradient
    ctx.lineTo(w - 10, h);
    ctx.lineTo(10, h);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, "rgba(0, 240, 255, 0.4)");
    grad.addColorStop(1, "rgba(0, 240, 255, 0.0)");
    ctx.fillStyle = grad;
    ctx.fill();
}

// -----------------------------------------------------------------------------
// 6. UI Event Listeners & Modals
// -----------------------------------------------------------------------------
function setupEventListeners() {
    // Preset dropdown
    const presetSelect = document.getElementById("presetSelect");
    presetSelect.addEventListener("change", (e) => {
        const val = e.target.value;
        const isCustom = val === "custom";
        document.getElementById("customBboxGroup").style.display = isCustom ? "block" : "none";
        document.getElementById("presetInfoCard").style.display = isCustom ? "none" : "block";

        if (!isCustom && state.presets[val]) {
            const p = state.presets[val];
            document.getElementById("presetGov").innerText = p.governorate;
            document.getElementById("presetDesc").innerText = p.description;
            document.getElementById("rainInput").value = p.default_rainfall_mm;
            document.getElementById("rainVal").innerText = `${p.default_rainfall_mm} ملم`;
            document.getElementById("cnInput").value = p.default_cn;
            document.getElementById("cnVal").innerText = p.default_cn;
        }
    });

    // Prepare Domain button
    document.getElementById("btnPrepareDomain").addEventListener("click", () => {
        const val = presetSelect.value;
        if (val === "custom") {
            const bbox = [
                parseFloat(document.getElementById("bboxWest").value),
                parseFloat(document.getElementById("bboxSouth").value),
                parseFloat(document.getElementById("bboxEast").value),
                parseFloat(document.getElementById("bboxNorth").value)
            ];
            prepareDomain("custom", bbox);
        } else {
            prepareDomain(val);
        }
    });

    // Sliders
    document.getElementById("rainInput").addEventListener("input", (e) => {
        document.getElementById("rainVal").innerText = `${e.target.value} ملم`;
    });
    document.getElementById("cnInput").addEventListener("input", (e) => {
        document.getElementById("cnVal").innerText = e.target.value;
    });
    document.getElementById("manningInput").addEventListener("input", (e) => {
        document.getElementById("manningVal").innerText = parseFloat(e.target.value).toFixed(3);
    });

    // Simulation buttons
    document.getElementById("btnRunSimulation").addEventListener("click", triggerSimulation);
    document.getElementById("btnCancelSimulation").addEventListener("click", cancelCurrentJob);

    // Time slider
    const timeSlider = document.getElementById("timeSlider");
    timeSlider.addEventListener("input", (e) => {
        updateTimeFrame(parseInt(e.target.value));
    });

    // Play/Pause
    const playBtn = document.getElementById("btnPlayPause");
    playBtn.addEventListener("click", togglePlayback);
    document.getElementById("btnStepNext").addEventListener("click", () => {
        if (!state.runSummary) return;
        const max = state.runSummary.snapshots.length - 1;
        if (state.currentFrameIndex < max) {
            timeSlider.value = state.currentFrameIndex + 1;
            updateTimeFrame(state.currentFrameIndex + 1);
        }
    });
    document.getElementById("btnStepPrev").addEventListener("click", () => {
        if (state.currentFrameIndex > 0) {
            timeSlider.value = state.currentFrameIndex - 1;
            updateTimeFrame(state.currentFrameIndex - 1);
        }
    });

    document.getElementById("playSpeedSelect").addEventListener("change", (e) => {
        state.playSpeedMs = parseInt(e.target.value);
        if (state.isPlaying) {
            clearInterval(state.playTimer);
            state.playTimer = setInterval(advanceFrame, state.playSpeedMs);
        }
    });

    // Basemap selector
    const basemapSelect = document.getElementById("basemapSelect");
    if (basemapSelect) {
        basemapSelect.addEventListener("change", (e) => {
            setBasemap(e.target.value);
        });
    }

    // Terrain toggle
    const layerTerrain = document.getElementById("layerTerrain");
    if (layerTerrain) {
        layerTerrain.addEventListener("change", async (e) => {
            if (e.target.checked) {
                if (state.cesiumIonToken && state.cesiumIonToken.length > 20) {
                    try {
                        const terrain = await Cesium.createWorldTerrainAsync({
                            requestWaterMask: true,
                            requestVertexNormals: true
                        });
                        state.viewer.terrainProvider = terrain;
                        state.viewer.scene.globe.depthTestAgainstTerrain = true;
                    } catch (err) {
                        console.warn("Could not load world terrain:", err);
                    }
                } else {
                    const statusEl = document.getElementById("domainStatus");
                    statusEl.innerText = "تنبيه: التضاريس ثلاثية الأبعاد الدقيقة مدمجة في حسابات المحاكاة الهيدروليكية (Copernicus DEM). لتجسيمها عالمياً على CesiumJS يمكن تزويد CESIUM_ION_TOKEN في الإعدادات.";
                    statusEl.style.color = "#fbbf24";
                }
            } else {
                state.viewer.terrainProvider = new Cesium.EllipsoidTerrainProvider();
                state.viewer.scene.globe.depthTestAgainstTerrain = false;
            }
        });
    }

    // Layers Toggles
    document.getElementById("layerWaterways").addEventListener("change", (e) => {
        if (state.osmWaterwaysDataSource) state.osmWaterwaysDataSource.show = e.target.checked;
    });
    document.getElementById("layerRoads").addEventListener("change", (e) => {
        if (state.osmRoadsDataSource) state.osmRoadsDataSource.show = e.target.checked;
    });
    document.getElementById("layerBuildings").addEventListener("change", (e) => {
        if (state.osmBuildingsDataSource) state.osmBuildingsDataSource.show = e.target.checked;
    });
    document.getElementById("layerFlood").addEventListener("change", () => {
        updateTimeFrame(state.currentFrameIndex);
    });
    document.getElementById("layerHazard").addEventListener("change", (e) => {
        toggleHazardLayer(e.target.checked);
    });

    // Camera 3D controls
    document.getElementById("btnZoomIn").addEventListener("click", () => {
        state.viewer.camera.zoomIn(1500);
    });
    document.getElementById("btnZoomOut").addEventListener("click", () => {
        state.viewer.camera.zoomOut(1500);
    });
    document.getElementById("btnResetView").addEventListener("click", () => {
        if (state.domainBounds) {
            const [w, s, e, n] = state.domainBounds;
            state.viewer.camera.flyTo({
                destination: Cesium.Rectangle.fromDegrees(w, s, e, n),
                duration: 1.5
            });
        } else {
            state.viewer.camera.flyTo({
                destination: Cesium.Rectangle.fromDegrees(58.50, 23.53, 58.59, 23.62),
                duration: 1.5
            });
        }
    });
    document.getElementById("btn3DTilt").addEventListener("click", () => {
        const curPitch = state.viewer.camera.pitch;
        const targetPitch = curPitch < -1.1 ? -0.7 : -1.5; // Toggle 40 deg vs 90 deg (2D/3D)
        state.viewer.camera.flyTo({
            destination: state.viewer.camera.position,
            orientation: { heading: 0, pitch: targetPitch, roll: 0 },
            duration: 1.0
        });
    });

    // Inspector close button
    document.getElementById("btnCloseInspector").addEventListener("click", () => {
        document.getElementById("pointInspectorCard").style.display = "none";
        if (state.clickPinEntity) state.viewer.entities.remove(state.clickPinEntity);
    });

    // Modals
    document.getElementById("btnOpenMetadata").addEventListener("click", openMetadataModal);
    document.getElementById("btnCloseMetadata").addEventListener("click", () => {
        document.getElementById("metadataModal").style.display = "none";
    });
    document.getElementById("btnOpenCompare").addEventListener("click", openCompareModal);
    document.getElementById("btnCloseCompare").addEventListener("click", () => {
        document.getElementById("compareModal").style.display = "none";
    });

    // Export GeoTIFF
    document.getElementById("btnExportMenu").addEventListener("click", () => {
        if (!state.currentRunId) {
            alert("يرجى تشغيل محاكاة أولاً لتصدير نتائجها.");
            return;
        }
        window.open(`/api/runs/${state.currentRunId}/download/max_depth.tif`, "_blank");
    });
}

function togglePlayback() {
    const playBtn = document.getElementById("btnPlayPause");
    if (state.isPlaying) {
        state.isPlaying = false;
        playBtn.innerText = "▶";
        clearInterval(state.playTimer);
    } else {
        if (!state.runSummary || !state.runSummary.snapshots) return;
        state.isPlaying = true;
        playBtn.innerText = "⏸";
        state.playTimer = setInterval(advanceFrame, state.playSpeedMs);
    }
}

function advanceFrame() {
    if (!state.runSummary) return;
    const max = state.runSummary.snapshots.length - 1;
    let next = state.currentFrameIndex + 1;
    if (next > max) next = 0; // Loop around
    document.getElementById("timeSlider").value = next;
    updateTimeFrame(next);
}

function toggleHazardLayer(show) {
    if (!state.currentRunId || !state.domainBounds) return;
    if (state.hazardImageryLayer) {
        state.viewer.imageryLayers.remove(state.hazardImageryLayer);
        state.hazardImageryLayer = null;
    }
    if (show) {
        const [w, s, e, n] = state.domainBounds;
        const rect = Cesium.Rectangle.fromDegrees(w, s, e, n);
        const provider = new Cesium.SingleTileImageryProvider({
            url: `/api/runs/${state.currentRunId}/layers/hazard.png?t=${Date.now()}`,
            rectangle: rect
        });
        state.hazardImageryLayer = state.viewer.imageryLayers.addImageryProvider(provider);
        state.hazardImageryLayer.alpha = 0.85;
    }
}

async function openMetadataModal() {
    const modal = document.getElementById("metadataModal");
    const container = document.getElementById("metadataModalContent");
    modal.style.display = "flex";
    container.innerHTML = "جاري استرجاع بطاقة البيانات الوصفية...";

    try {
        const resp = await fetch("/api/sources");
        const data = await resp.json();
        const sources = data.sources || {};

        let html = "";
        for (const [key, s] of Object.entries(sources)) {
            html += `
                <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                    <h4 style="margin: 0 0 6px 0; color: #00f0ff;">${s.name}</h4>
                    <p style="margin: 2px 0;"><b>المزود:</b> ${s.provider}</p>
                    <p style="margin: 2px 0;"><b>الدقة الأفقية / الرأسية:</b> ${s.horizontal_resolution || "غير محدد"} / ${s.vertical_accuracy || "غير محدد"}</p>
                    <p style="margin: 2px 0;"><b>نظام الإحداثيات / المسند:</b> ${s.horizontal_crs || "WGS84"} / ${s.vertical_datum || "EGM2008"}</p>
                    <p style="margin: 2px 0;"><b>الترخيص:</b> ${s.license || "مفتوح الاستخدام"}</p>
                    ${s.caveats ? `<p style="margin: 4px 0 0 0; color: #f59e0b; font-size: 11px;">⚠️ <b>محددات وتحذيرات:</b> ${s.caveats.join(" ")}</p>` : ""}
                </div>
            `;
        }
        container.innerHTML = html;
    } catch (e) {
        container.innerHTML = "تعذر تحميل البيانات الوصفية.";
    }
}

async function openCompareModal() {
    const modal = document.getElementById("compareModal");
    modal.style.display = "flex";

    const selA = document.getElementById("compareRunA");
    const selB = document.getElementById("compareRunB");
    selA.innerHTML = "";
    selB.innerHTML = "";

    try {
        const resp = await fetch("/api/runs");
        const runs = await resp.json();
        runs.forEach(r => {
            const opt = `<option value="${r.run_id}">${r.run_id} (${r.domain_id} - ${r.simulated_duration_hours}h)</option>`;
            selA.innerHTML += opt;
            selB.innerHTML += opt;
        });
        if (runs.length > 1) selB.selectedIndex = 1;
    } catch (e) {}

    document.getElementById("btnExecuteCompare").onclick = async () => {
        const rA = selA.value;
        const rB = selB.value;
        if (!rA || !rB) return;
        const compResp = await fetch(`/api/runs/compare?run_a=${rA}&run_b=${rB}`);
        const c = await compResp.json();
        document.getElementById("compareResultsBox").innerHTML = `
            <div style="background: rgba(0,240,255,0.06); border: 1px solid #00f0ff; border-radius: 8px; padding: 14px;">
                <h4>نتائج المقارنة المكانية بين السيناريوهين:</h4>
                <p>فرق مساحة الغمر الإجمالية (ΔArea): <b>${c.delta_area_km2} كم²</b></p>
                <p>مساحة الأراضي التي غُمرت حديثاً في السيناريو ب: <b>${c.newly_flooded_area_km2} كم²</b></p>
                <p>مساحة الأراضي التي انحسرت عنها المياه: <b>${c.receded_area_km2} كم²</b></p>
                <p>فرق أقصى عمق مياه (ΔDepth): <b>${c.delta_peak_depth_m} م</b></p>
            </div>
        `;
    };
}

// Start app on DOM load
window.addEventListener("DOMContentLoaded", initCesium);

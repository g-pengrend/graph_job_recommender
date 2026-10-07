/**
 * SINGAPORE SPATIAL JOB RADAR - CLIENT CONTROLLER
 * Full-bleed interactive Leaflet Map + 60fps Commute Radar + GraphSAGE API + Ollama Resume Inspector
 */

const state = {
  userLat: 1.298,
  userLon: 103.787,
  locationName: "One-North, Singapore",
  radiusKm: 10.0,
  priority: "Job Description",
  activeSubtab: "within",
  profileMode: "resume", // Default to "resume" as requested
  
  // Resume state
  rawResumeText: "",
  aiResumeText: "",
  activeResumeSource: "ai", // "ai" or "raw"
  currentViewingVersion: "ai", // "ai" or "raw"

  // Recommendations data
  withinJobs: [],
  outsideJobs: [],
  
  // Heatmap configuration
  heatmapDensity: 5.5,
  heatmapRadius: 13,
  heatmapBlur: 15,
  activePalette: "cyberpunk",
  heatmapRawPoints: [],

  // Leaflet references
  map: null,
  tileLayer: null,
  heatmapLayer: null,
  userMarker: null,
  radiusCircle: null,
  jobMarkersGroup: null,
  commuteVectorLine: null
};

// ----------------- Curated Multi-Stop Thermal Palettes ----------------- //
const HEATMAP_PALETTES = {
  cyberpunk: {
    name: "Cyberpunk Thermal",
    gradient: {
      0.08: '#00f2fe', // Electric Cyan (Sparse 1-3 jobs)
      0.20: '#00c6ff', // Deep Cyan / Sky Blue
      0.35: '#00e676', // Vivid Emerald Green (Active Suburban Hubs)
      0.52: '#ffea00', // Electric Gold / Yellow (Commercial Centers)
      0.70: '#ff9100', // Tangerine Orange (Dense Business Parks)
      0.88: '#ff1744', // Hot Coral / Crimson (High Volume Hubs)
      1.00: '#d500f9'  // Neon Magenta / Violet Peak (CBD & One-North Core)
    },
    cssGradient: "linear-gradient(to right, #00f2fe 8%, #00c6ff 20%, #00e676 35%, #ffea00 52%, #ff9100 70%, #ff1744 88%, #d500f9 100%)"
  },
  magma: {
    name: "Solar Magma",
    gradient: {
      0.08: '#3b0f70', // Deep Indigo
      0.25: '#8c2981', // Purple Violet
      0.50: '#de4968', // Coral Red
      0.75: '#fe9f6d', // Warm Peach
      0.90: '#fcfdbf', // Bright Solar Gold
      1.00: '#ffffff'  // White-hot Peak
    },
    cssGradient: "linear-gradient(to right, #3b0f70 8%, #8c2981 25%, #de4968 50%, #fe9f6d 75%, #fcfdbf 90%, #ffffff 100%)"
  },
  emerald: {
    name: "Emerald Matrix",
    gradient: {
      0.08: '#115e59', // Dark Teal
      0.25: '#06b6d4', // Cyan
      0.50: '#10b981', // Neon Emerald
      0.75: '#a3e635', // Lime
      0.90: '#facc15', // Amber Gold
      1.00: '#f97316'  // Orange Hotspot
    },
    cssGradient: "linear-gradient(to right, #115e59 8%, #06b6d4 25%, #10b981 50%, #a3e635 75%, #facc15 90%, #f97316 100%)"
  }
};

// ==========================================================================
// 1. INITIALIZATION & MAP SETUP
// ==========================================================================

document.addEventListener("DOMContentLoaded", async () => {
  setProfileMode("resume");
  await initMapWithConfig();
  loadPresets();
  loadHeatmap();
  geocodePostal(); // Default 138632
  setupDropzone();
});

async function initMapWithConfig() {
  state.map = L.map("map", {
    center: [state.userLat, state.userLon],
    zoom: 12,
    zoomControl: false,
    attributionControl: false
  });

  // Re-position zoom control to bottom right
  L.control.zoom({ position: "bottomright" }).addTo(state.map);

  // Layer groups
  state.jobMarkersGroup = L.featureGroup().addTo(state.map);

  // Fetch API config to determine tile provider
  let config = { mapbox_token: "", stadia_key: "", carto_key: "", ollama_model: "qwen2.5:3b" };
  try {
    const res = await fetch("/api/config");
    config = await res.json();
  } catch (e) {
    console.warn("Could not load config, using free OSM fallback:", e);
  }

  const providerStatus = document.getElementById("map-provider-status");

  // Determine Tile Layer (Mapbox -> Stadia -> Carto with key -> Keyless Dark OpenStreetMap)
  if (config.mapbox_token && config.mapbox_token.trim().length > 10) {
    state.tileLayer = L.tileLayer(
      `https://api.mapbox.com/styles/v1/mapbox/dark-v11/tiles/{z}/{x}/{y}?access_token=${config.mapbox_token}`,
      { maxZoom: 19, tileSize: 512, zoomOffset: -1 }
    );
    if (providerStatus) providerStatus.textContent = "Mapbox Dark Active";
  } else if (config.stadia_key && config.stadia_key.trim().length > 5) {
    state.tileLayer = L.tileLayer(
      `https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png?api_key=${config.stadia_key}`,
      { maxZoom: 19 }
    );
    if (providerStatus) providerStatus.textContent = "Stadia Dark Active";
  } else if (config.carto_key && config.carto_key.trim().length > 5) {
    state.tileLayer = L.tileLayer(
      `https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png?api_key=${config.carto_key}`,
      { maxZoom: 19, subdomains: "abcd" }
    );
    if (providerStatus) providerStatus.textContent = "CARTO Dark Active";
  } else {
    // 100% Free, Keyless Guaranteed Default: OpenStreetMap with custom dark CSS filter
    state.tileLayer = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      className: "dark-osm-tiles"
    });
    if (providerStatus) providerStatus.textContent = "OpenStreetMap (Keyless Dark Mode)";
  }

  state.tileLayer.addTo(state.map);

  // User Beacon & Radius Circle
  updateUserLocationMarker(state.userLat, state.userLon);

  // Toggle heatmap listener
  const heatmapToggle = document.getElementById("toggle-heatmap");
  if (heatmapToggle) {
    heatmapToggle.addEventListener("change", (e) => {
      if (!state.heatmapLayer) return;
      const countEl = document.getElementById("hud-point-count");
      const card = document.getElementById("hud-legend-card");
      if (e.target.checked) {
        state.map.addLayer(state.heatmapLayer);
        if (countEl) countEl.textContent = `${(state.heatmapRawPoints.length || 25610).toLocaleString()} jobs`;
        if (card) card.style.opacity = "1";
      } else {
        state.map.removeLayer(state.heatmapLayer);
        if (countEl) countEl.textContent = "Off (Hidden)";
        if (card) card.style.opacity = "0.6";
      }
    });
  }
}

function updateUserLocationMarker(lat, lon) {
  state.userLat = lat;
  state.userLon = lon;

  const beaconIcon = L.divIcon({
    className: "custom-user-beacon",
    html: `<div class="user-beacon-halo"></div><div class="user-beacon-core"></div>`,
    iconSize: [16, 16],
    iconAnchor: [8, 8]
  });

  if (state.userMarker) {
    state.userMarker.setLatLng([lat, lon]);
  } else {
    state.userMarker = L.marker([lat, lon], { icon: beaconIcon, zIndexOffset: 1000 }).addTo(state.map);
  }

  // 60fps Commute Radius Circle
  if (state.radiusCircle) {
    state.radiusCircle.setLatLng([lat, lon]);
    state.radiusCircle.setRadius(state.radiusKm * 1000);
  } else {
    state.radiusCircle = L.circle([lat, lon], {
      radius: state.radiusKm * 1000,
      color: "#10b981",
      weight: 2,
      opacity: 0.85,
      fillColor: "#10b981",
      fillOpacity: 0.08,
      interactive: false
    }).addTo(state.map);
  }
}

// Real-time 60fps Commute Radius Slider
function onRadiusSliderChange(val) {
  state.radiusKm = parseFloat(val);
  document.getElementById("radius-val").textContent = `${state.radiusKm} km`;
  
  if (state.radiusCircle) {
    state.radiusCircle.setRadius(state.radiusKm * 1000);
  }
}

function flyToHub(lat, lon, zoom, name) {
  state.map.flyTo([lat, lon], zoom, { duration: 1.2 });
}

// ==========================================================================
// 2. PROFILE MODE SWITCHER (DEMO vs RESUME vs MANUAL)
// ==========================================================================

function setProfileMode(mode) {
  state.profileMode = mode;

  document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
  document.querySelectorAll(".mode-panel").forEach(p => p.classList.add("hidden"));

  const targetRoleBox = document.getElementById("target-role-box");
  const priorityFocusCol = document.getElementById("priority-focus-col");

  if (mode === "resume") {
    document.getElementById("mode-resume-btn").classList.add("active");
    document.getElementById("panel-resume").classList.remove("hidden");
    if (targetRoleBox) targetRoleBox.classList.add("hidden");
    if (priorityFocusCol) priorityFocusCol.classList.add("hidden");
    const prioritySelect = document.getElementById("priority-select");
    if (prioritySelect) prioritySelect.value = "Job Description";
  } else if (mode === "persona") {
    document.getElementById("mode-persona-btn").classList.add("active");
    document.getElementById("panel-persona").classList.remove("hidden");
    if (targetRoleBox) targetRoleBox.classList.remove("hidden");
    if (priorityFocusCol) priorityFocusCol.classList.remove("hidden");
  } else if (mode === "manual") {
    document.getElementById("mode-manual-btn").classList.add("active");
    if (targetRoleBox) targetRoleBox.classList.remove("hidden");
    if (priorityFocusCol) priorityFocusCol.classList.remove("hidden");
  }
}

// ==========================================================================
// 3. PROMINENT RESUME PDF DROPZONE & DUAL-VERSION INSPECTOR
// ==========================================================================

function setupDropzone() {
  const dropzone = document.getElementById("resume-dropzone");
  if (!dropzone) return;

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('dragover');
    }, false);
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0 && files[0].type === "application/pdf") {
      processResumeUpload(files[0]);
    } else {
      alert("Please upload a valid PDF file.");
    }
  });
}

function handleResumeFile(e) {
  const file = e.target.files[0];
  if (file) {
    processResumeUpload(file);
  }
}

async function processResumeUpload(file) {
  const indicator = document.getElementById("resume-processing-indicator");
  const statusText = document.getElementById("processing-status-text");
  const inspector = document.getElementById("resume-inspector-card");

  indicator.classList.remove("hidden");
  inspector.classList.add("hidden");
  statusText.textContent = `Extracting "${file.name}" with PyPDF2 & querying Ollama LLM...`;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/parse-resume", {
      method: "POST",
      body: formData
    });
    const data = await res.json();

    if (data.success) {
      state.rawResumeText = data.raw_text;
      state.aiResumeText = data.ai_extracted;
      state.activeResumeSource = "ai";
      state.currentViewingVersion = "ai";

      // Populate Inspector Card
      document.getElementById("uploaded-filename").textContent = data.filename || file.name;
      document.getElementById("uploaded-wordcount").textContent = `(${data.word_count} words)`;
      
      const badge = document.getElementById("llm-model-badge");
      if (data.llm_success) {
        badge.textContent = `🤖 Ollama (${data.llm_used}) Active`;
        badge.style.background = "rgba(16, 185, 129, 0.2)";
        badge.style.color = "#34d399";
      } else {
        badge.textContent = `ℹ️ Raw Text Extracted`;
        badge.style.background = "rgba(56, 189, 248, 0.2)";
        badge.style.color = "#38bdf8";
      }

      // Update text preview area
      updateResumeInspectorDisplay();
      inspector.classList.remove("hidden");

      // Auto-set matching target description
      document.getElementById("job-desc-input").value = state.aiResumeText;

    } else {
      alert("Error parsing resume: " + (data.detail || "Could not read PDF."));
    }
  } catch (err) {
    alert("Error uploading resume: " + err);
  } finally {
    indicator.classList.add("hidden");
  }
}

function switchResumeView(version) {
  state.currentViewingVersion = version;
  document.getElementById("btn-view-ai").classList.toggle("active", version === "ai");
  document.getElementById("btn-view-raw").classList.toggle("active", version === "raw");
  
  const textarea = document.getElementById("resume-preview-textarea");
  textarea.value = version === "ai" ? state.aiResumeText : state.rawResumeText;
}

function toggleActiveResumeSource() {
  if (state.activeResumeSource === "ai") {
    state.activeResumeSource = "raw";
  } else {
    state.activeResumeSource = "ai";
  }
  updateResumeInspectorDisplay();
}

function updateResumeInspectorDisplay() {
  const bannerIcon = document.getElementById("version-banner-icon");
  const bannerText = document.getElementById("version-banner-text");
  const toggleBtn = document.getElementById("toggle-source-btn");
  const textarea = document.getElementById("resume-preview-textarea");

  if (state.activeResumeSource === "ai") {
    bannerIcon.textContent = "✓";
    bannerText.textContent = "Using AI-Structured Profile for Recommender";
    toggleBtn.textContent = "Switch to Raw Text";
    document.getElementById("job-desc-input").value = state.aiResumeText;
  } else {
    bannerIcon.textContent = "📄";
    bannerText.textContent = "Using Raw Original Text for Recommender";
    toggleBtn.textContent = "Switch to AI Profile";
    document.getElementById("job-desc-input").value = state.rawResumeText;
  }

  // Update viewing textarea
  textarea.value = state.currentViewingVersion === "ai" ? state.aiResumeText : state.rawResumeText;
}

function onPreviewTextEdited(val) {
  if (state.currentViewingVersion === "ai") {
    state.aiResumeText = val;
  } else {
    state.rawResumeText = val;
  }
  
  // Sync with matching input
  if (state.activeResumeSource === state.currentViewingVersion) {
    document.getElementById("job-desc-input").value = val;
  }
}

// ==========================================================================
// 4. DATA LOADING (PRESETS & AMBIENT HEATMAP)
// ==========================================================================

async function loadPresets() {
  try {
    const res = await fetch("/api/presets");
    const data = await res.json();

    const container = document.getElementById("persona-chips-container");
    container.innerHTML = "";

    data.personas.forEach((p, idx) => {
      const chip = document.createElement("button");
      chip.className = `persona-chip ${idx === 0 ? 'active' : ''}`;
      chip.textContent = p.title;
      chip.onclick = () => {
        document.querySelectorAll(".persona-chip").forEach(c => c.classList.remove("active"));
        chip.classList.add("active");
        document.getElementById("job-title-input").value = p.title;
        document.getElementById("job-desc-input").value = p.description;
      };
      container.appendChild(chip);
    });

    if (data.personas.length > 0 && state.profileMode !== "resume") {
      document.getElementById("job-title-input").value = data.personas[0].title;
      document.getElementById("job-desc-input").value = data.personas[0].description;
    }
  } catch (err) {
    console.warn("Could not load presets:", err);
  }
}

async function loadHeatmap() {
  try {
    const res = await fetch("/api/heatmap");
    const data = await res.json();
    
    if (window.L && L.heatLayer && data.points) {
      state.heatmapRawPoints = data.points;
      
      const countEl = document.getElementById("hud-point-count");
      if (countEl) countEl.textContent = `${data.points.length.toLocaleString()} jobs`;

      const pal = HEATMAP_PALETTES[state.activePalette];
      state.heatmapLayer = L.heatLayer(data.points, {
        radius: state.heatmapRadius,
        blur: state.heatmapBlur,
        maxZoom: 16,
        max: state.heatmapDensity,
        minOpacity: 0.18,
        gradient: pal.gradient
      });
      state.heatmapLayer.addTo(state.map);
      updateSpectrumBar(pal.cssGradient);
    }
  } catch (err) {
    console.warn("Heatmap loading error:", err);
  }
}

// ----------------- Interactive Heatmap & HUD Legend Functions ----------------- //

function onHeatmapDensityChange(val) {
  state.heatmapDensity = parseFloat(val);
  const tag = document.getElementById("hud-density-val");
  if (tag) {
    let desc = "Balanced";
    if (state.heatmapDensity <= 3.0) desc = "Intense / High Saturation";
    else if (state.heatmapDensity >= 7.5) desc = "Fine-Grain / Sparse";
    tag.textContent = `${desc} (${state.heatmapDensity.toFixed(1)})`;
  }
  applyHeatmapOptions();
}

function onHeatmapRadiusChange(val) {
  state.heatmapRadius = parseInt(val, 10);
  state.heatmapBlur = Math.round(state.heatmapRadius * 1.15);
  const tag = document.getElementById("hud-radius-val");
  if (tag) tag.textContent = `${state.heatmapRadius} px`;
  applyHeatmapOptions();
}

function setHeatmapPalette(palKey) {
  if (!HEATMAP_PALETTES[palKey]) return;
  state.activePalette = palKey;

  document.querySelectorAll(".palette-chip").forEach(btn => {
    btn.classList.toggle("active", btn.id === `pal-btn-${palKey}`);
  });

  const pal = HEATMAP_PALETTES[palKey];
  updateSpectrumBar(pal.cssGradient);
  applyHeatmapOptions();
}

function applyHeatmapOptions() {
  if (!state.heatmapLayer) return;
  const pal = HEATMAP_PALETTES[state.activePalette];
  state.heatmapLayer.setOptions({
    radius: state.heatmapRadius,
    blur: state.heatmapBlur,
    max: state.heatmapDensity,
    maxZoom: 16,
    minOpacity: 0.18,
    gradient: pal.gradient
  });
}

function updateSpectrumBar(cssGrad) {
  const bar = document.getElementById("hud-spectrum-bar");
  if (bar) bar.style.background = cssGrad;
}

function toggleHudLegend() {
  const card = document.getElementById("hud-legend-card");
  const btn = document.getElementById("hud-collapse-btn");
  if (!card) return;
  card.classList.toggle("collapsed");
  if (btn) btn.textContent = card.classList.contains("collapsed") ? "▴" : "▾";
}

// ==========================================================================
// 5. GEOCODING & USER LOCATION
// ==========================================================================

async function geocodePostal() {
  const query = document.getElementById("postal-input").value.trim();
  if (!query) return;

  const resolvedText = document.getElementById("resolved-location-text");
  resolvedText.textContent = "Locating on Singapore map...";

  try {
    const res = await fetch(`/api/geocode?q=${encodeURIComponent(query)}`);
    const data = await res.json();

    if (data.success) {
      state.userLat = data.lat;
      state.userLon = data.lon;
      state.locationName = data.address.split(",")[0];
      
      resolvedText.textContent = `📍 ${state.locationName}`;
      updateUserLocationMarker(data.lat, data.lon);
      state.map.flyTo([data.lat, data.lon], 13, { duration: 1.0 });
    } else {
      resolvedText.textContent = "Location not found, using current coordinates";
    }
  } catch (e) {
    resolvedText.textContent = "Geocoding error, using default";
  }
}

function locateUserGPS() {
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        updateUserLocationMarker(pos.coords.latitude, pos.coords.longitude);
        state.map.flyTo([pos.coords.latitude, pos.coords.longitude], 13);
        document.getElementById("resolved-location-text").textContent = "📍 Current GPS Location";
      },
      () => alert("GPS access denied or unavailable.")
    );
  }
}

// ==========================================================================
// 6. CORE RECOMMENDATION PIPELINE
// ==========================================================================

async function triggerRecommendation() {
  const btn = document.getElementById("run-radar-btn");
  btn.disabled = true;
  btn.innerHTML = `<span class="btn-radar-icon">⏳</span><span>Querying GraphSAGE Network...</span>`;

  // Determine active title, description and priority based on mode
  let activeTitle = "";
  let activeDesc = "";
  let activePriority = "Job Description";

  if (state.profileMode === "resume") {
    activeDesc = state.activeResumeSource === "ai" ? state.aiResumeText : state.rawResumeText;
    activeTitle = ""; // In resume mode, matching is driven by skills and description embeddings
    activePriority = "Job Description";

    if (!activeDesc || !activeDesc.trim()) {
      alert("Please upload a PDF resume first (or switch to '✨ Sample Personas' / '✍️ Manual Input').");
      btn.disabled = false;
      btn.innerHTML = `<span class="btn-radar-icon">⚡</span><span>Run Spatial Graph Recommender</span>`;
      return;
    }
  } else {
    activeTitle = document.getElementById("job-title-input").value.trim();
    activeDesc = document.getElementById("job-desc-input").value.trim();
    activePriority = document.getElementById("priority-select").value;
  }

  const payload = {
    job_title: activeTitle,
    job_description: activeDesc,
    user_lat: state.userLat,
    user_lon: state.userLon,
    max_distance_km: state.radiusKm,
    priority: activePriority,
    top_k_within: parseInt(document.getElementById("topk-select").value),
    top_k_outside: parseInt(document.getElementById("topk-select").value)
  };

  try {
    const res = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    state.withinJobs = data.within_range || [];
    state.outsideJobs = data.outside_range || [];

    document.getElementById("results-count-badge").textContent = state.withinJobs.length + state.outsideJobs.length;
    document.getElementById("within-count").textContent = state.withinJobs.length;
    document.getElementById("outside-count").textContent = state.outsideJobs.length;

    renderMapJobPins();
    renderResultsList(state.activeSubtab);
    switchTab("results-tab");

  } catch (err) {
    alert("Error fetching recommendations: " + err);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<span class="btn-radar-icon">⚡</span><span>Run Spatial Graph Recommender</span>`;
  }
}

// ==========================================================================
// 7. MAP PINS & COMMUTE VECTORS
// ==========================================================================

function renderMapJobPins() {
  state.jobMarkersGroup.clearLayers();

  // Within Commute Pins (Emerald Green)
  state.withinJobs.forEach((job) => {
    const pinIcon = L.divIcon({
      className: "custom-job-pin-container",
      html: `<div class="custom-job-pin pin-in" data-id="${job.id}">#${job.rank}</div>`,
      iconSize: [28, 28],
      iconAnchor: [14, 14]
    });

    const marker = L.marker([job.lat, job.lon], { icon: pinIcon }).addTo(state.jobMarkersGroup);
    marker.on("click", () => {
      openDrawer(job);
      highlightCard(job.id);
    });
    job._marker = marker;
  });

  // Outside Commute Pins (Amber Orange)
  state.outsideJobs.forEach((job) => {
    const pinIcon = L.divIcon({
      className: "custom-job-pin-container",
      html: `<div class="custom-job-pin pin-out" data-id="${job.id}">+${job.rank}</div>`,
      iconSize: [28, 28],
      iconAnchor: [14, 14]
    });

    const marker = L.marker([job.lat, job.lon], { icon: pinIcon }).addTo(state.jobMarkersGroup);
    marker.on("click", () => {
      openDrawer(job);
      highlightCard(job.id);
    });
    job._marker = marker;
  });

  if (state.jobMarkersGroup.getLayers().length > 0) {
    const group = L.featureGroup([state.jobMarkersGroup, state.userMarker]);
    state.map.flyToBounds(group.getBounds().pad(0.2), { duration: 1.2 });
  }
}

function showCommuteLine(job) {
  if (state.commuteVectorLine) {
    state.map.removeLayer(state.commuteVectorLine);
  }

  const isWithin = job.status === "within";
  const lineColor = isWithin ? "#10b981" : "#f59e0b";

  state.commuteVectorLine = L.polyline(
    [[state.userLat, state.userLon], [job.lat, job.lon]],
    {
      color: lineColor,
      weight: 3,
      opacity: 0.9,
      dashArray: "6, 8"
    }
  ).addTo(state.map);
}

function clearCommuteLine() {
  if (state.commuteVectorLine) {
    state.map.removeLayer(state.commuteVectorLine);
    state.commuteVectorLine = null;
  }
}

// ==========================================================================
// 8. RESULTS LIST & CARDS
// ==========================================================================

function renderResultsList(type) {
  const container = document.getElementById("results-list");
  container.innerHTML = "";

  const jobs = type === "within" ? state.withinJobs : state.outsideJobs;

  if (jobs.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">${type === 'within' ? '📍' : '🚗'}</div>
        <div class="empty-title">No jobs in this category</div>
        <div class="empty-desc">${type === 'within' ? 'Try expanding the commute distance slider.' : 'All high matches are within your commute range!'}</div>
      </div>
    `;
    return;
  }

  jobs.forEach((job) => {
    const isWithin = job.status === "within";
    const badgeClass = isWithin ? "badge-in" : "badge-out";
    const badgeText = isWithin ? `RANK #${job.rank} • WITHIN RANGE` : `RANK +${job.rank} • OUTSIDE RANGE`;

    const card = document.createElement("div");
    card.className = "job-item-card";
    card.id = `card-${job.id}`;

    card.innerHTML = `
      <div class="job-card-header">
        <span class="${badgeClass}">${badgeText}</span>
        <div class="job-score-val">${job.final_score.toFixed(3)}</div>
      </div>
      <div class="job-card-title">${job.title}</div>
      <div class="job-card-company">🏢 ${job.company}</div>
      
      <div class="job-card-meta">
        <span class="meta-chip meta-dist">📍 ${job.distance_km} km away</span>
        <span class="meta-chip">💼 ${job.job_type.join(", ")}</span>
        <span class="meta-chip">🏡 Remote: ${job.is_remote ? 'Yes' : 'No'}</span>
      </div>

      <div class="match-bar-bg">
        <div class="match-bar-fill" style="width: ${(job.similarity_score * 100).toFixed(0)}%;"></div>
      </div>
      <div class="match-bar-labels">
        <span>Semantic: <strong>${job.similarity_score.toFixed(3)}</strong> (${(job.sim_weight * 100).toFixed(0)}%)</span>
        <span>GraphSAGE: <strong>${job.graph_score.toFixed(3)}</strong></span>
      </div>

      <div class="card-actions-row">
        ${job.job_url_direct ? `<a class="card-btn card-btn-primary" href="${job.job_url_direct}" target="_blank" onclick="event.stopPropagation()">Apply Direct ↗</a>` : ''}
        ${job.job_url ? `<a class="card-btn card-btn-outline" href="${job.job_url}" target="_blank" onclick="event.stopPropagation()">Platform Link ↗</a>` : ''}
      </div>
    `;

    card.addEventListener("mouseenter", () => {
      showCommuteLine(job);
      if (job._marker) {
        job._marker._icon.querySelector(".custom-job-pin")?.classList.add("active-hover");
      }
    });

    card.addEventListener("mouseleave", () => {
      clearCommuteLine();
      if (job._marker) {
        job._marker._icon.querySelector(".custom-job-pin")?.classList.remove("active-hover");
      }
    });

    card.addEventListener("click", () => {
      openDrawer(job);
      state.map.flyTo([job.lat, job.lon], 14, { duration: 0.8 });
    });

    container.appendChild(card);
  });
}

function highlightCard(jobId) {
  document.querySelectorAll(".job-item-card").forEach(c => c.classList.remove("active-pin"));
  const card = document.getElementById(`card-${jobId}`);
  if (card) {
    card.classList.add("active-pin");
    card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

// ==========================================================================
// 9. JOB DETAIL DRAWER (MODAL SLIDE-OVER)
// ==========================================================================

function openDrawer(job) {
  const drawer = document.getElementById("detail-drawer");
  const rankElem = document.getElementById("drawer-rank");
  const content = document.getElementById("drawer-content");

  const isWithin = job.status === "within";
  rankElem.textContent = isWithin ? `RANK #${job.rank} • WITHIN COMMUTE RADIUS` : `RANK +${job.rank} • OUTSIDE COMMUTE RADIUS`;
  rankElem.style.color = isWithin ? "#10b981" : "#f59e0b";

  content.innerHTML = `
    <div>
      <h2 style="font-family: var(--font-heading); font-size: 22px; font-weight: 800; color: #ffffff; margin-bottom: 4px;">${job.title}</h2>
      <div style="font-size: 15px; font-weight: 600; color: #38bdf8;">🏢 ${job.company}</div>
    </div>

    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
      <span class="meta-chip meta-dist" style="font-size: 12px; padding: 4px 10px;">📍 ${job.distance_km} km Commute</span>
      <span class="meta-chip" style="font-size: 12px; padding: 4px 10px;">💼 ${job.job_type.join(", ")}</span>
      <span class="meta-chip" style="font-size: 12px; padding: 4px 10px;">🏡 Remote: ${job.is_remote ? 'Yes' : 'No'}</span>
    </div>

    <div class="section-box" style="margin-top: 6px;">
      <div class="section-title">📍 Physical Location & Address</div>
      <div style="font-size: 13px; color: #e2e8f0; line-height: 1.5;">${job.address || 'Singapore'}</div>
      <div style="font-size: 11px; color: var(--text-dim); margin-top: 4px;">Coordinates: (${job.lat.toFixed(4)}, ${job.lon.toFixed(4)})</div>
    </div>

    <div class="section-box">
      <div class="section-title">📊 GraphSAGE & NLP Match Decomposition</div>
      
      <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 10px;">
        <span style="font-size: 13px; color: var(--text-muted);">Composite Recommendation Score:</span>
        <span style="font-family: var(--font-heading); font-size: 24px; font-weight: 800; color: #38bdf8;">${job.final_score.toFixed(3)}</span>
      </div>

      <div style="margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; font-size: 11px; color: var(--text-muted); margin-bottom: 4px;">
          <span>Semantic Resume/Title Similarity (${(job.sim_weight * 100).toFixed(0)}%)</span>
          <strong>${job.similarity_score.toFixed(3)}</strong>
        </div>
        <div class="match-bar-bg" style="height: 8px;">
          <div class="match-bar-fill" style="width: ${(job.similarity_score * 100).toFixed(0)}%;"></div>
        </div>
      </div>

      <div>
        <div style="display: flex; justify-content: space-between; font-size: 11px; color: var(--text-muted); margin-bottom: 4px;">
          <span>Graph Network Topology & Centrality (${((1 - job.sim_weight) * 100).toFixed(0)}%)</span>
          <strong>${job.graph_score.toFixed(3)}</strong>
        </div>
        <div class="match-bar-bg" style="height: 8px;">
          <div class="match-bar-fill" style="width: ${(job.graph_score * 100).toFixed(0)}%; background: linear-gradient(90deg, #10b981, #06b6d4);"></div>
        </div>
      </div>
    </div>

    <div style="margin-top: auto; display: flex; gap: 10px; padding-top: 14px;">
      ${job.job_url_direct ? `<a class="card-btn card-btn-primary" style="flex:1; text-align:center; padding: 10px;" href="${job.job_url_direct}" target="_blank">Direct Apply ↗</a>` : ''}
      ${job.job_url ? `<a class="card-btn card-btn-outline" style="flex:1; text-align:center; padding: 10px;" href="${job.job_url}" target="_blank">Platform Posting ↗</a>` : ''}
    </div>
  `;

  drawer.classList.add("open");
}

function closeDrawer() {
  document.getElementById("detail-drawer").classList.remove("open");
}

// ==========================================================================
// 10. DOCK TABS & UI HELPERS
// ==========================================================================

function switchTab(tabId) {
  document.querySelectorAll(".dock-tab").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

  const tabBtn = document.querySelector(`.dock-tab[data-tab="${tabId}"]`);
  const pane = document.getElementById(tabId);

  if (tabBtn) tabBtn.classList.add("active");
  if (pane) pane.classList.add("active");
}

function switchResultsSubtab(type) {
  state.activeSubtab = type;
  document.getElementById("subtab-within").classList.toggle("active", type === "within");
  document.getElementById("subtab-outside").classList.toggle("active", type === "outside");
  renderResultsList(type);
}

function toggleDock() {
  const dock = document.getElementById("command-dock");
  const btn = document.getElementById("collapse-btn");
  dock.classList.toggle("collapsed");
  btn.textContent = dock.classList.contains("collapsed") ? "▶" : "◀";
}

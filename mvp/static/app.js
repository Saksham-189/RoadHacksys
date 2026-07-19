const state = {
  bootstrap: null,
  view: "extraction",
  map: null,
  layerData: new Map(),
  mapLayers: new Map(),
  resultLayers: [],
  actionLayer: null,
  actions: [],
  actionMode: "inspect",
  runMode: "preview",
  selectedFeature: null,
};

const byId = (id) => document.getElementById(id);
const pct = (value) => `${(100 * Number(value || 0)).toFixed(2)}%`;
const fixed = (value, digits = 3) => Number(value || 0).toFixed(digits);

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function api(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }
  return response.json();
}

function setHealth(payload) {
  const chip = byId("health-chip");
  chip.className = `health-chip ${payload.status}`;
  chip.querySelector("b").textContent =
    payload.status === "ready" ? "System ready" :
    payload.status === "degraded" ? "Fallback ready" : payload.status;
}

function initMap() {
  state.map = L.map("map", {
    zoomControl: true,
    attributionControl: false,
    preferCanvas: true,
  });
  state.map.createPane("roads");
  state.map.getPane("roads").style.zIndex = 410;
  state.map.createPane("criticality");
  state.map.getPane("criticality").style.zIndex = 430;
  state.map.createPane("impacts");
  state.map.getPane("impacts").style.zIndex = 450;
  state.actionLayer = L.layerGroup().addTo(state.map);
  state.map.on("click", (event) => {
    if (state.view !== "disruption" || state.actionMode !== "close_circle") return;
    const radius = Number(byId("radius-input").value);
    addAction({
      action: "close_circle",
      longitude: event.latlng.lng,
      latitude: event.latlng.lat,
      radius_m: radius,
    });
    L.circle(event.latlng, {
      radius,
      color: "#287aa3",
      fillColor: "#55b7d8",
      fillOpacity: 0.22,
      weight: 2,
    }).addTo(state.actionLayer);
  });
}

function flowColor(value) {
  if (value >= 1) return "#bf3b32";
  if (value >= 0.7) return "#e7a63c";
  if (value >= 0.35) return "#8cae43";
  return "#24764c";
}

function criticalColor(value) {
  if (value >= 0.25) return "#b92525";
  if (value >= 0.1) return "#e56a2c";
  if (value > 0) return "#e7b64a";
  return "#748087";
}

function layerOptions(name) {
  if (name === "node_criticality") {
    return {
      pane: "criticality",
      pointToLayer: (feature, latlng) => {
        const value = Number(feature.properties.flow_criticality || 0);
        return L.circleMarker(latlng, {
          pane: "criticality",
          radius: value > 0 ? 4 + 13 * Math.min(value / 0.35, 1) : 2,
          color: "#ffffff",
          weight: value > 0 ? 1 : 0,
          fillColor: criticalColor(value),
          fillOpacity: value > 0 ? 0.92 : 0.28,
        });
      },
      onEachFeature: bindFeature,
    };
  }
  return {
    pane: name.includes("criticality") ? "criticality" : "roads",
    style: (feature) => {
      const properties = feature.properties || {};
      if (name === "relative_flow") {
        return {
          color: flowColor(Number(properties.vc_ratio || 0)),
          weight: Math.min(1.1 + Number(properties.vc_ratio || 0) * 2.1, 5),
          opacity: 0.82,
        };
      }
      if (name === "edge_criticality") {
        const value = Number(properties.flow_criticality || 0);
        return {
          color: criticalColor(value),
          weight: value > 0 ? 2.4 + value * 8 : 1,
          opacity: value > 0 ? 0.9 : 0.15,
        };
      }
      return { color: "#879198", weight: 1.1, opacity: 0.62 };
    },
    onEachFeature: bindFeature,
  };
}

function bindFeature(feature, layer) {
  const properties = feature.properties || {};
  const label = properties.feature === "node"
    ? `Node ${properties.node_id}`
    : `Road ${properties.source}–${properties.target}`;
  layer.bindTooltip(label, { sticky: true, direction: "top" });
  layer.on("click", (event) => {
    L.DomEvent.stopPropagation(event);
    inspectFeature(feature);
    if (state.view !== "disruption") return;
    if (state.actionMode === "close_nodes" && properties.feature === "node") {
      addAction({ action: "close_nodes", node_ids: [String(properties.node_id)] });
      L.circleMarker(event.latlng, { radius: 10, color: "#b92525", weight: 3, fillOpacity: 0 }).addTo(state.actionLayer);
    }
    if (state.actionMode === "close_edges" && properties.feature === "edge") {
      addAction({
        action: "close_edges",
        edges: [{ source: String(properties.source), target: String(properties.target) }],
      });
      L.polyline(layer.getLatLngs(), { color: "#b92525", weight: 6, opacity: 0.9 }).addTo(state.actionLayer);
    }
    if (state.actionMode === "capacity_derating" && properties.feature === "edge") {
      addAction({
        action: "capacity_derating",
        edges: [{
          source: String(properties.source),
          target: String(properties.target),
          capacity_factor: Number(byId("capacity-input").value) / 100,
        }],
      });
      L.polyline(layer.getLatLngs(), { color: "#e56a2c", weight: 6, dashArray: "7 5" }).addTo(state.actionLayer);
    }
  });
}

function inspectFeature(feature) {
  state.selectedFeature = feature;
  const properties = feature.properties || {};
  const rows = Object.entries(properties)
    .filter(([, value]) => ["string", "number", "boolean"].includes(typeof value))
    .slice(0, 9)
    .map(([key, value]) => `<dt>${escapeHtml(key.replaceAll("_", " "))}</dt><dd>${escapeHtml(typeof value === "number" ? Number(value).toFixed(3) : value)}</dd>`)
    .join("");
  const inspector = byId("selection-inspector");
  inspector.className = "inspector";
  inspector.innerHTML = `<dl>${rows}</dl>`;
}

async function loadLayer(name, visible = true) {
  if (!state.layerData.has(name)) {
    const data = await api(`/api/v1/layers/${name}`);
    state.layerData.set(name, data);
  }
  if (state.mapLayers.has(name)) {
    const old = state.mapLayers.get(name);
    if (visible && !state.map.hasLayer(old)) old.addTo(state.map);
    if (!visible && state.map.hasLayer(old)) state.map.removeLayer(old);
    return old;
  }
  const layer = L.geoJSON(state.layerData.get(name), layerOptions(name));
  state.mapLayers.set(name, layer);
  if (visible) layer.addTo(state.map);
  return layer;
}

function clearMapLayers() {
  for (const layer of state.mapLayers.values()) {
    if (state.map.hasLayer(layer)) state.map.removeLayer(layer);
  }
}

async function configureViewLayers() {
  clearResultLayers();
  clearMapLayers();
  if (state.view === "criticality") {
    const checks = document.querySelectorAll("[data-layer]");
    for (const input of checks) {
      if (input.checked) await loadLayer(input.dataset.layer, true);
    }
  } else if (state.view === "disruption") {
    await loadLayer("relative_flow", true);
    await loadLayer("node_criticality", true);
  }
}

function switchView(view) {
  state.view = view;
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.view === view));
  document.querySelectorAll(".side-view").forEach((panel) => panel.classList.toggle("active", panel.dataset.panel === view));
  const extraction = view === "extraction";
  byId("inference-workspace").classList.toggle("active", extraction);
  byId("map-workspace").classList.toggle("active", !extraction);
  byId("metrics-strip").classList.toggle("hidden", view !== "disruption");
  if (!extraction) {
    setTimeout(() => {
      state.map.invalidateSize();
      state.map.fitBounds(state.bootstrap.bounds, { padding: [20, 20] });
      configureViewLayers().catch(showError);
    }, 20);
  }
  if (window.innerWidth <= 960) byId("sidebar").classList.remove("open");
}

function addAction(action) {
  state.actions.push(action);
  byId("preset-select").value = "";
  renderActions();
}

function actionLabel(action) {
  if (action.action === "close_nodes") return `Close node ${action.node_ids.join(", ")}`;
  if (action.action === "close_edges") return `Close road ${action.edges[0].source}–${action.edges[0].target}`;
  if (action.action === "capacity_derating") return `Road ${action.edges[0].source}–${action.edges[0].target} at ${pct(action.edges[0].capacity_factor)}`;
  return `Flood ${Math.round(action.radius_m)} m at ${action.latitude.toFixed(4)}, ${action.longitude.toFixed(4)}`;
}

function renderActions() {
  const target = byId("actions-list");
  if (!state.actions.length) {
    target.className = "actions-list empty";
    target.textContent = "No actions selected";
    return;
  }
  target.className = "actions-list";
  target.innerHTML = state.actions.map((action, index) =>
    `<div class="action-item"><span><b>${index + 1}</b> ${escapeHtml(actionLabel(action))}</span></div>`
  ).join("");
}

function clearActions() {
  state.actions = [];
  state.actionLayer.clearLayers();
  byId("preset-select").value = "";
  renderActions();
}

function clearResultLayers() {
  for (const layer of state.resultLayers) {
    if (state.map.hasLayer(layer)) state.map.removeLayer(layer);
  }
  state.resultLayers = [];
}

function resultStyle(name, feature) {
  const properties = feature.properties || {};
  if (name === "affected_zones.geojson") {
    const score = Number(properties.impact_score || 0);
    return { color: "#b92525", fillColor: "#e85d25", fillOpacity: Math.min(0.15 + score * 2.5, 0.78), weight: 1 };
  }
  if (name === "edge_rerouting.geojson") {
    const burden = Number(properties.rerouting_burden || 0);
    return { color: burden > 0 ? "#e85d25" : "#889299", weight: burden > 0 ? 2.5 + Math.min(burden, 5) : 1, opacity: burden > 0 ? 0.95 : 0.2 };
  }
  if (name === "route_examples.geojson") {
    return { color: properties.route_type === "disrupted" ? "#b92525" : "#287aa3", weight: 4, opacity: 0.8, dashArray: properties.route_type === "baseline" ? "7 5" : null };
  }
  return { color: "#657179", weight: 1, opacity: 0.35 };
}

async function showResult(result) {
  clearResultLayers();
  updateMetrics(result.summary, result.artifacts);
  const wanted = ["disrupted_network.geojson", "edge_rerouting.geojson", "affected_zones.geojson", "route_examples.geojson"];
  for (const name of wanted) {
    const url = result.artifacts[name];
    if (!url) continue;
    const data = await api(url);
    const layer = L.geoJSON(data, {
      pane: "impacts",
      style: (feature) => resultStyle(name, feature),
      pointToLayer: (feature, latlng) => L.circleMarker(latlng, { pane: "impacts", radius: 5, color: "#b92525" }),
    }).addTo(state.map);
    state.resultLayers.push(layer);
  }
}

function updateMetrics(summary, artifacts = {}) {
  byId("metric-served").textContent = pct(summary.served_demand_ratio);
  byId("metric-disconnected").textContent = pct(summary.disconnected_demand_ratio);
  byId("metric-resilience").textContent = fixed(summary.service_adjusted_resilience);
  byId("metric-band").textContent = summary.resilience_band || "Measured";
  byId("metric-path").textContent = pct(summary.path_length_increase);
  byId("metric-time").textContent = pct(summary.travel_time_increase);
  byId("metric-affected").textContent = pct(summary.affected_demand_ratio);
  byId("metric-runtime").textContent = `${fixed(summary.runtime_seconds, 2)} s · ${summary.mode || "cached"}`;
  const links = Object.entries(artifacts).filter(([name]) => ["summary.json", "od_impacts.csv", "affected_zones.geojson"].includes(name));
  byId("download-menu").innerHTML = links.length
    ? `<a href="${links[0][1]}" title="Download result"><i data-lucide="download"></i></a>`
    : "";
  lucide.createIcons();
}

async function loadPreset(scenarioId) {
  if (!scenarioId) {
    clearResultLayers();
    return;
  }
  setMapMessage(`Loading ${scenarioId}…`);
  const result = await api(`/api/v1/scenarios/${scenarioId}`);
  await showResult(result);
  setMapMessage(`${scenarioId} loaded`, 1800);
}

async function runScenario() {
  if (!state.actions.length) {
    showError(new Error("Add at least one disruption action or choose a preset."));
    return;
  }
  const button = byId("run-scenario");
  button.disabled = true;
  setMapMessage(state.runMode === "preview" ? "Calculating preview…" : "Running exact assignment…");
  const payload = {
    name: "Interactive map disruption",
    hazard_type: state.actions.some((action) => action.action === "close_circle") ? "flood" : "interactive",
    actions: state.actions,
  };
  try {
    if (state.runMode === "preview") {
      const result = await api("/api/v1/simulations/preview", { method: "POST", body: JSON.stringify(payload) });
      await showResult(result);
    } else {
      const job = await api("/api/v1/simulations/exact", { method: "POST", body: JSON.stringify(payload) });
      await pollJob(job.job_id);
    }
    setMapMessage("Simulation complete", 1800);
  } catch (error) {
    showError(error);
  } finally {
    button.disabled = false;
  }
}

async function pollJob(jobId) {
  while (true) {
    await new Promise((resolve) => setTimeout(resolve, 650));
    const job = await api(`/api/v1/jobs/${jobId}`);
    setMapMessage(job.status === "queued" ? "Exact simulation queued…" : "Exact MSA rerouting…");
    if (job.status === "completed") {
      await showResult(job.result);
      return;
    }
    if (job.status === "failed") throw new Error(job.error || "Exact simulation failed");
  }
}

function setMapMessage(message, timeout = 0) {
  const element = byId("map-message");
  element.textContent = message;
  element.classList.remove("hidden");
  if (timeout) setTimeout(() => element.classList.add("hidden"), timeout);
}

function showError(error) {
  setMapMessage(error.message || String(error), 5000);
  byId("inference-status").textContent = error.message || String(error);
}

async function runInference() {
  const button = byId("run-inference");
  button.disabled = true;
  byId("inference-status").textContent = "Running E013…";
  try {
    const payload = {
      tile_id: byId("tile-select").value,
      occlusion: byId("occlusion-select").value,
      seed: 42,
    };
    const result = await api("/api/v1/inference", { method: "POST", body: JSON.stringify(payload) });
    renderInference(result);
  } catch (error) {
    showError(error);
  } finally {
    button.disabled = false;
  }
}

function renderInference(result) {
  byId("inference-title").textContent = `${result.tile_id} · ${result.occlusion}`;
  const metrics = result.metrics;
  byId("inference-status").textContent = metrics
    ? `${result.mode} on ${result.device} · IoU ${fixed(metrics.iou)} · Dice ${fixed(metrics.dice)} · Recall ${fixed(metrics.recall)}`
    : `${result.mode} · cached diagnostic`;
  const titles = { input: "RGB input", target: "OSM target", probability: "Road probability", overlay: "Prediction overlay", diagnostic: "Cached four-panel diagnostic" };
  const entries = Object.entries(result.panels);
  const target = byId("inference-panels");
  target.className = "inference-panels";
  target.innerHTML = entries.map(([name, url]) =>
    `<article class="image-panel ${name === "diagnostic" ? "diagnostic" : ""}"><h3>${titles[name]}</h3><img src="${url}" alt="${titles[name]}"></article>`
  ).join("");
}

async function initialize() {
  initMap();
  const [health, bootstrap, tiles] = await Promise.all([
    api("/api/v1/health"),
    api("/api/v1/bootstrap"),
    api("/api/v1/inference/tiles"),
  ]);
  state.bootstrap = bootstrap;
  setHealth(health);
  window.setInterval(async () => {
    try {
      setHealth(await api("/api/v1/health"));
    } catch {
      // Keep the last known state during brief server restarts.
    }
  }, 5000);
  state.map.fitBounds(bootstrap.bounds, { padding: [20, 20] });
  byId("tile-select").innerHTML = tiles.slice(0, 80).map((tile) =>
    `<option value="${escapeHtml(tile.tile_id)}" ${tile.default ? "selected" : ""}>${escapeHtml(tile.tile_id)} · road ${(100 * tile.road_pixel_ratio).toFixed(1)}%</option>`
  ).join("");
  byId("preset-select").innerHTML += bootstrap.scenarios.map((scenario) =>
    `<option value="${scenario.scenario_id}">${scenario.scenario_id} · ${escapeHtml(scenario.name)}</option>`
  ).join("");
  updateMetrics({
    served_demand_ratio: 1,
    disconnected_demand_ratio: 0,
    service_adjusted_resilience: 1,
    resilience_band: "Baseline",
    path_length_increase: 0,
    travel_time_increase: 0,
    affected_demand_ratio: 0,
    runtime_seconds: 0,
  });
}

document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => switchView(tab.dataset.view)));
document.querySelectorAll("[data-layer]").forEach((input) => input.addEventListener("change", () => loadLayer(input.dataset.layer, input.checked).catch(showError)));
document.querySelectorAll(".action-button").forEach((button) => button.addEventListener("click", () => {
  state.actionMode = button.dataset.action;
  document.querySelectorAll(".action-button").forEach((item) => item.classList.toggle("active", item === button));
}));
document.querySelectorAll("[data-run-mode]").forEach((button) => button.addEventListener("click", () => {
  state.runMode = button.dataset.runMode;
  document.querySelectorAll("[data-run-mode]").forEach((item) => item.classList.toggle("active", item === button));
}));
byId("radius-input").addEventListener("input", (event) => byId("radius-value").textContent = `${event.target.value} m`);
byId("capacity-input").addEventListener("input", (event) => byId("capacity-value").textContent = `${event.target.value}%`);
byId("preset-select").addEventListener("change", (event) => loadPreset(event.target.value).catch(showError));
byId("run-inference").addEventListener("click", runInference);
byId("run-scenario").addEventListener("click", runScenario);
byId("clear-actions").addEventListener("click", clearActions);
byId("undo-action").addEventListener("click", () => {
  state.actions.pop();
  state.actionLayer.clearLayers();
  renderActions();
});
byId("reset-map").addEventListener("click", () => state.map.fitBounds(state.bootstrap.bounds, { padding: [20, 20] }));
byId("sidebar-toggle").addEventListener("click", () => byId("sidebar").classList.toggle("open"));

lucide.createIcons();
initialize().catch((error) => {
  setHealth({ status: "missing" });
  showError(error);
});

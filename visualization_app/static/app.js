const state = {
  graphs: [],
  activeDataset: null,
  activeSourceSet: "",
  activeSlug: null,
  datasetFilter: "",
  summary: null,
  steeringOptions: null,
  storageRecords: [],
  selectedStoredSupernodes: [],
};

const el = (id) => document.getElementById(id);

function graphKey(graph) {
  return [graph.dataset, graph.source_set, graph.slug].filter(Boolean).join("/");
}

function graphApiPath(dataset, slug, suffix = "", sourceSet = "") {
  const path = `/api/graphs/${encodeURIComponent(dataset)}/${encodeURIComponent(slug)}${suffix}`;
  if (!sourceSet) return path;
  return `${path}?source_set=${encodeURIComponent(sourceSet)}`;
}

function defaultShapPath(dataset, sourceSet = "") {
  if (sourceSet) return `dataset/${dataset}/${sourceSet}/shap_values.json`;
  if (dataset === "analogies" || dataset === "multihop") {
    return `dataset/${dataset}/shap_values.json`;
  }
  return "";
}

function isActiveGraph(graph) {
  return (
    graph.dataset === state.activeDataset &&
    (graph.source_set || "") === state.activeSourceSet &&
    graph.slug === state.activeSlug
  );
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || response.statusText);
  }
  return payload;
}

function numberValue(id) {
  const value = el(id).value;
  return value === "" ? null : Number(value);
}

function filteredGraphs() {
  if (!state.datasetFilter) return state.graphs;
  return state.graphs.filter((graph) => graph.dataset === state.datasetFilter);
}

function renderGraphs() {
  const list = el("graphList");
  list.innerHTML = "";
  const graphs = filteredGraphs();
  if (!graphs.length) {
    list.innerHTML = '<div class="muted">No local graphs found.</div>';
    return;
  }

  graphs.forEach((graph) => {
    const button = document.createElement("button");
    button.className = `graph-item ${isActiveGraph(graph) ? "active" : ""}`;
    button.innerHTML = `
      <div class="graph-name">${graphKey(graph)}</div>
      <div class="graph-meta">${graph.node_count} nodes - ${graph.link_count} edges - ${graph.has_pt ? ".pt" : "view-only"}${graph.has_summary ? " - summary" : ""}</div>
    `;
    button.addEventListener("click", () =>
      selectGraph(graph.dataset, graph.slug, false, graph.source_set || ""),
    );
    list.appendChild(button);
  });
}

async function refreshGraphs() {
  const payload = await api("/api/graphs");
  state.graphs = payload.graphs;
  renderGraphs();
}

function activeGraph() {
  return state.graphs.find((graph) => isActiveGraph(graph));
}

async function selectGraph(dataset, slug, useSummary = false, sourceSet = "") {
  state.activeDataset = dataset;
  state.activeSourceSet = sourceSet || "";
  state.activeSlug = slug;
  state.summary = null;
  renderGraphs();
  const graph = activeGraph();
  el("activeTitle").textContent = [dataset, state.activeSourceSet, slug].filter(Boolean).join("/");
  el("activeMeta").textContent = graph ? `${graph.scan || "unknown scan"} - ${graph.prompt || "no prompt metadata"}` : "";
  el("summaryBtn").disabled = !graph?.has_pt;
  el("showSummaryBtn").disabled = !graph?.has_summary;
  el("steerBtn").disabled = !graph?.has_summary;
  el("showRawBtn").disabled = false;
  state.steeringOptions = null;
  await openViewer(useSummary);
}

async function openViewer(useSummary = false) {
  if (!state.activeDataset || !state.activeSlug) return;
  const payload = await api(
    `${graphApiPath(
      state.activeDataset,
      state.activeSlug,
      "/viewer-url",
      state.activeSourceSet,
    )}${state.activeSourceSet ? "&" : "?"}summary=${useSummary ? "true" : "false"}`,
  );
  el("viewerFrame").src = payload.url;
  state.summary = payload.summary;
  renderSummaryDetails(payload.summary);
  renderClusterViz(payload.summary);
}

function renderSummaryDetails(summaryPayload) {
  const container = el("summaryDetails");
  if (!summaryPayload?.summary?.length) {
    container.className = "summary-details muted";
    container.textContent = "Summary roles and descriptions appear here.";
    return;
  }

  container.className = "summary-details";
  container.innerHTML = "";
  summaryPayload.summary.forEach((item) => {
    const card = document.createElement("div");
    card.className = "summary-card";
    card.innerHTML = `
      <strong>${item.name}</strong>
      <span>${item.role || item.type}</span>
      <div>${item.description || `${item.members.length} member node(s)`}</div>
    `;
    container.appendChild(card);
  });
}

function renderClusterViz(summaryPayload) {
  const panel = el("clusterVizPanel");
  const frame = el("clusterVizFrame");
  const figureHtml = summaryPayload?.figure_html || "";
  panel.hidden = !figureHtml;
  frame.srcdoc = figureHtml;
}

function updateProgress(progressEl, textEl, progress) {
  if (!progressEl || !textEl) return;
  const value = Math.max(0, Math.min(1, Number(progress) || 0));
  progressEl.value = value;
  textEl.textContent = `${Math.round(value * 100)}%`;
}

async function pollJob(job, statusEl, onComplete, progressEl = null, progressTextEl = null) {
  statusEl.textContent = `${job.kind}: ${job.status}`;
  updateProgress(progressEl, progressTextEl, job.progress);
  while (true) {
    await new Promise((resolve) => setTimeout(resolve, 1400));
    const payload = await api(`/api/jobs/${job.id}`);
    const current = payload.job;
    statusEl.textContent = `${current.kind}: ${current.message}`;
    updateProgress(progressEl, progressTextEl, current.progress);
    if (current.status === "completed") {
      statusEl.textContent = "Completed.";
      updateProgress(progressEl, progressTextEl, 1);
      await onComplete(current.result);
      return current.result;
    }
    if (current.status === "failed") {
      statusEl.textContent = current.error || "Failed.";
      throw new Error(current.error || "Job failed");
    }
  }
}

function tokenPills(tokens) {
  return tokens.map((token) => `<span class="pill">${token}</span>`).join("");
}

function nextTokenPills(tokens) {
  return tokens
    .map((item) => `<span class="pill">${item.token} ${item.probability.toFixed(3)}</span>`)
    .join("");
}

function generationPayload() {
  return {
    prompt: el("genPrompt").value,
    slug: el("genSlug").value,
    model_name: el("genModel").value,
    transcoder: el("genTranscoder").value,
    dtype: el("genDtype").value,
    backend: el("genBackend").value,
    max_n_logits: numberValue("genMaxLogits"),
    desired_logit_prob: numberValue("genDesiredProb"),
    max_feature_nodes: numberValue("genMaxNodes"),
    batch_size: numberValue("genBatch"),
    node_threshold: numberValue("genNodeThreshold"),
    edge_threshold: numberValue("genEdgeThreshold"),
    qwen_system: el("genQwenSystem").value,
    qwen_assistant: el("genQwenAssistant").value,
    qwen_enable_thinking: el("genQwenThinking").checked,
  };
}

function syncThetaModeInputs() {
  const adaptive = el("sumThetaMode").value === "adaptive";
  el("sumThetaPercentileRow").hidden = !adaptive;
  el("sumThetaFixedRow").hidden = adaptive;
}

function summaryThetaValue() {
  if (el("sumThetaMode").value === "adaptive") {
    return `p${numberValue("sumThetaPercentile")}`;
  }
  return numberValue("sumThetaFixed");
}

function summaryPayload() {
  return {
    logit_weights: el("sumLogitWeights").value,
    token_weights_source: el("sumTokenWeights").value,
    token_attr_model: el("sumTokenAttrModel").value,
    token_attr_normalize: el("sumTokenAttrNormalize").value,
    entmax_alpha: numberValue("sumEntmaxAlpha"),
    shap_values_path: el("sumShapValuesPath").value,
    device: el("sumDevice").value,
    node_threshold: numberValue("sumNodeThreshold"),
    edge_threshold: numberValue("sumEdgeThreshold"),
    combine_method: el("sumCombine").value,
    normalization: el("sumNormalization").value,
    alpha: numberValue("sumAlpha"),
    keep_all_tokens_and_logits: el("sumKeepAll").checked,
    filter_act_density: el("sumFilterAct").checked,
    max_layer_span: numberValue("sumLayerSpan"),
    max_sn: numberValue("sumMaxSn") || null,
    eps_causal: numberValue("sumEpsCausal"),
    theta: summaryThetaValue(),
    ilp_time_limit: numberValue("sumIlpTime"),
    label_supernodes: el("sumLabelGraph").checked,
    label_model: el("sumLabelModel").value,
    label_temperature: numberValue("sumLabelTemp"),
    thinking_effort: el("sumThinking").value,
  };
}

async function openSteeringDialog() {
  if (!state.activeDataset || !state.activeSlug) return;
  const status = el("steeringStatus");
  const list = el("steeringNodeList");
  const result = el("steeringResult");
  state.selectedStoredSupernodes = [];
  status.textContent = "Loading steering options...";
  result.className = "steering-result muted";
  result.textContent = "Steering output appears here.";
  el("steeringFigure").innerHTML = "";
  el("steeringProgressRow").hidden = true;
  el("supernodeStoragePanel").hidden = true;
  el("steeringDialog").showModal();
  try {
    const payload = await api(
      graphApiPath(
        state.activeDataset,
        state.activeSlug,
        "/steering-options",
        state.activeSourceSet,
      ),
    );
    state.steeringOptions = payload;
    el("steerModel").value = payload.model_name || "";
    el("steerTranscoder").value = payload.transcoder || "";
    renderSteeringNodes(payload.supernodes || []);
    status.textContent = `${payload.supernodes.length} feature supernode(s) available.`;
  } catch (error) {
    list.className = "steering-node-list muted";
    list.textContent = error.message;
    status.textContent = error.message;
  }
}

function defaultStoredTargetPos() {
  const tokens = state.steeringOptions?.prompt_tokens || [];
  return Math.max(0, tokens.length - 1);
}

function captureSteeringSelection() {
  const checked = [];
  const factors = {};
  el("steeringNodeList")
    .querySelectorAll("input[type='checkbox'][data-steer-name]")
    .forEach((checkbox) => {
      if (checkbox.checked) checked.push(checkbox.dataset.steerName);
    });
  el("steeringNodeList")
    .querySelectorAll("input[data-steer-factor]")
    .forEach((input) => {
      factors[input.dataset.steerFactor] = input.value;
    });
  return { checked, factors };
}

function renderSteeringNodes(supernodes) {
  const preSelection = captureSteeringSelection();
  const hadPriorNodes =
    el("steeringNodeList").querySelectorAll("input[data-steer-name]").length > 0;
  const list = el("steeringNodeList");
  list.innerHTML = "";
  const stored = state.selectedStoredSupernodes || [];
  if (!supernodes.length && !stored.length) {
    list.className = "steering-node-list muted";
    list.textContent = "No feature supernodes are available to steer.";
    return;
  }

  list.className = "steering-node-list";
  supernodes.forEach((supernode, index) => {
    const row = document.createElement("div");
    row.className = "steering-node";

    const checkboxLabel = document.createElement("label");
    checkboxLabel.className = "steering-node-main";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.dataset.steerName = supernode.name;
    checkbox.checked = hadPriorNodes
      ? preSelection.checked.includes(supernode.name)
      : index === 0;
    checkboxLabel.appendChild(checkbox);

    const text = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = supernode.name;
    const meta = document.createElement("small");
    meta.textContent = `${supernode.feature_count} feature(s) · L${supernode.layer_min}-${supernode.layer_max}${supernode.role ? ` · ${supernode.role}` : ""}`;
    text.appendChild(title);
    text.appendChild(meta);
    checkboxLabel.appendChild(text);

    const factor = document.createElement("input");
    factor.type = "number";
    factor.step = "0.5";
    factor.value = preSelection.factors[supernode.name] ?? "-1";
    factor.dataset.steerFactor = supernode.name;
    factor.setAttribute("aria-label", `factor for ${supernode.name}`);

    row.appendChild(checkboxLabel);
    row.appendChild(factor);
    list.appendChild(row);
  });

  stored.forEach((record) => {
    const row = document.createElement("div");
    row.className = "steering-node stored";

    const main = document.createElement("div");
    main.className = "steering-node-main";
    const marker = document.createElement("span");
    marker.className = "pill";
    marker.textContent = "Stored";
    main.appendChild(marker);

    const text = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = record.label;
    const meta = document.createElement("small");
    const sourceLabel = record.source_dataset
      ? `${record.source_dataset}/${record.source_slug}`
      : record.source_slug;
    meta.textContent = `${record.feature_count} feature(s) · L${record.layer_min}-${record.layer_max} · ${sourceLabel}${record.role ? ` · ${record.role}` : ""}`;
    text.appendChild(title);
    text.appendChild(meta);
    main.appendChild(text);

    const factor = document.createElement("input");
    factor.type = "number";
    factor.step = "0.5";
    factor.value = record.factor;
    factor.dataset.storedFactor = record.record_id;
    factor.setAttribute("aria-label", `factor for stored ${record.label}`);

    const target = document.createElement("input");
    target.type = "number";
    target.min = "0";
    target.step = "1";
    target.value = record.target_pos;
    target.dataset.storedTargetPos = record.record_id;
    target.setAttribute("aria-label", `target token position for stored ${record.label}`);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "secondary steering-node-remove";
    remove.textContent = "Remove";
    remove.addEventListener("click", () => {
      state.selectedStoredSupernodes = state.selectedStoredSupernodes.filter(
        (item) => item.record_id !== record.record_id,
      );
      renderSteeringNodes(state.steeringOptions?.supernodes || []);
    });

    const controls = document.createElement("div");
    controls.className = "steering-node-controls";
    controls.appendChild(factor);
    controls.appendChild(target);
    controls.appendChild(remove);

    row.appendChild(main);
    row.appendChild(controls);
    list.appendChild(row);
  });
}

function setAllSteeringNodes(checked) {
  el("steeringNodeList")
    .querySelectorAll("input[type='checkbox'][data-steer-name]")
    .forEach((input) => {
      input.checked = checked;
    });
}

async function loadSupernodeStorage({ rebuild = false } = {}) {
  el("supernodeStorageList").className = "storage-list muted";
  el("supernodeStorageList").textContent = rebuild
    ? "Rebuilding storage index from saved summaries..."
    : "Loading stored supernodes...";
  const params = new URLSearchParams();
  const label = el("storageLabelFilter").value.trim();
  const role = el("storageRoleFilter").value.trim();
  const description = el("storageDescriptionFilter").value.trim();
  const source = el("storageSourceFilter").value.trim();
  const modelName = el("steerModel").value.trim();
  const transcoder = el("steerTranscoder").value.trim();
  if (label) params.set("label", label);
  if (role) params.set("role", role);
  if (description) params.set("description", description);
  if (source) params.set("source_slug", source);
  if (modelName) params.set("model_name", modelName);
  if (transcoder) params.set("transcoder", transcoder);
  if (rebuild) params.set("rebuild", "true");
  const suffix = params.toString() ? `?${params.toString()}` : "";
  const payload = await api(`/api/supernode-storage${suffix}`);
  state.storageRecords = payload.records || [];
  renderSupernodeStorage(state.storageRecords);
}

function renderSupernodeStorage(records) {
  const container = el("supernodeStorageList");
  container.innerHTML = "";
  if (!records.length) {
    container.className = "storage-list muted";
    container.textContent = "No stored supernodes are indexed yet. Generate a summary or rebuild the index.";
    return;
  }

  container.className = "storage-list";
  records.forEach((record) => {
    const row = document.createElement("div");
    row.className = "storage-record";

    const main = document.createElement("div");
    main.className = "storage-record-main";
    const title = document.createElement("strong");
    title.textContent = record.label;
    const meta = document.createElement("small");
    const sourceLabel = record.source_dataset
      ? `${record.source_dataset}/${record.source_slug}`
      : record.source_slug;
    meta.textContent = `${record.role || "Feature"} · L${record.layer_min}-${record.layer_max} · ${record.feature_count} feature(s) · ${sourceLabel}`;
    const description = document.createElement("span");
    description.textContent = record.description || "";
    main.appendChild(title);
    main.appendChild(meta);
    if (description.textContent) {
      main.appendChild(description);
    }

    const button = document.createElement("button");
    button.type = "button";
    button.className = "secondary";
    button.textContent = "Add";
    button.addEventListener("click", () => addStoredSupernode(record));
    row.appendChild(main);
    row.appendChild(button);
    container.appendChild(row);
  });
}

function addStoredSupernode(record) {
  const exists = state.selectedStoredSupernodes.some(
    (item) => item.record_id === record.record_id,
  );
  if (exists) return;
  state.selectedStoredSupernodes.push({
    ...record,
    factor: -1,
    target_pos: defaultStoredTargetPos(),
  });
  renderSteeringNodes(state.steeringOptions?.supernodes || []);
}

function steeringPayload() {
  const factors = {};
  const factorInputs = new Map();
  el("steeringNodeList")
    .querySelectorAll("input[data-steer-factor]")
    .forEach((input) => {
      factorInputs.set(input.dataset.steerFactor, input);
    });

  el("steeringNodeList")
    .querySelectorAll("input[type='checkbox'][data-steer-name]")
    .forEach((checkbox) => {
      if (!checkbox.checked) return;
      const name = checkbox.dataset.steerName;
      const factorInput = factorInputs.get(name);
      factors[name] = Number(factorInput.value);
    });

  const targetInputs = new Map();
  el("steeringNodeList")
    .querySelectorAll("input[data-stored-target-pos]")
    .forEach((input) => {
      targetInputs.set(input.dataset.storedTargetPos, input);
    });
  const storedSupernodes = [];
  el("steeringNodeList")
    .querySelectorAll("input[data-stored-factor]")
    .forEach((input) => {
      const recordId = input.dataset.storedFactor;
      const targetInput = targetInputs.get(recordId);
      storedSupernodes.push({
        record_id: recordId,
        factor: Number(input.value),
        target_pos: Number(targetInput.value),
      });
    });

  return {
    factors,
    stored_supernodes: storedSupernodes,
    model_name: el("steerModel").value,
    transcoder: el("steerTranscoder").value,
    dtype: el("steerDtype").value,
    backend: el("steerBackend").value,
    freeze_attention: el("steerFreezeAttention").checked,
    layers_below: numberValue("steerLayersBelow"),
    layers_above: numberValue("steerLayersAbove"),
    edge_threshold: numberValue("steerEdgeThreshold"),
    top_k: numberValue("steerTopK"),
  };
}

function renderSteeringResult(result) {
  const container = el("steeringResult");
  container.className = "steering-result";
  container.innerHTML = "";
  const figurePanel = el("steeringFigure");
  figurePanel.innerHTML = "";

  const outputs = document.createElement("div");
  outputs.className = "steering-outputs";
  (result.top_outputs || []).forEach((item) => {
    const pill = document.createElement("span");
    pill.className = "pill";
    pill.textContent = `${item.token} ${Number(item.probability).toFixed(3)}`;
    outputs.appendChild(pill);
  });
  container.appendChild(outputs);

  if (result.figure_html) {
    const frame = document.createElement("iframe");
    frame.className = "steering-figure";
    frame.title = "steering visualization";
    frame.srcdoc = result.figure_html;
    figurePanel.appendChild(frame);
    return;
  }

  const svg = document.createElement("div");
  svg.className = "steering-svg";
  svg.innerHTML = result.svg || "";
  figurePanel.appendChild(svg);
}

el("refreshBtn").addEventListener("click", refreshGraphs);
el("datasetFilter").addEventListener("change", (event) => {
  state.datasetFilter = event.target.value;
  renderGraphs();
});
el("newGraphBtn").addEventListener("click", () => el("generateDialog").showModal());
el("uploadBtn").addEventListener("click", () => el("uploadDialog").showModal());
el("summaryBtn").addEventListener("click", () => {
  if (state.activeDataset) {
    el("sumShapValuesPath").value = defaultShapPath(state.activeDataset, state.activeSourceSet);
  }
  syncThetaModeInputs();
  el("summaryDialog").showModal();
});
el("sumThetaMode").addEventListener("change", syncThetaModeInputs);
el("showSummaryBtn").addEventListener("click", () => openViewer(true));
el("steerBtn").addEventListener("click", openSteeringDialog);
el("showRawBtn").addEventListener("click", () => openViewer(false));
el("steerAllBtn").addEventListener("click", () => setAllSteeringNodes(true));
el("unsteerAllBtn").addEventListener("click", () => setAllSteeringNodes(false));
el("addStoredSupernodeBtn").addEventListener("click", async () => {
  const panel = el("supernodeStoragePanel");
  panel.hidden = !panel.hidden;
  if (!panel.hidden) {
    el("supernodeStorageList").className = "storage-list muted";
    el("supernodeStorageList").textContent = "Loading stored supernodes...";
    try {
      await loadSupernodeStorage();
    } catch (error) {
      el("supernodeStorageList").className = "storage-list muted";
      el("supernodeStorageList").textContent = error.message;
    }
  }
});
el("storageSearchBtn").addEventListener("click", async () => {
  try {
    await loadSupernodeStorage();
  } catch (error) {
    el("supernodeStorageList").className = "storage-list muted";
    el("supernodeStorageList").textContent = error.message;
  }
});
el("storageRebuildBtn").addEventListener("click", async () => {
  try {
    await loadSupernodeStorage({ rebuild: true });
  } catch (error) {
    el("supernodeStorageList").className = "storage-list muted";
    el("supernodeStorageList").textContent = error.message;
  }
});

el("previewBtn").addEventListener("click", async () => {
  const out = el("previewOut");
  out.textContent = "Starting preview...";
  const req = {
    prompt: el("genPrompt").value,
    model_name: el("genModel").value,
    transcoder: el("genTranscoder").value,
    dtype: el("genDtype").value,
    backend: el("genBackend").value,
    top_k: 5,
    qwen_system: el("genQwenSystem").value,
    qwen_assistant: el("genQwenAssistant").value,
    qwen_enable_thinking: el("genQwenThinking").checked,
  };
  const payload = await api("/api/graphs/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  await pollJob(payload.job, out, async (result) => {
    out.innerHTML = `<div>${result.tokens.length} tokens</div>${tokenPills(result.tokens)}<div class="muted">Next token</div>${nextTokenPills(result.next_tokens)}`;
  });
});

el("startGenerateBtn").addEventListener("click", async () => {
  const status = el("generateStatus");
  status.textContent = "Starting generation...";
  const payload = await api("/api/graphs/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(generationPayload()),
  });
  await pollJob(payload.job, status, async (result) => {
    await refreshGraphs();
    el("generateDialog").close();
    await selectGraph(result.graph.dataset, result.graph.slug, false, result.graph.source_set || "");
  });
});

el("startUploadBtn").addEventListener("click", async () => {
  const status = el("uploadStatus");
  const file = el("uploadFile").files[0];
  if (!file) {
    status.textContent = "Choose a .pt file first.";
    return;
  }
  const data = new FormData();
  data.append("file", file);
  data.append("slug", el("uploadSlug").value);
  data.append("scan", el("uploadScan").value);
  status.textContent = "Uploading...";
  const payload = await api("/api/graphs/upload", { method: "POST", body: data });
  await refreshGraphs();
  el("uploadDialog").close();
  await selectGraph(payload.graph.dataset, payload.graph.slug, false, payload.graph.source_set || "");
});

el("startSummaryBtn").addEventListener("click", async () => {
  if (!state.activeDataset || !state.activeSlug) return;
  const status = el("summaryStatus");
  const progressRow = el("summaryProgressRow");
  const progressBar = el("summaryProgress");
  const progressText = el("summaryProgressText");
  const startButton = el("startSummaryBtn");
  startButton.disabled = true;
  progressRow.hidden = false;
  updateProgress(progressBar, progressText, 0);
  status.textContent = "Starting summary...";
  try {
    const payload = await api(
      graphApiPath(state.activeDataset, state.activeSlug, "/summary", state.activeSourceSet),
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(summaryPayload()),
      },
    );
    await pollJob(payload.job, status, async () => {
      await refreshGraphs();
      el("summaryDialog").close();
      await selectGraph(state.activeDataset, state.activeSlug, true, state.activeSourceSet);
    }, progressBar, progressText);
  } finally {
    startButton.disabled = false;
  }
});

el("startSteeringBtn").addEventListener("click", async () => {
  if (!state.activeDataset || !state.activeSlug) return;
  const status = el("steeringStatus");
  const progressRow = el("steeringProgressRow");
  const progressBar = el("steeringProgress");
  const progressText = el("steeringProgressText");
  const startButton = el("startSteeringBtn");
  const req = steeringPayload();
  if (!Object.keys(req.factors).length && !req.stored_supernodes.length) {
    status.textContent = "Select at least one feature supernode.";
    return;
  }

  startButton.disabled = true;
  progressRow.hidden = false;
  updateProgress(progressBar, progressText, 0);
  status.textContent = "Starting steering...";
  try {
    const payload = await api(
      graphApiPath(state.activeDataset, state.activeSlug, "/steer", state.activeSourceSet),
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req),
      },
    );
    await pollJob(payload.job, status, async (result) => {
      renderSteeringResult(result);
    }, progressBar, progressText);
  } finally {
    startButton.disabled = false;
  }
});

refreshGraphs().catch((error) => {
  el("graphList").textContent = error.message;
});

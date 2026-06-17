const state = {
  graphs: [],
  activeSlug: null,
  summary: null,
};

const el = (id) => document.getElementById(id);

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

function renderGraphs() {
  const list = el("graphList");
  list.innerHTML = "";
  if (!state.graphs.length) {
    list.innerHTML = '<div class="muted">No local graphs found.</div>';
    return;
  }

  state.graphs.forEach((graph) => {
    const button = document.createElement("button");
    button.className = `graph-item ${graph.slug === state.activeSlug ? "active" : ""}`;
    button.innerHTML = `
      <div class="graph-name">${graph.slug}</div>
      <div class="graph-meta">${graph.node_count} nodes - ${graph.link_count} edges - ${graph.has_pt ? ".pt" : "view-only"}${graph.has_summary ? " - summary" : ""}</div>
    `;
    button.addEventListener("click", () => selectGraph(graph.slug));
    list.appendChild(button);
  });
}

async function refreshGraphs() {
  const payload = await api("/api/graphs");
  state.graphs = payload.graphs;
  renderGraphs();
}

function activeGraph() {
  return state.graphs.find((graph) => graph.slug === state.activeSlug);
}

async function selectGraph(slug, useSummary = false) {
  state.activeSlug = slug;
  state.summary = null;
  renderGraphs();
  const graph = activeGraph();
  el("activeTitle").textContent = slug;
  el("activeMeta").textContent = graph ? `${graph.scan || "unknown scan"} - ${graph.prompt || "no prompt metadata"}` : "";
  el("summaryBtn").disabled = !graph?.has_pt;
  el("showSummaryBtn").disabled = !graph?.has_summary;
  el("showRawBtn").disabled = false;
  await openViewer(useSummary);
}

async function openViewer(useSummary = false) {
  if (!state.activeSlug) return;
  const payload = await api(`/api/graphs/${encodeURIComponent(state.activeSlug)}/viewer-url?summary=${useSummary ? "true" : "false"}`);
  el("viewerFrame").src = payload.url;
  state.summary = payload.summary;
  renderSummaryDetails(payload.summary);
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

async function pollJob(job, statusEl, onComplete) {
  statusEl.textContent = `${job.kind}: ${job.status}`;
  while (true) {
    await new Promise((resolve) => setTimeout(resolve, 1400));
    const payload = await api(`/api/jobs/${job.id}`);
    const current = payload.job;
    statusEl.textContent = `${current.kind}: ${current.message}`;
    if (current.status === "completed") {
      statusEl.textContent = "Completed.";
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
  };
}

function summaryPayload() {
  return {
    logit_weights: el("sumLogitWeights").value,
    token_weights_source: el("sumTokenWeights").value,
    node_threshold: numberValue("sumNodeThreshold"),
    edge_threshold: numberValue("sumEdgeThreshold"),
    combine_method: el("sumCombine").value,
    normalization: el("sumNormalization").value,
    alpha: numberValue("sumAlpha"),
    keep_all_tokens_and_logits: el("sumKeepAll").checked,
    filter_act_density: el("sumFilterAct").checked,
    cluster_method: el("sumClusterMethod").value,
    target_k: numberValue("sumTargetK"),
    max_layer_span: numberValue("sumLayerSpan"),
    max_sn: numberValue("sumMaxSn") || null,
    mean_method: el("sumMean").value,
    normalize_weights: el("sumNormalizeWeights").checked,
    decay_rate: numberValue("sumDecay"),
    random_state: numberValue("sumRandom"),
    n_init: numberValue("sumNInit"),
    ilp_time_limit: numberValue("sumIlpTime"),
    label_supernodes: el("sumDoLabel").checked,
    label_model: el("sumLabelModel").value,
    label_temperature: numberValue("sumLabelTemp"),
    thinking_effort: el("sumThinking").value,
  };
}

el("refreshBtn").addEventListener("click", refreshGraphs);
el("newGraphBtn").addEventListener("click", () => el("generateDialog").showModal());
el("uploadBtn").addEventListener("click", () => el("uploadDialog").showModal());
el("summaryBtn").addEventListener("click", () => el("summaryDialog").showModal());
el("showSummaryBtn").addEventListener("click", () => openViewer(true));
el("showRawBtn").addEventListener("click", () => openViewer(false));

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
    await selectGraph(result.graph.slug);
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
  await selectGraph(payload.graph.slug);
});

el("startSummaryBtn").addEventListener("click", async () => {
  if (!state.activeSlug) return;
  const status = el("summaryStatus");
  status.textContent = "Starting summary...";
  const payload = await api(`/api/graphs/${encodeURIComponent(state.activeSlug)}/summary`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(summaryPayload()),
  });
  await pollJob(payload.job, status, async () => {
    await refreshGraphs();
    el("summaryDialog").close();
    await selectGraph(state.activeSlug, true);
  });
});

refreshGraphs().catch((error) => {
  el("graphList").textContent = error.message;
});

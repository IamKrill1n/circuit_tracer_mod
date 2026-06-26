from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from visualization_app import services

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
VIEWER_FRONTEND_DIR = Path(str(files("circuit_tracer") / "frontend/assets"))


class JobRecord(BaseModel):
    id: str
    kind: str
    status: str = "queued"
    message: str = "Queued"
    progress: float = 0.0
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class GenerateGraphRequest(BaseModel):
    prompt: str
    slug: str = ""
    model_name: str = "google/gemma-2-2b"
    transcoder: str = "mntss/clt-gemma-2-2b-2.5M"
    dtype: str = "bfloat16"
    backend: str = "transformerlens"
    max_n_logits: int = 15
    desired_logit_prob: float = 0.99
    max_feature_nodes: int = 8192
    batch_size: int = 256
    node_threshold: float = 0.8
    edge_threshold: float = 0.98
    qwen_system: str = "You are a helpful assistant."
    qwen_assistant: str = ""
    qwen_enable_thinking: bool = False


class PreviewRequest(BaseModel):
    prompt: str
    model_name: str = "google/gemma-2-2b"
    transcoder: str = "mntss/clt-gemma-2-2b-2.5M"
    dtype: str = "bfloat16"
    backend: str = "transformerlens"
    top_k: int = 5
    qwen_system: str = "You are a helpful assistant."
    qwen_assistant: str = ""
    qwen_enable_thinking: bool = False


class SummaryRequest(BaseModel):
    model_name: str = "google/gemma-2-2b"
    logit_weights: str = "target"
    token_weights_source: str = "shap"
    token_attr_model: str = ""
    token_attr_normalize: str = "entmax"
    entmax_alpha: float = 1.25
    shap_values_path: str = ""
    device: str = "cuda"
    node_threshold: float = 0.02
    edge_threshold: float = 0.9
    combine_method: str = "geometric"
    normalization: str = "rank"
    alpha: float = 0.5
    keep_all_tokens_and_logits: bool = False
    filter_act_density: bool = True
    act_density_lb: float = 2e-5
    act_density_ub: float = 0.1
    max_layer_span: int = 7
    max_sn: int | None = 20
    eps_causal: float | None = 0.05
    theta: float | str = "p65"
    ilp_time_limit: float = 30.0
    label_supernodes: bool = True
    label_model: str = "gemma-4-31b-it"
    label_temperature: float = 0.2
    thinking_effort: str = "off"


class StoredSupernodeRequest(BaseModel):
    record_id: str
    factor: float = -1.0
    target_pos: int


class SteeringRequest(BaseModel):
    factors: dict[str, float] = Field(default_factory=dict)
    stored_supernodes: list[StoredSupernodeRequest] = Field(default_factory=list)
    model_name: str = ""
    transcoder: str = ""
    dtype: str = "bfloat16"
    backend: str = "transformerlens"
    freeze_attention: bool = True
    layers_below: int = 0
    layers_above: int = 1
    edge_threshold: float = 0.1
    top_k: int = 5


app = FastAPI(title="Circuit Tracer Visualization App")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_executor = ThreadPoolExecutor(max_workers=1)
_jobs: dict[str, JobRecord] = {}
_jobs_lock = threading.Lock()
_viewer_lock = threading.Lock()
_viewer_dir: str | None = None


def _json_job(job: JobRecord) -> dict[str, Any]:
    return job.model_dump()


def _set_job(job_id: str, **updates: Any) -> None:
    with _jobs_lock:
        job = _jobs[job_id]
        for key, value in updates.items():
            setattr(job, key, value)
        job.updated_at = time.time()


def _submit_job(
    kind: str,
    fn: Callable[[Callable[[str, float | None], None]], dict[str, Any]],
) -> JobRecord:
    job_id = uuid.uuid4().hex
    job = JobRecord(id=job_id, kind=kind)
    with _jobs_lock:
        _jobs[job_id] = job

    def run() -> None:
        def progress(message: str, value: float | None = None) -> None:
            updates: dict[str, Any] = {"message": message}
            if value is not None:
                updates["progress"] = max(0.0, min(1.0, float(value)))
            _set_job(job_id, **updates)

        _set_job(job_id, status="running", message="Running", progress=0.0)
        try:
            result = fn(progress)
        except Exception as exc:
            _set_job(job_id, status="failed", message="Failed", error=str(exc))
        else:
            _set_job(job_id, status="completed", message="Completed", progress=1.0, result=result)

    future: Future[None] = _executor.submit(run)
    future.add_done_callback(lambda _future: None)
    return job


def _set_viewer_dir(viewer_dir: Path) -> str:
    global _viewer_dir
    with _viewer_lock:
        _viewer_dir = str(viewer_dir)
    return "/viewer"


def _active_viewer_dir() -> Path:
    with _viewer_lock:
        viewer_dir = _viewer_dir
    if viewer_dir is None:
        raise HTTPException(status_code=404, detail="No viewer graph has been selected.")
    return Path(viewer_dir)


def _viewer_file_response(base_dir: Path, relative_path: str) -> FileResponse:
    path = (base_dir / relative_path).resolve()
    base = base_dir.resolve()
    if not path.is_file() or base not in path.parents and path != base:
        raise HTTPException(status_code=404, detail="Viewer file not found.")
    return FileResponse(path)


@app.get("/viewer/data/{relative_path:path}")
def viewer_data(relative_path: str) -> FileResponse:
    return _viewer_file_response(_active_viewer_dir(), relative_path)


@app.get("/viewer/graph_data/{relative_path:path}")
def viewer_graph_data(relative_path: str) -> FileResponse:
    return _viewer_file_response(_active_viewer_dir(), relative_path)


@app.get("/viewer/{relative_path:path}")
def viewer_static(relative_path: str = "index.html") -> FileResponse:
    path = relative_path or "index.html"
    return _viewer_file_response(VIEWER_FRONTEND_DIR, path)


@app.post("/save_graph/{slug}")
async def save_viewer_graph(slug: str, request: Request) -> dict[str, bool]:
    # The stock viewer writes qParams here. Keep the endpoint available when
    # serving the viewer through the FastAPI app instead of local_server.
    graph_path = (_active_viewer_dir() / f"{services.slugify(slug)}.json").resolve()
    viewer_dir = _active_viewer_dir().resolve()
    if not graph_path.is_file() or viewer_dir not in graph_path.parents:
        raise HTTPException(status_code=404, detail="Viewer graph not found.")

    payload = await request.json()
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    graph["qParams"] = payload["qParams"]
    graph_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    return {"ok": True}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/graphs")
def graphs() -> dict[str, Any]:
    return {"graphs": [asdict(record) for record in services.list_graphs()]}


@app.get("/api/supernode-storage")
def supernode_storage(
    label: str = "",
    role: str = "",
    description: str = "",
    source_slug: str = "",
    model_name: str = "",
    transcoder: str = "",
    rebuild: bool = False,
) -> dict[str, Any]:
    if rebuild:
        services.rebuild_supernode_storage()
    return services.list_supernode_storage(
        label=label,
        role=role,
        description=description,
        source_slug=source_slug,
        model_name=model_name,
        transcoder=transcoder,
    )


@app.post("/api/graphs/upload")
def upload_graph(
    file: UploadFile = File(...),
    slug: str = Form(""),
    scan: str = Form(""),
    node_threshold: float = Form(0.8),
    edge_threshold: float = Form(0.98),
) -> dict[str, Any]:
    if not file.filename or not file.filename.endswith(".pt"):
        raise HTTPException(status_code=400, detail="Only .pt attribution graphs are supported.")

    safe_slug = services.slugify(slug or Path(file.filename).stem)
    upload_dir = services.graph_dir(safe_slug, services.CUSTOM_DATASET)
    upload_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = upload_dir / f"{safe_slug}.upload.pt"
    with tmp_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        if not scan:
            _, scan = services.infer_graph_model_and_scan(tmp_path)
        record = services.convert_pt_to_viewer(
            tmp_path,
            slug=safe_slug,
            dataset=services.CUSTOM_DATASET,
            scan=scan,
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
        )
    finally:
        if tmp_path.exists() and tmp_path.name.endswith(".upload.pt"):
            tmp_path.unlink()

    return {"graph": asdict(record)}


@app.post("/api/graphs/preview")
def preview_graph(req: PreviewRequest) -> dict[str, Any]:
    job = _submit_job(
        "preview",
        lambda progress: (
            progress("Loading preview model", 0.15)
            or services.preview_prompt(
                prompt=req.prompt,
                model_name=req.model_name,
                transcoder=req.transcoder,
                dtype=req.dtype,  # type: ignore[arg-type]
                backend=req.backend,  # type: ignore[arg-type]
                top_k=req.top_k,
                qwen_system=req.qwen_system,
                qwen_assistant=req.qwen_assistant,
                qwen_enable_thinking=req.qwen_enable_thinking,
            )
        ),
    )
    return {"job": _json_job(job)}


@app.post("/api/graphs/generate")
def generate_graph(req: GenerateGraphRequest) -> dict[str, Any]:
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required.")

    job = _submit_job(
        "generate",
        lambda progress: (
            progress("Generating attribution graph", 0.15)
            or {
                "graph": asdict(
                    services.generate_graph(
                        prompt=req.prompt,
                        slug=req.slug or req.prompt[:40],
                        model_name=req.model_name,
                        transcoder=req.transcoder,
                        dtype=req.dtype,  # type: ignore[arg-type]
                        backend=req.backend,  # type: ignore[arg-type]
                        max_n_logits=req.max_n_logits,
                        desired_logit_prob=req.desired_logit_prob,
                        max_feature_nodes=req.max_feature_nodes,
                        batch_size=req.batch_size,
                        node_threshold=req.node_threshold,
                        edge_threshold=req.edge_threshold,
                        qwen_system=req.qwen_system,
                        qwen_assistant=req.qwen_assistant,
                        qwen_enable_thinking=req.qwen_enable_thinking,
                    )
                )
            }
        ),
    )
    return {"job": _json_job(job)}


@app.post("/api/graphs/{dataset}/{slug}/summary")
def summarize_graph(
    dataset: str,
    slug: str,
    req: SummaryRequest,
    source_set: str = "",
) -> dict[str, Any]:
    safe_slug = services.slugify(slug)
    safe_dataset = services.validate_dataset(dataset)
    safe_source_set = services.validate_source_set(source_set)
    try:
        services.load_graph_record(safe_slug, safe_dataset, source_set=safe_source_set)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    settings = req.model_dump()
    if not settings.get("shap_values_path"):
        settings["shap_values_path"] = services.default_shap_path(safe_dataset, safe_source_set)

    job = _submit_job(
        "summary",
        lambda progress: services.run_summary(
            slug=safe_slug,
            dataset=safe_dataset,
            source_set=safe_source_set,
            settings=settings,
            progress=progress,
        ),
    )
    return {"job": _json_job(job)}


@app.get("/api/graphs/{dataset}/{slug}/steering-options")
def steering_options(dataset: str, slug: str, source_set: str = "") -> dict[str, Any]:
    safe_slug = services.slugify(slug)
    safe_dataset = services.validate_dataset(dataset)
    safe_source_set = services.validate_source_set(source_set)
    try:
        return services.steering_options(safe_slug, safe_dataset, source_set=safe_source_set)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/graphs/{dataset}/{slug}/steer")
def steer_graph(
    dataset: str,
    slug: str,
    req: SteeringRequest,
    source_set: str = "",
) -> dict[str, Any]:
    safe_slug = services.slugify(slug)
    safe_dataset = services.validate_dataset(dataset)
    safe_source_set = services.validate_source_set(source_set)
    try:
        services.load_graph_record(safe_slug, safe_dataset, source_set=safe_source_set)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    job = _submit_job(
        "steering",
        lambda progress: services.run_steering(
            slug=safe_slug,
            dataset=safe_dataset,
            source_set=safe_source_set,
            factors=req.factors,
            stored_supernodes=[item.model_dump() for item in req.stored_supernodes],
            model_name=req.model_name,
            transcoder=req.transcoder,
            dtype=req.dtype,  # type: ignore[arg-type]
            backend=req.backend,  # type: ignore[arg-type]
            freeze_attention=req.freeze_attention,
            layers_below=req.layers_below,
            layers_above=req.layers_above,
            edge_threshold=req.edge_threshold,
            top_k=req.top_k,
            progress=progress,
        ),
    )
    return {"job": _json_job(job)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job.")
    return {"job": _json_job(job)}


@app.get("/api/graphs/{dataset}/{slug}/viewer-url")
def get_viewer_url(
    dataset: str,
    slug: str,
    summary: bool = False,
    source_set: str = "",
) -> dict[str, Any]:
    safe_slug = services.slugify(slug)
    safe_dataset = services.validate_dataset(dataset)
    safe_source_set = services.validate_source_set(source_set)
    try:
        record = services.load_graph_record(safe_slug, safe_dataset, source_set=safe_source_set)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    base_url = _set_viewer_dir(Path(record.directory))

    extra_params = None
    summary_data = None
    if summary:
        path = services.app_summary_path(safe_slug, safe_dataset, source_set=safe_source_set)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Summary has not been generated.")
        from summarization.summarize import SummaryGraph

        sng = SummaryGraph.load(str(path))
        pinned_ids, supernodes, stats = services.summary_graph_viewer_payload(sng)
        extra_params = services.summary_query_params(pinned_ids, supernodes)
        summary_data = {
            "stats": stats,
            "summary": services.summary_metadata(sng),
        }

    return {
        "url": services.viewer_url(base_url, safe_slug, extra_params),
        "graph": asdict(record),
        "summary": summary_data,
    }

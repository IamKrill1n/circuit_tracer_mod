from __future__ import annotations

import shutil
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from circuit_tracer.frontend.local_server import serve

from visualization_app import services

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
VIEWER_PORT = 8032


class JobRecord(BaseModel):
    id: str
    kind: str
    status: str = "queued"
    message: str = "Queued"
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


class PreviewRequest(BaseModel):
    prompt: str
    model_name: str = "google/gemma-2-2b"
    transcoder: str = "mntss/clt-gemma-2-2b-2.5M"
    dtype: str = "bfloat16"
    backend: str = "transformerlens"
    top_k: int = 5


class SummaryRequest(BaseModel):
    model_name: str = "google/gemma-2-2b"
    logit_weights: str = "target"
    token_weights_source: str = "uniform"
    token_attr_model: str = ""
    token_attr_normalize: str = "softmax"
    entmax_alpha: float = 1.25
    device: str = "cuda"
    node_threshold: float = 0.8
    edge_threshold: float = 0.98
    combine_method: str = "geometric"
    normalization: str = "rank"
    alpha: float = 0.5
    keep_all_tokens_and_logits: bool = False
    filter_act_density: bool = False
    act_density_lb: float = 2e-5
    act_density_ub: float = 0.1
    cluster_method: str = "spectral"
    target_k: int = 7
    max_layer_span: int = 4
    max_sn: int | None = None
    mean_method: str = "arith"
    normalize_weights: bool = False
    decay_rate: float | None = 1.0
    random_state: int = 42
    n_init: int = 20
    lambda_causal: float = 1.0
    eps_causal: float | None = None
    ilp_time_limit: float = 30.0
    label_supernodes: bool = True
    label_model: str = "gemini-2.5-flash"
    label_temperature: float = 0.2
    thinking_effort: str = "off"


app = FastAPI(title="Circuit Tracer Visualization App")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_executor = ThreadPoolExecutor(max_workers=1)
_jobs: dict[str, JobRecord] = {}
_jobs_lock = threading.Lock()
_viewer_lock = threading.Lock()
_viewer_server = None
_viewer_dir: str | None = None


def _json_job(job: JobRecord) -> dict[str, Any]:
    return job.model_dump()


def _set_job(job_id: str, **updates: Any) -> None:
    with _jobs_lock:
        job = _jobs[job_id]
        for key, value in updates.items():
            setattr(job, key, value)
        job.updated_at = time.time()


def _submit_job(kind: str, fn: Callable[[], dict[str, Any]]) -> JobRecord:
    job_id = uuid.uuid4().hex
    job = JobRecord(id=job_id, kind=kind)
    with _jobs_lock:
        _jobs[job_id] = job

    def run() -> None:
        _set_job(job_id, status="running", message="Running")
        try:
            result = fn()
        except Exception as exc:
            _set_job(job_id, status="failed", message="Failed", error=str(exc))
        else:
            _set_job(job_id, status="completed", message="Completed", result=result)

    future: Future[None] = _executor.submit(run)
    future.add_done_callback(lambda _future: None)
    return job


def _ensure_viewer_server(viewer_dir: Path) -> str:
    global _viewer_server, _viewer_dir

    viewer_dir_str = str(viewer_dir)
    with _viewer_lock:
        if _viewer_server is not None and _viewer_dir == viewer_dir_str:
            return f"http://localhost:{VIEWER_PORT}"

        if _viewer_server is not None:
            _viewer_server.stop()

        _viewer_server = serve(data_dir=viewer_dir_str, port=VIEWER_PORT)
        _viewer_dir = viewer_dir_str
        return f"http://localhost:{VIEWER_PORT}"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/graphs")
def graphs() -> dict[str, Any]:
    return {"graphs": [asdict(record) for record in services.list_graphs()]}


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
    upload_dir = services.graph_dir(safe_slug)
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
        lambda: services.preview_prompt(
            prompt=req.prompt,
            model_name=req.model_name,
            transcoder=req.transcoder,
            dtype=req.dtype,  # type: ignore[arg-type]
            backend=req.backend,  # type: ignore[arg-type]
            top_k=req.top_k,
        ),
    )
    return {"job": _json_job(job)}


@app.post("/api/graphs/generate")
def generate_graph(req: GenerateGraphRequest) -> dict[str, Any]:
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required.")

    job = _submit_job(
        "generate",
        lambda: {
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
                )
            )
        },
    )
    return {"job": _json_job(job)}


@app.post("/api/graphs/{slug}/summary")
def summarize_graph(slug: str, req: SummaryRequest) -> dict[str, Any]:
    safe_slug = services.slugify(slug)
    try:
        services.load_graph_record(safe_slug)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    job = _submit_job(
        "summary",
        lambda: services.run_summary(slug=safe_slug, settings=req.model_dump()),
    )
    return {"job": _json_job(job)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job.")
    return {"job": _json_job(job)}


@app.get("/api/graphs/{slug}/viewer-url")
def get_viewer_url(slug: str, summary: bool = False) -> dict[str, Any]:
    safe_slug = services.slugify(slug)
    record = services.load_graph_record(safe_slug)
    base_url = _ensure_viewer_server(Path(record.directory))

    extra_params = None
    summary_data = None
    if summary:
        path = services.summary_path(safe_slug)
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

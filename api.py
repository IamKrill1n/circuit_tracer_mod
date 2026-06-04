import requests
from typing import Tuple, List, Optional, Any
from pathlib import Path
import zlib
from config import NEURONPEDIA_API_KEY
import json

from circuit_tracer.frontend.feature_models import Model as FeatureDashboard

BASE_URL = "https://www.neuronpedia.org"
CLOUDFRONT_FEATURES = "https://d1fk9w8oratjix.cloudfront.net/features"
HF_BASE = "https://huggingface.co"


def get_feature(modelId: str, layer: str, index: int) -> Tuple[int, str]:
    """Fetch a feature from the Neuronpedia API.

    Args:
        modelId: Model identifier (e.g., "gemma-2-2b")
        layer: Layer identifier (e.g., "10-clt-hp")
        index: Feature index

    Returns:
        Tuple of (status_code, response_body)
    """
    url = f"{BASE_URL}/api/feature/{modelId}/{layer}/{index}"
    headers = {"x-api-key": NEURONPEDIA_API_KEY}

    resp = requests.get(url, headers=headers, timeout=30)
    return resp.status_code, resp.text


def _cantor_pair(x: int, y: int) -> int:
    # Matches Node.feature_node in circuit_tracer/frontend/graph_models.py
    return (x + y) * (x + y + 1) // 2 + y


def _hf_features_url(scan: str, path: str) -> str:
    # Mirrors featureUrl() in circuit_tracer/frontend/assets/feature_examples/init-feature-examples.js
    # scan formats: "repo/id", "repo/id@rev", "repo/id//subdir", "repo/id//subdir@rev"
    if "//" in scan:
        repo_part, rest = scan.split("//", 1)
        if "@" in rest:
            file_path, revision = rest.split("@", 1)
        else:
            file_path, revision = rest, "main"
        repo_id = repo_part.split("@")[0]
    else:
        if "@" in scan:
            repo_id, revision = scan.split("@", 1)
        else:
            repo_id, revision = scan, "main"
        file_path = ""
    prefix = f"{file_path.rstrip('/')}/" if file_path else ""
    return f"{HF_BASE}/{repo_id}/resolve/{revision}/{prefix}features/{path}"


def _parse_bin_payload(raw: bytes) -> bytes:
    # First 4 bytes = little-endian payload length; rest is zlib/gzip-deflated JSON.
    # Mirrors the 'bin' branch of getFile() in circuit_tracer/frontend/assets/util.js
    data_len = int.from_bytes(raw[:4], "little")
    payload = raw[4:4 + data_len]
    return zlib.decompress(payload, wbits=zlib.MAX_WBITS | 32)  # auto-detect zlib/gzip


# The HF index (offsets of every feature in every layer) is identical for all features of a scan.
# Cache it so a graph's worth of feature lookups downloads it once, not once per feature.
# Value is the parsed index, or None for scans not hosted on HF (→ Cloudfront fallback).
_HF_INDEX_CACHE: dict[str, list | dict | None] = {}


def _get_hf_index(scan: str) -> list | dict | None:
    if scan not in _HF_INDEX_CACHE:
        resp = requests.get(_hf_features_url(scan, "index.json.gz"), timeout=30)
        if resp.status_code == 200:
            _HF_INDEX_CACHE[scan] = json.loads(zlib.decompress(resp.content, wbits=zlib.MAX_WBITS | 32))
        else:
            _HF_INDEX_CACHE[scan] = None
    return _HF_INDEX_CACHE[scan]


def get_feature_dashboard(
    scan: str,
    layer: int,
    feat_idx: int,
    features_dir: str | Path | None = None,
) -> FeatureDashboard:
    """Fetch a feature dashboard JSON. Tries local dir → HuggingFace binary → Cloudfront.

    Args:
        scan: Transcoder scan id from the graph's metadata (e.g., "mntss/clt-gemma-2-2b-2.5M").
        layer: Layer index.
        feat_idx: Feature index within the layer.
        features_dir: Optional local directory of feature JSONs. Layout: {features_dir}/{scan}/{cantor_index}.json.
    """
    feature_index = _cantor_pair(layer, feat_idx)

    # 1) Local
    if features_dir is not None:
        local_path = Path(features_dir) / scan / f"{feature_index}.json"
        if local_path.exists():
            return FeatureDashboard.model_validate_json(local_path.read_text())

    # 2) HuggingFace binary chunks (preferred for HF-hosted scans like mntss/clt-gemma-2-2b-2.5M)
    index_data = _get_hf_index(scan)
    if index_data is not None:
        # index may be a list (indexed by layer) or dict (string-keyed)
        entry = index_data[layer] if isinstance(index_data, list) else index_data[str(layer)]
        offsets = entry["offsets"]
        bin_filename = entry["filename"]
        start, end = offsets[feat_idx], offsets[feat_idx + 1]
        bin_url = _hf_features_url(scan, bin_filename)
        resp = requests.get(bin_url, headers={"Range": f"bytes={start}-{end}"}, timeout=30)
        resp.raise_for_status()
        decoded = _parse_bin_payload(resp.content)
        return FeatureDashboard.model_validate_json(decoded)

    # 3) Cloudfront fallback (Anthropic-hosted scans)
    cf_url = f"{CLOUDFRONT_FEATURES}/{scan}/{feature_index}.json"
    resp = requests.get(cf_url, timeout=30)
    resp.raise_for_status()
    return FeatureDashboard.model_validate_json(resp.text)


def generate_autointerp(
    modelId: str,
    layer: str,
    index: int,
    explanationModelName: str = "gemini-2.5-flash",
    explanationType: str = "oai_token-act-pair"
) -> Tuple[int, str]:
    """Generate auto-interpretation for a feature.
    
    Args:
        modelId: Model identifier
        layer: Layer identifier
        index: Feature index
        explanationModelName: Model to use for explanation generation
        explanationType: Type of explanation to generate
        
    Returns:
        Tuple of (status_code, response_body)
    """
    url = f"{BASE_URL}/api/explanation/generate"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": NEURONPEDIA_API_KEY
    }
    payload = {
        "modelId": modelId,
        "layer": layer,
        "index": index,
        "explanationType": explanationType,
        "explanationModelName": explanationModelName
    }
    
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    return resp.status_code, resp.text


def generate_graph(
    modelId: str,
    prompt: str,
    slug: str,
    sourceSetName: str,
    desiredLogitProb: float = 0.99,
    edgeThreshold: float = 1,
    maxFeatureNodes: int = 10000,
    maxNLogits: int = 15,
    nodeThreshold: float = 1
) -> Tuple[int, str]:
    """Generate an attribution graph via the Neuronpedia API.
    
    Args:
        modelId: Model identifier (e.g., "gemma-2-2b")
        prompt: Input text prompt
        slug: Unique identifier for the graph
        sourceSetName: Source set name (e.g., "clt-hp")
        desiredLogitProb: Desired logit probability threshold
        edgeThreshold: Edge weight threshold
        maxFeatureNodes: Maximum number of feature nodes
        maxNLogits: Maximum number of logits
        nodeThreshold: Node importance threshold
        
    Returns:
        Tuple of (status_code, response_body)
    """
    url = f"{BASE_URL}/api/graph/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "modelId": modelId,
        "prompt": prompt,
        "slug": slug,
        "sourceSetName": sourceSetName,
        "desiredLogitProb": desiredLogitProb,
        "edgeThreshold": edgeThreshold,
        "maxFeatureNodes": maxFeatureNodes,
        "maxNLogits": maxNLogits,
        "nodeThreshold": nodeThreshold
    }
    
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    return resp.status_code, resp.text


def save_subgraph(
    modelId: str,
    slug: str,
    displayName: str,
    pinnedIds: List[str],
    supernodes: Optional[List[List[str]]] = None,
    clerps: Optional[List[str]] = None,
    pruningThreshold: float = 0.8,
    densityThreshold: float = 0.99,
    overwriteId: str = "",
) -> Tuple[int, str]:
    """Save a subgraph to Neuronpedia.

    Args:
        modelId: Model identifier (e.g., "gemma-2-2b")
        slug: Slug of the parent graph
        displayName: Human-readable name for the subgraph
        pinnedIds: List of node IDs to pin in the subgraph
        supernodes: List of supernode groups, each is [label, node_id, ...]
        clerps: List of custom clerp overrides
        pruningThreshold: Pruning threshold used to produce this subgraph
        densityThreshold: Density threshold used to produce this subgraph
        overwriteId: If non-empty, overwrite an existing subgraph with this ID

    Returns:
        Tuple of (status_code, response_body)
    """
    url = f"{BASE_URL}/api/graph/subgraph/save"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": NEURONPEDIA_API_KEY
    }
    print(headers)
    payload = {
        "modelId": modelId,
        "slug": slug,
        "displayName": displayName,
        "pinnedIds": pinnedIds,
        "supernodes": supernodes if supernodes is not None else [],
        "clerps": clerps if clerps is not None else [],
        "pruningThreshold": pruningThreshold,
        "densityThreshold": densityThreshold,
        "overwriteId": overwriteId,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    return resp.status_code, resp.text


def steer_logits(
    modelId: str,
    prompt: str,
    features: List[dict[str, Any]],
    sourceSetName: Optional[str] = None,
    nTokens: int = 1,
    topK: int = 5,
    freezeAttention: bool = True,
    temperature: float = 0.0,
    freqPenalty: float = 0.0,
    seed: Optional[int] = 16,
    steeredOutputOnly: bool = False,
) -> Tuple[int, str]:
    """Steer model logits via Neuronpedia's `/api/steer-logits` endpoint.

    Args:
        modelId: Model identifier.
        prompt: Input prompt to continue from.
        features: Steering features list with objects shaped like:
            {
              "layer": int,
              "index": int,
              "token_active_position": int,
              "steer_position": Optional[int],
              "steer_generated_tokens": bool,
              "delta": Optional[float],
              "ablate": bool,
            }
        sourceSetName: Optional source set name.
        nTokens: Number of completion tokens to generate.
        topK: Number of top logits returned per token.
        freezeAttention: Whether to freeze attention during generation.
        temperature: Sampling temperature.
        freqPenalty: Frequency penalty.
        seed: Optional random seed.
        steeredOutputOnly: If True, only returns steered output.

    Returns:
        Tuple of (status_code, response_body)
    """
    url = f"{BASE_URL}/api/steer-logits"
    headers = {"Content-Type": "application/json"}
    payload = {
        "modelId": modelId,
        "sourceSetName": sourceSetName,
        "prompt": prompt,
        "features": features,
        "nTokens": nTokens,
        "topK": topK,
        "freezeAttention": freezeAttention,
        "temperature": temperature,
        "freqPenalty": freqPenalty,
        "seed": seed,
        "steeredOutputOnly": steeredOutputOnly,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    return resp.status_code, resp.text


if __name__ == "__main__":
    # features = [
    #     {
    #         "layer": 20,
    #         "index": 15589,
    #         "token_active_position": 9,
    #         "steer_position": 9,
    #         "steer_generated_tokens": False,
    #         "delta": -1.0,
    #         "ablate": False,
    #     }
    # ]
    # status, data = steer_logits(
    #     modelId="gemma-2-2b",
    #     prompt="Fact: The capital of the state containing Dallas is",
    #     features=features,
    #     sourceSetName="clt-hp",
    #     nTokens=1,
    #     topK=5,
    # )
    # print(f"Status: {status}")
    # print(data)
    # print(_get_api_key())
    # Example: get feature
    status, data = get_feature("gemma-2-2b", "1-clt-hp", 89326)
    print(f"Status: {status}")
    # dict_data = json.loads(data)
    # print(dict_data['explanations'])
    # print(dict_data['vector'])
    # print(dict_data['neuron_alignment_indices'])
    # Example: generate auto-interpretation
    # status, data = generate_autointerp(
    #     modelId="gemma-2-2b",
    #     layer="10-clt-hp",
    #     index=512,
    #     explanationModelName="gemini-2.5-flash",
    #     explanationType="oai_token-act-pair"
    # )
    # print(f"Status: {status}")
    # print(data)


    # Example: generate graph (uncomment to use)
    # status, data = generate_graph(
    #     modelId="gemma-2-2b",
    #     prompt="If dog is to bark, then cat is to",
    #     slug="meow",
    #     sourceSetName="clt-hp",
    #     desiredLogitProb=0.9,
    #     edgeThreshold=0.8,
    #     maxFeatureNodes=5000,
    #     maxNLogits=10,
    #     nodeThreshold=0.5
    # )
    # print(f"Status: {status}")
    # print(data)
    # print(_get_api_key())
    # status, data = save_subgraph(
    #     modelId="gemma-2-2b",
    #     slug="my doggo graph",
    #     displayName="test save subgraph",
    #     pinnedIds = [
    #         "2_15681_2",
    #         "E_2_0",
    #         "4_14735_2",
    #         "E_5929_2",
    #         "19_9180_3",
    #         "27_6784_5"
    #     ],
    #     supernodes=[
    #         [
    #         "supernode",
    #         "4_14735_2",
    #         "19_9180_3"
    #         ]
    #     ],
    #     pruningThreshold=0.8,
    #     densityThreshold=0.99,
        
    # )
    # print(f"Status: {status}")
    # print(data)
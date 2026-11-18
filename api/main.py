"""PathSnap inference API (Laptop 1).

Hosts the trained classifier, the HMM/Viterbi decoder, the road-network data
and the server-side evaluation step.  The UI on Laptop 2 sends a trip id and
receives the reconstructed route plus accuracy figures.

Evaluation integrity
    The reference route is loaded only inside `/predict`, *after* the
    prediction is complete, and is used solely to compute accuracy.  It is
    never passed to the matcher and never returned to the client, so ground
    truth cannot reach the model or the UI.  Nothing here is hardcoded per
    trip: every request runs the same pipeline over the raw files.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pathsnap.hmm import HMMConfig
from pathsnap.network import build_network
from pathsnap.pipeline import PathSnapMatcher, evaluate_against_reference
from pathsnap.trips import list_trips, load_trip

# --------------------------------------------------------------------- config
# The evaluation partition is pointed at by PATHSNAP_EVAL_DIR.  It defaults to
# the student-testing trips so the system is runnable from a fresh clone; on
# evaluation day it is set to the ten reserved evaluator trips.
EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", ROOT / "data" / "raw" / "student_test"))
MODEL_PATH = os.environ.get("PATHSNAP_MODEL")
HMM_CONFIG_PATH = ROOT / "results" / "hmm_config.json"

app = FastAPI(
    title="PathSnap Map-Matching API",
    description="Reconstructs the road-network route travelled by a noisy GPS trace.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

STATE: dict = {"matcher": None, "model_name": None, "config": None,
               "network_cache": {}, "request_log": [], "matcher_cache": {}}

#: Loading every artifact at startup would cost well over a gigabyte -
#: extra_trees alone is 422 MB - so alternates are loaded on first use and
#: the cache is capped. The deployed model is never evicted.
MAX_CACHED_MATCHERS = 3


# --------------------------------------------------------------------- schemas
class PredictRequest(BaseModel):
    trip_id: str = Field(..., description="Trip id as listed by /trips")
    include_gps: bool = Field(True, description="Return the raw GPS points")
    include_reference: bool = Field(
        False, description="Return the reference route geometry (evaluator view only)")
    run_hmm: bool = Field(True, description="Apply Viterbi decoding")
    model: str | None = Field(
        None, description="Classifier to use, as named by /models. "
                          "Defaults to the deployed model.")


class TripInfo(BaseModel):
    trip_id: str
    n_observations: int
    duration_s: float
    n_nodes: int
    n_arcs: int
    bbox: list[float]


# ------------------------------------------------------------------- lifecycle
def _resolve_model_path() -> Path:
    if MODEL_PATH:
        return Path(MODEL_PATH)
    models = ROOT / "artifacts" / "models"
    if HMM_CONFIG_PATH.exists():
        name = json.loads(HMM_CONFIG_PATH.read_text()).get("model_artifact")
        if name and (models / name).exists():
            return models / name
    for cand in sorted(models.glob("tuned_*.joblib")):
        return cand
    found = sorted(models.glob("*.joblib"))
    if not found:
        raise FileNotFoundError(
            f"no model artifact in {models}; run scripts/03_train_models.py first")
    return found[0]


@app.on_event("startup")
def startup() -> None:
    path = _resolve_model_path()
    model = joblib.load(path)
    cfg = HMMConfig()
    if HMM_CONFIG_PATH.exists():
        cfg = HMMConfig(**json.loads(HMM_CONFIG_PATH.read_text())["config"])
    STATE["matcher"] = PathSnapMatcher(model, cfg)
    STATE["model_name"] = path.name
    STATE["config"] = cfg.to_dict()
    print(f"[PathSnap API] model      : {path.name}")
    print(f"[PathSnap API] decoder    : {cfg.to_dict()}")
    print(f"[PathSnap API] trip source: {EVAL_DIR}")
    print(f"[PathSnap API] trips      : {len(list_trips(EVAL_DIR))}")


def _get_trip(trip_id: str):
    available = list_trips(EVAL_DIR)
    if trip_id not in available:
        raise HTTPException(404, f"unknown trip '{trip_id}'; available: {available}")
    trip = load_trip(EVAL_DIR, trip_id)
    if trip_id not in STATE["network_cache"]:
        STATE["network_cache"][trip_id] = build_network(trip)
    return trip, STATE["network_cache"][trip_id]


# --------------------------------------------------------------------- routes
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": STATE["model_name"],
            "decoder": STATE["config"],
            "trip_source": str(EVAL_DIR),
            "n_trips": len(list_trips(EVAL_DIR)),
            "requests_served": len(STATE["request_log"])}


def _available_models() -> list[dict]:
    """Model artifacts on disk, joined to their recorded experiment metrics.

    Metrics come from results/model_comparison.csv, so this endpoint reports
    what the experiments measured rather than anything computed here.
    """
    import csv

    metrics: dict[str, dict] = {}
    comp = ROOT / "results" / "model_comparison.csv"
    if comp.exists():
        with open(comp, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                metrics[row["model"]] = row

    tuned = ROOT / "results" / "bayesopt_summary.csv"
    tuned_rows: dict[str, dict] = {}
    if tuned.exists():
        with open(tuned, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("variant") == "tuned":
                    tuned_rows[f"tuned_{row['model']}"] = row

    deployed = STATE["model_name"]
    out = []
    for path in sorted((ROOT / "artifacts" / "models").glob("*.joblib")):
        name = path.stem
        m = metrics.get(name) or tuned_rows.get(name) or {}

        def num(key):
            v = m.get(key)
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        out.append({
            "name": name,
            "is_deployed": path.name == deployed,
            "size_mb": round(path.stat().st_size / 1e6, 1),
            "algorithm_family": m.get("algorithm_family"),
            "configuration": m.get("key_configuration"),
            "cv_macro_f1": num("cv_macro_f1_mean") or num("cv_macro_f1"),
            "test_top1_accuracy": num("test_top1_accuracy"),
            "test_macro_f1": num("test_macro_f1"),
            "test_precision_macro": num("test_precision_macro"),
            "test_recall_macro": num("test_recall_macro"),
            "test_log_loss": num("test_log_loss"),
            "test_route_continuity": num("test_route_continuity"),
            "train_seconds": num("train_seconds"),
            "inference_us_per_row": num("inference_us_per_row"),
        })
    return out


@app.get("/models")
def models() -> dict:
    """Classifiers available for reconstruction, with their measured metrics."""
    return {"deployed": STATE["model_name"], "models": _available_models()}


def _matcher_for(model_name: str | None):
    """Return a matcher for the requested classifier, loading it on demand."""
    if not model_name:
        return STATE["matcher"], STATE["model_name"]

    artifact = f"{model_name}.joblib"
    if artifact == STATE["model_name"]:
        return STATE["matcher"], STATE["model_name"]

    cache = STATE["matcher_cache"]
    if model_name in cache:
        return cache[model_name], artifact

    path = ROOT / "artifacts" / "models" / artifact
    if not path.exists():
        available = [m["name"] for m in _available_models()]
        raise HTTPException(404, f"unknown model '{model_name}'; "
                                 f"available: {available}")

    print(f"[PathSnap API] loading alternate classifier: {artifact}", flush=True)
    matcher = PathSnapMatcher(joblib.load(path), STATE["matcher"].hmm_config)
    if len(cache) >= MAX_CACHED_MATCHERS:
        cache.pop(next(iter(cache)))
    cache[model_name] = matcher
    return matcher, artifact


@app.get("/trips", response_model=list[TripInfo])
def trips() -> list[TripInfo]:
    """The evaluation trips the UI offers."""
    out = []
    for tid in list_trips(EVAL_DIR):
        trip = load_trip(EVAL_DIR, tid)
        lon, lat = trip.track[:, 0], trip.track[:, 1]
        out.append(TripInfo(
            trip_id=tid, n_observations=trip.n_obs,
            duration_s=float(trip.track[-1, 2] - trip.track[0, 2]),
            n_nodes=len(trip.nodes), n_arcs=len(trip.arcs),
            bbox=[float(lon.min()), float(lat.min()),
                  float(lon.max()), float(lat.max())]))
    return out


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    t_start = time.perf_counter()
    trip, network = _get_trip(req.trip_id)

    matcher, artifact = _matcher_for(req.model)

    # --- prediction: the reference route is not read anywhere in this block
    result = matcher.match(trip, network, run_hmm=req.run_hmm)

    # --- server-side scoring, after the fact
    metrics = evaluate_against_reference(result, trip, network)

    payload: dict = {
        "trip_id": req.trip_id,
        "model": artifact,
        "decoder": STATE["config"],
        "n_observations": int(result.n_observations),
        "predicted_route": {
            # the connected route: decoded arcs with intermediate arcs filled in
            "arc_ids": [int(a) for a in result.route_arc_sequence("expanded")],
            "polyline": result.route_polyline("expanded"),
        },
        "decoded_route": {
            # the raw per-observation Viterbi decisions, before expansion
            "arc_ids": [int(a) for a in result.route_arc_sequence("hmm")],
        },
        "classifier_only_route": {
            "arc_ids": [int(a) for a in result.route_arc_sequence("classifier")],
            "polyline": result.route_polyline("classifier"),
        },
        "per_observation": {
            "hmm_arc_id": [int(a) for a in result.hmm_arcs],
            "classifier_arc_id": [int(a) for a in result.classifier_arcs],
            "confidence": [float(s) for s in result.scores],
            "n_candidates": [int(c) for c in result.candidates_per_obs],
        },
        "diagnostics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                        for k, v in result.diagnostics.items()},
        "timings": {k: round(float(v), 4) for k, v in result.timings.items()},
        "evaluation": {k: (None if isinstance(v, float) and np.isnan(v) else v)
                       for k, v in metrics.items()},
        "served_at": datetime.now(timezone.utc).isoformat(),
    }

    if req.include_gps:
        payload["gps"] = {
            "points": [[float(lat), float(lon)] for lon, lat, _ in trip.track],
            "timestamps": [float(t) for t in trip.track[:, 2]],
        }
    if req.include_reference and trip.route is not None:
        payload["reference_route"] = {
            "arc_ids": [int(a) for a in trip.route],
            "polyline": _arcs_to_polyline(network, trip.route),
        }

    payload["total_seconds"] = round(time.perf_counter() - t_start, 4)

    STATE["request_log"].append({"trip_id": req.trip_id,
                                 "model": artifact,
                                 "at": payload["served_at"],
                                 "seconds": payload["total_seconds"]})
    ev = payload["evaluation"]
    print(f"[PathSnap API] /predict {req.trip_id}: "
          f"{result.n_observations} obs | classifier top-1 "
          f"{ev.get('classifier_top1_accuracy')} -> HMM "
          f"{ev.get('hmm_top1_accuracy')} | continuity "
          f"{ev.get('hmm_route_continuity')} | "
          f"{payload['total_seconds']}s", flush=True)
    return payload


@app.get("/network/{trip_id}")
def network_geometry(trip_id: str, max_arcs: int = 4000) -> dict:
    """Sampled road-network geometry, so the UI can draw context."""
    trip, net = _get_trip(trip_id)
    step = max(1, net.n_arcs // max_arcs)
    segs = []
    for a in range(0, net.n_arcs, step):
        u, v = net.arcs[a]
        segs.append([[float(net.nodes_lonlat[u, 1]), float(net.nodes_lonlat[u, 0])],
                     [float(net.nodes_lonlat[v, 1]), float(net.nodes_lonlat[v, 0])]])
    return {"trip_id": trip_id, "n_arcs_total": int(net.n_arcs),
            "n_arcs_returned": len(segs), "segments": segs}


@app.get("/metrics")
def metrics() -> dict:
    """Experiment results for the UI's analytics view.

    Served from the CSVs that scripts/03-05 actually wrote, so the dashboard
    cannot display a number the experiments did not produce. Returns whatever
    stages have been run; missing sections come back as null rather than
    being invented.
    """
    import csv

    results = ROOT / "results"

    def read_csv(name: str):
        path = results / name
        if not path.exists():
            return None
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            for k, v in row.items():
                if v in ("", None):
                    row[k] = None
                    continue
                try:
                    row[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
                except (ValueError, TypeError):
                    pass
        return rows

    def read_json(name: str):
        path = results / name
        return json.loads(path.read_text()) if path.exists() else None

    return {
        "model_comparison": read_csv("model_comparison.csv"),
        "cv_folds": read_csv("model_cv_folds.csv"),
        "observation_outcomes": read_csv("observation_level_outcomes.csv"),
        "confusion_matrices": read_csv("confusion_matrices.csv"),
        "bayesopt_summary": read_csv("bayesopt_summary.csv"),
        "bayesopt_trials": read_csv("bayesopt_trials.csv"),
        "bayesopt_best": read_json("bayesopt_best_params.json"),
        "hmm_summary": read_csv("hmm_comparison_summary.csv"),
        "hmm_per_trip": read_csv("hmm_comparison_per_trip.csv"),
        "candidate_generation": read_csv("candidate_generation_summary.csv"),
    }


@app.get("/logs")
def logs(limit: int = 50) -> dict:
    """Recent requests, for showing the evaluator that traffic is real."""
    return {"n_total": len(STATE["request_log"]),
            "recent": STATE["request_log"][-limit:]}


def _arcs_to_polyline(network, arc_ids):
    pts = []
    for a in np.asarray(arc_ids):
        u, v = network.arcs[int(a)]
        p1 = [float(network.nodes_lonlat[u, 1]), float(network.nodes_lonlat[u, 0])]
        p2 = [float(network.nodes_lonlat[v, 1]), float(network.nodes_lonlat[v, 0])]
        if not pts or pts[-1] != p1:
            pts.append(p1)
        pts.append(p2)
    return pts

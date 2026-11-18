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
               "network_cache": {}, "request_log": []}


# --------------------------------------------------------------------- schemas
class PredictRequest(BaseModel):
    trip_id: str = Field(..., description="Trip id as listed by /trips")
    include_gps: bool = Field(True, description="Return the raw GPS points")
    include_reference: bool = Field(
        False, description="Return the reference route geometry (evaluator view only)")
    run_hmm: bool = Field(True, description="Apply Viterbi decoding")


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

    # --- prediction: the reference route is not read anywhere in this block
    result = STATE["matcher"].match(trip, network, run_hmm=req.run_hmm)

    # --- server-side scoring, after the fact
    metrics = evaluate_against_reference(result, trip, network)

    payload: dict = {
        "trip_id": req.trip_id,
        "model": STATE["model_name"],
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

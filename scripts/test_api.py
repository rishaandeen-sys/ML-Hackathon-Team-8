"""Smoke test for the inference API, without needing a running server.

Exercises /health, /trips and /predict in-process with FastAPI's TestClient,
and asserts the evaluation-integrity properties: the reference route must not
leak into the response unless it is explicitly requested.

Run:  python scripts/test_api.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient

import api.main as api_module


def main() -> int:
    failures = []

    with TestClient(api_module.app) as client:
        health = client.get("/health")
        assert health.status_code == 200, health.text
        h = health.json()
        print(f"/health      : model={h['model']} trips={h['n_trips']}")
        print(f"               decoder={h['decoder']}")

        trips = client.get("/trips")
        assert trips.status_code == 200, trips.text
        tj = trips.json()
        print(f"/trips       : {len(tj)} trips, first={tj[0]['trip_id']} "
              f"({tj[0]['n_observations']} obs)")
        if len(tj) != 10:
            failures.append(f"expected 10 evaluation trips, got {len(tj)}")

        # smallest trip, so the smoke test stays quick
        smallest = min(tj, key=lambda t: t["n_observations"])["trip_id"]
        print(f"\n/predict     : {smallest} ...", flush=True)
        r = client.post("/predict", json={"trip_id": smallest,
                                          "include_gps": True,
                                          "include_reference": False,
                                          "run_hmm": True})
        assert r.status_code == 200, r.text
        p = r.json()
        ev = p["evaluation"]
        print(f"  observations        : {p['n_observations']:,}")
        print(f"  predicted route arcs: {len(p['predicted_route']['arc_ids']):,}")
        print(f"  polyline points     : {len(p['predicted_route']['polyline']):,}")
        print(f"  classifier top-1    : {ev['classifier_top1_accuracy']:.4f}")
        print(f"  HMM top-1           : {ev['hmm_top1_accuracy']:.4f} "
              f"({ev['hmm_improvement_pp']:+.2f} pp)")
        print(f"  continuity clf->hmm : {ev['classifier_route_continuity']:.4f} "
              f"-> {ev['hmm_route_continuity']:.4f}")
        print(f"  timings             : {p['timings']}")

        # --- evaluation-integrity assertions
        if "reference_route" in p:
            failures.append("reference route leaked into the default response")
        if not p["predicted_route"]["polyline"]:
            failures.append("empty predicted polyline")
        if len(p["per_observation"]["hmm_arc_id"]) != p["n_observations"]:
            failures.append("per-observation arc list length mismatch")

        r2 = client.post("/predict", json={"trip_id": smallest,
                                           "include_reference": True})
        if "reference_route" not in r2.json():
            failures.append("reference route missing when explicitly requested")

        bad = client.post("/predict", json={"trip_id": "does_not_exist"})
        if bad.status_code != 404:
            failures.append(f"unknown trip returned {bad.status_code}, expected 404")

        net = client.get(f"/network/{smallest}")
        assert net.status_code == 200
        print(f"/network     : {net.json()['n_arcs_returned']:,} of "
              f"{net.json()['n_arcs_total']:,} arcs returned")

        logs = client.get("/logs").json()
        print(f"/logs        : {logs['n_total']} requests served")

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("All API checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

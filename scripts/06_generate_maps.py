"""Stage 6: generate the interactive Folium maps.

For each selected trip the map carries four toggleable layers -- road network,
raw GPS observations, reference route and predicted route -- plus the
classifier-only route so the effect of Viterbi decoding is visible directly on
the map rather than only in a table.

Run:  python scripts/06_generate_maps.py [--trips ID ...] [--partition P]
Out:  results/maps/match_<partition>_<trip>.html
      results/maps/index.html
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import pandas as pd

from pathsnap.hmm import HMMConfig
from pathsnap.network import build_network
from pathsnap.pipeline import PathSnapMatcher, evaluate_against_reference
from pathsnap.trips import list_trips, load_trip
from pathsnap.viz import folium_trip_map

RESULTS = ROOT / "results"
MAPS = RESULTS / "maps"
RAW = ROOT / "data" / "raw"


def load_matcher():
    cfg_path = RESULTS / "hmm_config.json"
    cfg_json = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    cfg = HMMConfig(**cfg_json["config"]) if "config" in cfg_json else HMMConfig()
    models = ROOT / "artifacts" / "models"
    name = cfg_json.get("model_artifact")
    path = models / name if name and (models / name).exists() else None
    if path is None:
        tuned = sorted(models.glob("tuned_*.joblib"))
        path = tuned[0] if tuned else sorted(models.glob("*.joblib"))[0]
    print(f"Model: {path.name}  |  decoder: beta={cfg.beta}, "
          f"penalty={cfg.unreachable_penalty}")
    return PathSnapMatcher(joblib.load(path), cfg), path.name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="student_test")
    ap.add_argument("--trips", nargs="*", default=None)
    ap.add_argument("--train-examples", type=int, default=2,
                    help="additional example maps from the training partition")
    args = ap.parse_args()

    MAPS.mkdir(parents=True, exist_ok=True)
    matcher, model_name = load_matcher()

    jobs = [(args.partition, t) for t in
            (args.trips or list_trips(RAW / args.partition))]
    if args.train_examples:
        jobs += [("train", t) for t in list_trips(RAW / "train")[:args.train_examples]]

    index_rows = []
    for partition, tid in jobs:
        trip = load_trip(RAW / partition, tid)
        net = build_network(trip)
        res = matcher.match(trip, net, run_hmm=True)
        ev = evaluate_against_reference(res, trip, net)

        title = (f"{partition} / trip {tid} — top-1 "
                 f"{ev.get('hmm_top1_accuracy', float('nan')) * 100:.1f}%, "
                 f"continuity {ev.get('hmm_route_continuity', float('nan')) * 100:.1f}%")
        m = folium_trip_map(trip, net,
                            predicted_arcs=res.hmm_arcs,
                            classifier_arcs=res.classifier_arcs,
                            title=title)
        out = MAPS / f"match_{partition}_{tid}.html"
        m.save(str(out))
        index_rows.append({
            "partition": partition, "trip_id": tid, "file": out.name,
            "n_observations": trip.n_obs,
            "classifier_top1": ev.get("classifier_top1_accuracy"),
            "hmm_top1": ev.get("hmm_top1_accuracy"),
            "classifier_continuity": ev.get("classifier_route_continuity"),
            "hmm_continuity": ev.get("hmm_route_continuity"),
        })
        print(f"  {out.name}: top-1 {ev.get('hmm_top1_accuracy', 0) * 100:.2f}%, "
              f"continuity {ev.get('hmm_route_continuity', 0) * 100:.2f}%", flush=True)

    idx = pd.DataFrame(index_rows)
    idx.to_csv(RESULTS / "map_index.csv", index=False)
    write_index_page(idx, model_name)
    print(f"\n{len(idx)} maps -> {MAPS}")


def write_index_page(idx: pd.DataFrame, model_name: str) -> None:
    rows = "\n".join(
        f'<tr><td>{r.partition}</td><td><a href="{r.file}">{r.trip_id}</a></td>'
        f'<td>{r.n_observations:,}</td>'
        f'<td>{r.classifier_top1 * 100:.2f}%</td><td>{r.hmm_top1 * 100:.2f}%</td>'
        f'<td>{r.classifier_continuity * 100:.2f}%</td>'
        f'<td>{r.hmm_continuity * 100:.2f}%</td></tr>'
        for r in idx.itertuples())
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>PathSnap maps</title>
<style>
 body{{font:15px/1.5 system-ui,sans-serif;margin:40px auto;max-width:900px;color:#111}}
 h1{{margin-bottom:4px}} p.sub{{color:#666;margin-top:0}}
 table{{border-collapse:collapse;width:100%;margin-top:18px}}
 th,td{{padding:8px 10px;border-bottom:1px solid #e5e7eb;text-align:left}}
 th{{background:#f9fafb;font-weight:600}} a{{color:#1f6fd0}}
</style></head><body>
<h1>PathSnap route reconstructions</h1>
<p class="sub">Interactive Folium maps, one per trip. Model: <code>{model_name}</code></p>
<table><tr><th>Partition</th><th>Trip</th><th>GPS points</th>
<th>Top-1 (classifier)</th><th>Top-1 (HMM)</th>
<th>Continuity (classifier)</th><th>Continuity (HMM)</th></tr>
{rows}</table></body></html>"""
    (MAPS / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()

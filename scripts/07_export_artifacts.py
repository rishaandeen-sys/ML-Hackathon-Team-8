"""Stage 7: export the deployable inference bundle.

Collects everything the API needs into artifacts/ and writes a manifest with
SHA-256 checksums, the headline metrics and the exact configuration, so the
system that gets demonstrated can be tied back to the experiments that
produced it.

Run:  python scripts/07_export_artifacts.py
Out:  artifacts/pathsnap_model.joblib
      artifacts/inference_config.json
      artifacts/manifest.json
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from pathsnap.candidates import (DEFAULT_MAX_CANDIDATES, DEFAULT_RADIUS_M,
                                 FEATURE_NAMES)
from pathsnap.dataset import LABEL_RESIDUAL_MAX_M
from pathsnap.hmm import HMMConfig

RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    hmm_cfg_path = RESULTS / "hmm_config.json"
    if not hmm_cfg_path.exists():
        raise SystemExit("run scripts/05_hmm_viterbi.py first")
    hmm_json = json.loads(hmm_cfg_path.read_text())

    src = ARTIFACTS / "models" / hmm_json["model_artifact"]
    dst = ARTIFACTS / "pathsnap_model.joblib"

    # Re-dump with compression rather than copying: an uncompressed forest can
    # exceed GitHub's 100 MB per-file limit, and the deployed bundle is meant
    # to be committed.  joblib reloads a compressed file transparently.
    import joblib
    model = joblib.load(src)
    joblib.dump(model, dst, compress=("zlib", 3))
    size_mb = dst.stat().st_size / 1e6
    print(f"model  : {src.name} ({src.stat().st_size / 1e6:.1f} MB) -> "
          f"{dst.name} ({size_mb:.1f} MB, zlib-3)")
    if size_mb > 90:
        print(f"  WARNING: {size_mb:.0f} MB exceeds GitHub's comfortable file size. "
              f"Consider a lighter final model, or distribute this artifact "
              f"outside git and document where to fetch it.")

    config = {
        "model_artifact": dst.name,
        "source_artifact": src.name,
        "model_name": hmm_json["model"],
        "feature_names": FEATURE_NAMES,
        "candidate_generation": {
            "radius_m": DEFAULT_RADIUS_M,
            "max_candidates": DEFAULT_MAX_CANDIDATES,
            "arc_sample_spacing_m": 25.0,
        },
        "label_residual_max_m": LABEL_RESIDUAL_MAX_M,
        "hmm": hmm_json["config"],
        "projection": "local equirectangular about the trip centroid, metres",
    }
    (ARTIFACTS / "inference_config.json").write_text(json.dumps(config, indent=2))

    headline = {}
    comp_path = RESULTS / "model_comparison.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path).sort_values("test_top1_accuracy", ascending=False)
        headline["best_untuned_model"] = comp.iloc[0]["model"]
        headline["models_compared"] = int(len(comp))
    bo_path = RESULTS / "bayesopt_summary.csv"
    if bo_path.exists():
        bo = pd.read_csv(bo_path).set_index("variant")
        headline["tuned_cv_top1"] = float(bo.loc["tuned", "cv_top1_accuracy"])
        headline["untuned_cv_top1"] = float(bo.loc["untuned", "cv_top1_accuracy"])
        headline["tuned_test_top1"] = float(bo.loc["tuned", "test_top1_accuracy"])
    hmm_path = RESULTS / "hmm_comparison_summary.csv"
    if hmm_path.exists():
        hs = pd.read_csv(hmm_path).set_index("stage")
        headline["classifier_only_test_top1"] = float(hs.loc["classifier_only", "top1_accuracy_weighted"])
        headline["classifier_plus_hmm_test_top1"] = float(hs.loc["classifier_plus_hmm", "top1_accuracy_weighted"])
        headline["classifier_only_continuity"] = float(hs.loc["classifier_only", "route_continuity"])
        headline["classifier_plus_hmm_continuity"] = float(hs.loc["classifier_plus_hmm", "route_continuity"])

    tracked = [dst, ARTIFACTS / "inference_config.json"]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "headline_metrics": headline,
        "config": config,
        "checksums": {p.name: sha256(p) for p in tracked},
        "files": {p.name: p.stat().st_size for p in tracked},
    }
    (ARTIFACTS / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("config : artifacts/inference_config.json")
    print("manifest: artifacts/manifest.json")
    print("\nheadline metrics:")
    for k, v in headline.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Run the whole PathSnap pipeline, stage by stage.
#
#   ./scripts/run_all.sh           full run
#   QUICK=1 ./scripts/run_all.sh   smaller forests / fewer trials, for a sanity pass
#
# Each stage writes to results/ and can also be run on its own.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="$ROOT/.venv/Scripts/python.exe"
[ -x "$PYTHON" ] || PYTHON="python3"

if [ ! -d "$ROOT/data/raw/train" ]; then
    echo "data/raw/train not found - unpack the dataset first (see README section 3)." >&2
    exit 1
fi

QUICK_FLAG=()
TRIALS=30
if [ "${QUICK:-0}" = "1" ]; then
    QUICK_FLAG=(--quick)
    TRIALS=8
fi

stage() {
    local name="$1"; shift
    echo
    printf '%.0s=' {1..70}; echo
    echo "  $name"
    printf '%.0s=' {1..70}; echo
    local t0=$SECONDS
    "$PYTHON" -I "$@"
    printf '  done in %.1f min\n' "$(echo "scale=2; ($SECONDS - $t0) / 60" | bc)"
}

stage "1/8  Build candidate dataset"    scripts/01_build_dataset.py
stage "2/8  Exploratory data analysis"  scripts/02_eda.py
stage "3/8  Train and compare models"   scripts/03_train_models.py "${QUICK_FLAG[@]}"
stage "3b/8 Confusion matrices"         scripts/03b_confusion_matrices.py
stage "4/8  Bayesian optimisation"      scripts/04_bayesian_optimisation.py --trials "$TRIALS"
stage "5/8  HMM and Viterbi"            scripts/05_hmm_viterbi.py
stage "6/8  Generate Folium maps"       scripts/06_generate_maps.py
stage "7/8  Export inference artifacts" scripts/07_export_artifacts.py
stage "8/8  Build the report skeleton"  scripts/08_build_report.py

cat <<'EOF'

Pipeline complete.
  results/model_comparison.csv          model comparison table
  results/bayesopt_summary.csv          tuned vs untuned
  results/hmm_comparison_per_trip.csv   classifier vs HMM
  results/maps/index.html               interactive maps
  results/report/technical_report.md    report skeleton

Serve it:  uvicorn api.main:app --host 0.0.0.0 --port 8000
EOF

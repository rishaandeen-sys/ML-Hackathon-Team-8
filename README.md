# PathSnap — Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning Hackathon · PES University, Department of CSE (RR Campus)

PathSnap reconstructs the road-network route a vehicle actually travelled from a
noisy GPS trace. It combines a supervised candidate-ranking classifier with an
HMM/Viterbi decoder that enforces road-network connectivity, and serves the
result over a FastAPI backend to a separate evaluator UI.

```
raw .track/.nodes/.arcs  ->  candidate generation  ->  feature engineering
     ->  classifier (ranks candidate arcs)  ->  HMM + Viterbi (enforces connectivity)
     ->  reconstructed route  ->  FastAPI  ->  evaluator UI
```

---

## 1. Problem

GPS fixes drift off the road: urban canyons, multipath, interference and low
sampling rates push an observation onto a parallel street or across a building.
Map matching is the task of deciding, for every observation, which directed road
segment the vehicle was really on, and of stitching those decisions into a route
that a vehicle could physically have driven.

The two halves of PathSnap address the two halves of that problem:

- **The classifier** answers *"which nearby arc best explains this fix?"* using
  geometry, heading and network context. It is accurate per observation but
  knows nothing about time: consecutive observations can land on disconnected
  streets.
- **The HMM/Viterbi decoder** answers *"which sequence of arcs is jointly most
  likely?"*, using the classifier's scores as emissions and road-network travel
  distance as transitions. It cannot fix a candidate set that never contained
  the right arc, but it removes the physically impossible jumps.

---

## 2. Repository layout

```
PathSnap/
├── src/pathsnap/              core library (importable, used by every stage)
│   ├── geo.py                 projection and planar geometry
│   ├── trips.py               raw-file loading and the Trip container
│   ├── network.py             road network, spatial index, bounded Dijkstra
│   ├── labeling.py            monotonic DP alignment: observation -> route arc
│   ├── candidates.py          candidate generation + the 26 features
│   ├── dataset.py             partition -> training table
│   ├── metrics.py             top-1, macro F1, log loss, continuity, sequence
│   ├── hmm.py                 HMM formulation and Viterbi decoder
│   ├── pipeline.py            end-to-end matcher used by stage 5, maps and API
│   └── viz.py                 Folium map construction
├── scripts/                   the pipeline, in order
│   ├── 01_build_dataset.py    candidate tables for both partitions
│   ├── 02_eda.py              inventory, data quality, plots, Folium map
│   ├── 03_train_models.py     ten models, group CV, comparison + plots
│   ├── 04_bayesian_optimisation.py   Optuna TPE search on the chosen model
│   ├── 05_hmm_viterbi.py      decoder tuning + classifier-vs-HMM comparison
│   ├── 06_generate_maps.py    per-trip interactive maps
│   └── 07_export_artifacts.py deployable bundle + manifest
├── notebooks/
│   └── pathsnap_walkthrough.ipynb   narrated end-to-end walkthrough
├── api/main.py                FastAPI backend (Laptop 1)
├── ui/app.py                  Streamlit evaluator interface (Laptop 2)
├── artifacts/                 trained model, inference config, manifest
├── results/                   every CSV, figure, map and log reported on
├── docs/
│   ├── DEMO_CHECKLIST.md      evaluation-day runbook for the two laptops
│   └── provided/              the organisers' supporting documents
├── requirements.txt
└── README.md
```

---

## 3. Installation

Python 3.12 is what the reported results were produced with.

```bash
git clone <this repository>
cd PathSnap

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Dataset

The raw dataset is not committed (≈91 MB, and it is the organisers' to
distribute). Unpack `Student folder.zip` so the tree looks like this:

```
data/raw/train/00000001/00000001.{track,nodes,arcs,route}
data/raw/train/...                              (80 trips)
data/raw/student_test/00000000/...              (10 trips)
```

On evaluation day the ten reserved evaluator trips go in their own directory
and the API is pointed at it with `PATHSNAP_EVAL_DIR` (see §7).

---

## 4. Running the pipeline

Each stage writes its outputs to `results/` and can be run independently once
the stage before it has run.

```bash
python scripts/01_build_dataset.py          #  ~45 s
python scripts/02_eda.py                    #  ~4 min
python scripts/03_train_models.py           #  ~1-2 h  (ten models x 5 folds)
python scripts/04_bayesian_optimisation.py  #  ~1 h    (30 TPE trials)
python scripts/05_hmm_viterbi.py            #  ~20 min
python scripts/06_generate_maps.py          #  ~3 min
python scripts/07_export_artifacts.py       #  seconds
```

Or run everything in one go:

```powershell
.\scripts\run_all.ps1           # Windows
```
```bash
./scripts/run_all.sh            # macOS / Linux
```

Useful flags: `03_train_models.py --quick` (smaller forests, for a fast sanity
run), `--models NAME ...` (subset), `04_...py --trials N`, `05_...py
--skip-tuning`.

---

## 5. Method

### 5.1 Data understanding

Every trip ships **its own road sub-network**, so node and arc ids are local to
a trip and must never be compared across trips. Arcs are **directed**: a two-way
street appears twice, with opposite bearing, and the reference route is a
sequence of directed arc ids — so the direction of travel is part of what has
to be predicted, which is why heading features matter.

`.track` is nominally 1 Hz (94% of training steps are exactly 1 s apart),
`.nodes` is indexed by row, `.arcs` gives `from_node to_node` with the row index
as the arc id, and `.route` is the chronological sequence of traversed arc ids.

Distances use a **local equirectangular projection to metres** centred on each
trip. At city scale the error is well under 0.1%, it is exactly invertible, and
it is far cheaper than per-pair geodesics — degrees of longitude and latitude
are not interchangeable units of distance, so working in raw degrees would
distort every distance and heading in the feature set.

### 5.2 Deriving per-observation labels

`.route` says which arcs were traversed, not which observation belongs to which
arc, and the two sequences have different lengths. Choosing the nearest route
arc independently per observation lets GPS noise push the assignment backwards
and forwards along the route. Instead `labeling.py` solves a **monotonic
alignment by dynamic programming**, forcing the assigned route index to be
non-decreasing in time:

```
f[t, r] = cost[t, r] + min over r' <= r of f[t-1, r']
```

A running prefix-minimum makes this O(T × R).

**A documented limitation.** The resulting residuals are sharply bimodal: 75% of
training observations sit within 11 m of their assigned arc, but a tail beyond
~90 m comes from trips whose supplied `.route` covers only part of the journey —
trips `00000033`, `00000054` and `00000002` cover just 13–19% of their GPS
track. For those stretches there is no reference arc to learn from. Observations
whose residual exceeds **30 m** are therefore flagged `label_valid = 0`, kept in
the table for inspection, and excluded from fitting and scoring. That keeps
87.5% of training observations and 99.2% of student-testing observations, and
the candidate miss rate on what remains (0.26%) matches the student-testing
partition exactly — which is the evidence that the cut separates bad labels
rather than hard examples.

### 5.3 Candidate generation and features

For each observation, arcs within **60 m** are retrieved from a KD-tree built
over points sampled every 25 m along each arc (sampling, rather than indexing
midpoints, is what stops a long arc being missed when the query point is near
its middle). The nearest **10** are kept, giving ~7.2 candidates per observation
and a candidate miss rate of 0.26% on reliably-labelled training observations —
that miss rate is the hard ceiling on top-1 accuracy.

Each (observation, candidate) pair carries 26 features in four groups:

| Group | Features |
|---|---|
| Geometry | `dist_m`, `dist_rank`, `dist_minus_best`, `dist_ratio_best`, `dist_z_in_obs`, `arc_length`, `proj_t`, `proj_is_endpoint` |
| Heading | `heading_diff`, `heading_cos`, `heading_valid`, `turn_angle` |
| Kinematics | `speed_mps`, `accel_mps2`, `dt_prev`, `dt_next`, `step_prev_m`, `step_next_m` |
| Network / context | `arc_out_degree`, `arc_in_degree`, `has_reverse_arc`, `n_candidates`, `local_arc_density`, `dist_prev_obs_to_arc`, `dist_next_obs_to_arc`, `obs_frac` |

`heading_cos` is the one that separates the two directions of a two-way street,
which pure distance never can: both directed arcs have identical geometry and
therefore identical `dist_m`.

### 5.4 Models

Ten models across seven algorithm families are trained with `GroupKFold(5)`
grouped by trip, so no trip is ever split across folds. Full results, including
every per-fold score, are in `results/model_comparison.csv` and
`results/model_cv_folds.csv`; the plots are in `results/figures/`.

### 5.5 Bayesian hyperparameter optimisation

Optuna's **TPE sampler** — a Bayesian method: it models the density of good
versus bad trials and samples where their ratio is highest, so each trial uses
what the previous ones revealed. The objective is **group-CV observation-level
top-1 accuracy**, because that is what the HMM consumes and what route quality
depends on; a model can lift its binary macro F1 by rejecting obviously-wrong
candidates without ever improving the choice that matters. Search records are in
`results/bayesopt_trials.csv`, the tuned-versus-untuned comparison in
`results/bayesopt_summary.csv`.

### 5.6 HMM and Viterbi

- **Hidden states** — the candidate arcs of each observation. The state space
  changes from step to step: a lattice, not a fixed alphabet.
- **Emissions** — the classifier's scores, normalised over each observation's
  candidates.
- **Transitions** — Newson & Krumm (2009). Compare on-road travel distance
  `d_route = (1 - t_i)·len_i + shortest_path(head_i, tail_j) + t_j·len_j`
  against straight-line `d_gps`; a real move has them close, a teleport does
  not. `P(j | i) ∝ exp(-|d_route - d_gps| / beta)`.
- **Unreachable pairs** take a fixed penalty rather than being forbidden, so a
  thin candidate set or a GPS dropout can never make the lattice dead-end.
- **Gaps** — when `dt` exceeds 5 s the search radius grows and beta is relaxed,
  letting the decoder bridge the hole instead of forcing a local jump.
- Shortest paths come from a **bounded Dijkstra**: at 1 Hz consecutive fixes are
  metres apart, so the explored subgraph stays tiny and no all-pairs
  precomputation is needed.

Decoder parameters (`beta`, unreachable penalty) are selected on **training
trips only** — `results/hmm_parameter_search.csv` — so the student-testing
numbers stay honest.

---

## 6. Results

All figures below are produced by the scripts in this repository and are
reproducible from the committed code. See `results/` for the underlying CSVs.

- `results/model_comparison.csv` — the ten-model comparison table
- `results/model_cv_folds.csv` — every per-fold score
- `results/bayesopt_summary.csv` — tuned vs untuned
- `results/hmm_comparison_per_trip.csv` — classifier-only vs classifier + HMM
- `results/maps/index.html` — all interactive maps, with per-trip scores

---

## 7. Two-laptop deployment

### Laptop 1 — model and API

```bash
# point the API at the evaluation trips (defaults to data/raw/student_test)
# Windows
set PATHSNAP_EVAL_DIR=C:\path\to\evaluator_test
# macOS / Linux
export PATHSNAP_EVAL_DIR=/path/to/evaluator_test

uvicorn api.main:app --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` is what makes the API reachable from Laptop 2. Find the
machine's LAN address with `ipconfig` (Windows) or `ip addr` (Linux). The API
prints a line per request, which is the log to show the evaluator.

| Endpoint | Purpose |
|---|---|
| `GET /health` | model name, decoder config, trip count, requests served |
| `GET /trips` | the evaluation trips the UI lists |
| `POST /predict` | run the pipeline on one trip, return route + server-side metrics |
| `GET /network/{trip_id}` | sampled road-network geometry for map context |
| `GET /logs` | recent requests |

### Laptop 2 — evaluator UI

```bash
streamlit run ui/app.py
# then set the API base URL in the sidebar, e.g. http://192.168.1.50:8000
```

The UI holds no model, no road network and no ground truth. It lists the ten
trips, sends the selected id, and draws what comes back.

### Evaluation integrity

- The reference route is read **only inside `/predict`, after** the prediction
  is complete, and only to compute accuracy. It is never passed to the matcher
  and is omitted from the response unless `include_reference` is explicitly set.
- Nothing is hardcoded per trip: every request runs the same pipeline over the
  raw files, and the same `PathSnapMatcher` class is used by the training-time
  scripts and the API.
- `PATHSNAP_EVAL_DIR` is the only thing that changes between the student-testing
  and evaluator partitions.

---

## 8. Reproducibility

- Seed `42` throughout (`numpy`, scikit-learn, LightGBM, XGBoost, Optuna).
- `GroupKFold` by trip in every cross-validation, so no trip crosses folds.
- `requirements.txt` pins the exact versions used.
- `artifacts/manifest.json` records SHA-256 checksums of the deployed model and
  config alongside the headline metrics and the platform they came from.
- Every number in the report is traceable to a CSV in `results/`; no metric is
  typed in by hand.

## 9. Known limitations

- Three training trips have reference routes covering only 13–19% of their GPS
  track; those stretches are excluded from fitting and scoring rather than
  silently trained on (§5.2).
- Candidate generation imposes a ~0.26% ceiling on top-1 accuracy. Widening the
  radius lowers it slightly, at a roughly linear cost in features and decoding.
- The reference routes are *reference* information, not independently verified
  ground truth; reported accuracies are agreement with them, not with physical
  truth.
- The local equirectangular projection is appropriate for this city-scale
  dataset and would need revisiting for data spanning a much larger area.

## 10. References

- Newson, P. & Krumm, J. (2009). *Hidden Markov Map Matching Through Noise and
  Sparseness.* ACM SIGSPATIAL.
- Viterbi, A. (1967). *Error bounds for convolutional codes and an
  asymptotically optimum decoding algorithm.* IEEE Trans. Information Theory.
- Bergstra, J. et al. (2011). *Algorithms for Hyper-Parameter Optimization.*
  NeurIPS. (TPE)

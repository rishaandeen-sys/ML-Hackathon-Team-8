"""Stage 8: assemble the technical-report skeleton from the actual results.

Every table and number in the output is read from the CSVs in results/, so
nothing can drift from what the experiments produced. The written analysis is
deliberately NOT generated: each section that calls for the team's own
interpretation is left as an explicit TODO block.

That split is intentional. The assessment policy prohibits submitting
generated analysis as evidence of experiments, and requires every team member
to be able to defend the claims in the report. Auto-filling the numbers
removes transcription errors; the reasoning has to be the team's own.

Run:  python scripts/08_build_report.py
Out:  results/report/technical_report.md
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

RESULTS = ROOT / "results"
OUT = RESULTS / "report"


def read_csv(name: str) -> pd.DataFrame | None:
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


def read_json(name: str) -> dict | None:
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def table(df: pd.DataFrame, cols: list[str] | None = None, nd: int = 4) -> str:
    if df is None or df.empty:
        return "_(not available — run the corresponding stage)_\n"
    d = df[cols] if cols else df
    return d.round(nd).to_markdown(index=False) + "\n"


def todo(*prompts: str) -> str:
    body = "\n".join(f"> - {p}" for p in prompts)
    return f"\n> **TEAM ANALYSIS REQUIRED — write this yourself:**\n{body}\n"


def fig(name: str, caption: str) -> str:
    return f"\n![{caption}](../figures/{name})\n\n*{caption}*\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    cand = read_csv("candidate_generation_summary.csv")
    eda_part = read_csv("eda_partition_summary.csv")
    eda_dq = read_csv("eda_data_quality.csv")
    trips = read_csv("eda_trip_inventory.csv")
    comp = read_csv("model_comparison.csv")
    folds = read_csv("model_cv_folds.csv")
    bo_sum = read_csv("bayesopt_summary.csv")
    bo_trials = read_csv("bayesopt_trials.csv")
    bo_best = read_json("bayesopt_best_params.json")
    hmm_trip = read_csv("hmm_comparison_per_trip.csv")
    hmm_sum = read_csv("hmm_comparison_summary.csv")
    hmm_cfg = read_json("hmm_config.json")
    maps = read_csv("map_index.csv")

    L: list[str] = []
    A = L.append

    A("# PathSnap — Technical Report\n")
    A("**UE24CS352A Machine Learning Hackathon · PES University, "
      "Department of CSE (RR Campus)**\n")
    A(f"Team: _______________  ·  Members: _______________  ·  "
      f"Generated: {date.today().isoformat()}\n")
    A("Repository: https://github.com/rishaandeen-sys/PathSnap\n")
    A("---\n")
    A("> **How to use this file.** Every table and figure below is filled in "
      "automatically from the CSVs in `results/`, so the numbers match the "
      "experiments exactly. The blocks marked **TEAM ANALYSIS REQUIRED** are "
      "deliberately empty — write those in your own words, then export this "
      "file to PDF.\n")
    A("---\n")

    # ---------------------------------------------------------------- 2. EDA
    A("## 2. Dataset Understanding and Exploratory Data Analysis\n")
    A("### 2.1 Partition summary\n")
    A(table(eda_part, nd=3))
    A("### 2.2 Data-quality checks\n")
    if eda_dq is not None:
        A(eda_dq.set_index("partition").T.to_markdown() + "\n")
    A("### 2.3 Candidate generation\n")
    A(table(cand, nd=5))
    if cand is not None:
        tr = cand[cand.partition == "train"].iloc[0]
        A(f"\nThe candidate miss rate of **{tr.candidate_miss_rate:.4%}** on "
          f"reliably-labelled training observations is the hard ceiling on "
          f"top-1 accuracy: an observation whose reference arc was never "
          f"retrieved cannot be classified correctly by any model.\n")
        A(f"\n**Label reliability.** {tr.frac_observations_reliably_labelled:.2%} "
          f"of training observations have a reference arc within "
          f"{tr.label_residual_max_m:.0f} m. The remainder come from trips whose "
          f"supplied `.route` covers only part of the GPS track and are excluded "
          f"from fitting and scoring.\n")

    A("### 2.4 Preprocessing\n")
    A("- Local equirectangular projection to metres, centred per trip.\n"
      "- Per-observation reference arc recovered by monotonic DP alignment "
      "against the `.route` sequence.\n"
      "- Observations whose aligned arc is further than 30 m away flagged and "
      "excluded.\n"
      "- Candidates: arcs within 60 m, nearest 10 kept.\n"
      "- Median imputation for the neighbour-dependent features at the first "
      "and last fix; standardisation for the distance- and gradient-based "
      "models only.\n")
    A(todo("How did you preprocess the data, and why those choices?",
           "What patterns did you see during exploration?",
           "Which features did you extract and why those?",
           "Which partition was used for each analysis?"))

    A(fig("eda_overview.png", "Dataset overview"))
    A(fig("eda_trajectories.png", "Raw GPS trajectories, first 12 training trips"))
    A(fig("eda_gaps.png", "Sampling gaps and step sizes per trip"))
    A(fig("eda_extents.png", "Geographic extent of every trip"))
    A("\n**Folium map:** `results/maps/eda_trip_*.html` — include a screenshot here.\n")

    # ------------------------------------------------- 3. Model comparison
    A("\n---\n\n## 3. Model Experimentation and Comparison\n")
    A("### 3.1 Overall model comparison\n")
    if comp is not None:
        A(table(comp, ["model", "algorithm_family", "key_configuration",
                       "cv_macro_f1_mean", "test_top1_accuracy",
                       "test_route_continuity", "test_macro_f1",
                       "test_log_loss", "inference_seconds"]))
        A(f"\n{len(comp)} models were trained and compared, spanning "
          f"{comp.algorithm_family.nunique()} algorithm families. "
          f"Cross-validation is `GroupKFold(5)` grouped by trip, so no trip is "
          f"split across folds.\n")
    else:
        A("_(run scripts/03_train_models.py)_\n")

    A("\n### 3.2 Per-fold scores\n")
    if folds is not None:
        piv = folds.pivot(index="model", columns="fold", values="macro_f1")
        A(piv.round(4).to_markdown() + "\n")

    A(fig("model_cv_macro_f1.png", "Cross-validation macro F1 across all models"))
    A(fig("model_test_performance.png",
          "Student-testing top-1 accuracy and macro F1"))
    A(fig("model_log_loss.png", "Student-testing log loss"))
    A(fig("model_fold_stability.png", "Per-fold stability"))
    A(fig("model_route_continuity.png", "Route continuity before HMM decoding"))
    A(fig("model_accuracy_vs_cost.png", "Accuracy against training cost"))

    A("\n### 3.3 Per-model questions\n")
    if comp is not None:
        for _, r in comp.iterrows():
            A(f"\n#### {r.model} — {r.algorithm_family}\n")
            A(f"Configuration: `{r.key_configuration}`\n")
            A(f"\n| metric | value |\n|---|---|\n"
              f"| CV macro F1 | {r.cv_macro_f1_mean:.4f} ± {r.cv_macro_f1_std:.4f} |\n"
              f"| CV top-1 accuracy | {r.cv_top1_accuracy_mean:.4f} |\n"
              f"| Student-test top-1 accuracy | {r.test_top1_accuracy:.4f} |\n"
              f"| Student-test macro F1 | {r.test_macro_f1:.4f} |\n"
              f"| Student-test precision / recall (macro) | "
              f"{r.test_precision_macro:.4f} / {r.test_recall_macro:.4f} |\n"
              f"| Student-test log loss | {r.test_log_loss:.4f} |\n"
              f"| Route continuity | {r.test_route_continuity:.4f} |\n"
              f"| Train / inference time (s) | {r.train_seconds:.1f} / "
              f"{r.inference_seconds:.2f} |\n")
            A(todo(f"Why is {r.model} suitable for the PathSnap problem?",
                   "What are its strengths and limitations *for this task*?",
                   "Which hyperparameters mattered, and what were their initial values?",
                   "What loss or optimisation objective does it use?",
                   "Did you observe overfitting or underfitting? What is the evidence "
                   "(compare the CV and student-test columns above)?"))

    # ---------------------------------------------------- 4. Bayesian search
    A("\n---\n\n## 4. Bayesian Hyperparameter Optimisation\n")
    if bo_best:
        A(f"- **Model tuned:** `{bo_best['model']}`\n"
          f"- **Sampler:** {bo_best['sampler']} — TPE models the density of good "
          f"versus bad trials and samples where their ratio is highest, so each "
          f"trial uses what the previous ones revealed.\n"
          f"- **Objective:** {bo_best['objective']}\n"
          f"- **Trials:** {bo_best['n_trials']} "
          f"({bo_best['search_folds']}-fold group CV inside the search)\n"
          f"- **Search wall time:** {bo_best['search_seconds'] / 60:.1f} min\n"
          f"- **Best CV top-1:** {bo_best['best_cv_top1']:.4f}\n")
        A("\n**Best hyperparameters**\n\n```json\n"
          + json.dumps(bo_best["best_params"], indent=2) + "\n```\n")
    if bo_sum is not None:
        A("\n### Tuned vs untuned (identical protocol for both)\n")
        A(table(bo_sum))
    if bo_trials is not None:
        A(f"\nAll {len(bo_trials)} trials are recorded in "
          f"`results/bayesopt_trials.csv`. Top 10 by objective:\n")
        cols = [c for c in ("trial", "cv_top1_accuracy", "cv_macro_f1", "seconds")
                if c in bo_trials.columns]
        A(table(bo_trials.nlargest(10, "cv_top1_accuracy"), cols))
    A(fig("bayesopt_history.png", "TPE search history"))
    A(fig("bayesopt_params.png", "Objective against each searched hyperparameter"))
    A(fig("bayesopt_tuned_vs_untuned.png", "Tuned vs untuned"))
    A(todo("Which hyperparameters did you search, and why those?",
           "What ranges/distributions did you choose, and on what grounds?",
           "Did the tuned model beat the untuned one? By how much, and is the "
           "gap larger than the fold-to-fold variation?",
           "What did the search cost, and was it worth it?"))

    # ------------------------------------------------------------- 5. HMM
    A("\n---\n\n## 5. HMM and Viterbi Route Reconstruction\n")
    if hmm_cfg:
        A("**Decoder configuration** (selected on training trips only — "
          f"{', '.join(hmm_cfg['tuning_trips'])}):\n\n```json\n"
          + json.dumps(hmm_cfg["config"], indent=2) + "\n```\n")
    A("\n- **Hidden states:** the candidate arcs of each observation — a "
      "lattice whose state space changes per step, not a fixed alphabet.\n"
      "- **Emissions:** classifier scores, normalised over each observation's "
      "candidates.\n"
      "- **Transitions:** `P(j|i) ∝ exp(-|d_route - d_gps| / beta)`, with "
      "`d_route` the on-road travel distance via bounded Dijkstra "
      "(Newson & Krumm, 2009).\n"
      "- **Unreachable pairs:** fixed penalty rather than prohibition, so the "
      "lattice can never dead-end.\n"
      "- **GPS gaps:** above 5 s the search radius grows and beta is relaxed.\n"
      "- **Route expansion:** intermediate arcs the vehicle crossed between "
      "fixes are filled in by shortest path, turning per-observation decisions "
      "into a drivable route.\n")
    if hmm_sum is not None:
        A("\n### Classifier-only vs classifier + HMM\n")
        A(table(hmm_sum))
    if hmm_trip is not None:
        A("\n### Per-trip detail\n")
        cols = [c for c in ("trip_id", "n_observations", "classifier_top1_accuracy",
                            "hmm_top1_accuracy", "hmm_improvement_pp",
                            "classifier_route_continuity", "hmm_route_continuity",
                            "viterbi_seconds") if c in hmm_trip.columns]
        A(table(hmm_trip, cols))
        helped = int((hmm_trip.hmm_improvement_pp > 0).sum())
        hurt = int((hmm_trip.hmm_improvement_pp < 0).sum())
        A(f"\nDecoding improved **{helped}** trips and degraded **{hurt}** "
          f"of {len(hmm_trip)}.\n")
    A(fig("hmm_per_trip_comparison.png", "Top-1 accuracy and continuity per trip"))
    A(fig("hmm_improvement.png", "Effect of Viterbi decoding per trip"))
    A(fig("hmm_continuity_scatter.png", "Continuity before and after decoding"))
    A(fig("hmm_parameter_search.png", "Decoder parameter search on training trips"))
    A(todo("What are the hidden states in your implementation?",
           "How are emission probabilities calculated?",
           "How are transition probabilities calculated?",
           "How are missing candidates and large GPS gaps handled?",
           "What happens when consecutive observations suggest disconnected segments?",
           "Did decoding improve the route? Where did it NOT help, and why?"))

    # ----------------------------------------------------------- 6. Folium
    A("\n---\n\n## 6. Geographical Visualisation Using Folium\n")
    if maps is not None:
        A(table(maps, ["partition", "trip_id", "n_observations",
                       "classifier_top1", "hmm_top1", "hmm_continuity"]))
    A("\nInteractive maps: `results/maps/index.html`. Each map carries "
      "toggleable layers for the road network, raw GPS observations, the "
      "reference route, the classifier-only route and the predicted route, "
      "with a legend and layer control.\n")
    A(todo("Insert at least one map screenshot and describe what it shows: "
           "where the raw GPS drifts off-road and how the predicted route "
           "recovers it."))

    # ------------------------------------------------------ 7. API / UI
    A("\n---\n\n## 7. API, UI and Deployment\n")
    A("**Laptop 1 (model + API)**\n\n```bash\nset PATHSNAP_EVAL_DIR=...\\evaluator_test\n"
      "uvicorn api.main:app --host 0.0.0.0 --port 8000\n```\n")
    A("\n**Laptop 2 (evaluator UI)**\n\n```bash\nstreamlit run ui/app.py\n```\n")
    A("\n| Endpoint | Purpose |\n|---|---|\n"
      "| `GET /health` | model, decoder config, trip count |\n"
      "| `GET /trips` | the ten evaluation trips |\n"
      "| `POST /predict` | run the pipeline, return route + server-side metrics |\n"
      "| `GET /network/{trip_id}` | road-network geometry for map context |\n"
      "| `GET /logs` | recent requests |\n")
    A("\n**Evaluation integrity.** The reference route is read only inside "
      "`/predict`, after prediction, and only to compute accuracy. It is never "
      "passed to the matcher and is omitted from the response unless explicitly "
      "requested. Nothing is hardcoded per trip.\n")
    A(todo("Insert screenshots of the working UI.",
           "Insert a screenshot of the API logs on Laptop 1 showing the "
           "evaluator's requests arriving."))

    # ------------------------------------------------------- 8. Integrity
    A("\n---\n\n## 8. Academic Integrity and AI Use\n")
    A(todo("State which parts were AI-assisted and which were the team's own "
           "work, as the event's policy requires.",
           "Confirm every member can explain the code, experiments and results."))

    A("\n---\n\n## Appendix: reproducibility\n")
    cfgj = read_json("experiment_config.json")
    if cfgj:
        A(f"- Seed: `{cfgj['seed']}`, folds: `{cfgj['n_folds']}`\n"
          f"- Features ({len(cfgj['features'])}): "
          f"`{', '.join(cfgj['features'])}`\n"
          f"- Training rows: {cfgj['n_train_rows']:,}; "
          f"student-test rows: {cfgj['n_test_rows']:,}\n")
    A("- Source CSVs for every table above are in `results/`.\n")

    out = OUT / "technical_report.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out}")
    print(f"  {len(L)} blocks, {out.stat().st_size / 1024:.1f} KB")
    missing = [n for n, d in [("model_comparison", comp),
                              ("bayesopt_summary", bo_sum),
                              ("hmm_comparison_summary", hmm_sum),
                              ("map_index", maps)] if d is None]
    if missing:
        print(f"  still pending: {', '.join(missing)}")


if __name__ == "__main__":
    main()

"""Stage 3b: confusion matrices and threshold diagnostics per model.

Reads the saved student-testing predictions from stage 3, so it needs no
refitting and can be re-run at any time.

Two views per model, because they answer different questions:

  * the **candidate-level** confusion matrix is the binary problem the model
    was actually fitted on -- is this (observation, candidate) pair the right
    one?  It is dominated by the negative class, since only about one
    candidate in seven is correct.
  * the **observation-level** outcome is what the route depends on -- for each
    observation, was the top-ranked candidate the reference arc, was it a
    different candidate, or was the reference arc never retrieved at all?

Run:  python scripts/03b_confusion_matrices.py
Out:  results/figures/confusion_matrices.png
      results/figures/threshold_sensitivity.png
      results/confusion_matrices.csv
      results/observation_level_outcomes.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9})


def model_files() -> dict[str, Path]:
    out = {}
    for p in sorted(RESULTS.glob("test_predictions_*.parquet")):
        name = p.stem.replace("test_predictions_", "")
        out[name] = p
    return out


def observation_outcomes(df: pd.DataFrame) -> dict:
    """Split observations into correct / wrong-candidate / not-retrieved."""
    grp = df.groupby(["trip_id", "obs_idx"])
    best_idx = grp["score"].idxmax()
    best = df.loc[best_idx]
    retrieved = grp["label"].max()
    n = len(best)
    correct = int(best["label"].sum())
    not_retrieved = int((retrieved == 0).sum())
    wrong = n - correct - not_retrieved
    return {"n_observations": n, "top1_correct": correct,
            "top1_wrong_candidate": wrong,
            "reference_not_retrieved": not_retrieved,
            "top1_accuracy": correct / n,
            "wrong_candidate_rate": wrong / n,
            "candidate_miss_rate": not_retrieved / n}


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    files = model_files()
    if not files:
        raise SystemExit("no test_predictions_*.parquet; run scripts/03 first")
    print(f"Found predictions for {len(files)} models")

    cm_rows, obs_rows = [], []
    frames = {}
    for name, path in files.items():
        df = pd.read_parquet(path)
        frames[name] = df
        y = df["label"].to_numpy()
        pred = (df["score"].to_numpy() >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        cm_rows.append({"model": name, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                        "precision": tp / (tp + fp) if tp + fp else np.nan,
                        "recall": tp / (tp + fn) if tp + fn else np.nan,
                        "specificity": tn / (tn + fp) if tn + fp else np.nan,
                        "macro_f1": f1_score(y, pred, average="macro",
                                             zero_division=0)})
        obs_rows.append({"model": name, **observation_outcomes(df)})
        print(f"  {name:28s} top-1 {obs_rows[-1]['top1_accuracy']:.4f}")

    cm = pd.DataFrame(cm_rows).sort_values("macro_f1", ascending=False)
    obs = pd.DataFrame(obs_rows).sort_values("top1_accuracy", ascending=False)
    cm.to_csv(RESULTS / "confusion_matrices.csv", index=False)
    obs.to_csv(RESULTS / "observation_level_outcomes.csv", index=False)

    # ---- grid of normalised candidate-level confusion matrices
    n = len(cm)
    ncol = min(5, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 2.9 * nrow))
    for ax, (_, r) in zip(np.ravel(np.atleast_1d(axes)), cm.iterrows()):
        mat = np.array([[r.tn, r.fp], [r.fn, r.tp]], dtype=float)
        norm = mat / mat.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{norm[i, j]:.2f}\n({int(mat[i, j]):,})",
                        ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1],
               xticklabels=["pred 0", "pred 1"],
               yticklabels=["true 0", "true 1"],
               title=f"{r.model}\nmacro F1 {r.macro_f1:.3f}")
        ax.tick_params(labelsize=7)
        ax.title.set_fontsize(8)
    for ax in np.ravel(np.atleast_1d(axes))[n:]:
        ax.axis("off")
    fig.suptitle("Candidate-level confusion matrices, student-testing partition "
                 "(row-normalised)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "confusion_matrices.png")
    plt.close(fig)

    # ---- observation-level outcome breakdown
    fig, ax = plt.subplots(figsize=(9, 4.2))
    idx = np.arange(len(obs))
    ax.barh(idx, obs.top1_accuracy, color="#55A868", label="top-1 correct")
    ax.barh(idx, obs.wrong_candidate_rate, left=obs.top1_accuracy,
            color="#DD8452", label="wrong candidate chosen")
    ax.barh(idx, obs.candidate_miss_rate,
            left=obs.top1_accuracy + obs.wrong_candidate_rate,
            color="#C44E52", label="reference never retrieved")
    ax.set(yticks=idx, yticklabels=obs.model, xlim=(0, 1),
           xlabel="share of student-testing observations",
           title="Where each model's observations end up")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "observation_outcomes.png")
    plt.close(fig)

    # ---- threshold sensitivity of macro F1
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    thresholds = np.linspace(0.05, 0.95, 19)
    for name, df in frames.items():
        y = df["label"].to_numpy()
        s = df["score"].to_numpy()
        ax.plot(thresholds,
                [f1_score(y, (s >= t).astype(int), average="macro",
                          zero_division=0) for t in thresholds],
                lw=1.3, marker="o", ms=3, label=name)
    ax.axvline(0.5, color="k", ls="--", lw=0.8)
    ax.set(xlabel="decision threshold", ylabel="macro F1",
           title="Macro F1 against decision threshold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "threshold_sensitivity.png")
    plt.close(fig)

    print("\n=== observation-level outcomes ===")
    print(obs.round(4).to_string(index=False))
    print(f"\nfigures -> {FIGS}")


if __name__ == "__main__":
    main()

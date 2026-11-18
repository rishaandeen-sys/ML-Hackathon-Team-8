# Evaluation-day checklist

Two laptops on the same network. Laptop 1 holds the model, the road-network
data and the ground truth; Laptop 2 holds only the interface.

---

## Before the evaluator arrives

### Laptop 1 — model and API

1. **Place the evaluator trips.** Unpack the ten reserved trips into their own
   directory, in the same layout as the other partitions:

   ```
   data/raw/evaluator_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}
   ```

2. **Confirm the artifacts are present.**

   ```bash
   python -c "import json; print(json.load(open('artifacts/manifest.json'))['headline_metrics'])"
   ```

   If `artifacts/pathsnap_model.joblib` is missing, run
   `python scripts/07_export_artifacts.py`.

3. **Point the API at the evaluator trips and start it.**

   ```powershell
   # Windows PowerShell
   $env:PATHSNAP_EVAL_DIR = "C:\path\to\PathSnap\data\raw\evaluator_test"
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

   ```bash
   # macOS / Linux
   export PATHSNAP_EVAL_DIR=/path/to/PathSnap/data/raw/evaluator_test
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

   `--host 0.0.0.0` is what makes it reachable from Laptop 2. Binding to
   `localhost` is the single most common reason the UI cannot connect.

4. **Check the startup banner** lists the model, the decoder config and
   `trips: 10`. If it says a different number, `PATHSNAP_EVAL_DIR` is wrong.

5. **Note the LAN address.**

   ```powershell
   ipconfig        # look for IPv4 Address, e.g. 192.168.1.50
   ```
   ```bash
   ip addr         # or: hostname -I
   ```

6. **Allow the port through the firewall** if Windows prompts — choose
   *Private networks*. To pre-authorise:

   ```powershell
   New-NetFirewallRule -DisplayName "PathSnap API" -Direction Inbound `
     -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private
   ```

7. **Keep the API terminal visible.** It prints one line per request — that is
   the log the evaluator must be shown.

### Laptop 2 — evaluator interface

1. ```bash
   streamlit run ui/app.py
   ```
2. In the sidebar, set **API base URL** to `http://<Laptop-1-IP>:8000`.
3. Press **Check connection**. You should see the model name.

---

## Smoke test, before the evaluator sits down

From Laptop 2:

```bash
curl http://<Laptop-1-IP>:8000/health
curl http://<Laptop-1-IP>:8000/trips
```

Then run **one full trip through the UI** and confirm:

- the map draws the GPS points and the predicted route;
- the metrics panel fills in;
- a `/predict` line appears in the Laptop 1 terminal.

Run the **largest** trip too — it is the slowest, and you want to know its
timing before you are being watched.

---

## During the demonstration

1. Evaluator opens the UI on Laptop 2.
2. UI lists the ten reserved trips.
3. Evaluator picks any one.
4. UI sends the trip id to the API on Laptop 1.
5. API runs the full pipeline and computes accuracy against its own reference
   data, server-side.
6. UI displays the predicted route and the results.
7. **Show the API terminal** so the request is visibly arriving over the
   network.

Every one of the ten trips must work, not just a rehearsed one.

---

## Points worth making when questioned

- **Why two stages?** The classifier ranks candidates per observation but has
  no notion of time, so consecutive fixes can land on disconnected streets.
  The HMM uses road-network travel distance to pick the best *sequence*.
  `results/hmm_comparison_per_trip.csv` has the per-trip effect.
- **Hidden states** are the candidate arcs of each observation — the state
  space changes per step, so it is a lattice rather than a fixed alphabet.
- **Emissions** are the classifier's scores normalised per observation;
  **transitions** compare on-road travel distance against straight-line
  distance, `exp(-|d_route - d_gps| / beta)`.
- **Why top-1 accuracy as the tuning objective** rather than macro F1: a model
  can raise macro F1 by rejecting obviously-wrong candidates without ever
  improving the choice that reaches the route.
- **Candidate miss rate** is the ceiling on top-1: if the reference arc was
  never retrieved, no model can pick it.
- **The three broken training trips** (`00000033`, `00000054`, `00000002`)
  have reference routes covering only 13–19% of their GPS track. Observations
  further than 30 m from their aligned arc are flagged and excluded rather
  than trained on.
- **Why no ground-truth leakage:** the reference route is read only inside
  `/predict`, after the prediction is complete, purely to score it. The UI
  never receives it unless `include_reference` is explicitly set.

---

## If something breaks

| Symptom | Cause | Fix |
|---|---|---|
| UI cannot reach the API | bound to `localhost` | restart with `--host 0.0.0.0` |
| UI cannot reach the API | firewall | add the inbound rule above |
| `trips: 0` at startup | `PATHSNAP_EVAL_DIR` wrong | check the path and the per-trip subdirectories |
| `no model artifact` | stage 7 not run | `python scripts/07_export_artifacts.py` |
| A trip is slow | large network, many fixes | expected; `/predict` reports a per-stage timing breakdown |
| Map is blank | no internet for tiles | the route still draws; mention that tiles are fetched online |

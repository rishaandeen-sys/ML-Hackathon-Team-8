# PathSnap — Technical Report

**UE24CS352A Machine Learning Hackathon · PES University, Department of CSE (RR Campus)**

Team: _______________  ·  Members: _______________  ·  Generated: 2026-11-18

Repository: https://github.com/rishaandeen-sys/PathSnap

---

> **How to use this file.** Every table and figure below is filled in automatically from the CSVs in `results/`, so the numbers match the experiments exactly. The blocks marked **TEAM ANALYSIS REQUIRED** are deliberately empty — write those in your own words, then export this file to PDF.

---

## 2. Dataset Understanding and Exploratory Data Analysis

### 2.1 Partition summary

| partition    |   n_trips |   total_observations |   obs_per_trip_mean |   obs_per_trip_min |   obs_per_trip_max |   total_track_km |   mean_speed_kmh |   frac_dt_eq_1s |   trips_with_gaps |   mean_arcs |   label_residual_median_m |
|:-------------|----------:|---------------------:|--------------------:|-------------------:|-------------------:|-----------------:|-----------------:|----------------:|------------------:|------------:|--------------------------:|
| student_test |        10 |                19160 |             1916    |                262 |               3958 |          296.968 |           53.527 |           0.982 |                 2 |     18504.1 |                     4.745 |
| train        |        80 |               208417 |             2605.21 |                361 |              19313 |         2978.41  |           48.6   |           0.938 |                45 |     48461.3 |                     4.629 |

### 2.2 Data-quality checks

|                                 |   train |   student_test |
|:--------------------------------|--------:|---------------:|
| total_observations              |  208417 |          19160 |
| nan_values_in_track             |       0 |              0 |
| duplicate_timestamps            |       0 |              0 |
| non_increasing_timestamps       |       0 |              0 |
| coordinates_out_of_range        |       0 |              0 |
| consecutive_identical_positions |       6 |              0 |
| self_loop_arcs                  |      22 |              2 |
| arcs_with_invalid_node_id       |       0 |              0 |
| zero_length_arcs                |     100 |              6 |
| route_arc_ids_out_of_range      |       0 |              0 |

### 2.3 Candidate generation

| partition    |   n_trips |   n_observations |   n_candidate_rows |   mean_candidates_per_obs |   n_observations_reliably_labelled |   frac_observations_reliably_labelled |   label_residual_max_m |   candidate_miss_rate |   candidate_miss_rate_all_obs |   positive_rate |   median_label_residual_m |   build_seconds |   radius_m |   max_candidates |
|:-------------|----------:|-----------------:|-------------------:|--------------------------:|-----------------------------------:|--------------------------------------:|-----------------------:|----------------------:|------------------------------:|----------------:|--------------------------:|----------------:|-----------:|-----------------:|
| train        |        80 |           208415 |            1497156 |                   7.18353 |                             182351 |                               0.87494 |                     30 |               0.00262 |                       0.07742 |         0.12843 |                   5.9081  |        38.9883  |         60 |               10 |
| student_test |        10 |            19160 |             140908 |                   7.35428 |                              19013 |                               0.99233 |                     30 |               0.00084 |                       0.00256 |         0.13563 |                   5.23226 |         3.27683 |         60 |               10 |


The candidate miss rate of **0.2616%** on reliably-labelled training observations is the hard ceiling on top-1 accuracy: an observation whose reference arc was never retrieved cannot be classified correctly by any model.


**Label reliability.** 87.49% of training observations have a reference arc within 30 m. The remainder come from trips whose supplied `.route` covers only part of the GPS track and are excluded from fitting and scoring.

### 2.4 Preprocessing

- Local equirectangular projection to metres, centred per trip.
- Per-observation reference arc recovered by monotonic DP alignment against the `.route` sequence.
- Observations whose aligned arc is further than 30 m away flagged and excluded.
- Candidates: arcs within 60 m, nearest 10 kept.
- Median imputation for the neighbour-dependent features at the first and last fix; standardisation for the distance- and gradient-based models only.


> **TEAM ANALYSIS REQUIRED — write this yourself:**
> - How did you preprocess the data, and why those choices?
> - What patterns did you see during exploration?
> - Which features did you extract and why those?
> - Which partition was used for each analysis?


![Dataset overview](../figures/eda_overview.png)

*Dataset overview*


![Raw GPS trajectories, first 12 training trips](../figures/eda_trajectories.png)

*Raw GPS trajectories, first 12 training trips*


![Sampling gaps and step sizes per trip](../figures/eda_gaps.png)

*Sampling gaps and step sizes per trip*


![Geographic extent of every trip](../figures/eda_extents.png)

*Geographic extent of every trip*


**Folium map:** `results/maps/eda_trip_*.html` — include a screenshot here.


---

## 3. Model Experimentation and Comparison

### 3.1 Overall model comparison

| model                  | algorithm_family            | key_configuration                                               |   cv_macro_f1_mean |   test_top1_accuracy |   test_route_continuity |   test_macro_f1 |   test_log_loss |   inference_seconds |
|:-----------------------|:----------------------------|:----------------------------------------------------------------|-------------------:|---------------------:|------------------------:|----------------:|----------------:|--------------------:|
| mlp                    | Neural network              | hidden=(64, 32), relu, adam, early stopping, standardised       |             0.9068 |               0.8767 |                  0.9342 |          0.9206 |          0.0909 |              0.0725 |
| xgboost                | Boosting (GBDT)             | 200 trees, lr=0.05, max_depth=7, scale_pos_weight tuned         |             0.914  |               0.8742 |                  0.9343 |          0.9249 |          0.0854 |              0.097  |
| lightgbm               | Boosting (GBDT)             | 200 trees, lr=0.05, num_leaves=63, is_unbalance=True            |             0.8911 |               0.8736 |                  0.9333 |          0.898  |          0.1257 |              0.19   |
| random_forest          | Ensemble (bagging)          | 200 trees, max_depth=20, min_samples_leaf=5, balanced_subsample |             0.9092 |               0.8711 |                  0.9305 |          0.9147 |          0.103  |              1.7745 |
| hist_gradient_boosting | Boosting                    | lr=0.1, max_iter=300, max_leaf_nodes=31, early stopping         |             0.9122 |               0.8707 |                  0.9317 |          0.9225 |          0.089  |              0.6148 |
| extra_trees            | Ensemble (randomised trees) | 200 trees, max_depth=20, min_samples_leaf=5                     |             0.8941 |               0.8643 |                  0.9269 |          0.8963 |          0.1337 |              1.7864 |
| decision_tree          | Tree-based                  | gini, max_depth=14, min_samples_leaf=50, balanced               |             0.8793 |               0.8614 |                  0.9231 |          0.8842 |          0.1855 |              0.1144 |
| logistic_regression    | Linear / probabilistic      | lbfgs, C=1.0, class_weight=balanced, standardised               |             0.872  |               0.8608 |                  0.9345 |          0.8765 |          0.1802 |              0.0389 |
| knn_distance           | Instance-based              | k=15, distance-weighted, standardised, 150k-row subsample       |             0.9031 |               0.855  |                  0.9205 |          0.9111 |          0.2263 |             32.7458 |
| gaussian_nb            | Probabilistic               | var_smoothing=1e-9, standardised                                |             0.7882 |               0.8342 |                  0.9083 |          0.7826 |          1.0714 |              0.0895 |


10 models were trained and compared, spanning 9 algorithm families. Cross-validation is `GroupKFold(5)` grouped by trip, so no trip is split across folds.


### 3.2 Per-fold scores

| model                  |      1 |      2 |      3 |      4 |      5 |
|:-----------------------|-------:|-------:|-------:|-------:|-------:|
| decision_tree          | 0.8758 | 0.8802 | 0.9083 | 0.8679 | 0.864  |
| extra_trees            | 0.8903 | 0.8933 | 0.9221 | 0.8843 | 0.8807 |
| gaussian_nb            | 0.7957 | 0.8085 | 0.7779 | 0.7678 | 0.7912 |
| hist_gradient_boosting | 0.9103 | 0.9157 | 0.9417 | 0.9    | 0.8935 |
| knn_distance           | 0.9008 | 0.902  | 0.931  | 0.8991 | 0.8826 |
| lightgbm               | 0.8881 | 0.8912 | 0.919  | 0.8776 | 0.8798 |
| logistic_regression    | 0.8703 | 0.8727 | 0.8999 | 0.8578 | 0.8593 |
| mlp                    | 0.9071 | 0.9081 | 0.9345 | 0.8955 | 0.8888 |
| random_forest          | 0.9035 | 0.911  | 0.9367 | 0.9    | 0.8947 |
| xgboost                | 0.9112 | 0.9162 | 0.9434 | 0.9043 | 0.8948 |


![Cross-validation macro F1 across all models](../figures/model_cv_macro_f1.png)

*Cross-validation macro F1 across all models*


![Student-testing top-1 accuracy and macro F1](../figures/model_test_performance.png)

*Student-testing top-1 accuracy and macro F1*


![Student-testing log loss](../figures/model_log_loss.png)

*Student-testing log loss*


![Per-fold stability](../figures/model_fold_stability.png)

*Per-fold stability*


![Route continuity before HMM decoding](../figures/model_route_continuity.png)

*Route continuity before HMM decoding*


![Accuracy against training cost](../figures/model_accuracy_vs_cost.png)

*Accuracy against training cost*


### 3.3 Per-model questions


#### mlp — Neural network

Configuration: `hidden=(64, 32), relu, adam, early stopping, standardised`


| metric | value |
|---|---|
| CV macro F1 | 0.9068 ± 0.0156 |
| CV top-1 accuracy | 0.8517 |
| Student-test top-1 accuracy | 0.8767 |
| Student-test macro F1 | 0.9206 |
| Student-test precision / recall (macro) | 0.9277 / 0.9138 |
| Student-test log loss | 0.0909 |
| Route continuity | 0.9342 |
| Train / inference time (s) | 46.4 / 0.07 |


> **TEAM ANALYSIS REQUIRED — write this yourself:**
> - Why is mlp suitable for the PathSnap problem?
> - What are its strengths and limitations *for this task*?
> - Which hyperparameters mattered, and what were their initial values?
> - What loss or optimisation objective does it use?
> - Did you observe overfitting or underfitting? What is the evidence (compare the CV and student-test columns above)?


#### xgboost — Boosting (GBDT)

Configuration: `200 trees, lr=0.05, max_depth=7, scale_pos_weight tuned`


| metric | value |
|---|---|
| CV macro F1 | 0.9140 ± 0.0164 |
| CV top-1 accuracy | 0.8578 |
| Student-test top-1 accuracy | 0.8742 |
| Student-test macro F1 | 0.9249 |
| Student-test precision / recall (macro) | 0.9330 / 0.9172 |
| Student-test log loss | 0.0854 |
| Route continuity | 0.9343 |
| Train / inference time (s) | 11.1 / 0.10 |


> **TEAM ANALYSIS REQUIRED — write this yourself:**
> - Why is xgboost suitable for the PathSnap problem?
> - What are its strengths and limitations *for this task*?
> - Which hyperparameters mattered, and what were their initial values?
> - What loss or optimisation objective does it use?
> - Did you observe overfitting or underfitting? What is the evidence (compare the CV and student-test columns above)?


#### lightgbm — Boosting (GBDT)

Configuration: `200 trees, lr=0.05, num_leaves=63, is_unbalance=True`


| metric | value |
|---|---|
| CV macro F1 | 0.8911 ± 0.0148 |
| CV top-1 accuracy | 0.8561 |
| Student-test top-1 accuracy | 0.8736 |
| Student-test macro F1 | 0.8980 |
| Student-test precision / recall (macro) | 0.8615 / 0.9500 |
| Student-test log loss | 0.1257 |
| Route continuity | 0.9333 |
| Train / inference time (s) | 6.9 / 0.19 |


> **TEAM ANALYSIS REQUIRED — write this yourself:**
> - Why is lightgbm suitable for the PathSnap problem?
> - What are its strengths and limitations *for this task*?
> - Which hyperparameters mattered, and what were their initial values?
> - What loss or optimisation objective does it use?
> - Did you observe overfitting or underfitting? What is the evidence (compare the CV and student-test columns above)?


#### random_forest — Ensemble (bagging)

Configuration: `200 trees, max_depth=20, min_samples_leaf=5, balanced_subsample`


| metric | value |
|---|---|
| CV macro F1 | 0.9092 ± 0.0148 |
| CV top-1 accuracy | 0.8569 |
| Student-test top-1 accuracy | 0.8711 |
| Student-test macro F1 | 0.9147 |
| Student-test precision / recall (macro) | 0.8911 / 0.9433 |
| Student-test log loss | 0.1030 |
| Route continuity | 0.9305 |
| Train / inference time (s) | 152.8 / 1.77 |


> **TEAM ANALYSIS REQUIRED — write this yourself:**
> - Why is random_forest suitable for the PathSnap problem?
> - What are its strengths and limitations *for this task*?
> - Which hyperparameters mattered, and what were their initial values?
> - What loss or optimisation objective does it use?
> - Did you observe overfitting or underfitting? What is the evidence (compare the CV and student-test columns above)?


#### hist_gradient_boosting — Boosting

Configuration: `lr=0.1, max_iter=300, max_leaf_nodes=31, early stopping`


| metric | value |
|---|---|
| CV macro F1 | 0.9122 ± 0.0166 |
| CV top-1 accuracy | 0.8549 |
| Student-test top-1 accuracy | 0.8707 |
| Student-test macro F1 | 0.9225 |
| Student-test precision / recall (macro) | 0.9291 / 0.9162 |
| Student-test log loss | 0.0890 |
| Route continuity | 0.9317 |
| Train / inference time (s) | 21.0 / 0.61 |


> **TEAM ANALYSIS REQUIRED — write this yourself:**
> - Why is hist_gradient_boosting suitable for the PathSnap problem?
> - What are its strengths and limitations *for this task*?
> - Which hyperparameters mattered, and what were their initial values?
> - What loss or optimisation objective does it use?
> - Did you observe overfitting or underfitting? What is the evidence (compare the CV and student-test columns above)?


#### extra_trees — Ensemble (randomised trees)

Configuration: `200 trees, max_depth=20, min_samples_leaf=5`


| metric | value |
|---|---|
| CV macro F1 | 0.8941 ± 0.0147 |
| CV top-1 accuracy | 0.8545 |
| Student-test top-1 accuracy | 0.8643 |
| Student-test macro F1 | 0.8963 |
| Student-test precision / recall (macro) | 0.8616 / 0.9449 |
| Student-test log loss | 0.1337 |
| Route continuity | 0.9269 |
| Train / inference time (s) | 70.6 / 1.79 |


> **TEAM ANALYSIS REQUIRED — write this yourself:**
> - Why is extra_trees suitable for the PathSnap problem?
> - What are its strengths and limitations *for this task*?
> - Which hyperparameters mattered, and what were their initial values?
> - What loss or optimisation objective does it use?
> - Did you observe overfitting or underfitting? What is the evidence (compare the CV and student-test columns above)?


#### decision_tree — Tree-based

Configuration: `gini, max_depth=14, min_samples_leaf=50, balanced`


| metric | value |
|---|---|
| CV macro F1 | 0.8793 ± 0.0156 |
| CV top-1 accuracy | 0.8492 |
| Student-test top-1 accuracy | 0.8614 |
| Student-test macro F1 | 0.8842 |
| Student-test precision / recall (macro) | 0.8456 / 0.9424 |
| Student-test log loss | 0.1855 |
| Route continuity | 0.9231 |
| Train / inference time (s) | 55.8 / 0.11 |


> **TEAM ANALYSIS REQUIRED — write this yourself:**
> - Why is decision_tree suitable for the PathSnap problem?
> - What are its strengths and limitations *for this task*?
> - Which hyperparameters mattered, and what were their initial values?
> - What loss or optimisation objective does it use?
> - Did you observe overfitting or underfitting? What is the evidence (compare the CV and student-test columns above)?


#### logistic_regression — Linear / probabilistic

Configuration: `lbfgs, C=1.0, class_weight=balanced, standardised`


| metric | value |
|---|---|
| CV macro F1 | 0.8720 ± 0.0151 |
| CV top-1 accuracy | 0.8539 |
| Student-test top-1 accuracy | 0.8608 |
| Student-test macro F1 | 0.8765 |
| Student-test precision / recall (macro) | 0.8370 / 0.9378 |
| Student-test log loss | 0.1802 |
| Route continuity | 0.9345 |
| Train / inference time (s) | 5.2 / 0.04 |


> **TEAM ANALYSIS REQUIRED — write this yourself:**
> - Why is logistic_regression suitable for the PathSnap problem?
> - What are its strengths and limitations *for this task*?
> - Which hyperparameters mattered, and what were their initial values?
> - What loss or optimisation objective does it use?
> - Did you observe overfitting or underfitting? What is the evidence (compare the CV and student-test columns above)?


#### knn_distance — Instance-based

Configuration: `k=15, distance-weighted, standardised, 150k-row subsample`


| metric | value |
|---|---|
| CV macro F1 | 0.9031 ± 0.0156 |
| CV top-1 accuracy | 0.8438 |
| Student-test top-1 accuracy | 0.8550 |
| Student-test macro F1 | 0.9111 |
| Student-test precision / recall (macro) | 0.9106 / 0.9116 |
| Student-test log loss | 0.2263 |
| Route continuity | 0.9205 |
| Train / inference time (s) | 1.1 / 32.75 |


> **TEAM ANALYSIS REQUIRED — write this yourself:**
> - Why is knn_distance suitable for the PathSnap problem?
> - What are its strengths and limitations *for this task*?
> - Which hyperparameters mattered, and what were their initial values?
> - What loss or optimisation objective does it use?
> - Did you observe overfitting or underfitting? What is the evidence (compare the CV and student-test columns above)?


#### gaussian_nb — Probabilistic

Configuration: `var_smoothing=1e-9, standardised`


| metric | value |
|---|---|
| CV macro F1 | 0.7882 ± 0.0142 |
| CV top-1 accuracy | 0.8351 |
| Student-test top-1 accuracy | 0.8342 |
| Student-test macro F1 | 0.7826 |
| Student-test precision / recall (macro) | 0.7432 / 0.8942 |
| Student-test log loss | 1.0714 |
| Route continuity | 0.9083 |
| Train / inference time (s) | 2.9 / 0.09 |


> **TEAM ANALYSIS REQUIRED — write this yourself:**
> - Why is gaussian_nb suitable for the PathSnap problem?
> - What are its strengths and limitations *for this task*?
> - Which hyperparameters mattered, and what were their initial values?
> - What loss or optimisation objective does it use?
> - Did you observe overfitting or underfitting? What is the evidence (compare the CV and student-test columns above)?


---

## 4. Bayesian Hyperparameter Optimisation


![TPE search history](../figures/bayesopt_history.png)

*TPE search history*


![Objective against each searched hyperparameter](../figures/bayesopt_params.png)

*Objective against each searched hyperparameter*


![Tuned vs untuned](../figures/bayesopt_tuned_vs_untuned.png)

*Tuned vs untuned*


> **TEAM ANALYSIS REQUIRED — write this yourself:**
> - Which hyperparameters did you search, and why those?
> - What ranges/distributions did you choose, and on what grounds?
> - Did the tuned model beat the untuned one? By how much, and is the gap larger than the fold-to-fold variation?
> - What did the search cost, and was it worth it?


---

## 5. HMM and Viterbi Route Reconstruction


- **Hidden states:** the candidate arcs of each observation — a lattice whose state space changes per step, not a fixed alphabet.
- **Emissions:** classifier scores, normalised over each observation's candidates.
- **Transitions:** `P(j|i) ∝ exp(-|d_route - d_gps| / beta)`, with `d_route` the on-road travel distance via bounded Dijkstra (Newson & Krumm, 2009).
- **Unreachable pairs:** fixed penalty rather than prohibition, so the lattice can never dead-end.
- **GPS gaps:** above 5 s the search radius grows and beta is relaxed.
- **Route expansion:** intermediate arcs the vehicle crossed between fixes are filled in by shortest path, turning per-observation decisions into a drivable route.


![Top-1 accuracy and continuity per trip](../figures/hmm_per_trip_comparison.png)

*Top-1 accuracy and continuity per trip*


![Effect of Viterbi decoding per trip](../figures/hmm_improvement.png)

*Effect of Viterbi decoding per trip*


![Continuity before and after decoding](../figures/hmm_continuity_scatter.png)

*Continuity before and after decoding*


![Decoder parameter search on training trips](../figures/hmm_parameter_search.png)

*Decoder parameter search on training trips*


> **TEAM ANALYSIS REQUIRED — write this yourself:**
> - What are the hidden states in your implementation?
> - How are emission probabilities calculated?
> - How are transition probabilities calculated?
> - How are missing candidates and large GPS gaps handled?
> - What happens when consecutive observations suggest disconnected segments?
> - Did decoding improve the route? Where did it NOT help, and why?


---

## 6. Geographical Visualisation Using Folium

| partition    |   trip_id |   n_observations |   classifier_top1 |   hmm_top1 |   hmm_continuity |
|:-------------|----------:|-----------------:|------------------:|-----------:|-----------------:|
| student_test |        52 |              262 |            0.9618 |     0.9962 |           0.9617 |
| student_test |         7 |             1354 |            0.9231 |     0.905  |           1      |


Interactive maps: `results/maps/index.html`. Each map carries toggleable layers for the road network, raw GPS observations, the reference route, the classifier-only route and the predicted route, with a legend and layer control.


> **TEAM ANALYSIS REQUIRED — write this yourself:**
> - Insert at least one map screenshot and describe what it shows: where the raw GPS drifts off-road and how the predicted route recovers it.


---

## 7. API, UI and Deployment

**Laptop 1 (model + API)**

```bash
set PATHSNAP_EVAL_DIR=...\evaluator_test
uvicorn api.main:app --host 0.0.0.0 --port 8000
```


**Laptop 2 (evaluator UI)**

```bash
streamlit run ui/app.py
```


| Endpoint | Purpose |
|---|---|
| `GET /health` | model, decoder config, trip count |
| `GET /trips` | the ten evaluation trips |
| `POST /predict` | run the pipeline, return route + server-side metrics |
| `GET /network/{trip_id}` | road-network geometry for map context |
| `GET /logs` | recent requests |


**Evaluation integrity.** The reference route is read only inside `/predict`, after prediction, and only to compute accuracy. It is never passed to the matcher and is omitted from the response unless explicitly requested. Nothing is hardcoded per trip.


> **TEAM ANALYSIS REQUIRED — write this yourself:**
> - Insert screenshots of the working UI.
> - Insert a screenshot of the API logs on Laptop 1 showing the evaluator's requests arriving.


---

## 8. Academic Integrity and AI Use


> **TEAM ANALYSIS REQUIRED — write this yourself:**
> - State which parts were AI-assisted and which were the team's own work, as the event's policy requires.
> - Confirm every member can explain the code, experiments and results.


---

## Appendix: reproducibility

- Seed: `42`, folds: `5`
- Features (26): `dist_m, dist_rank, dist_minus_best, dist_ratio_best, dist_z_in_obs, n_candidates, arc_length, proj_t, proj_is_endpoint, heading_diff, heading_cos, heading_valid, speed_mps, accel_mps2, dt_prev, dt_next, step_prev_m, step_next_m, dist_prev_obs_to_arc, dist_next_obs_to_arc, arc_out_degree, arc_in_degree, has_reverse_arc, local_arc_density, obs_frac, turn_angle`
- Training rows: 1,270,733; student-test rows: 140,155

- Source CSVs for every table above are in `results/`.

"""PathSnap evaluator interface (Laptop 2).

A geospatial analytics dashboard over the inference API on Laptop 1. The UI
holds no model, no road network and no ground truth: everything it renders
comes from an API response, and every number shown is one the backend
actually computed.

Workflow: check connectivity, review the evaluation set, select a trip,
reconstruct it, then compare the raw GPS trace against the classifier-only
and HMM-decoded routes on the map.

Run:
    streamlit run ui/app.py
    (set the API URL in the sidebar, e.g. http://192.168.1.50:8000)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import folium
import pandas as pd
import requests
import streamlit as st
import xyzservices.providers as xyz
from streamlit_folium import st_folium

from theme import (ACCENT, AMBER, BASEMAPS, EMERALD, MAP_COLOURS, ROSE, TEXT_DIM,
                   VIOLET, css, legend, metric_card)

DEFAULT_API = os.environ.get("PATHSNAP_API", "http://localhost:8000")

st.set_page_config(page_title="PathSnap — Map Matching",
                   page_icon="◈", layout="wide",
                   initial_sidebar_state="expanded")
# st.html bypasses the markdown parser entirely, so CSS cannot be
# mangled by link/emphasis syntax inside selectors
st.html(css())


# ----------------------------------------------------------------- API client
def api_get(base: str, path: str, timeout: int = 30):
    r = requests.get(f"{base.rstrip('/')}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def api_post(base: str, path: str, payload: dict, timeout: int = 900):
    r = requests.post(f"{base.rstrip('/')}{path}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=120, show_spinner=False)
def fetch_trips(base: str):
    return api_get(base, "/trips")


@st.cache_data(ttl=120, show_spinner=False)
def fetch_metrics(base: str):
    try:
        return api_get(base, "/metrics")
    except Exception:
        return None


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def pct(x, nd=2):
    return "—" if x is None else f"{x * 100:.{nd}f}%"


# --------------------------------------------------------------------- sidebar
with st.sidebar:
    st.html(
        '<div class="ps-brandmark"><span class="dot"></span>PathSnap</div>'
        '<div style="font-size:1.08rem;font-weight:700;letter-spacing:-.02em;'
        'margin-bottom:3px">Map Matching</div>'
        '<div style="font-size:.80rem;color:var(--text-faint);line-height:1.45;'
        'margin-bottom:18px">Intelligent map matching &amp; route '
        'reconstruction</div>')

    st.html('<div class="ps-panel-title">Connection</div>')
    api_url = st.text_input("API endpoint", value=DEFAULT_API,
                            label_visibility="collapsed",
                            help="Address of the inference API on Laptop 1")

    health, health_err = None, None
    try:
        health = api_get(api_url, "/health", timeout=5)
    except Exception as exc:
        health_err = str(exc)

    if health:
        st.html(
            f'<div class="ps-status ps-ok" style="margin:9px 0 4px 0">'
            f'<span class="dot" style="width:7px;height:7px;border-radius:50%;'
            f'background:{EMERALD};box-shadow:0 0 9px {EMERALD}"></span>'
            f'Connected</div>')
        st.html(
            f'<div style="font-family:JetBrains Mono,monospace;font-size:.70rem;'
            f'color:var(--text-faint);margin:9px 0 2px 0;line-height:1.7">'
            f'model &nbsp;<span style="color:var(--text-dim)">'
            f'{health.get("model", "—")}</span><br>'
            f'trips &nbsp;&nbsp;<span style="color:var(--text-dim)">'
            f'{health.get("n_trips", "—")}</span><br>'
            f'served <span style="color:var(--text-dim)">'
            f'{health.get("requests_served", 0)}</span></div>')
    else:
        st.html('<div class="ps-status ps-err" style="margin:9px 0">'
                    'Disconnected</div>')

    if st.button("Test connection", width="stretch"):
        fetch_trips.clear()
        fetch_metrics.clear()
        st.rerun()

    st.html('<div style="height:14px"></div>')
    st.html('<div class="ps-panel-title">Map</div>')
    basemap_name = st.selectbox("Basemap", list(BASEMAPS), index=0,
                                label_visibility="collapsed")

    show_predicted = st.checkbox("Predicted route (HMM)", value=True)
    show_classifier = st.checkbox("Classifier-only route", value=True)
    show_gps = st.checkbox("Raw GPS observations", value=True)
    show_reference = st.checkbox("Reference route", value=False,
                                 help="Served only when explicitly requested")
    show_network = st.checkbox("Road network context", value=False)

    st.html('<div style="height:14px"></div>')
    st.html('<div class="ps-panel-title">Inference</div>')
    run_hmm = st.checkbox("Apply HMM / Viterbi decoding", value=True,
                          help="Turn off to see classifier-only output")

    if health and health.get("decoder"):
        d = health["decoder"]
        rows = "".join(
            f'<div style="display:flex;justify-content:space-between;gap:10px">'
            f'<span style="color:var(--text-faint)">{k}</span>'
            f'<span style="color:var(--text-dim)">{v}</span></div>'
            for k, v in list(d.items())[:6])
        with st.expander("Decoder configuration", expanded=False):
            st.html(
                f'<div style="font-family:JetBrains Mono,monospace;'
                f'font-size:.70rem;line-height:1.85">{rows}</div>')


# ------------------------------------------------------------------- header
status_chip = (
    f'<span class="ps-status ps-ok">API online</span>' if health
    else f'<span class="ps-status ps-err">API offline</span>')

st.html(
    f'<div class="ps-hero">'
    f'<div class="ps-brandmark"><span class="dot"></span>PathSnap</div>'
    f'<h1>Map Matching &amp; Route Reconstruction</h1>'
    f'<div class="tag">Reconstructs the road-network route a vehicle actually '
    f'travelled from a noisy GPS trace, by ranking candidate road segments and '
    f'decoding the most likely connected sequence.</div>'
    f'<div style="margin-top:15px">{status_chip}</div>'
    f'</div>')

if not health:
    st.html(
        f'<div class="ps-panel"><div class="ps-panel-title">Not connected</div>'
        f'<div style="color:var(--text-dim);font-size:.90rem;line-height:1.65">'
        f'Could not reach <span class="mono" style="color:var(--accent)">'
        f'{api_url}</span>.<br><br>Start the API on Laptop 1:</div>'
        f'<div style="font-family:JetBrains Mono,monospace;font-size:.80rem;'
        f'background:rgba(10,14,26,.7);border:1px solid var(--border);'
        f'border-radius:10px;padding:13px 15px;margin-top:13px;'
        f'color:var(--accent)">uvicorn api.main:app --host 0.0.0.0 --port 8000'
        f'</div></div>')
    if health_err:
        with st.expander("Error detail"):
            st.code(health_err)
    st.stop()

# -------------------------------------------------------------- evaluation set
try:
    trips = fetch_trips(api_url)
except Exception as exc:
    st.error(f"Could not load the trip list: {exc}")
    st.stop()

tdf = pd.DataFrame(trips)
total_obs = int(tdf.n_observations.sum())
total_nodes = int(tdf.n_nodes.sum())
total_arcs = int(tdf.n_arcs.sum())
mean_dur = float(tdf.duration_s.mean())

st.html('<div class="ps-panel-title">Evaluation set</div>')
st.html(
    '<div class="ps-grid">'
    + metric_card("Evaluation trips", f"{len(tdf)}", spark=ACCENT)
    + metric_card("GPS observations", f"{total_obs:,}", spark=VIOLET)
    + metric_card("Network nodes", f"{total_nodes:,}", spark=EMERALD)
    + metric_card("Network arcs", f"{total_arcs:,}", spark=AMBER)
    + metric_card("Mean duration", fmt_duration(mean_dur), spark=ROSE)
    + '</div>')

# -------------------------------------------------------------- trip selection
st.html('<div class="ps-panel-title">Select a trip</div>')
sel_col, btn_col = st.columns([3, 1])
with sel_col:
    options = list(tdf.trip_id)
    selected = st.selectbox(
        "Trip", options, label_visibility="collapsed",
        format_func=lambda t: (
            f"Trip {t}   ·   "
            f"{int(tdf.loc[tdf.trip_id == t, 'n_observations'].iloc[0]):,} GPS pts   ·   "
            f"{fmt_duration(tdf.loc[tdf.trip_id == t, 'duration_s'].iloc[0])}   ·   "
            f"{int(tdf.loc[tdf.trip_id == t, 'n_arcs'].iloc[0]):,} arcs"))
with btn_col:
    go = st.button("Reconstruct route", type="primary",
                   width="stretch")

row = tdf[tdf.trip_id == selected].iloc[0]
st.html(
    '<div class="ps-grid" style="margin-top:4px">'
    + metric_card("Trip", str(row.trip_id), spark=ACCENT)
    + metric_card("GPS points", f"{int(row.n_observations):,}", spark=VIOLET)
    + metric_card("Duration", fmt_duration(row.duration_s), spark=EMERALD)
    + metric_card("Network arcs", f"{int(row.n_arcs):,}", spark=AMBER)
    + metric_card("Network nodes", f"{int(row.n_nodes):,}", spark=ROSE)
    + '</div>')

for key in ("result", "elapsed", "network", "result_trip"):
    st.session_state.setdefault(key, None)

if go:
    with st.spinner(f"Reconstructing trip {selected}…"):
        try:
            t0 = time.time()
            st.session_state.result = api_post(api_url, "/predict", {
                "trip_id": selected, "include_gps": True,
                "include_reference": show_reference, "run_hmm": run_hmm})
            st.session_state.elapsed = time.time() - t0
            st.session_state.result_trip = selected
            st.session_state.network = (
                api_get(api_url, f"/network/{selected}") if show_network else None)
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            st.session_state.result = None

res = st.session_state.result
if res is None:
    st.html(
        '<div class="ps-panel" style="text-align:center;padding:46px 22px">'
        '<div style="font-size:2.1rem;opacity:.26;margin-bottom:10px">◈</div>'
        '<div style="color:var(--text-dim);font-size:.95rem;font-weight:600">'
        'No route reconstructed yet</div>'
        '<div style="color:var(--text-faint);font-size:.85rem;margin-top:7px">'
        'Select a trip above and press Reconstruct route.</div></div>')
    st.stop()

# ------------------------------------------------------------------- results
ev = res.get("evaluation") or {}
clf_acc = ev.get("classifier_top1_accuracy")
hmm_acc = ev.get("hmm_top1_accuracy")
improve = ev.get("hmm_improvement_pp")

st.html(f'<div class="ps-panel-title">Results — trip '
            f'{res["trip_id"]}</div>')
st.html(
    '<div class="ps-grid">'
    + metric_card("Top-1 accuracy", pct(hmm_acc), spark=ACCENT,
                  delta=(None if improve is None else f"{improve:+.2f} pp vs classifier"),
                  delta_kind=("up" if (improve or 0) > 0
                              else "down" if (improve or 0) < 0 else "flat"))
    + metric_card("Classifier only", pct(clf_acc), spark=AMBER)
    + metric_card("Route continuity", pct(ev.get("hmm_route_continuity")),
                  spark=EMERALD)
    + metric_card("Route arcs", f"{len(res['predicted_route']['arc_ids']):,}",
                  spark=VIOLET)
    + metric_card("Inference", f"{res.get('total_seconds', 0):.2f}", unit="s",
                  spark=ROSE,
                  delta=(f"{st.session_state.elapsed:.2f}s round trip"
                         if st.session_state.elapsed else None),
                  delta_kind="flat")
    + '</div>')

tab_map, tab_perf, tab_trip, tab_raw = st.tabs(
    ["Map", "Model performance", "Trip analysis", "API response"])

# ----------------------------------------------------------------- map tab
with tab_map:
    entries = []
    if show_gps:
        entries.append(("Raw GPS observations", MAP_COLOURS["gps"], False))
    if show_classifier:
        entries.append(("Classifier only", MAP_COLOURS["classifier"], True))
    if show_predicted:
        entries.append(("Predicted route (HMM)", MAP_COLOURS["predicted"], False))
    if show_reference and res.get("reference_route"):
        entries.append(("Reference route", MAP_COLOURS["reference"], False))
    if show_network and st.session_state.network:
        entries.append(("Road network", MAP_COLOURS["network"], False))
    st.html(legend(entries))

    poly = res["predicted_route"]["polyline"]
    gps_pts = (res.get("gps") or {}).get("points") or []
    anchor = poly or gps_pts
    centre = ([sum(p[0] for p in anchor) / len(anchor),
               sum(p[1] for p in anchor) / len(anchor)] if anchor else [49.87, 8.65])

    prov = xyz.flatten()[BASEMAPS[basemap_name]]
    # tiles=None plus an explicit named TileLayer. Passing the URL straight to
    # folium.Map makes the layer control label the basemap with its raw tile
    # URL, and control=False keeps the basemap out of the control entirely.
    m = folium.Map(location=centre, zoom_start=14, control_scale=False,
                   tiles=None, zoom_control=True, prefer_canvas=True)
    folium.TileLayer(prov.build_url(), attr="Esri", name=basemap_name,
                     control=False).add_to(m)

    if show_network and st.session_state.network:
        g = folium.FeatureGroup(name="Road network", show=True)
        folium.PolyLine(st.session_state.network["segments"],
                        color=MAP_COLOURS["network"], weight=1,
                        opacity=0.42).add_to(g)
        g.add_to(m)

    if show_gps and gps_pts:
        g = folium.FeatureGroup(name="Raw GPS observations", show=True)
        step = max(1, len(gps_pts) // 1200)
        ts = (res.get("gps") or {}).get("timestamps") or []
        for i in range(0, len(gps_pts), step):
            tip = f"obs {i}"
            if i < len(ts):
                tip += f" · t={ts[i]:.0f}s"
            folium.CircleMarker(gps_pts[i], radius=2.1,
                                color=MAP_COLOURS["gps"], weight=0,
                                fill=True, fill_opacity=0.80,
                                tooltip=tip).add_to(g)
        g.add_to(m)

    if show_classifier and res.get("classifier_only_route", {}).get("polyline"):
        g = folium.FeatureGroup(name="Classifier-only route", show=True)
        folium.PolyLine(res["classifier_only_route"]["polyline"],
                        color=MAP_COLOURS["classifier"], weight=3.4,
                        opacity=0.88, dash_array="7,7",
                        tooltip="Classifier only — no sequence reasoning"
                        ).add_to(g)
        g.add_to(m)

    if show_reference and res.get("reference_route"):
        g = folium.FeatureGroup(name="Reference route", show=True)
        folium.PolyLine(res["reference_route"]["polyline"],
                        color=MAP_COLOURS["reference"], weight=6,
                        opacity=0.60, tooltip="Reference route").add_to(g)
        g.add_to(m)

    if show_predicted and poly:
        g = folium.FeatureGroup(name="Predicted route (HMM)", show=True)
        # a wide translucent pass under the line reads as a glow on satellite
        folium.PolyLine(poly, color=MAP_COLOURS["predicted"], weight=9,
                        opacity=0.20).add_to(g)
        folium.PolyLine(poly, color=MAP_COLOURS["predicted"], weight=3.6,
                        opacity=0.97,
                        tooltip="Predicted route — HMM + Viterbi").add_to(g)
        g.add_to(m)

    if anchor:
        folium.CircleMarker(anchor[0], radius=7, color="#FFFFFF", weight=2,
                            fill=True, fill_color=EMERALD, fill_opacity=1,
                            tooltip="Start").add_to(m)
        folium.CircleMarker(anchor[-1], radius=7, color="#FFFFFF", weight=2,
                            fill=True, fill_color=ROSE, fill_opacity=1,
                            tooltip="End").add_to(m)
        lats = [p[0] for p in anchor]
        lons = [p[1] for p in anchor]
        m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]],
                     padding=(28, 28))

    folium.LayerControl(collapsed=True, position="topright").add_to(m)

    # The map renders in an iframe, so the page stylesheet cannot reach it.
    # Leaflet's default chrome is a white box with a long attribution string,
    # which reads as unfinished against the dark theme.
    m.get_root().html.add_child(folium.Element("""
<style>
  .leaflet-control-attribution {
    background: rgba(10,14,26,.62) !important;
    color: rgba(190,200,220,.40) !important;
    font: 9px system-ui, sans-serif !important;
    padding: 1px 6px !important;
    border-radius: 4px 0 0 0 !important;
    box-shadow: none !important;
  }
  .leaflet-control-attribution a {
    color: rgba(190,200,220,.55) !important; text-decoration: none !important;
  }
  .leaflet-control-attribution svg { display: none !important; }
  .leaflet-bar, .leaflet-control-layers {
    background: rgba(18,24,41,.90) !important;
    border: 1px solid rgba(148,170,220,.20) !important;
    border-radius: 7px !important;
    box-shadow: 0 2px 10px rgba(0,0,0,.35) !important;
    backdrop-filter: blur(8px);
  }
  .leaflet-bar a {
    background: transparent !important;
    color: #C8D2E4 !important;
    border-bottom: 1px solid rgba(148,170,220,.14) !important;
    width: 26px !important; height: 26px !important; line-height: 26px !important;
    font-size: 15px !important;
  }
  .leaflet-bar a:last-child { border-bottom: 0 !important; }
  .leaflet-bar a:hover { background: rgba(110,231,249,.14) !important; color: #fff !important; }
  .leaflet-control-layers-toggle {
    background-color: rgba(18,24,41,.90) !important;
    width: 28px !important; height: 28px !important;
  }
  .leaflet-control-layers-expanded {
    padding: 8px 10px !important; color: #C8D2E4 !important;
    font: 11.5px system-ui, sans-serif !important; min-width: 0 !important;
  }
  .leaflet-control-layers-expanded label { margin: 3px 0 !important; font-weight: 400; }
  .leaflet-control-layers-separator { border-top: 1px solid rgba(148,170,220,.16) !important; }
</style>"""))

    st_folium(m, use_container_width=True, height=660, returned_objects=[])

# -------------------------------------------------------- model performance
with tab_perf:
    metrics = fetch_metrics(api_url)
    if not metrics:
        st.html('<div class="ps-panel"><div class="ps-panel-title">'
                    'Unavailable</div><div style="color:var(--text-dim);'
                    'font-size:.88rem">The API did not return experiment '
                    'metrics. Run the training stages to populate them.</div>'
                    '</div>')
    else:
        comp = metrics.get("model_comparison")
        if comp:
            cdf = pd.DataFrame(comp).sort_values("test_top1_accuracy",
                                                 ascending=False)
            st.html('<div class="ps-panel-title">Model comparison — '
                        'student-testing partition</div>')
            chart = cdf.set_index("model")[["test_top1_accuracy",
                                            "cv_macro_f1_mean",
                                            "test_macro_f1"]]
            chart.columns = ["Test top-1", "CV macro F1", "Test macro F1"]
            st.bar_chart(chart, height=300,
                         color=[ACCENT, VIOLET, EMERALD])
            show = cdf[["model", "algorithm_family", "cv_macro_f1_mean",
                        "test_top1_accuracy", "test_macro_f1", "test_log_loss",
                        "test_route_continuity", "train_seconds",
                        "inference_us_per_row"]].copy()
            show.columns = ["Model", "Family", "CV macro F1", "Test top-1",
                            "Test macro F1", "Log loss", "Continuity",
                            "Train (s)", "Inference (µs/row)"]
            st.dataframe(show.round(4), hide_index=True,
                         width="stretch")

        hmm = metrics.get("hmm_summary")
        if hmm:
            hdf = pd.DataFrame(hmm)
            st.html('<div class="ps-panel-title">Classifier only vs '
                        'classifier + HMM</div>')
            clf_row = hdf[hdf.stage == "classifier_only"].iloc[0]
            hmm_row = hdf[hdf.stage == "classifier_plus_hmm"].iloc[0]
            gain = (hmm_row.top1_accuracy_weighted
                    - clf_row.top1_accuracy_weighted) * 100
            st.html(
                '<div class="ps-grid">'
                + metric_card("Classifier only",
                              pct(clf_row.top1_accuracy_weighted), spark=AMBER)
                + metric_card("Classifier + HMM",
                              pct(hmm_row.top1_accuracy_weighted), spark=ACCENT,
                              delta=f"{gain:+.2f} pp", delta_kind="up")
                + metric_card("Continuity (HMM)",
                              pct(hmm_row.route_continuity), spark=EMERALD)
                + metric_card("Arc-set Jaccard",
                              pct(hmm_row.arc_set_jaccard), spark=VIOLET)
                + '</div>')

        per_trip = metrics.get("hmm_per_trip")
        if per_trip:
            pdf = pd.DataFrame(per_trip)
            pdf["trip"] = pdf.trip_id.apply(lambda t: f"{int(t):08d}")
            st.html('<div class="ps-panel-title">Per-trip effect of '
                        'Viterbi decoding</div>')
            st.bar_chart(
                pdf.set_index("trip")[["classifier_top1_accuracy",
                                       "hmm_top1_accuracy"]].rename(
                    columns={"classifier_top1_accuracy": "Classifier only",
                             "hmm_top1_accuracy": "Classifier + HMM"}),
                height=290, color=[AMBER, ACCENT])

        bo = metrics.get("bayesopt_summary")
        if bo:
            bdf = pd.DataFrame(bo)
            st.html('<div class="ps-panel-title">Bayesian hyperparameter '
                        'optimisation</div>')
            cols = [c for c in ["variant", "cv_top1_accuracy", "cv_macro_f1",
                                "test_top1_accuracy", "test_macro_f1",
                                "test_log_loss", "train_seconds"]
                    if c in bdf.columns]
            st.dataframe(bdf[cols].round(5), hide_index=True,
                         width="stretch")

        outcomes = metrics.get("observation_outcomes")
        if outcomes:
            odf = pd.DataFrame(outcomes).sort_values("top1_accuracy",
                                                     ascending=False)
            with st.expander("Observation-level outcome breakdown"):
                st.caption(
                    "Where each model's observations end up: ranked correctly, "
                    "ranked a wrong candidate first, or the reference arc was "
                    "never retrieved at all. The last column is a property of "
                    "candidate generation, not of the classifier — which is "
                    "why it is identical across every model.")
                st.dataframe(odf.round(5), hide_index=True,
                             width="stretch")

# ------------------------------------------------------------- trip analysis
with tab_trip:
    diag = res.get("diagnostics") or {}
    timings = res.get("timings") or {}

    st.html('<div class="ps-panel-title">Reconstruction detail</div>')
    st.html(
        '<div class="ps-grid">'
        + metric_card("Observations", f"{res['n_observations']:,}", spark=ACCENT)
        + metric_card("Mean candidates / obs",
                      f"{diag.get('mean_candidates_per_obs', 0):.2f}",
                      spark=VIOLET)
        + metric_card("Scored observations",
                      f"{ev.get('n_scored_observations', 0):,}", spark=EMERALD)
        + metric_card("Decoded route arcs",
                      f"{int(diag.get('expanded_route_arcs', 0) or 0):,}",
                      spark=AMBER)
        + metric_card("Unreachable transitions",
                      f"{int(diag.get('hmm_unreachable_transitions', 0) or 0):,}",
                      spark=ROSE)
        + '</div>')

    c1, c2 = st.columns(2)
    with c1:
        st.html('<div class="ps-panel-title">Accuracy breakdown</div>')
        acc_rows = [
            ("Classifier top-1", pct(ev.get("classifier_top1_accuracy"))),
            ("HMM top-1", pct(ev.get("hmm_top1_accuracy"))),
            ("Classifier continuity", pct(ev.get("classifier_route_continuity"))),
            ("HMM continuity", pct(ev.get("hmm_route_continuity"))),
            ("Arc-set precision", pct(ev.get("hmm_arc_set_precision"))),
            ("Arc-set recall", pct(ev.get("hmm_arc_set_recall"))),
            ("Arc-set Jaccard", pct(ev.get("hmm_arc_set_jaccard"))),
        ]
        st.dataframe(pd.DataFrame(acc_rows, columns=["Metric", "Value"]),
                     hide_index=True, width="stretch")
    with c2:
        st.html('<div class="ps-panel-title">Pipeline timings</div>')
        tim = pd.DataFrame(
            [(k.replace("_s", "").replace("_", " ").title(), f"{v:.3f} s")
             for k, v in timings.items()], columns=["Stage", "Seconds"])
        st.dataframe(tim, hide_index=True, width="stretch")
        if timings:
            st.bar_chart(
                pd.Series({k.replace("_s", "").title(): v
                           for k, v in timings.items() if k != "total_s"}),
                height=190, color=ACCENT)

    conf = (res.get("per_observation") or {}).get("confidence")
    if conf:
        with st.expander("Per-observation classifier confidence", expanded=False):
            st.caption("Confidence of the top-ranked candidate at each GPS "
                       "observation. Dips mark places the classifier found the "
                       "choice ambiguous — typically junctions and parallel roads.")
            st.line_chart(pd.DataFrame({"confidence": conf}), height=210,
                          color=ACCENT)

    ncand = (res.get("per_observation") or {}).get("n_candidates")
    if ncand:
        with st.expander("Candidates per observation", expanded=False):
            st.caption("How many road segments competed at each observation. "
                       "Higher counts mean denser road network around that fix.")
            st.area_chart(pd.DataFrame({"candidates": ncand}), height=210,
                          color=VIOLET)

# ------------------------------------------------------------------- raw tab
with tab_raw:
    st.html(
        f'<div style="font-family:JetBrains Mono,monospace;font-size:.76rem;'
        f'color:var(--text-faint);margin-bottom:11px">'
        f'model <span style="color:var(--accent)">{res.get("model")}</span>'
        f' &nbsp;·&nbsp; served '
        f'<span style="color:var(--text-dim)">{res.get("served_at", "")}</span>'
        f'</div>')
    st.json({k: v for k, v in res.items()
             if k not in ("gps", "per_observation")}, expanded=False)
    with st.expander("Per-observation arrays (large)"):
        st.json(res.get("per_observation", {}), expanded=False)

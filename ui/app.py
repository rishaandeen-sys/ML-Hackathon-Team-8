"""PathSnap evaluator interface (Laptop 2).

Lists the evaluation trips served by the API on Laptop 1, sends the selected
trip id over HTTP and draws the returned route on an interactive map next to
the accuracy figures the API computed server-side.

The UI holds no model, no road network and no ground truth: everything it
shows comes from the API response.

Run:
    streamlit run ui/app.py
    (set the API URL in the sidebar, e.g. http://192.168.1.50:8000)
"""
from __future__ import annotations

import os
import time

import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

DEFAULT_API = os.environ.get("PATHSNAP_API", "http://localhost:8000")

COLOURS = {"gps": "#1f6fd0", "reference": "#17a558",
           "predicted": "#d93025", "classifier": "#f29900",
           "network": "#9aa0a6"}

st.set_page_config(page_title="PathSnap Evaluator", page_icon="🛰️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1500px;}
  div[data-testid="stMetricValue"] {font-size: 1.5rem;}
  .ps-sub {color:#6b7280; font-size:0.92rem; margin-top:-0.6rem;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ API calls
def api_get(base: str, path: str, **kw):
    r = requests.get(f"{base.rstrip('/')}{path}", timeout=kw.pop("timeout", 30), **kw)
    r.raise_for_status()
    return r.json()


def api_post(base: str, path: str, payload: dict, timeout: int = 600):
    r = requests.post(f"{base.rstrip('/')}{path}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


# --------------------------------------------------------------------- layout
st.title("PathSnap — Map Matching & Route Reconstruction")
st.markdown('<p class="ps-sub">Evaluator interface. Select a trip; the '
            'request is processed by the inference API on Laptop 1.</p>',
            unsafe_allow_html=True)

with st.sidebar:
    st.header("Connection")
    api_url = st.text_input("API base URL", value=DEFAULT_API,
                            help="Address of the inference API running on Laptop 1")
    if st.button("Check connection", use_container_width=True):
        try:
            h = api_get(api_url, "/health")
            st.success(f"Connected — model {h['model']}")
            st.json(h)
        except Exception as exc:
            st.error(f"Cannot reach the API: {exc}")

    st.divider()
    st.header("Display")
    show_network = st.checkbox("Road network context", value=False)
    show_classifier = st.checkbox("Classifier-only route", value=True)
    show_reference = st.checkbox("Reference route (if served)", value=False)
    show_gps = st.checkbox("Raw GPS observations", value=True)
    run_hmm = st.checkbox("Apply HMM / Viterbi decoding", value=True)

# ------------------------------------------------------------- trip selection
try:
    trips = api_get(api_url, "/trips")
except Exception as exc:
    st.error(f"Could not load the trip list from {api_url}: {exc}")
    st.info("Start the API on Laptop 1:  `uvicorn api.main:app --host 0.0.0.0 --port 8000`")
    st.stop()

st.subheader(f"Evaluation trips ({len(trips)})")
trip_table = pd.DataFrame(trips)[["trip_id", "n_observations", "duration_s",
                                  "n_nodes", "n_arcs"]]
trip_table = trip_table.rename(columns={
    "trip_id": "Trip", "n_observations": "GPS points", "duration_s": "Duration (s)",
    "n_nodes": "Network nodes", "n_arcs": "Network arcs"})

left, right = st.columns([1, 2])
with left:
    selected = st.selectbox("Select a trip to reconstruct",
                            [t["trip_id"] for t in trips])
    go = st.button("Reconstruct route", type="primary", use_container_width=True)
with right:
    st.dataframe(trip_table, hide_index=True, use_container_width=True, height=240)

if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.elapsed = None

if go:
    with st.spinner(f"Sending trip {selected} to the API..."):
        try:
            t0 = time.time()
            st.session_state.result = api_post(api_url, "/predict", {
                "trip_id": selected, "include_gps": show_gps,
                "include_reference": show_reference, "run_hmm": run_hmm})
            st.session_state.elapsed = time.time() - t0
            st.session_state.network = (api_get(api_url, f"/network/{selected}")
                                        if show_network else None)
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            st.session_state.result = None

res = st.session_state.result
if res is None:
    st.info("Select a trip and press **Reconstruct route**.")
    st.stop()

# ------------------------------------------------------------------- results
ev = res.get("evaluation", {}) or {}
st.divider()
st.subheader(f"Results — trip {res['trip_id']}")

c = st.columns(5)
c[0].metric("Observations", f"{res['n_observations']:,}")
c[1].metric("Top-1 accuracy (HMM)",
            "n/a" if ev.get("hmm_top1_accuracy") is None
            else f"{ev['hmm_top1_accuracy'] * 100:.2f}%",
            None if ev.get("hmm_improvement_pp") is None
            else f"{ev['hmm_improvement_pp']:+.2f} pp vs classifier")
c[2].metric("Route continuity",
            "n/a" if ev.get("hmm_route_continuity") is None
            else f"{ev['hmm_route_continuity'] * 100:.2f}%")
c[3].metric("Arcs in predicted route",
            f"{len(res['predicted_route']['arc_ids']):,}")
c[4].metric("Round trip", f"{st.session_state.elapsed:.2f}s",
            f"API {res.get('total_seconds', 0):.2f}s")

tab_map, tab_metrics, tab_raw = st.tabs(["Map", "Metrics", "API response"])

with tab_map:
    poly = res["predicted_route"]["polyline"]
    if poly:
        lats = [p[0] for p in poly]; lons = [p[1] for p in poly]
        centre = [sum(lats) / len(lats), sum(lons) / len(lons)]
    elif res.get("gps"):
        pts = res["gps"]["points"]
        centre = [sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)]
    else:
        centre = [49.87, 8.65]

    m = folium.Map(location=centre, zoom_start=14, tiles="OpenStreetMap",
                   control_scale=True)

    net = st.session_state.get("network")
    if show_network and net:
        g = folium.FeatureGroup(name="Road network", show=True)
        for seg in net["segments"]:
            folium.PolyLine(seg, color=COLOURS["network"], weight=1,
                            opacity=0.45).add_to(g)
        g.add_to(m)

    if show_gps and res.get("gps"):
        g = folium.FeatureGroup(name="Raw GPS observations", show=True)
        for i, p in enumerate(res["gps"]["points"]):
            folium.CircleMarker(p, radius=2, color=COLOURS["gps"], fill=True,
                                fill_opacity=0.7, weight=0,
                                tooltip=f"obs {i}").add_to(g)
        g.add_to(m)

    if show_reference and res.get("reference_route"):
        g = folium.FeatureGroup(name="Reference route", show=True)
        folium.PolyLine(res["reference_route"]["polyline"],
                        color=COLOURS["reference"], weight=6,
                        opacity=0.7, tooltip="reference").add_to(g)
        g.add_to(m)

    if show_classifier and res.get("classifier_only_route", {}).get("polyline"):
        g = folium.FeatureGroup(name="Classifier-only route", show=True)
        folium.PolyLine(res["classifier_only_route"]["polyline"],
                        color=COLOURS["classifier"], weight=4, opacity=0.8,
                        dash_array="6,6", tooltip="classifier only").add_to(g)
        g.add_to(m)

    if poly:
        g = folium.FeatureGroup(name="Predicted route", show=True)
        folium.PolyLine(poly, color=COLOURS["predicted"], weight=5, opacity=0.9,
                        tooltip="predicted route (HMM + Viterbi)").add_to(g)
        g.add_to(m)
        folium.Marker(poly[0], tooltip="start",
                      icon=folium.Icon(color="green", icon="play")).add_to(m)
        folium.Marker(poly[-1], tooltip="end",
                      icon=folium.Icon(color="red", icon="stop")).add_to(m)
        m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    folium.LayerControl(collapsed=False).add_to(m)

    legend_rows = [("Raw GPS observations", COLOURS["gps"]),
                   ("Predicted route (HMM)", COLOURS["predicted"])]
    if show_classifier:
        legend_rows.insert(1, ("Classifier only", COLOURS["classifier"]))
    if show_reference and res.get("reference_route"):
        legend_rows.append(("Reference route", COLOURS["reference"]))
    items = "".join(
        f'<div style="margin:2px 0"><span style="display:inline-block;width:18px;'
        f'height:4px;background:{col};vertical-align:middle;margin-right:6px">'
        f'</span>{name}</div>' for name, col in legend_rows)
    m.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;'
        'background:rgba(255,255,255,.94);padding:10px 12px;border-radius:6px;'
        'border:1px solid #ccc;font:12px system-ui,sans-serif">'
        f'<div style="font-weight:600;margin-bottom:6px">Trip {res["trip_id"]}</div>'
        f'{items}</div>'))

    st_folium(m, width=None, height=620, returned_objects=[])

with tab_metrics:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Server-side evaluation**")
        rows = [("Observations", ev.get("n_observations")),
                ("Scored observations", ev.get("n_scored_observations")),
                ("Classifier top-1", ev.get("classifier_top1_accuracy")),
                ("HMM top-1", ev.get("hmm_top1_accuracy")),
                ("Classifier continuity", ev.get("classifier_route_continuity")),
                ("HMM continuity", ev.get("hmm_route_continuity")),
                ("Arc-set precision (HMM)", ev.get("hmm_arc_set_precision")),
                ("Arc-set recall (HMM)", ev.get("hmm_arc_set_recall")),
                ("Arc-set Jaccard (HMM)", ev.get("hmm_arc_set_jaccard"))]
        st.dataframe(pd.DataFrame(rows, columns=["Metric", "Value"]),
                     hide_index=True, use_container_width=True)
    with col2:
        st.markdown("**Pipeline timings (seconds)**")
        st.dataframe(pd.DataFrame(list(res.get("timings", {}).items()),
                                  columns=["Stage", "Seconds"]),
                     hide_index=True, use_container_width=True)
        st.markdown("**Decoder configuration**")
        st.json(res.get("decoder", {}), expanded=False)

    conf = res.get("per_observation", {}).get("confidence")
    if conf:
        st.markdown("**Per-observation classifier confidence**")
        st.line_chart(pd.DataFrame({"confidence": conf}), height=200)

with tab_raw:
    st.caption(f"Model artifact served by the API: `{res.get('model')}`")
    trimmed = {k: v for k, v in res.items() if k not in ("gps", "per_observation")}
    st.json(trimmed, expanded=False)

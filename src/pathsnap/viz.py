"""Folium and matplotlib visualisation helpers.

Every map built here uses the same layer vocabulary so that maps from
different stages of the pipeline can be read side by side:

    grey    road network (context)
    blue    raw GPS observations
    green   reference route, where supplied
    red     predicted route
"""
from __future__ import annotations

import numpy as np

COLOURS = {
    "network": "#9aa0a6",
    "gps": "#1f6fd0",
    "reference": "#17a558",
    "predicted": "#d93025",
    "classifier": "#f29900",
}


def _centre(trip):
    return [float(np.mean(trip.track[:, 1])), float(np.mean(trip.track[:, 0]))]


def _arc_latlon(network, arc_ids):
    """[[lat, lon], [lat, lon]] pairs for a list of arc ids, rounded for size."""
    out = []
    for a in np.asarray(arc_ids):
        if a < 0:
            continue
        u, v = network.arcs[int(a)]
        out.append([[round(float(network.nodes_lonlat[u, 1]), 6),
                     round(float(network.nodes_lonlat[u, 0]), 6)],
                    [round(float(network.nodes_lonlat[v, 1]), 6),
                     round(float(network.nodes_lonlat[v, 0]), 6)]])
    return out


def _collapse(seq):
    """Drop consecutive duplicates from an arc sequence, ignoring -1."""
    seq = np.asarray(seq)
    seq = seq[seq >= 0]
    if len(seq) == 0:
        return seq
    return seq[np.concatenate([[True], seq[1:] != seq[:-1]])]


#: OpenStreetMap needs no API key.  CartoDB's tiles started requiring one,
#: which would leave every saved map with a blank background.
TILES = "OpenStreetMap"


def folium_trip_map(trip, network=None, predicted_arcs=None,
                    classifier_arcs=None, title: str = "",
                    network_sample: int = 5000, gps_sample: int | None = None):
    """Interactive map of one trip with toggleable layers.

    Parameters
    ----------
    trip            : Trip
    network         : RoadNetwork (built on demand when omitted)
    predicted_arcs  : arc sequence from the full pipeline (classifier + HMM)
    classifier_arcs : arc sequence from the classifier alone, for comparison
    """
    import folium

    if network is None:
        from .network import build_network
        network = build_network(trip)

    # keep the saved HTML manageable: one marker per ~1500 observations
    if gps_sample is None:
        gps_sample = max(1, trip.n_obs // 1500)

    m = folium.Map(location=_centre(trip), zoom_start=14, tiles=TILES,
                   control_scale=True)

    # --- road network context (sampled: full networks reach 165 k arcs)
    net_layer = folium.FeatureGroup(name="Road network", show=True)
    step = max(1, network.n_arcs // network_sample)
    # a single multi-polyline: thousands of separate PolyLine objects would
    # inflate the saved HTML by an order of magnitude
    folium.PolyLine(_arc_latlon(network, np.arange(0, network.n_arcs, step)),
                    color=COLOURS["network"], weight=1,
                    opacity=0.45).add_to(net_layer)
    net_layer.add_to(m)

    # --- raw GPS observations
    gps_layer = folium.FeatureGroup(name="Raw GPS observations", show=True)
    pts = trip.track[::gps_sample]
    for i, (lon, lat, ts) in enumerate(pts):
        folium.CircleMarker(
            [round(float(lat), 6), round(float(lon), 6)], radius=2,
            color=COLOURS["gps"], fill=True, fill_opacity=0.7, weight=0,
            tooltip=f"obs {i * gps_sample} | t={ts:.0f}s").add_to(gps_layer)
    gps_layer.add_to(m)

    folium.PolyLine([[round(float(lat), 6), round(float(lon), 6)]
                     for lon, lat, _ in trip.track],
                    color=COLOURS["gps"], weight=1.5, opacity=0.5,
                    tooltip="GPS trajectory (raw order)").add_to(
        folium.FeatureGroup(name="GPS trajectory line", show=False).add_to(m))

    # --- reference route
    if trip.route is not None and len(trip.route):
        ref = folium.FeatureGroup(name="Reference route", show=True)
        folium.PolyLine(_arc_latlon(network, trip.route),
                        color=COLOURS["reference"], weight=5, opacity=0.75,
                        tooltip="reference route").add_to(ref)
        ref.add_to(m)

    # --- classifier-only prediction
    if classifier_arcs is not None:
        lay = folium.FeatureGroup(name="Classifier-only prediction", show=False)
        folium.PolyLine(_arc_latlon(network, _collapse(classifier_arcs)),
                        color=COLOURS["classifier"], weight=4, opacity=0.8,
                        dash_array="6,6",
                        tooltip="classifier only").add_to(lay)
        lay.add_to(m)

    # --- final predicted route
    if predicted_arcs is not None:
        lay = folium.FeatureGroup(name="Predicted route (HMM + Viterbi)", show=True)
        folium.PolyLine(_arc_latlon(network, _collapse(predicted_arcs)),
                        color=COLOURS["predicted"], weight=4, opacity=0.9,
                        tooltip="predicted route").add_to(lay)
        lay.add_to(m)

    folium.Marker([trip.track[0, 1], trip.track[0, 0]], tooltip="start",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([trip.track[-1, 1], trip.track[-1, 0]], tooltip="end",
                  icon=folium.Icon(color="red", icon="stop")).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[trip.track[:, 1].min(), trip.track[:, 0].min()],
                  [trip.track[:, 1].max(), trip.track[:, 0].max()]])
    _add_legend(m, title, predicted_arcs is not None, classifier_arcs is not None,
                trip.route is not None)
    return m


def _add_legend(m, title, has_pred, has_clf, has_ref):
    import folium
    rows = [("Road network", COLOURS["network"]),
            ("Raw GPS observations", COLOURS["gps"])]
    if has_ref:
        rows.append(("Reference route", COLOURS["reference"]))
    if has_clf:
        rows.append(("Classifier only", COLOURS["classifier"]))
    if has_pred:
        rows.append(("Predicted route (HMM)", COLOURS["predicted"]))

    items = "".join(
        f'<div style="margin:2px 0"><span style="display:inline-block;width:18px;'
        f'height:4px;background:{c};vertical-align:middle;margin-right:6px"></span>'
        f'{name}</div>' for name, c in rows)
    html = (
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;'
        'background:rgba(255,255,255,0.94);padding:10px 12px;border-radius:6px;'
        'border:1px solid #ccc;font:12px system-ui,sans-serif;'
        'box-shadow:0 1px 4px rgba(0,0,0,.2)">'
        f'<div style="font-weight:600;margin-bottom:6px">{title or "PathSnap"}</div>'
        f'{items}</div>')
    m.get_root().html.add_child(folium.Element(html))

"""Stage 2: exploratory data analysis over the raw dataset.

Produces the dataset inventory, data-quality checks, network-connectivity
statistics, the static plots and one interactive Folium map.

Run:  python scripts/02_eda.py
Out:  results/eda_trip_inventory.csv
      results/eda_partition_summary.csv
      results/eda_network_summary.csv
      results/eda_data_quality.csv
      results/figures/*.png
      results/maps/eda_trip_<id>.html
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

from pathsnap.geo import haversine_m
from pathsnap.labeling import align_observations_to_route
from pathsnap.network import build_network
from pathsnap.trips import list_trips, load_trip
from pathsnap.viz import folium_trip_map

RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
MAPS = RESULTS / "maps"
PARTITIONS = ("train", "student_test")

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})


def inventory() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-trip inventory plus per-trip road-network statistics."""
    trip_rows, net_rows = [], []
    for partition in PARTITIONS:
        pdir = RAW / partition
        for tid in list_trips(pdir):
            trip = load_trip(pdir, tid)
            lon, lat, ts = trip.track[:, 0], trip.track[:, 1], trip.track[:, 2]
            dt = np.diff(ts)
            seg = haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])

            net = build_network(trip)
            summary = net.connectivity_summary()
            net_rows.append({"partition": partition, "trip_id": tid, **summary})

            # a reference route is supplied for both partitions here; the
            # evaluator partition is held back by the organisers
            if trip.route is not None:
                obs_xy = net.projection.to_xy(lon, lat)
                _, residual, assign = align_observations_to_route(obs_xy, net, trip.route)
                route_len = float(net.arc_length[trip.route].sum())
                res_median = float(np.median(residual))
                res_p95 = float(np.percentile(residual, 95))
                route_used = len(np.unique(assign)) / len(trip.route)
            else:
                route_len = res_median = res_p95 = route_used = np.nan

            trip_rows.append({
                "partition": partition,
                "trip_id": tid,
                "n_observations": trip.n_obs,
                "duration_s": float(ts[-1] - ts[0]),
                "n_route_arcs": 0 if trip.route is None else len(trip.route),
                "route_length_m": route_len,
                "track_length_m": float(seg.sum()),
                "lon_min": float(lon.min()), "lon_max": float(lon.max()),
                "lat_min": float(lat.min()), "lat_max": float(lat.max()),
                "bbox_width_m": float(haversine_m(lon.min(), lat.mean(), lon.max(), lat.mean())),
                "bbox_height_m": float(haversine_m(lon.mean(), lat.min(), lon.mean(), lat.max())),
                "dt_min_s": float(dt.min()), "dt_median_s": float(np.median(dt)),
                "dt_max_s": float(dt.max()),
                "frac_dt_eq_1s": float(np.mean(np.isclose(dt, 1.0))),
                "n_gaps_gt_5s": int(np.sum(dt > 5.0)),
                "step_median_m": float(np.median(seg)),
                "step_p99_m": float(np.percentile(seg, 99)),
                "step_max_m": float(seg.max()),
                "mean_speed_kmh": float(seg.sum() / max(ts[-1] - ts[0], 1e-9) * 3.6),
                "n_nodes": summary["n_nodes"], "n_arcs": summary["n_arcs"],
                "label_residual_median_m": res_median,
                "label_residual_p95_m": res_p95,
                "route_arc_coverage": route_used,
            })
            print(f"  {partition}/{tid}: {trip.n_obs} obs, "
                  f"{summary['n_arcs']} arcs", flush=True)

    return pd.DataFrame(trip_rows), pd.DataFrame(net_rows)


def data_quality(trips: pd.DataFrame) -> pd.DataFrame:
    """Explicit missing-value and validity checks on the raw files."""
    checks = []
    for partition in PARTITIONS:
        pdir = RAW / partition
        n_nan = n_dup_ts = n_nonmono = n_out_of_range = n_dup_pts = 0
        n_self_loop = n_bad_arc = n_zero_len = n_route_oob = 0
        total_obs = 0
        for tid in list_trips(pdir):
            trip = load_trip(pdir, tid)
            lon, lat, ts = trip.track[:, 0], trip.track[:, 1], trip.track[:, 2]
            total_obs += len(ts)
            n_nan += int(np.isnan(trip.track).sum())
            n_dup_ts += int(len(ts) - len(np.unique(ts)))
            n_nonmono += int(np.sum(np.diff(ts) <= 0))
            n_out_of_range += int(np.sum((lon < -180) | (lon > 180) |
                                         (lat < -90) | (lat > 90)))
            n_dup_pts += int(np.sum((np.diff(lon) == 0) & (np.diff(lat) == 0)))
            n_self_loop += int(np.sum(trip.arcs[:, 0] == trip.arcs[:, 1]))
            n_bad_arc += int(np.sum((trip.arcs < 0) |
                                    (trip.arcs >= len(trip.nodes))))
            net = build_network(trip)
            n_zero_len += int(np.sum(net.arc_length == 0))
            if trip.route is not None:
                n_route_oob += int(np.sum((trip.route < 0) |
                                          (trip.route >= len(trip.arcs))))
        checks.append({
            "partition": partition, "total_observations": total_obs,
            "nan_values_in_track": n_nan,
            "duplicate_timestamps": n_dup_ts,
            "non_increasing_timestamps": n_nonmono,
            "coordinates_out_of_range": n_out_of_range,
            "consecutive_identical_positions": n_dup_pts,
            "self_loop_arcs": n_self_loop,
            "arcs_with_invalid_node_id": n_bad_arc,
            "zero_length_arcs": n_zero_len,
            "route_arc_ids_out_of_range": n_route_oob,
        })
    return pd.DataFrame(checks)


def make_plots(trips: pd.DataFrame, nets: pd.DataFrame) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    tr = trips[trips.partition == "train"]
    te = trips[trips.partition == "student_test"]

    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    ax[0, 0].hist([tr.n_observations, te.n_observations], bins=18,
                  label=["train", "student_test"], color=["#4C72B0", "#DD8452"])
    ax[0, 0].set(title="GPS observations per trip", xlabel="observations", ylabel="trips")
    ax[0, 0].legend()

    ax[0, 1].hist(trips.dt_median_s, bins=20, color="#4C72B0")
    ax[0, 1].set(title="Median sampling interval", xlabel="seconds", ylabel="trips")

    ax[0, 2].hist(trips.frac_dt_eq_1s * 100, bins=20, color="#55A868")
    ax[0, 2].set(title="Share of steps exactly 1 s apart", xlabel="% of steps", ylabel="trips")

    ax[1, 0].scatter(trips.n_observations, trips.n_route_arcs, s=18,
                     c=np.where(trips.partition == "train", "#4C72B0", "#DD8452"))
    ax[1, 0].set(title="Observations vs reference-route length",
                 xlabel="observations", ylabel="route arcs")

    ax[1, 1].hist(np.log10(nets.n_arcs), bins=20, color="#C44E52")
    ax[1, 1].set(title="Road-network size per trip", xlabel="log10(arcs)", ylabel="trips")

    ax[1, 2].hist(trips.label_residual_median_m.dropna(), bins=20, color="#8172B3")
    ax[1, 2].set(title="Median GPS-to-reference-arc residual",
                 xlabel="metres", ylabel="trips")
    fig.suptitle("PathSnap dataset overview", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "eda_overview.png")
    plt.close(fig)

    # sampling-gap detail
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].bar(range(len(trips)), trips.dt_max_s.to_numpy(),
              color=np.where(trips.partition == "train", "#4C72B0", "#DD8452"))
    ax[0].set(title="Largest time gap per trip", xlabel="trip", ylabel="seconds", yscale="log")
    ax[1].bar(range(len(trips)), trips.step_max_m.to_numpy(),
              color=np.where(trips.partition == "train", "#4C72B0", "#DD8452"))
    ax[1].set(title="Largest step between fixes", xlabel="trip", ylabel="metres", yscale="log")
    fig.tight_layout()
    fig.savefig(FIGS / "eda_gaps.png")
    plt.close(fig)

    # trajectory shapes for a sample of trips
    sample = list_trips(RAW / "train")[:12]
    fig, axes = plt.subplots(3, 4, figsize=(13, 9))
    for axx, tid in zip(axes.ravel(), sample):
        trip = load_trip(RAW / "train", tid)
        axx.plot(trip.track[:, 0], trip.track[:, 1], lw=0.8, color="#4C72B0")
        axx.scatter(trip.track[0, 0], trip.track[0, 1], s=18, color="green", zorder=3)
        axx.scatter(trip.track[-1, 0], trip.track[-1, 1], s=18, color="red", zorder=3)
        axx.set_title(f"{tid} ({trip.n_obs} obs)", fontsize=8)
        axx.tick_params(labelsize=6)
    fig.suptitle("Raw GPS trajectories, first 12 training trips "
                 "(green = start, red = end)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "eda_trajectories.png")
    plt.close(fig)

    # geographic extent of every trip
    fig, ax = plt.subplots(figsize=(7, 6))
    for _, r in trips.iterrows():
        colour = "#4C72B0" if r.partition == "train" else "#DD8452"
        ax.add_patch(plt.Rectangle((r.lon_min, r.lat_min),
                                   r.lon_max - r.lon_min, r.lat_max - r.lat_min,
                                   fill=False, lw=0.7, ec=colour, alpha=0.7))
    ax.set(xlim=(trips.lon_min.min() - 0.01, trips.lon_max.max() + 0.01),
           ylim=(trips.lat_min.min() - 0.01, trips.lat_max.max() + 0.01),
           xlabel="longitude", ylabel="latitude",
           title="Geographic extent of each trip (blue = train, orange = student_test)")
    fig.tight_layout()
    fig.savefig(FIGS / "eda_extents.png")
    plt.close(fig)
    print(f"  figures -> {FIGS}")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    MAPS.mkdir(parents=True, exist_ok=True)

    print("Building trip inventory...")
    trips, nets = inventory()
    trips.to_csv(RESULTS / "eda_trip_inventory.csv", index=False)
    nets.to_csv(RESULTS / "eda_network_summary.csv", index=False)

    agg = trips.groupby("partition").agg(
        n_trips=("trip_id", "count"),
        total_observations=("n_observations", "sum"),
        obs_per_trip_mean=("n_observations", "mean"),
        obs_per_trip_min=("n_observations", "min"),
        obs_per_trip_max=("n_observations", "max"),
        total_track_km=("track_length_m", lambda s: s.sum() / 1000),
        mean_speed_kmh=("mean_speed_kmh", "mean"),
        frac_dt_eq_1s=("frac_dt_eq_1s", "mean"),
        trips_with_gaps=("n_gaps_gt_5s", lambda s: int((s > 0).sum())),
        mean_arcs=("n_arcs", "mean"),
        label_residual_median_m=("label_residual_median_m", "median"),
    ).reset_index()
    agg.to_csv(RESULTS / "eda_partition_summary.csv", index=False)

    print("\nRunning data-quality checks...")
    dq = data_quality(trips)
    dq.to_csv(RESULTS / "eda_data_quality.csv", index=False)

    print("\nPlotting...")
    make_plots(trips, nets)

    print("\nBuilding Folium map for a representative trip...")
    tid = trips[trips.partition == "train"].sort_values("n_observations").iloc[
        len(trips[trips.partition == "train"]) // 2].trip_id
    trip = load_trip(RAW / "train", tid)
    m = folium_trip_map(trip, title=f"EDA: training trip {tid}")
    out = MAPS / f"eda_trip_{tid}.html"
    m.save(str(out))
    print(f"  map -> {out}")

    print("\n=== partition summary ===")
    print(agg.to_string(index=False))
    print("\n=== data quality ===")
    print(dq.T.to_string())


if __name__ == "__main__":
    main()

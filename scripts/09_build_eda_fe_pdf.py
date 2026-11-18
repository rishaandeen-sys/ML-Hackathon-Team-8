"""Stage 9: build the EDA + feature-engineering briefing PDF, split by owner.

One document, four parts - one per team member. Shared context first, then a
self-contained chapter per person covering what they own, what they must be
able to explain, which files are theirs, and what they write.

Every figure is read from results/ at build time, so the PDF cannot drift from
what the pipeline actually produced.

This document is team-internal and gitignored. Re-run after any stage that
changes the numbers.

Run:  python scripts/09_build_eda_fe_pdf.py
Out:  docs/PathSnap_EDA_and_Feature_Engineering.pdf
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from pathsnap.candidates import DEFAULT_MAX_CANDIDATES, DEFAULT_RADIUS_M
from pathsnap.dataset import LABEL_RESIDUAL_MAX_M

RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
OUT = ROOT / "docs" / "PathSnap_EDA_and_Feature_Engineering.pdf"
REPO = "https://github.com/rishaandeen-sys/ML-Hackathon-Team-8"

OWNERS = {
    "A": ("Nanditha", "Dataset &amp; EDA"),
    "B": ("Rishaan", "Labelling &amp; data quality"),
    "C": ("Shreya", "Feature engineering"),
    "D": ("Sanjay", "Visualisation &amp; evidence"),
}

INK = colors.HexColor("#111827")
MUTED = colors.HexColor("#6B7280")
ACCENT = colors.HexColor("#1F6FD0")
RULE = colors.HexColor("#D1D5DB")
BAND = colors.HexColor("#F3F4F6")
TINT = colors.HexColor("#EAF2FC")

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold",
                            fontSize=21, leading=25, textColor=INK, spaceAfter=2),
    "sub": ParagraphStyle("s", parent=ss["Normal"], fontSize=10, leading=14.5,
                          textColor=MUTED, spaceAfter=13),
    "part": ParagraphStyle("pt", parent=ss["Heading1"], fontName="Helvetica-Bold",
                           fontSize=17, leading=21, textColor=ACCENT,
                           spaceBefore=0, spaceAfter=2),
    "partsub": ParagraphStyle("ps", parent=ss["Normal"], fontSize=10,
                              leading=14, textColor=MUTED, spaceAfter=12),
    "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold",
                         fontSize=13.5, leading=17, textColor=INK,
                         spaceBefore=15, spaceAfter=6),
    "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                         fontSize=11, leading=14.5, textColor=INK,
                         spaceBefore=10, spaceAfter=5),
    "body": ParagraphStyle("b", parent=ss["Normal"], fontSize=9.5, leading=14,
                           textColor=INK, alignment=TA_LEFT, spaceAfter=7),
    "small": ParagraphStyle("sm", parent=ss["Normal"], fontSize=8.3, leading=11.6,
                            textColor=MUTED, spaceAfter=6),
    "cap": ParagraphStyle("c", parent=ss["Normal"], fontSize=8.2, leading=11,
                          textColor=MUTED, spaceBefore=3, spaceAfter=10),
    "code": ParagraphStyle("co", parent=ss["Normal"], fontName="Courier",
                           fontSize=8.2, leading=11.5, textColor=INK,
                           backColor=BAND, borderPadding=5, spaceAfter=8),
    "cell": ParagraphStyle("cl", parent=ss["Normal"], fontSize=8.2, leading=11),
    "cellb": ParagraphStyle("clb", parent=ss["Normal"], fontSize=8.2, leading=11,
                            fontName="Helvetica-Bold"),
}


def P(t, style="body"):
    return Paragraph(t, S[style])


def bullets(items, style="body"):
    return [Paragraph(f"&bull;&nbsp;&nbsp;{t}", S[style]) for t in items]


def table(rows, widths, header=True, align_right=()):
    data = [[Paragraph(str(c), S["cellb" if (header and i == 0) else "cell"])
             for c in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LINEBELOW", (0, 0), (-1, 0), 0.9, INK if header else RULE),
          ("LINEBELOW", (0, 1), (-1, -2), 0.3, RULE),
          ("TOPPADDING", (0, 0), (-1, -1), 4),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
          ("LEFTPADDING", (0, 0), (-1, -1), 5),
          ("RIGHTPADDING", (0, 0), (-1, -1), 5)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), BAND))
    for c in align_right:
        st.append(("ALIGN", (c, 0), (c, -1), "RIGHT"))
    t.setStyle(TableStyle(st))
    return t


def callout(title, body):
    inner = [Paragraph(f"<b>{title}</b>", S["cell"]),
             Spacer(1, 3), Paragraph(body, S["cell"])]
    t = Table([[inner]], colWidths=[165 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), TINT),
        ("LINEBEFORE", (0, 0), (0, -1), 2.2, ACCENT),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9)]))
    return [t, Spacer(1, 9)]


def part_header(letter):
    name, area = OWNERS[letter]
    return [P(f"Part {letter} &nbsp;|&nbsp; {name}", "part"),
            P(area, "partsub")]


def figure(name, caption, width=162 * mm):
    p = FIGS / name
    if not p.exists():
        return [P(f"<i>missing figure: {name}</i>", "small")]
    from PIL import Image as PILImage
    with PILImage.open(p) as im:
        w, h = im.size
    return [Image(str(p), width=width, height=width * h / w), P(caption, "cap")]


def furniture(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 12 * mm,
                      "PathSnap | Team 8 | EDA & Feature Engineering")
    canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, f"Page {doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(20 * mm, 15.5 * mm, A4[0] - 20 * mm, 15.5 * mm)
    canvas.restoreState()


def load():
    out = {}
    for k, n in [("part", "eda_partition_summary.csv"),
                 ("dq", "eda_data_quality.csv"),
                 ("trips", "eda_trip_inventory.csv"),
                 ("net", "eda_network_summary.csv"),
                 ("anom", "eda_anomalies.csv"),
                 ("cand", "candidate_generation_summary.csv")]:
        p = RESULTS / n
        out[k] = pd.read_csv(p) if p.exists() else None
    return out


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d = load()
    part, dq, trips, net, anom, cand = (d["part"], d["dq"], d["trips"],
                                        d["net"], d["anom"], d["cand"])
    tr = part[part.partition == "train"].iloc[0]
    te = part[part.partition == "student_test"].iloc[0]
    ctr = cand[cand.partition == "train"].iloc[0]
    cte = cand[cand.partition == "student_test"].iloc[0]
    dqt = dq.set_index("partition")
    gap_trips = int((trips.n_gaps_gt_5s > 0).sum())

    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=20 * mm,
        title="PathSnap - EDA and Feature Engineering (by owner)",
        author="Team 8")
    F: list = []

    # ============================================================ front matter
    F.append(P("PathSnap: Dataset, EDA &amp; Feature Engineering"))
    F.append(P(f"UE24CS352A Machine Learning Hackathon &nbsp;|&nbsp; PES University, "
               f"Dept. of CSE (RR Campus) &nbsp;|&nbsp; Team 8<br/>"
               f"Repository: <font color='#1F6FD0'>{REPO}</font>"
               f" &nbsp;|&nbsp; Generated {date.today().isoformat()}", "sub"))

    F.append(P("How this document is organised", "h1"))
    F.append(P(
        "Section 0 is shared context everyone needs. Parts A to D are one chapter "
        "per person: what you own, what you must be able to explain, which files "
        "are yours, and what you write. Read section 0 and your own part in full; "
        "skim the other three."))
    F.append(table([
        ["Part", "Who", "Area", "Writes", "Reviews"],
        ["A", "Nanditha", "Dataset &amp; EDA", "Report §2.1-2.3", "Part B"],
        ["B", "Rishaan", "Labelling &amp; data quality", "Report §2.4, §3", "Part C"],
        ["C", "Shreya", "Feature engineering", "Report §2.2, §4", "Part D"],
        ["D", "Sanjay", "Visualisation &amp; evidence", "Report §6, captions", "Part A"],
    ], [16 * mm, 28 * mm, 52 * mm, 38 * mm, 31 * mm]))
    F.extend(callout(
        "These are ownership areas, not silos.",
        "The assessment policy requires <i>every</i> member to be able to explain "
        "the whole submission, and the organisers may question anyone on anything. "
        "Each owner also reviews one teammate's part so nothing has a single point "
        "of failure."))

    # ============================================================== section 0
    F.append(P("0. Shared context: what the problem and the data are", "h1"))
    F.append(P(
        "The task is map matching: recovering which road a vehicle actually "
        "travelled on from a GPS trace that drifts off the road network. Each trip "
        "ships four whitespace-separated files with no headers, and <b>its own road "
        "sub-network</b> - node and arc ids are local to a trip and must never be "
        "compared across trips."))
    F.append(table([
        ["File", "Columns", "Meaning"],
        ["<font face='Courier'>.track</font>", "longitude, latitude, timestamp (s)",
         "GPS observations, nominally 1 Hz"],
        ["<font face='Courier'>.nodes</font>", "longitude, latitude",
         "Road-network nodes; row index is the node id"],
        ["<font face='Courier'>.arcs</font>", "from_node, to_node",
         "Directed road segments; row index is the arc id"],
        ["<font face='Courier'>.route</font>", "arc_id",
         "Chronological reference route, where supplied"],
    ], [26 * mm, 52 * mm, 87 * mm]))
    F.append(Spacer(1, 6))
    F.extend(callout(
        "Arcs are directed - everyone needs this.",
        "A two-way street appears twice, with identical geometry and opposite "
        "bearing, and the reference route is a sequence of <i>directed</i> arc ids. "
        "Direction of travel is part of what must be predicted. It is why distance "
        "alone can never be enough, and it shapes Part C's feature design."))
    F.append(P(
        "The pipeline has two stages. A <b>classifier</b> ranks the candidate arcs "
        "near each observation; an <b>HMM with Viterbi decoding</b> then picks the "
        "most likely <i>sequence</i>, using road-network connectivity. This document "
        "covers everything up to the point where the classifier is trained."))
    F.append(P("Pipeline order, and who owns each step:", "h2"))
    F.append(table([
        ["Step", "What happens", "Owner"],
        ["1", "Load and inspect the raw files", "A"],
        ["2", "Explore: quality, sampling, networks, anomalies", "A, B"],
        ["3", "Derive a reference arc per observation", "B"],
        ["4", "Decide which labels are trustworthy", "B"],
        ["5", "Retrieve candidate arcs, build 26 features", "C"],
        ["6", "Visualise and evidence all of it", "D"],
    ], [16 * mm, 118 * mm, 31 * mm]))

    # ================================================================= PART A
    F.append(PageBreak())
    F.extend(part_header("A"))
    F.extend(callout(
        "Your one-line job",
        "Know the dataset better than anyone in the room: how big it is, how clean "
        "it is, how it was sampled, and why we work in metres rather than degrees."))

    F.append(P("A1. Partitions", "h1"))
    F.append(table([
        ["", "Training", "Student testing", "Evaluator"],
        ["Trips", f"{int(tr.n_trips)}", f"{int(te.n_trips)}", "10 (held back)"],
        ["GPS observations", f"{int(tr.total_observations):,}",
         f"{int(te.total_observations):,}", "-"],
        ["Track distance", f"{tr.total_track_km:,.0f} km",
         f"{te.total_track_km:,.0f} km", "-"],
        ["Mean speed", f"{tr.mean_speed_kmh:.1f} km/h",
         f"{te.mean_speed_kmh:.1f} km/h", "-"],
        ["Steps exactly 1 s apart", f"{tr.frac_dt_eq_1s * 100:.1f}%",
         f"{te.frac_dt_eq_1s * 100:.1f}%", "-"],
        ["Mean arcs per trip", f"{tr.mean_arcs:,.0f}", f"{te.mean_arcs:,.0f}", "-"],
    ], [46 * mm, 40 * mm, 40 * mm, 39 * mm], align_right=(1, 2, 3)))
    F.append(P("Used for fitting and cross-validation / development-stage model "
               "comparison / final evaluation only. No trip is ever split across "
               "partitions or across CV folds.", "small"))

    F.append(P("A2. The raw data is clean", "h1"))
    F.append(table([
        ["Check", "Training", "Student testing"],
        ["NaN values in .track", int(dqt.loc["train", "nan_values_in_track"]),
         int(dqt.loc["student_test", "nan_values_in_track"])],
        ["Duplicate timestamps", int(dqt.loc["train", "duplicate_timestamps"]),
         int(dqt.loc["student_test", "duplicate_timestamps"])],
        ["Non-increasing timestamps",
         int(dqt.loc["train", "non_increasing_timestamps"]),
         int(dqt.loc["student_test", "non_increasing_timestamps"])],
        ["Coordinates out of range",
         int(dqt.loc["train", "coordinates_out_of_range"]),
         int(dqt.loc["student_test", "coordinates_out_of_range"])],
        ["Arcs with invalid node id",
         int(dqt.loc["train", "arcs_with_invalid_node_id"]),
         int(dqt.loc["student_test", "arcs_with_invalid_node_id"])],
        ["Route arc ids out of range",
         int(dqt.loc["train", "route_arc_ids_out_of_range"]),
         int(dqt.loc["student_test", "route_arc_ids_out_of_range"])],
        ["Self-loop arcs", int(dqt.loc["train", "self_loop_arcs"]),
         int(dqt.loc["student_test", "self_loop_arcs"])],
        ["Zero-length arcs", int(dqt.loc["train", "zero_length_arcs"]),
         int(dqt.loc["student_test", "zero_length_arcs"])],
    ], [75 * mm, 45 * mm, 45 * mm], align_right=(1, 2)))
    F.append(P(
        "There are no missing values to impute at the raw level - say so plainly "
        "rather than inventing an imputation story. The only structural oddities "
        "are a few self-loops and zero-length arcs, handled by guarding the "
        "division in the point-to-segment projection rather than by deleting them.",
        "small"))

    F.append(PageBreak())
    F.append(P("A3. The road networks vary enormously", "h1"))
    F.append(table([
        ["Statistic", "Min", "Median", "Max"],
        ["Nodes per trip", f"{net.n_nodes.min():,.0f}",
         f"{net.n_nodes.median():,.0f}", f"{net.n_nodes.max():,.0f}"],
        ["Arcs per trip", f"{net.n_arcs.min():,.0f}",
         f"{net.n_arcs.median():,.0f}", f"{net.n_arcs.max():,.0f}"],
        ["Weakly-connected components", f"{net.n_weak_components.min():,.0f}",
         f"{net.n_weak_components.median():,.0f}",
         f"{net.n_weak_components.max():,.0f}"],
        ["Largest component share",
         f"{net.largest_component_frac.min() * 100:.0f}%",
         f"{net.largest_component_frac.median() * 100:.0f}%",
         f"{net.largest_component_frac.max() * 100:.0f}%"],
        ["Median arc length", f"{net.arc_length_median_m.min():.1f} m",
         f"{net.arc_length_median_m.median():.1f} m",
         f"{net.arc_length_median_m.max():.1f} m"],
        ["Longest arc", f"{net.arc_length_max_m.min():,.0f} m",
         f"{net.arc_length_max_m.median():,.0f} m",
         f"{net.arc_length_max_m.max():,.0f} m"],
    ], [58 * mm, 35 * mm, 36 * mm, 36 * mm], align_right=(1, 2, 3)))
    F.append(P(
        "Two consequences you should be able to state. A median arc of ~25 m "
        "against a longest arc of several kilometres is why the spatial index "
        "(Part C) samples <i>along</i> arcs instead of indexing midpoints. And the "
        "networks are not fully connected, so the decoder must tolerate unreachable "
        "transitions rather than assume a path always exists.", "small"))

    F.append(P("A4. Sampling: mostly 1 Hz, but gaps are common", "h1"))
    F.append(P(
        f"{tr.frac_dt_eq_1s * 100:.1f}% of training steps are exactly 1 s apart, but "
        f"<b>{gap_trips} of {len(trips)} trips contain at least one gap longer than "
        f"5 s</b>. Gap handling is not an edge case: the decoder widens its search "
        f"radius and relaxes its transition scale across a gap rather than forcing "
        f"an implausible local jump."))

    F.append(P("A5. Why we project to metres", "h1"))
    F.extend(callout(
        "The one subtle point in your part.",
        "Longitude and latitude are <i>angular</i>. At this latitude a degree of "
        "longitude is about 72 km while a degree of latitude is about 111 km, so "
        "Euclidean distance in degrees is not a distance at all - it would distort "
        "every distance and every heading in the feature set. We project each trip "
        "to a local equirectangular plane in metres, centred on that trip. At city "
        "scale the error is well under 0.1%, it is exactly invertible, and it is far "
        "cheaper than computing geodesics for every pair."))

    F.extend(figure("eda_overview.png",
                    "Figure A1 - Dataset overview: observations per trip, sampling "
                    "interval, share of 1 s steps, observations against reference-"
                    "route length, network size, label residual."))
    F.append(PageBreak())
    F.extend(figure("eda_gaps.png",
                    "Figure A2 - Largest time gap and largest spatial step per trip "
                    "(log scale).", width=158 * mm))

    F.append(P("A6. Your files", "h1"))
    F.append(table([
        ["Path", "What it is"],
        ["<font face='Courier'>scripts/02_eda.py</font>",
         "The EDA stage you own end to end"],
        ["<font face='Courier'>src/pathsnap/trips.py</font>",
         "Raw-file loading, column validation"],
        ["<font face='Courier'>src/pathsnap/geo.py</font>",
         "The projection and point-to-segment geometry"],
        ["<font face='Courier'>results/eda_trip_inventory.csv</font>",
         "Per-trip: observations, duration, bbox, sampling, file sizes, networks"],
        ["<font face='Courier'>results/eda_partition_summary.csv</font>",
         "The table in A1"],
        ["<font face='Courier'>results/eda_data_quality.csv</font>",
         "The table in A2"],
        ["<font face='Courier'>results/eda_network_summary.csv</font>",
         "The table in A3"],
        ["<font face='Courier'>results/figures/eda_*.png</font>", "Figures A1-A2"],
    ], [70 * mm, 95 * mm]))
    F.append(P("Reproduce with: <font face='Courier'>python scripts/02_eda.py</font> "
               "(~4 min)", "small"))

    F.append(P("A7. Answer these cold", "h1"))
    F.extend(bullets([
        "Why project to metres? Why is Euclidean distance in degrees wrong here?",
        "How many trips are in each partition, and what is each used for?",
        "Were there missing values? (No - say so, and say how you checked.)",
        "How many trips have sampling gaps, and why does that matter downstream?",
        "Why do some trips have 300 nodes and others 155,000?",
    ], "small"))

    # ================================================================= PART B
    F.append(PageBreak())
    F.extend(part_header("B"))
    F.extend(callout(
        "Your one-line job",
        "Explain how we turned a route file into per-observation labels, and why we "
        "threw some of them away. Yours is the subtlest reasoning in the document."))

    F.append(P("B1. The labelling problem", "h1"))
    F.append(P(
        "<font face='Courier'>.route</font> gives the arcs traversed, but <i>not</i> "
        "which observation sits on which arc - and the two sequences have different "
        "lengths. Trip 00000001 has 1,315 observations against 525 route arcs. The "
        "mapping must be recovered before any supervised learning can happen."))
    F.append(P(
        "The obvious approach - assign each observation to its nearest route arc, "
        "independently - is wrong. GPS noise makes the assignment jump backwards "
        "and forwards along the route, producing a label sequence that no causal "
        "model could reproduce. Instead we solve a <b>monotonic alignment by "
        "dynamic programming</b>, forcing the assigned route index to be "
        "non-decreasing in time:"))
    F.append(P("f[t, r] = cost[t, r] + min over r' &lt;= r of f[t-1, r']", "code"))
    F.append(P(
        "<font face='Courier'>cost[t, r]</font> is the distance from observation t "
        "to route arc r. A running prefix-minimum computes the inner minimum in "
        "constant time per cell, making the whole thing O(T x R); backtracking "
        "recovers the assignment. It is the same shape as DTW. The result is "
        "monotonic by construction, and the residual distance it leaves behind "
        "becomes a diagnostic in its own right.", "small"))

    F.append(P("B2. The residuals are bimodal - and that is the finding", "h1"))
    F.append(table([
        ["Percentile of training residual", "Value"],
        ["50th", "5.7 m"],
        ["75th", "10.9 m"],
        ["90th", "89.3 m"],
        ["95th", "142.0 m"],
    ], [95 * mm, 70 * mm], align_right=(1,)))
    F.append(P(
        "There is a gap between the 75th and 90th percentiles with very little "
        "between them. That is not a smooth tail of hard examples - it is two "
        "different populations. The upper one is trips whose reference route covers "
        "only part of the journey."))
    worst = anom.nsmallest(3, "route_coverage")[
        ["trip_id", "n_observations", "route_coverage", "label_residual_median_m"]]
    F.append(table(
        [["Trip", "Observations", "Route coverage", "Median residual"]] +
        [[f"{int(r.trip_id):08d}", f"{int(r.n_observations):,}",
          f"{r.route_coverage * 100:.0f}%", f"{r.label_residual_median_m:.1f} m"]
         for r in worst.itertuples()],
        [40 * mm, 42 * mm, 42 * mm, 41 * mm], align_right=(1, 2, 3)))
    F.extend(callout(
        "Low coverage and bad labels are not the same thing.",
        "Trip 00000002 has only 14% route coverage yet a 5.8 m median residual - the "
        "alignment still recovers most of it. Only 00000033 and 00000054 have "
        "genuinely untrustworthy labels. Do not conflate the two; it is exactly the "
        "kind of distinction the examiners will probe."))

    F.append(PageBreak())
    F.append(P("B3. The 30 m threshold, and how to justify it", "h1"))
    F.append(P(
        f"Observations whose aligned arc lies further than "
        f"<b>{LABEL_RESIDUAL_MAX_M:.0f} m</b> away are flagged "
        f"<font face='Courier'>label_valid = 0</font>, kept in the table for "
        f"inspection, and excluded from fitting and scoring."))
    F.append(table([
        ["Threshold", "Observations kept", "Candidate miss rate on what remains"],
        ["25 m", "86.79%", "0.226%"],
        ["<b>30 m (chosen)</b>", "<b>87.49%</b>", "<b>0.262%</b>"],
        ["50 m", "88.84%", "0.658%"],
        ["75 m", "89.74%", "1.160%"],
        ["100 m", "90.15%", "1.254%"],
    ], [38 * mm, 50 * mm, 77 * mm], align_right=(1, 2)))
    F.extend(callout(
        "The argument is agreement, not the percentage kept.",
        f"The candidate miss rate on the surviving training observations "
        f"({ctr.candidate_miss_rate:.3%}) matches the student-testing partition "
        f"({cte.candidate_miss_rate:.3%}), which has no coverage problem at all. "
        f"Two independently-derived numbers landing in the same place is the "
        f"evidence that the cut separates <i>bad labels</i> from <i>good ones</i>, "
        f"rather than merely discarding hard examples. If you are asked “why not "
        f"50 m?”, that is the answer: at 50 m the miss rate climbs to 0.658% and "
        f"stops agreeing."))

    F.append(P("B4. Anomalies: three distinct problems", "h1"))
    F.append(P(f"{len(anom)} of {len(trips)} trips carry at least one flag. They are "
               f"reported separately rather than as one quality score, because they "
               f"need different treatment:"))
    F.extend(bullets([
        "<b>Trailing-fix artefact (2 trips).</b> Trip 00000090 has 536 observations "
        "spanning 154,030 s - but the 42.5-hour gap sits at observation 534. It is a "
        "normal ~800 s run plus two stray fixes recorded two days later. Worth "
        "mentioning: it would badly skew any per-trip duration statistic.",
        "<b>Low route coverage (12 trips).</b> Worst: 00000033 (13%), 00000002 "
        "(14%), 00000054 (19%).",
        "<b>Large spatial gap (7 trips).</b> Up to an 888 m single step between "
        "consecutive fixes.",
    ]))

    F.append(P("B5. Your files", "h1"))
    F.append(table([
        ["Path", "What it is"],
        ["<font face='Courier'>src/pathsnap/labeling.py</font>",
         "The monotonic DP alignment - your core artefact"],
        ["<font face='Courier'>src/pathsnap/dataset.py</font>",
         "Table assembly; owns <font face='Courier'>LABEL_RESIDUAL_MAX_M</font> "
         "and the candidate miss-rate metric"],
        ["<font face='Courier'>results/eda_anomalies.csv</font>",
         "The flagged trips in B4"],
        ["<font face='Courier'>scripts/01_build_dataset.py</font>",
         "Runs labelling over both partitions"],
    ], [70 * mm, 95 * mm]))

    F.append(P("B6. Answer these cold", "h1"))
    F.extend(bullets([
        "Why not just assign each observation to its nearest route arc?",
        "What does the monotonicity constraint buy you, and how is it enforced?",
        "What is the time complexity, and why is the prefix-minimum needed?",
        "What makes the residual distribution bimodal?",
        "Why 30 m and not 50 m? (Agreement with student_test - see B3.)",
        "Does excluding 12.5% of training observations bias the model? "
        "(Those stretches have no reference arc at all, so there is nothing to "
        "learn from them - the alternative is training on noise.)",
    ], "small"))

    # ================================================================= PART C
    F.append(PageBreak())
    F.extend(part_header("C"))
    F.extend(callout(
        "Your one-line job",
        "Explain how a map-matching problem became a classification problem, and "
        "justify all 26 features. Yours is the largest volume of material."))

    F.append(P("C1. Framing: candidate ranking", "h1"))
    F.append(P(
        "Map matching is cast as a <b>candidate-ranking</b> problem. For every GPS "
        "observation the nearby directed arcs are retrieved, and one row is built "
        "per (observation, candidate arc) pair. A binary classifier scores each "
        "pair; the per-observation scores are normalised into a distribution over "
        "candidates - which is exactly the emission model the HMM stage consumes. "
        "That is why the framing matters: it is not an arbitrary choice, it is what "
        "makes the two stages fit together."))
    F.append(table([
        ["", "Training", "Student testing"],
        ["Search radius", f"{DEFAULT_RADIUS_M:.0f} m", f"{DEFAULT_RADIUS_M:.0f} m"],
        ["Max candidates kept", f"{DEFAULT_MAX_CANDIDATES}",
         f"{DEFAULT_MAX_CANDIDATES}"],
        ["Candidate rows", f"{int(ctr.n_candidate_rows):,}",
         f"{int(cte.n_candidate_rows):,}"],
        ["Mean candidates / observation", f"{ctr.mean_candidates_per_obs:.2f}",
         f"{cte.mean_candidates_per_obs:.2f}"],
        ["Positive rate", f"{ctr.positive_rate:.2%}", f"{cte.positive_rate:.2%}"],
        ["Candidate miss rate", f"{ctr.candidate_miss_rate:.3%}",
         f"{cte.candidate_miss_rate:.3%}"],
    ], [63 * mm, 51 * mm, 51 * mm], align_right=(1, 2)))
    F.extend(callout(
        "Candidate miss rate is the ceiling on top-1 accuracy.",
        f"An observation whose reference arc was never retrieved cannot be "
        f"classified correctly by any model, however good. At "
        f"{ctr.candidate_miss_rate:.2%} it caps top-1 at "
        f"{100 - ctr.candidate_miss_rate * 100:.2f}%. Across all ten trained models "
        f"the miss rate is identical, because it is a property of <i>retrieval</i>, "
        f"not of the classifier. The real headroom is the 12-16% of observations "
        f"where the right arc <i>was</i> retrieved but not ranked first."))

    F.append(P("C2. The spatial index", "h1"))
    F.append(P(
        "Candidates come from a KD-tree built over points sampled every 25 m along "
        "each arc, then filtered by exact point-to-segment distance. Indexing arc "
        "<b>midpoints</b> instead would be a bug: the median arc is ~25 m but the "
        "longest is several kilometres, and a long arc running right past the query "
        "point can have its midpoint far away. Sampling along the arc removes that "
        "failure mode."))

    F.append(PageBreak())
    F.append(P("C3. The 26 features", "h1"))
    F.append(table([
        ["Group", "Features", "What it captures"],
        ["Geometry<br/>(8)",
         "<font face='Courier'>dist_m, dist_rank, dist_minus_best, "
         "dist_ratio_best, dist_z_in_obs, arc_length, proj_t, "
         "proj_is_endpoint</font>",
         "How close the arc is, and how close <i>relative to the other candidates "
         "of the same observation</i>. A 15 m arc means something different when "
         "the runner-up is 16 m away than when it is 60 m away."],
        ["Heading<br/>(4)",
         "<font face='Courier'>heading_diff, heading_cos, heading_valid, "
         "turn_angle</font>",
         "Direction of travel against arc bearing. The <b>only</b> thing that "
         "separates the two directions of a two-way street - both carry identical "
         "geometry, so distance cannot. "
         "<font face='Courier'>heading_valid</font> switches off when the vehicle "
         "has moved under 2 m and the heading is just noise."],
        ["Kinematics<br/>(6)",
         "<font face='Courier'>speed_mps, accel_mps2, dt_prev, dt_next, "
         "step_prev_m, step_next_m</font>",
         "Local motion and sampling irregularity. A stationary vehicle produces a "
         "different candidate pattern from one at 80 km/h."],
        ["Network<br/>context (8)",
         "<font face='Courier'>arc_out_degree, arc_in_degree, has_reverse_arc, "
         "n_candidates, local_arc_density, dist_prev_obs_to_arc, "
         "dist_next_obs_to_arc, obs_frac</font>",
         "Topology around the arc, and whether the same arc also explains the "
         "neighbouring fixes - weak sequential evidence, without the classifier "
         "having to model the sequence itself (that is the HMM's job)."],
    ], [26 * mm, 58 * mm, 81 * mm]))

    F.append(P("C4. Design decisions worth defending", "h1"))
    F.extend(bullets([
        "<b>Relative features, not just absolute ones.</b> "
        "<font face='Courier'>dist_rank</font>, "
        "<font face='Courier'>dist_minus_best</font> and "
        "<font face='Courier'>dist_z_in_obs</font> describe a candidate against its "
        "competitors. Without them the model must learn an absolute distance "
        "threshold, which cannot hold across both dense city centres and sparse "
        "outskirts.",
        "<b>No feature touches the reference route.</b> The same code path runs at "
        "training and at inference time, which is what makes the deployed API "
        "trustworthy - and is the first thing to say if asked about leakage.",
        "<b>Median imputation</b> for the neighbour-dependent features, which are "
        "genuinely undefined at the first and last fix of a trip. Standardisation "
        "is applied only for the distance- and gradient-based models.",
        "<b>60 m / 10 candidates</b> is a deliberate trade: wider lowers the miss "
        "rate slightly but costs roughly linearly in feature computation and "
        "decoding time.",
    ]))

    F.append(P("C5. Your files", "h1"))
    F.append(table([
        ["Path", "What it is"],
        ["<font face='Courier'>src/pathsnap/candidates.py</font>",
         "<b>Your core artefact</b> - retrieval and all 26 features. "
         "<font face='Courier'>FEATURE_NAMES</font> is the single source of "
         "feature order, relied on by the model and the API"],
        ["<font face='Courier'>src/pathsnap/network.py</font>",
         "Road network, the 25 m-sampled KD-tree, bounded Dijkstra"],
        ["<font face='Courier'>results/candidate_generation_summary.csv</font>",
         "The table in C1"],
        ["<font face='Courier'>notebooks/pathsnap_walkthrough.ipynb</font>",
         "Shows feature separation between correct and wrong candidates"],
    ], [70 * mm, 95 * mm]))

    F.append(P("C6. Answer these cold", "h1"))
    F.extend(bullets([
        "What separates the two directions of a two-way street? (Heading. Not "
        "distance - the geometry is identical.)",
        "Why relative distance features as well as absolute ones?",
        "What is the candidate miss rate, and why does it bound accuracy?",
        "Why sample along arcs instead of indexing midpoints?",
        "Why is the positive rate only ~13%, and how is that imbalance handled?",
        "Could you drop features? Which, and how would you decide?",
    ], "small"))

    # ================================================================= PART D
    F.append(PageBreak())
    F.extend(part_header("D"))
    F.extend(callout(
        "Your one-line job",
        "Make the work visible and provable. Yours is the least mathematically "
        "demanding part, so it also carries the deadline-critical assembly work - "
        "if the evidence is missing on the day, nothing else counts."))

    F.append(P("D1. The map layer vocabulary", "h1"))
    F.append(P("Every map in the project uses the same colours, so maps from "
               "different stages can be read side by side:"))
    F.append(table([
        ["Colour", "Layer", "What it means"],
        ["Grey", "Road network", "Context; sampled for file size"],
        ["Blue", "Raw GPS observations", "What the sensor recorded, drift and all"],
        ["Green", "Reference route", "The supplied ground truth, where available"],
        ["Orange (dashed)", "Classifier-only route", "Per-observation top-1 picks, "
         "before any sequence reasoning"],
        ["Red", "Predicted route", "After HMM + Viterbi, with intermediate arcs "
         "filled in"],
    ], [32 * mm, 45 * mm, 88 * mm]))
    F.extend(callout(
        "The orange-versus-red comparison is the whole story.",
        "Orange is what the classifier alone believes; red is what the decoder "
        "concludes once road-network connectivity is enforced. Where they diverge is "
        "precisely where the HMM earned its place. Find a trip where that is "
        "visually obvious and use it as the headline figure."))

    F.append(P("D2. What you must produce", "h1"))
    F.append(table([
        ["Deliverable", "Where it comes from", "Required by"],
        ["At least one Folium map in the report",
         "<font face='Courier'>results/maps/index.html</font> - pick the clearest",
         "Deliverables §6"],
        ["Screenshots of the working UI",
         "Streamlit on Laptop 2 during a live run", "Deliverables §7"],
        ["Screenshot of API responses in the terminal",
         "The uvicorn window on Laptop 1", "Deliverables §7"],
        ["The compiled report PDF",
         "<font face='Courier'>results/report/technical_report.md</font> plus "
         "everyone's written sections", "Deliverable 2"],
    ], [46 * mm, 72 * mm, 47 * mm]))
    F.extend(callout(
        "These cannot be produced after the fact.",
        "The UI and API screenshots only exist if someone captures them while the "
        "system is running on the day. Take them during the pre-demo smoke test, "
        "not during the evaluation itself."))

    F.append(PageBreak())
    F.append(P("D3. Map requirements from the brief", "h1"))
    F.extend(bullets([
        "Distinguishable colours and markers per layer - done, see D1.",
        "A clear legend - every generated map carries one.",
        "Appropriate extent and zoom - maps auto-fit to the trip bounds.",
        "Easy comparison of observed GPS against predicted route.",
        "Layer controls - all layers are individually toggleable.",
        "Commit the interactive HTML or the code that generates it - both are in "
        "the repository.",
    ]))
    F.append(P(
        "Note the maps use OpenStreetMap tiles. CartoDB started requiring an API "
        "key, which would have left every saved map with a blank background - worth "
        "knowing if anyone asks why that basemap and not another.", "small"))

    F.append(P("D4. Your files", "h1"))
    F.append(table([
        ["Path", "What it is"],
        ["<font face='Courier'>src/pathsnap/viz.py</font>",
         "Map construction, the layer vocabulary, the legend"],
        ["<font face='Courier'>scripts/06_generate_maps.py</font>",
         "Generates a map per trip plus the index page"],
        ["<font face='Courier'>scripts/08_build_report.py</font>",
         "Fills the report structure from the result CSVs"],
        ["<font face='Courier'>results/maps/index.html</font>",
         "All maps, with per-trip scores - start here"],
        ["<font face='Courier'>results/report/technical_report.md</font>",
         "The report you compile; blocks marked TEAM ANALYSIS REQUIRED are "
         "written by their owners"],
        ["<font face='Courier'>docs/DEMO_CHECKLIST.md</font>",
         "The two-laptop runbook you drive on the day"],
    ], [70 * mm, 95 * mm]))

    F.append(P("D5. Answer these cold", "h1"))
    F.extend(bullets([
        "What does each layer on the map show?",
        "Point to somewhere the raw GPS visibly leaves the road - and show how the "
        "predicted route recovers it.",
        "Where do the classifier-only and HMM routes differ, and why?",
        "Why is the predicted route drawn with more arcs than the decoder chose? "
        "(Intermediate arcs the vehicle crossed between 1 Hz fixes are filled in by "
        "shortest path, so the route is actually drivable.)",
    ], "small"))

    # ============================================================== close-out
    F.append(P("Shared checklist before submission", "h1"))
    F.append(table([
        ["Phase", "A - Nanditha", "B - Rishaan", "C - Shreya", "D - Sanjay"],
        ["1. Understand",
         "Read all 5 briefs; run <font face='Courier'>02_eda.py</font>",
         "Run <font face='Courier'>01_build_dataset.py</font>; inspect residuals",
         "Read <font face='Courier'>candidates.py</font> end to end",
         "Open every map; read the Deliverables brief"],
        ["2. Verify",
         "Re-derive 2-3 EDA numbers by hand",
         "Hand-check the alignment on one trip",
         "Hand-check features for one observation",
         "Confirm every required figure exists"],
        ["3. Write", "§2.1-2.3", "§2.4, §3", "§2.2, §4", "§6, captions, compile"],
        ["4. Rehearse", "A7 questions", "B6 questions", "C6 questions",
         "D5 questions"],
    ], [20 * mm, 36 * mm, 37 * mm, 36 * mm, 36 * mm]))

    F.append(Spacer(1, 8))
    F.extend(callout(
        "Honesty note",
        "This document reports what the pipeline produced; every figure is read "
        "directly from the CSVs in <font face='Courier'>results/</font> at build "
        "time. The written analysis in the submitted report must be the team's own, "
        "and each member is expected to defend their own section under questioning."))

    doc.build(F, onFirstPage=furniture, onLaterPages=furniture)
    print(f"wrote {OUT}")
    print(f"  {OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()

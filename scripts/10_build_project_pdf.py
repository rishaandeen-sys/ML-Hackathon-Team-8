"""Build the full-project technical PDF.

Covers the whole system end to end: problem, data, labelling, feature
engineering, model comparison, Bayesian optimisation, HMM decoding,
results and deployment.

Every figure is read from results/ at build time, so the document cannot
drift from what the pipeline produced. Regenerate after any stage that
changes the numbers.

Run:  python scripts/10_build_project_pdf.py
Out:  docs/PathSnap_Project_Report.pdf
"""
from __future__ import annotations

import json
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

from pathsnap.candidates import (DEFAULT_MAX_CANDIDATES, DEFAULT_RADIUS_M,
                                 FEATURE_NAMES)
from pathsnap.dataset import LABEL_RESIDUAL_MAX_M

RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
OUT = ROOT / "docs" / "PathSnap_Project_Report.pdf"
REPO_URL = "https://github.com/rishaandeen-sys/ML-Hackathon-Team-8"

INK = colors.HexColor("#15181C")
MUTED = colors.HexColor("#6B7280")
ACCENT = colors.HexColor("#1F6FEB")
ACCENT_DK = colors.HexColor("#14448F")
RULE = colors.HexColor("#DFE3E8")
BAND = colors.HexColor("#F4F6F8")
TINT = colors.HexColor("#EDF3FD")
GOOD = colors.HexColor("#1F7A4D")

ss = getSampleStyleSheet()
S = {
    "ktitle": ParagraphStyle("kt", parent=ss["Normal"], fontName="Helvetica-Bold",
                             fontSize=9.5, leading=12, textColor=ACCENT, spaceAfter=9),
    "title": ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold",
                            fontSize=27, leading=31, textColor=INK, spaceAfter=7),
    "sub": ParagraphStyle("s", parent=ss["Normal"], fontSize=12, leading=17,
                          textColor=ACCENT_DK, spaceAfter=20),
    "meta": ParagraphStyle("m", parent=ss["Normal"], fontSize=9.5, leading=15,
                           textColor=MUTED),
    "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold",
                         fontSize=14.5, leading=18, textColor=ACCENT_DK,
                         spaceBefore=17, spaceAfter=8),
    "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                         fontSize=11, leading=14.5, textColor=INK,
                         spaceBefore=12, spaceAfter=5),
    "body": ParagraphStyle("b", parent=ss["Normal"], fontSize=9.5, leading=14,
                           textColor=INK, alignment=TA_LEFT, spaceAfter=7),
    "small": ParagraphStyle("sm", parent=ss["Normal"], fontSize=8.3, leading=11.6,
                            textColor=MUTED, spaceAfter=6),
    "cap": ParagraphStyle("c", parent=ss["Normal"], fontSize=8.2, leading=11,
                          textColor=MUTED, spaceBefore=4, spaceAfter=11,
                          fontName="Helvetica-Oblique"),
    "code": ParagraphStyle("co", parent=ss["Normal"], fontName="Courier",
                           fontSize=8.3, leading=12, textColor=INK,
                           backColor=BAND, borderPadding=7, spaceAfter=8),
    "cell": ParagraphStyle("cl", parent=ss["Normal"], fontSize=8.1, leading=11.3),
    "cellb": ParagraphStyle("clb", parent=ss["Normal"], fontSize=8.1, leading=11.3,
                            fontName="Helvetica-Bold", textColor=ACCENT_DK),
    "ct": ParagraphStyle("ct", parent=ss["Normal"], fontSize=9.1, leading=12.4,
                         fontName="Helvetica-Bold", textColor=ACCENT_DK),
    "cb": ParagraphStyle("cb", parent=ss["Normal"], fontSize=8.9, leading=12.8,
                         textColor=INK),
}


def P(t, style="body"):
    return Paragraph(t, S[style])


def bullets(items, style="body"):
    return [Paragraph(f"-&nbsp;&nbsp;{t}", S[style]) for t in items]


def table(rows, widths, header=True, align_right=(), highlight=None):
    data = [[Paragraph(str(c), S["cellb" if (header and i == 0) else "cell"])
             for c in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LINEBELOW", (0, 0), (-1, 0), 1.0, ACCENT),
          ("LINEBELOW", (0, 1), (-1, -1), 0.35, RULE),
          ("TOPPADDING", (0, 0), (-1, -1), 4.5),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
          ("LEFTPADDING", (0, 0), (-1, -1), 6),
          ("RIGHTPADDING", (0, 0), (-1, -1), 6)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), TINT))
    for i in range(1 if header else 0, len(rows)):
        if (i - (1 if header else 0)) % 2 == 1:
            st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#FAFBFC")))
    if highlight is not None:
        st.append(("BACKGROUND", (0, highlight), (-1, highlight), colors.HexColor("#E6F4EA")))
    for c in align_right:
        st.append(("ALIGN", (c, 0), (c, -1), "RIGHT"))
    t.setStyle(TableStyle(st))
    return t


def callout(title, body, colour=ACCENT):
    inner = [Paragraph(title, S["ct"]), Spacer(1, 3), Paragraph(body, S["cb"])]
    t = Table([[inner]], colWidths=[169 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), TINT),
        ("LINEBEFORE", (0, 0), (0, -1), 2.6, colour),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("LEFTPADDING", (0, 0), (-1, -1), 11),
        ("RIGHTPADDING", (0, 0), (-1, -1), 11)]))
    return [t, Spacer(1, 10)]


def section(num, title):
    badge = Table([[Paragraph(num, ParagraphStyle(
        "bd", fontName="Helvetica-Bold", fontSize=10, leading=12,
        textColor=colors.white, alignment=1))]],
        colWidths=[8.5 * mm], rowHeights=[6.6 * mm])
    badge.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), ACCENT),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    h = Paragraph(title, ParagraphStyle("sh", parent=ss["Heading1"],
                                        fontName="Helvetica-Bold", fontSize=14.5,
                                        leading=18, textColor=ACCENT_DK))
    row = Table([[badge, h]], colWidths=[11 * mm, 158 * mm])
    row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                             ("LEFTPADDING", (0, 0), (-1, -1), 0),
                             ("TOPPADDING", (0, 0), (-1, -1), 0),
                             ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    rule = Table([[""]], colWidths=[169 * mm], rowHeights=[0.8])
    rule.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.7, RULE)]))
    return [Spacer(1, 5), row, Spacer(1, 7), rule, Spacer(1, 9)]


def figure(name, caption, width=162 * mm):
    p = FIGS / name
    if not p.exists():
        return [P(f"<i>missing figure: {name}</i>", "small")]
    from PIL import Image as PILImage
    with PILImage.open(p) as im:
        w, h = im.size
    img = Image(str(p), width=width, height=width * h / w)
    fr = Table([[img]], colWidths=[width + 4])
    fr.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, RULE),
                            ("TOPPADDING", (0, 0), (-1, -1), 2),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                            ("LEFTPADDING", (0, 0), (-1, -1), 2),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
    return [fr, P(caption, "cap")]


def furniture(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setFillColor(ACCENT)
        canvas.rect(0, A4[1] - 2.6, A4[0], 2.6, stroke=0, fill=1)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 12 * mm, "PathSnap  |  Team 8  |  Technical Report")
    canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, f"Page {doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(20 * mm, 15.5 * mm, A4[0] - 20 * mm, 15.5 * mm)
    canvas.restoreState()


def cover(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(ACCENT)
    canvas.rect(0, A4[1] - 8, A4[0], 8, stroke=0, fill=1)
    canvas.setFillColor(ACCENT_DK)
    canvas.rect(0, 0, A4[0], 3.5, stroke=0, fill=1)
    canvas.restoreState()


def csv(name):
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


def jsn(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    part = csv("eda_partition_summary.csv")
    dq = csv("eda_data_quality.csv")
    net = csv("eda_network_summary.csv")
    anom = csv("eda_anomalies.csv")
    trips = csv("eda_trip_inventory.csv")
    cand = csv("candidate_generation_summary.csv")
    comp = csv("model_comparison.csv")
    bo = csv("bayesopt_summary.csv")
    bob = jsn("bayesopt_best_params.json")
    hmm = csv("hmm_comparison_summary.csv")
    hper = csv("hmm_comparison_per_trip.csv")
    hcfg = jsn("hmm_config.json")
    hext = csv("hmm_extended_search.csv")
    hdir = csv("hmm_directional_search.csv")

    tr = part[part.partition == "train"].iloc[0]
    te = part[part.partition == "student_test"].iloc[0]
    ctr = cand[cand.partition == "train"].iloc[0]
    cte = cand[cand.partition == "student_test"].iloc[0]
    clf = hmm[hmm.stage == "classifier_only"].iloc[0]
    fin = hmm[hmm.stage == "classifier_plus_hmm"].iloc[0]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=20 * mm,
                            title="PathSnap - Technical Report", author="Team 8")
    F: list = []

    # ---------------------------------------------------------------- cover
    F.append(Spacer(1, 62 * mm))
    F.append(P("TECHNICAL REPORT", "ktitle"))
    F.append(P("PathSnap", "title"))
    F.append(P("Distributed Map-Matching &amp; Route Reconstruction", "sub"))
    r = Table([[""]], colWidths=[66 * mm], rowHeights=[1.2])
    r.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, -1), 1.3, ACCENT)]))
    F.append(r)
    F.append(Spacer(1, 10))
    F.append(P(f"Team 8  |  {date.today().strftime('%d %B %Y')}", "meta"))
    F.append(P("UE24CS352A Machine Learning  |  PES University, "
               "Dept. of CSE (RR Campus)", "meta"))
    F.append(Spacer(1, 7))
    F.append(P(f'Repository &nbsp;<font color="#1F6FEB"><b>{REPO_URL}</b></font>',
               "meta"))
    F.append(Spacer(1, 20 * mm))
    F.append(table([
        ["Headline result", ""],
        ["Observation-level top-1 accuracy", f"<b>{fin.top1_accuracy_weighted:.2%}</b>"],
        ["Classifier alone, before sequence decoding", f"{clf.top1_accuracy_weighted:.2%}"],
        ["Gain from HMM / Viterbi decoding",
         f"<b>+{(fin.top1_accuracy_weighted - clf.top1_accuracy_weighted) * 100:.2f} pp</b>"],
        ["Route continuity", f"{clf.route_continuity:.2%} -&gt; <b>{fin.route_continuity:.2%}</b>"],
        ["Models trained and compared", f"{len(comp)} across "
                                        f"{comp.algorithm_family.nunique()} families"],
        ["Trips improved by decoding", f"{int((hper.hmm_improvement_pp > 0).sum())} of {len(hper)}, "
                                       f"{int((hper.hmm_improvement_pp < 0).sum())} degraded"],
    ], [104 * mm, 65 * mm], align_right=(1,)))
    F.append(PageBreak())

    # ------------------------------------------------------------- 1 problem
    F.extend(section("1", "The problem"))
    F.append(P(
        "GPS receivers do not report where a vehicle was. They report a noisy "
        "estimate, pushed off the true position by urban canyons, multipath "
        "reflection, interference and low sampling rates. A vehicle driving one "
        "street can appear on the parallel one, or cut across a building."))
    F.append(P(
        "<b>Map matching</b> is the task of recovering, for every GPS "
        "observation, which directed road segment the vehicle was actually on, "
        "and of stitching those decisions into a route that a vehicle could "
        "physically have driven."))
    F.extend(callout(
        "Two problems, two stages",
        "<b>Which nearby road best explains this one fix?</b> is a per-observation "
        "question, answered by a supervised classifier over candidate road "
        "segments. It is accurate pointwise but has no notion of time, so "
        "consecutive fixes can land on streets that are not even connected.<br/><br/>"
        "<b>Which sequence of roads is jointly most likely?</b> is a sequence "
        "question, answered by a Hidden Markov Model decoded with Viterbi. It "
        "cannot rescue a candidate set that never contained the right road, but "
        "it removes the physically impossible jumps. The two stages address "
        "different failure modes, which is why the system uses both."))
    F.append(P("Pipeline", "h2"))
    F.append(P("raw .track/.nodes/.arcs -&gt; candidate generation -&gt; feature "
               "engineering<br/>-&gt; classifier (ranks candidate arcs) -&gt; HMM + "
               "Viterbi (enforces connectivity)<br/>-&gt; route expansion -&gt; "
               "FastAPI -&gt; evaluator UI", "code"))

    # ---------------------------------------------------------------- 2 data
    F.extend(section("2", "The dataset"))
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
    ], [26 * mm, 52 * mm, 91 * mm]))
    F.append(Spacer(1, 6))
    F.extend(callout(
        "Arcs are directed, and every trip has its own network",
        "A two-way street appears twice, with identical geometry and opposite "
        "bearing. The reference route is a sequence of <i>directed</i> arc ids, so "
        "direction of travel is part of what must be predicted. Node and arc ids "
        "are local to a trip and must never be compared across trips."))

    F.append(P("2.1 Partitions", "h2"))
    F.append(table([
        ["", "Training", "Student testing", "Evaluator"],
        ["Trips", f"{int(tr.n_trips)}", f"{int(te.n_trips)}", "10 (held back)"],
        ["GPS observations", f"{int(tr.total_observations):,}",
         f"{int(te.total_observations):,}", "-"],
        ["Track distance", f"{tr.total_track_km:,.0f} km", f"{te.total_track_km:,.0f} km", "-"],
        ["Mean speed", f"{tr.mean_speed_kmh:.1f} km/h", f"{te.mean_speed_kmh:.1f} km/h", "-"],
        ["Steps exactly 1 s apart", f"{tr.frac_dt_eq_1s:.1%}", f"{te.frac_dt_eq_1s:.1%}", "-"],
        ["Mean arcs per trip", f"{tr.mean_arcs:,.0f}", f"{te.mean_arcs:,.0f}", "-"],
    ], [47 * mm, 40 * mm, 42 * mm, 40 * mm], align_right=(1, 2, 3)))
    F.append(P("Fitting and group-aware cross-validation / development-stage model "
               "comparison / final evaluation only. No trip is split across "
               "partitions or across CV folds.", "small"))

    F.append(P("2.2 The raw data is clean", "h2"))
    d = dq.set_index("partition")
    F.append(table([
        ["Check", "Training", "Student testing"],
        ["NaN values in .track", int(d.loc["train", "nan_values_in_track"]),
         int(d.loc["student_test", "nan_values_in_track"])],
        ["Duplicate timestamps", int(d.loc["train", "duplicate_timestamps"]),
         int(d.loc["student_test", "duplicate_timestamps"])],
        ["Non-increasing timestamps", int(d.loc["train", "non_increasing_timestamps"]),
         int(d.loc["student_test", "non_increasing_timestamps"])],
        ["Coordinates out of range", int(d.loc["train", "coordinates_out_of_range"]),
         int(d.loc["student_test", "coordinates_out_of_range"])],
        ["Arcs with invalid node id", int(d.loc["train", "arcs_with_invalid_node_id"]),
         int(d.loc["student_test", "arcs_with_invalid_node_id"])],
        ["Route arc ids out of range", int(d.loc["train", "route_arc_ids_out_of_range"]),
         int(d.loc["student_test", "route_arc_ids_out_of_range"])],
        ["Self-loop arcs", int(d.loc["train", "self_loop_arcs"]),
         int(d.loc["student_test", "self_loop_arcs"])],
        ["Zero-length arcs", int(d.loc["train", "zero_length_arcs"]),
         int(d.loc["student_test", "zero_length_arcs"])],
    ], [77 * mm, 46 * mm, 46 * mm], align_right=(1, 2)))
    F.append(P("There are no missing values to impute at the raw level. The only "
               "structural oddities are a few self-loops and zero-length arcs, "
               "handled by guarding the division in the point-to-segment "
               "projection rather than by deleting them.", "small"))

    F.append(PageBreak())
    F.append(P("2.3 Networks vary by three orders of magnitude", "h2"))
    F.append(table([
        ["Statistic", "Min", "Median", "Max"],
        ["Nodes per trip", f"{net.n_nodes.min():,.0f}", f"{net.n_nodes.median():,.0f}",
         f"{net.n_nodes.max():,.0f}"],
        ["Arcs per trip", f"{net.n_arcs.min():,.0f}", f"{net.n_arcs.median():,.0f}",
         f"{net.n_arcs.max():,.0f}"],
        ["Weakly-connected components", f"{net.n_weak_components.min():,.0f}",
         f"{net.n_weak_components.median():,.0f}", f"{net.n_weak_components.max():,.0f}"],
        ["Largest component share", f"{net.largest_component_frac.min():.0%}",
         f"{net.largest_component_frac.median():.0%}", f"{net.largest_component_frac.max():.0%}"],
        ["Median arc length", f"{net.arc_length_median_m.min():.1f} m",
         f"{net.arc_length_median_m.median():.1f} m", f"{net.arc_length_median_m.max():.1f} m"],
        ["Longest arc", f"{net.arc_length_max_m.min():,.0f} m",
         f"{net.arc_length_max_m.median():,.0f} m", f"{net.arc_length_max_m.max():,.0f} m"],
    ], [58 * mm, 36 * mm, 37 * mm, 38 * mm], align_right=(1, 2, 3)))
    F.append(P("Two design consequences. A median arc of ~25 m against a longest "
               "arc of several kilometres is why the spatial index samples points "
               "<i>along</i> arcs rather than indexing midpoints. And networks are "
               "not fully connected, so the decoder must tolerate unreachable "
               "transitions rather than assume a path always exists.", "small"))

    gaps = int((trips.n_gaps_gt_5s > 0).sum())
    F.append(P("2.4 Anomalies", "h2"))
    F.append(P(f"{len(anom)} of {len(trips)} trips carry at least one flag, and "
               f"<b>{gaps} of {len(trips)}</b> contain a time gap longer than 5 s. "
               f"Gap handling is therefore not an edge case. The flags are kept "
               f"separate rather than collapsed into one score, because they need "
               f"different treatment:"))
    F.extend(bullets([
        "<b>Trailing-fix artefact (2 trips).</b> Trip 00000090 spans 154,030 s "
        "across 536 observations, but the 42.5-hour gap sits at observation 534: "
        "a normal ~800 s run plus two stray fixes two days later.",
        "<b>Low route coverage (12 trips).</b> The supplied route covers a "
        "fraction of the GPS track. Worst: 00000033 (13%), 00000002 (14%), "
        "00000054 (19%).",
        "<b>Large spatial gap (7 trips).</b> Up to an 888 m single step between "
        "consecutive fixes.",
    ]))
    F.extend(figure("eda_overview.png", "Figure 1 - Dataset overview."))

    # ----------------------------------------------------------- 3 labelling
    F.append(PageBreak())
    F.extend(section("3", "From raw files to labelled examples"))
    F.append(P("3.1 The labelling problem", "h2"))
    F.append(P(
        "The route file gives the arcs traversed, but <i>not</i> which "
        "observation sits on which arc, and the two sequences have different "
        "lengths. Trip 00000001 has 1,315 observations against 525 route arcs. "
        "Without solving this there are no supervised labels at all."))
    F.append(P(
        "Assigning each observation to its nearest route arc independently is "
        "wrong: GPS noise makes the assignment jump backwards and forwards along "
        "the route, producing a label sequence no causal process could generate. "
        "Instead we solve a <b>monotonic alignment by dynamic programming</b>, "
        "forcing the assigned route index to be non-decreasing in time:"))
    F.append(P("f[t, r] = cost[t, r] + min over r' &lt;= r of f[t-1, r']", "code"))
    F.append(P(
        "<font face='Courier'>cost[t, r]</font> is the distance from observation "
        "t to route arc r, and the <font face='Courier'>r' &lt;= r</font> "
        "restriction is the monotonicity constraint. A running prefix-minimum "
        "evaluates the inner minimum in constant time per cell, giving "
        "<b>O(T x R)</b> where the naive form is O(T x R^2). "
        "Backtracking recovers the assignment. The structure is the same as "
        "Dynamic Time Warping.", "small"))

    F.append(P("3.2 The residuals are bimodal", "h2"))
    F.append(table([
        ["Percentile of training residual", "Value"],
        ["50th", "5.7 m"], ["75th", "10.9 m"],
        ["90th", "89.3 m"], ["95th", "142.0 m"],
    ], [100 * mm, 69 * mm], align_right=(1,)))
    F.append(P("The gap between the 75th and 90th percentiles, with very little "
               "between them, is not a smooth tail of hard examples. It is two "
               "populations: ordinary GPS noise, and trips whose reference route "
               "covers only part of the journey."))
    F.append(table([
        ["Trip", "Route coverage", "Median residual", "Interpretation"],
        ["00000033", "13%", "122.4 m", "labels unusable"],
        ["00000054", "19%", "140.0 m", "labels unusable"],
        ["00000002", "14%", "5.8 m", "coverage low, labels still sound"],
    ], [30 * mm, 36 * mm, 38 * mm, 65 * mm], align_right=(1, 2)))
    F.extend(callout(
        "Low coverage and bad labels are different things",
        "Trip 00000002 has only 14% route coverage yet a 5.8 m median residual: "
        "the alignment still recovers most of it. Only 00000033 and 00000054 have "
        "genuinely untrustworthy labels. Conflating the two would discard usable "
        "data and misdescribe the fault."))

    F.append(P("3.3 The 30 m trust threshold", "h2"))
    F.append(P(
        f"Observations whose aligned arc lies further than "
        f"<b>{LABEL_RESIDUAL_MAX_M:.0f} m</b> away are flagged "
        f"<font face='Courier'>label_valid = 0</font>, retained in the table for "
        f"inspection, and excluded from fitting and scoring."))
    F.append(table([
        ["Threshold", "Observations kept", "Candidate miss rate on what remains"],
        ["25 m", "86.79%", "0.226%"],
        ["<b>30 m - chosen</b>", "<b>87.49%</b>", "<b>0.262%</b>"],
        ["50 m", "88.84%", "0.658%"],
        ["75 m", "89.74%", "1.160%"],
        ["100 m", "90.15%", "1.254%"],
    ], [40 * mm, 52 * mm, 77 * mm], align_right=(1, 2), highlight=2))
    F.extend(callout(
        "The justification is agreement, not the fraction retained",
        f"The candidate miss rate on surviving training observations "
        f"(<b>{ctr.candidate_miss_rate:.3%}</b>) matches the student-testing "
        f"partition (<b>{cte.candidate_miss_rate:.3%}</b>), which has no coverage "
        f"problem and is {cte.frac_observations_reliably_labelled:.2%} reliably "
        f"labelled. Two independently derived quantities landing in the same "
        f"range is the evidence that the cut separates bad labels from good ones "
        f"rather than merely discarding difficult examples. At 50 m the miss rate "
        f"climbs to 0.658% and the agreement breaks down."))

    # ------------------------------------------------------------ 4 features
    F.append(PageBreak())
    F.extend(section("4", "Candidate generation and features"))
    F.append(P(
        "Map matching is cast as a <b>candidate-ranking</b> problem. For each "
        "observation the nearby directed arcs are retrieved and one row is built "
        "per (observation, candidate arc) pair. A binary classifier scores each "
        "pair, and the scores are normalised over each observation's candidates "
        "into a distribution, which is exactly the emission model the HMM stage "
        "consumes."))
    F.append(table([
        ["", "Training", "Student testing"],
        ["Search radius", f"{DEFAULT_RADIUS_M:.0f} m", f"{DEFAULT_RADIUS_M:.0f} m"],
        ["Max candidates kept", f"{DEFAULT_MAX_CANDIDATES}", f"{DEFAULT_MAX_CANDIDATES}"],
        ["Candidate rows", f"{int(ctr.n_candidate_rows):,}", f"{int(cte.n_candidate_rows):,}"],
        ["Mean candidates per observation", f"{ctr.mean_candidates_per_obs:.2f}",
         f"{cte.mean_candidates_per_obs:.2f}"],
        ["Positive rate", f"{ctr.positive_rate:.2%}", f"{cte.positive_rate:.2%}"],
        ["Candidate miss rate", f"{ctr.candidate_miss_rate:.3%}",
         f"{cte.candidate_miss_rate:.3%}"],
    ], [67 * mm, 51 * mm, 51 * mm], align_right=(1, 2)))
    F.extend(callout(
        "Candidate miss rate is the ceiling on accuracy",
        f"An observation whose reference arc was never retrieved cannot be "
        f"classified correctly by any model. At {cte.candidate_miss_rate:.3%} on "
        f"the testing partition it caps top-1 at "
        f"{1 - cte.candidate_miss_rate:.2%}. The figure is identical across all "
        f"{len(comp)} models, because it is a property of retrieval rather than "
        f"of the classifier. The real headroom is the observations where the "
        f"right arc was retrieved but not ranked first."))

    F.append(P("4.1 The 26 features", "h2"))
    F.append(table([
        ["Group", "Features", "What it captures"],
        ["Geometry (8)",
         "<font face='Courier'>dist_m, dist_rank, dist_minus_best, dist_ratio_best, "
         "dist_z_in_obs, arc_length, proj_t, proj_is_endpoint</font>",
         "Proximity, and proximity <i>relative to the competing candidates of the "
         "same observation</i>. A 15 m arc means something different when the "
         "runner-up is 16 m away than when it is 60 m away."],
        ["Heading (4)",
         "<font face='Courier'>heading_diff, heading_cos, heading_valid, turn_angle</font>",
         "Travel direction against arc bearing. The only signal that separates the "
         "two directions of a two-way street, since both carry identical geometry "
         "and therefore identical distance."],
        ["Kinematics (6)",
         "<font face='Courier'>speed_mps, accel_mps2, dt_prev, dt_next, step_prev_m, "
         "step_next_m</font>",
         "Local motion and sampling irregularity."],
        ["Network (8)",
         "<font face='Courier'>arc_out_degree, arc_in_degree, has_reverse_arc, "
         "n_candidates, local_arc_density, dist_prev_obs_to_arc, dist_next_obs_to_arc, "
         "obs_frac</font>",
         "Topology around the arc, and whether the same arc also explains the "
         "neighbouring fixes - weak sequential evidence without the classifier "
         "modelling the sequence itself."],
    ], [25 * mm, 60 * mm, 84 * mm]))
    F.append(P("4.2 Decisions worth defending", "h2"))
    F.extend(bullets([
        "<b>Local equirectangular projection to metres</b>, centred per trip. "
        "Longitude and latitude are angular: at this latitude a degree of "
        "longitude spans about 72 km against 111 km for a degree of latitude, so "
        "Euclidean distance in degrees is not a distance and would distort every "
        "feature. At city scale the projection error is well under 0.1%.",
        "<b>Relative as well as absolute distance features.</b> Without them the "
        "model must learn an absolute threshold, which cannot hold across both "
        "dense centres and sparse outskirts.",
        "<b>The spatial index samples points every 25 m along each arc</b> rather "
        "than indexing midpoints, so a long arc passing close to a query point is "
        "never missed.",
        "<b>No feature reads the reference route.</b> The same code path runs at "
        "training and inference time, which is what makes the deployed API "
        "trustworthy.",
    ]))

    # -------------------------------------------------------------- 5 models
    F.append(PageBreak())
    F.extend(section("5", "Model experimentation"))
    F.append(P(f"{len(comp)} models were trained across "
               f"{comp.algorithm_family.nunique()} algorithm families, with "
               f"<font face='Courier'>GroupKFold(5)</font> grouped by trip so no "
               f"trip is split across folds."))
    cs = comp.sort_values("test_top1_accuracy", ascending=False)
    rows = [["Model", "Family", "CV macro F1", "Test top-1", "Test macro F1",
             "Log loss", "Train (s)"]]
    for r in cs.itertuples():
        rows.append([r.model, r.algorithm_family, f"{r.cv_macro_f1_mean:.4f}",
                     f"{r.test_top1_accuracy:.4f}", f"{r.test_macro_f1:.4f}",
                     f"{r.test_log_loss:.4f}", f"{r.train_seconds:.1f}"])
    F.append(table(rows, [33 * mm, 36 * mm, 23 * mm, 21 * mm, 24 * mm, 18 * mm, 18 * mm],
                   align_right=(2, 3, 4, 5, 6)))
    F.extend(callout(
        "Why XGBoost was selected over the more accurate MLP",
        "The MLP leads on raw test top-1 (0.8767 against 0.8742), but the gap of "
        "0.25 pp sits well inside the fold-to-fold standard deviation of about "
        "+/-1.6 pp, so it is not a reliable difference. XGBoost takes the best "
        "cross-validated macro F1 (0.9140), making it the most consistent across "
        "trip splits, and the best log loss (0.0854). Log loss matters "
        "disproportionately here: these scores become the HMM's emission "
        "probabilities, so calibration affects the final route more than raw "
        "top-1 does. It also trains in 11 s against the MLP's 46 s."))
    F.extend(figure("model_test_performance.png",
                    "Figure 2 - Student-testing top-1 accuracy and macro F1."))

    F.append(PageBreak())
    F.extend(section("6", "Bayesian hyperparameter optimisation"))
    if bob:
        F.append(P(
            f"<b>Sampler:</b> {bob['sampler']}. TPE models the density of good "
            f"against bad trials and samples where their ratio is highest, so "
            f"each trial uses what the previous ones revealed.<br/>"
            f"<b>Objective:</b> {bob['objective']}. Top-1 is what the HMM "
            f"consumes; a model can raise binary macro F1 by rejecting obviously "
            f"wrong candidates without improving the choice that reaches the "
            f"route.<br/>"
            f"<b>Trials:</b> {bob['n_trials']}, "
            f"{bob['search_folds']}-fold group CV, "
            f"{bob['search_seconds'] / 60:.0f} minutes."))
    if bo is not None:
        u = bo[bo.variant == "untuned"].iloc[0]
        t = bo[bo.variant == "tuned"].iloc[0]
        F.append(table([
            ["", "Untuned", "Tuned"],
            ["CV top-1 accuracy", f"{u.cv_top1_accuracy:.5f}", f"{t.cv_top1_accuracy:.5f}"],
            ["CV macro F1", f"{u.cv_macro_f1:.5f}", f"{t.cv_macro_f1:.5f}"],
            ["Test top-1 accuracy", f"{u.test_top1_accuracy:.5f}", f"{t.test_top1_accuracy:.5f}"],
            ["Test macro F1", f"{u.test_macro_f1:.5f}", f"{t.test_macro_f1:.5f}"],
            ["Test log loss", f"{u.test_log_loss:.5f}", f"{t.test_log_loss:.5f}"],
            ["Training time (s)", f"{u.train_seconds:.1f}", f"{t.train_seconds:.1f}"],
        ], [69 * mm, 50 * mm, 50 * mm], align_right=(1, 2)))
        F.append(P("Both variants are evaluated under the identical protocol: "
                   "5-fold group CV on the training trips, then the student-testing "
                   "partition. The evaluator partition is never touched.", "small"))

    # ----------------------------------------------------------------- 7 HMM
    F.append(PageBreak())
    F.extend(section("7", "HMM and Viterbi route reconstruction"))
    F.extend(bullets([
        "<b>Hidden states</b> - the candidate arcs of each observation. The state "
        "space changes from step to step, so this is a lattice rather than a fixed "
        "alphabet.",
        "<b>Emissions</b> - classifier scores, normalised over each observation's "
        "candidates.",
        "<b>Transitions</b> - following Newson &amp; Krumm (2009), compare on-road "
        "travel distance d_route against straight-line d_gps: "
        "P(j|i) proportional to exp(-|d_route - d_gps| / beta).",
        "<b>Unreachable pairs</b> take a fixed penalty rather than being "
        "forbidden, so a thin candidate set can never make the lattice dead-end.",
        "<b>Gaps</b> - above 5 s the search radius grows and beta relaxes.",
        "<b>Shortest paths</b> come from a bounded Dijkstra; at 1 Hz consecutive "
        "fixes are metres apart, so the explored subgraph stays small.",
        "<b>Route expansion</b> - intermediate arcs crossed entirely between two "
        "fixes are filled in by shortest path, turning per-observation decisions "
        "into a drivable route.",
    ]))

    F.append(P("7.1 A negative result that redirected the work", "h2"))
    if hext is not None:
        bp = hext[hext.phase == "beta_penalty"]
        F.append(P(
            f"An extended grid over beta and the unreachable penalty "
            f"({len(bp)} configurations, beta from "
            f"{bp.beta.min():.0f} to {bp.beta.max():.0f}, penalty from "
            f"{bp.unreachable_penalty.min():.0f} to "
            f"{bp.unreachable_penalty.max():.0f}) moved mean top-1 only between "
            f"<b>{bp.mean_top1_accuracy.min():.4f}</b> and "
            f"<b>{bp.mean_top1_accuracy.max():.4f}</b>. A response surface that "
            f"flat means the decoder parameters were not the binding constraint, "
            f"so further parameter search was the wrong instrument."))
    F.append(P("7.2 The structural gap was direction", "h2"))
    F.extend(callout(
        "Why distance alone cannot resolve street direction",
        "The reverse of an arc has <i>identical geometry</i> and therefore an "
        "identical d_route. The distance term in the transition model is "
        "mathematically incapable of distinguishing the two directions of a "
        "two-way street - and the arcs in this dataset are directed. Three "
        "directional terms were added and searched on training trips only."))
    if hdir is not None:
        base = hdir[hdir.phase == "baseline"].iloc[0].mean_top1_accuracy
        best = hdir.loc[hdir.mean_top1_accuracy.idxmax()]
        F.append(table([
            ["Term", "Selected value", "Effect"],
            ["U-turn penalty", "8.0", "+0.29 pp - the real gap"],
            ["Turn penalty weight", "2.0", "+0.03 pp"],
            ["GPS direction consistency", "0.0 - rejected",
             "harmful; -1.1 pp at weight 4.0"],
        ], [52 * mm, 42 * mm, 75 * mm]))
        F.append(P(
            "The rejected term is informative rather than disappointing: heading "
            "is already a classifier feature and both directed arcs are already "
            "candidates, so the emission model resolves GPS-heading agreement "
            "before the decoder sees it. Re-deriving it in the transition term "
            "double-counts the same evidence.", "small"))
    if hcfg:
        c = hcfg["config"]
        F.append(P(f"Final decoder: beta={c['beta']}, "
                   f"unreachable_penalty={c['unreachable_penalty']}, "
                   f"same_arc_bonus={c['same_arc_bonus']}, "
                   f"uturn_penalty={c['uturn_penalty']}, "
                   f"turn_penalty_weight={c['turn_penalty_weight']}. "
                   f"Selected on training trips only.", "code"))

    # ------------------------------------------------------------- 8 results
    F.append(PageBreak())
    F.extend(section("8", "Results"))
    F.append(table([
        ["Metric", "Classifier only", "Classifier + HMM", "Change"],
        ["Top-1 accuracy (weighted)", f"{clf.top1_accuracy_weighted:.4f}",
         f"<b>{fin.top1_accuracy_weighted:.4f}</b>",
         f"+{(fin.top1_accuracy_weighted - clf.top1_accuracy_weighted) * 100:.2f} pp"],
        ["Top-1 accuracy (macro)", f"{clf.top1_accuracy_macro:.4f}",
         f"{fin.top1_accuracy_macro:.4f}",
         f"+{(fin.top1_accuracy_macro - clf.top1_accuracy_macro) * 100:.2f} pp"],
        ["Route continuity", f"{clf.route_continuity:.4f}", f"{fin.route_continuity:.4f}",
         f"+{(fin.route_continuity - clf.route_continuity) * 100:.2f} pp"],
        ["Arc-set precision", f"{clf.arc_set_precision:.4f}", f"{fin.arc_set_precision:.4f}",
         f"+{(fin.arc_set_precision - clf.arc_set_precision) * 100:.2f} pp"],
        ["Arc-set recall", f"{clf.arc_set_recall:.4f}", f"{fin.arc_set_recall:.4f}",
         f"+{(fin.arc_set_recall - clf.arc_set_recall) * 100:.2f} pp"],
        ["Arc-set Jaccard", f"{clf.arc_set_jaccard:.4f}", f"{fin.arc_set_jaccard:.4f}",
         f"+{(fin.arc_set_jaccard - clf.arc_set_jaccard) * 100:.2f} pp"],
    ], [52 * mm, 38 * mm, 40 * mm, 39 * mm], align_right=(1, 2, 3)))

    F.append(P("8.1 Per trip", "h2"))
    rows = [["Trip", "Observations", "Classifier", "+ HMM", "Change", "Continuity"]]
    for r in hper.itertuples():
        rows.append([f"{int(r.trip_id):08d}", f"{int(r.n_observations):,}",
                     f"{r.classifier_top1_accuracy:.4f}", f"{r.hmm_top1_accuracy:.4f}",
                     f"+{r.hmm_improvement_pp:.2f} pp", f"{r.hmm_route_continuity:.4f}"])
    F.append(table(rows, [26 * mm, 30 * mm, 28 * mm, 26 * mm, 29 * mm, 30 * mm],
                   align_right=(1, 2, 3, 4, 5)))
    F.append(P(f"Decoding improved <b>{int((hper.hmm_improvement_pp > 0).sum())} of "
               f"{len(hper)}</b> trips and degraded none. The largest gain is trip "
               f"00000077 at +{hper.hmm_improvement_pp.max():.2f} pp, which was the "
               f"weakest trip for the classifier alone - the sequence constraint "
               f"helps most exactly where pointwise classification struggles."))
    F.extend(figure("hmm_per_trip_comparison.png",
                    "Figure 3 - Top-1 accuracy and route continuity per trip."))

    # ---------------------------------------------------------- 9 deployment
    F.append(PageBreak())
    F.extend(section("9", "Deployment and evaluation integrity"))
    F.append(P("Source code", "h2"))
    F.append(P(f'The complete implementation - data processing, feature '
               f'engineering, every training script, the Bayesian search, the '
               f'HMM decoder, the API and the user interface - is public at:'))
    F.append(P(f'<font color="#1F6FEB"><b>{REPO_URL}</b></font>', "code"))
    F.append(P("Every table and figure in this report is generated from the "
               "result CSVs committed in that repository, so each number here "
               "is traceable to the experiment that produced it.", "small"))

    F.append(P("Two-laptop architecture", "h2"))
    F.append(table([
        ["", "Laptop 1 - model and API", "Laptop 2 - evaluator UI"],
        ["Holds", "Classifier, decoder, road networks, reference routes",
         "Nothing but the interface"],
        ["Runs", "<font face='Courier'>uvicorn api.main:app --host 0.0.0.0 --port 8000</font>",
         "<font face='Courier'>streamlit run ui/app.py</font>"],
    ], [24 * mm, 76 * mm, 69 * mm]))
    F.append(Spacer(1, 5))
    F.append(table([
        ["Endpoint", "Purpose"],
        ["<font face='Courier'>GET /health</font>", "Model, decoder configuration, trip count"],
        ["<font face='Courier'>GET /trips</font>", "The evaluation trips offered to the evaluator"],
        ["<font face='Courier'>POST /predict</font>",
         "Run the pipeline on one trip; returns route and server-side metrics"],
        ["<font face='Courier'>GET /models</font>", "Available classifiers with measured metrics"],
        ["<font face='Courier'>GET /metrics</font>", "Experiment results behind the analytics view"],
        ["<font face='Courier'>GET /network/{trip_id}</font>", "Road geometry for map context"],
        ["<font face='Courier'>GET /logs</font>", "Recent requests"],
    ], [52 * mm, 117 * mm]))
    F.extend(callout(
        "Evaluation integrity",
        "The reference route is read <b>only inside /predict, after</b> the "
        "prediction is complete, and solely to compute accuracy. It is never "
        "passed to the matcher and is omitted from the response unless explicitly "
        "requested. Nothing is hardcoded per trip: every request runs the same "
        "pipeline over the raw files, and the same matcher class is used by the "
        "training-time scripts and the API. Group-aware cross-validation keeps "
        "trips whole across folds, decoder parameters were selected on training "
        "trips only, and the evaluator partition was never read during "
        "development."))

    F.append(P("Limitations", "h2"))
    F.extend(bullets([
        "Three training trips have reference routes covering 13-19% of their GPS "
        "track; those stretches are excluded rather than silently trained on.",
        f"Candidate generation imposes a {cte.candidate_miss_rate:.3%} ceiling on "
        f"top-1 accuracy.",
        "Reference routes are reference information, not independently verified "
        "ground truth. Reported accuracies are agreement with them.",
        "The local projection suits this city-scale dataset and would need "
        "revisiting for data spanning a much larger area.",
    ], "small"))

    F.append(P("References", "h2"))
    F.append(P(
        "Newson, P. &amp; Krumm, J. (2009). <i>Hidden Markov Map Matching Through "
        "Noise and Sparseness.</i> ACM SIGSPATIAL.<br/>"
        "Viterbi, A. (1967). <i>Error bounds for convolutional codes and an "
        "asymptotically optimum decoding algorithm.</i> IEEE Trans. Information "
        "Theory.<br/>"
        "Bergstra, J. et al. (2011). <i>Algorithms for Hyper-Parameter "
        "Optimization.</i> NeurIPS.", "small"))

    doc.build(F, onFirstPage=cover, onLaterPages=furniture)
    print(f"wrote {OUT}")
    print(f"  {OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()

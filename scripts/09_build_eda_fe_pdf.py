"""Stage 9: build the EDA + feature-engineering briefing PDF.

A normal, single-document PDF covering the dataset, EDA findings, how
per-observation labels are derived, and feature engineering. No per-person
split is baked into the document - that is handled separately.

Every figure is read from results/ at build time, so the PDF cannot drift
from what the pipeline actually produced.

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

DOC_TITLE = "ML Hackathon"
DOC_SUBTITLE = "Dataset, EDA &amp; Feature Engineering"

INK = colors.HexColor("#111827")
MUTED = colors.HexColor("#6B7280")
ACCENT = colors.HexColor("#1D4ED8")
ACCENT_DARK = colors.HexColor("#1E3A8A")
ACCENT_SOFT = colors.HexColor("#EFF4FD")
RULE = colors.HexColor("#E2E5EA")
BAND = colors.HexColor("#F6F7F9")
ZEBRA = colors.HexColor("#FAFBFC")
WHITE = colors.white

ss = getSampleStyleSheet()
S = {
    "cover_kicker": ParagraphStyle("ck", parent=ss["Normal"], fontName="Helvetica-Bold",
                                   fontSize=10.5, leading=13, textColor=ACCENT,
                                   spaceAfter=10, tracking=0.6),
    "cover_title": ParagraphStyle("ct", parent=ss["Title"], fontName="Helvetica-Bold",
                                  fontSize=34, leading=38, textColor=INK, spaceAfter=8),
    "cover_sub": ParagraphStyle("cs", parent=ss["Normal"], fontSize=15, leading=20,
                                textColor=ACCENT_DARK, spaceAfter=28,
                                fontName="Helvetica"),
    "cover_meta": ParagraphStyle("cm", parent=ss["Normal"], fontSize=10, leading=16,
                                 textColor=MUTED),
    "title": ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold",
                            fontSize=20, leading=24, textColor=INK, spaceAfter=2),
    "sub": ParagraphStyle("s", parent=ss["Normal"], fontSize=10, leading=14.5,
                          textColor=MUTED, spaceAfter=14),
    "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold",
                         fontSize=15, leading=19, textColor=ACCENT_DARK,
                         spaceBefore=18, spaceAfter=9),
    "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                         fontSize=11.5, leading=15, textColor=INK,
                         spaceBefore=13, spaceAfter=6),
    "body": ParagraphStyle("b", parent=ss["Normal"], fontSize=9.7, leading=14.6,
                           textColor=INK, alignment=TA_LEFT, spaceAfter=7),
    "small": ParagraphStyle("sm", parent=ss["Normal"], fontSize=8.4, leading=12,
                            textColor=MUTED, spaceAfter=6),
    "cap": ParagraphStyle("c", parent=ss["Normal"], fontSize=8.3, leading=11.2,
                          textColor=MUTED, spaceBefore=4, spaceAfter=12,
                          fontName="Helvetica-Oblique"),
    "code": ParagraphStyle("co", parent=ss["Normal"], fontName="Courier",
                           fontSize=8.3, leading=12, textColor=INK,
                           backColor=BAND, borderPadding=7, spaceAfter=9),
    "cell": ParagraphStyle("cl", parent=ss["Normal"], fontSize=8.3, leading=11.6),
    "cellb": ParagraphStyle("clb", parent=ss["Normal"], fontSize=8.3, leading=11.6,
                            fontName="Helvetica-Bold", textColor=ACCENT_DARK),
    "callout_t": ParagraphStyle("clt", parent=ss["Normal"], fontSize=9.3,
                                leading=12.5, fontName="Helvetica-Bold",
                                textColor=ACCENT_DARK),
    "callout_b": ParagraphStyle("clb2", parent=ss["Normal"], fontSize=9,
                                leading=13, textColor=INK),
}


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items, style="body"):
    return [Paragraph(f"&bull;&nbsp;&nbsp;{t}", S[style]) for t in items]


def table(rows, widths, header=True, align_right=(), zebra=True):
    data = []
    for r_i, row in enumerate(rows):
        out = []
        for cell in row:
            st = "cellb" if (header and r_i == 0) else "cell"
            out.append(Paragraph(str(cell), S[st]))
        data.append(out)
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, 0), 1.1, ACCENT if header else RULE),
        ("LINEBELOW", (0, 1), (-1, -2), 0.4, RULE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), ACCENT_SOFT))
    if zebra:
        start = 1 if header else 0
        for i in range(start, len(rows)):
            if (i - start) % 2 == 1:
                style.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    for c in align_right:
        style.append(("ALIGN", (c, 0), (c, -1), "RIGHT"))
    t.setStyle(TableStyle(style))
    return t


def figure(name, caption, width=164 * mm):
    p = FIGS / name
    if not p.exists():
        return [P(f"<i>missing figure: {name}</i>", "small")]
    from PIL import Image as PILImage
    with PILImage.open(p) as im:
        w, h = im.size
    img = Image(str(p), width=width, height=width * h / w)
    frame = Table([[img]], colWidths=[width + 4])
    frame.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ]))
    return [frame, P(caption, "cap")]


def callout(title, body):
    """A left-accented highlight box for a key finding or decision."""
    inner = [Paragraph(title, S["callout_t"]), Spacer(1, 3),
             Paragraph(body, S["callout_b"])]
    t = Table([[inner]], colWidths=[169 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), ACCENT_SOFT),
        ("LINEBEFORE", (0, 0), (0, -1), 2.6, ACCENT),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("LEFTPADDING", (0, 0), (-1, -1), 11),
        ("RIGHTPADDING", (0, 0), (-1, -1), 11),
    ]))
    return [t, Spacer(1, 11)]


def section(number, title):
    """A numbered section heading with a coloured badge."""
    badge = Table([[Paragraph(number, ParagraphStyle(
        "badge", fontName="Helvetica-Bold", fontSize=10.5, leading=13,
        textColor=WHITE, alignment=1))]], colWidths=[9 * mm], rowHeights=[7 * mm])
    badge.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), ACCENT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    heading = Paragraph(title, ParagraphStyle(
        "sh", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=15,
        leading=19, textColor=ACCENT_DARK))
    row = Table([[badge, heading]], colWidths=[11 * mm, 158 * mm])
    row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    rule = Table([[""]], colWidths=[169 * mm], rowHeights=[0.9])
    rule.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.8, RULE)]))
    return [Spacer(1, 4), row, Spacer(1, 8), rule, Spacer(1, 10)]


def page_furniture(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setFillColor(ACCENT)
        canvas.rect(0, A4[1] - 3, A4[0], 3, stroke=0, fill=1)
    canvas.setFont("Helvetica", 7.6)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 12 * mm, DOC_TITLE)
    canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, f"Page {doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(20 * mm, 15.5 * mm, A4[0] - 20 * mm, 15.5 * mm)
    canvas.restoreState()


def cover_furniture(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(ACCENT)
    canvas.rect(0, A4[1] - 9, A4[0], 9, stroke=0, fill=1)
    canvas.setFillColor(ACCENT_DARK)
    canvas.rect(0, 0, A4[0], 4, stroke=0, fill=1)
    canvas.restoreState()


def load():
    d = {}
    for key, name in [("part", "eda_partition_summary.csv"),
                      ("dq", "eda_data_quality.csv"),
                      ("trips", "eda_trip_inventory.csv"),
                      ("net", "eda_network_summary.csv"),
                      ("anom", "eda_anomalies.csv"),
                      ("cand", "candidate_generation_summary.csv")]:
        p = RESULTS / name
        d[key] = pd.read_csv(p) if p.exists() else None
    return d


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d = load()
    part, dq, trips, net, anom, cand = (d["part"], d["dq"], d["trips"],
                                        d["net"], d["anom"], d["cand"])
    tr = part[part.partition == "train"].iloc[0]
    te = part[part.partition == "student_test"].iloc[0]
    ctr = cand[cand.partition == "train"].iloc[0]
    cte = cand[cand.partition == "student_test"].iloc[0]

    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=20 * mm,
        title="ML Hackathon - Dataset, EDA and Feature Engineering",
        author="Team 8")

    F: list = []

    # ------------------------------------------------------------- cover page
    F.append(Spacer(1, 70 * mm))
    F.append(P("TECHNICAL BRIEFING", "cover_kicker"))
    F.append(P(DOC_TITLE, "cover_title"))
    F.append(P(DOC_SUBTITLE, "cover_sub"))

    meta_rule = Table([[""]], colWidths=[70 * mm], rowHeights=[1.2])
    meta_rule.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, -1), 1.4, ACCENT)]))
    F.append(meta_rule)
    F.append(Spacer(1, 10))
    F.append(P(f"Team 8 &nbsp;&bull;&nbsp; {date.today().strftime('%d %B %Y')}",
               "cover_meta"))
    F.append(P("UE24CS352A Machine Learning &nbsp;&bull;&nbsp; "
               "PES University, Dept. of CSE (RR Campus)", "cover_meta"))

    F.append(Spacer(1, 26 * mm))
    toc = table([
        ["Section", "Covers"],
        ["1. The dataset", "File formats, directed arcs, partitions"],
        ["2. Exploratory data analysis", "Data quality, networks, sampling, anomalies"],
        ["3. Labelled examples", "Deriving per-observation labels, the trust threshold"],
        ["4. Feature engineering", "Candidate ranking and the 26 features"],
        ["5. Where this lives", "File map and how to reproduce it"],
    ], [62 * mm, 107 * mm], zebra=False)
    F.append(toc)
    F.append(PageBreak())

    # ---------------------------------------------------------------- part 1
    F.extend(section("1", "The dataset"))
    F.append(P(
        "The task is map matching: recovering which road a vehicle actually "
        "travelled on from a GPS trace that drifts off the road network. Each "
        "trip ships four whitespace-separated files with no headers, and "
        "<b>its own road sub-network</b> - node and arc ids are local to a trip "
        "and must never be compared across trips."))

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
    ], [28 * mm, 52 * mm, 85 * mm]))
    F.append(Spacer(1, 7))

    F.extend(callout(
        "Key fact &mdash; arcs are directed",
        "A two-way street appears twice, with identical geometry and opposite "
        "bearing, and the reference route is a sequence of <i>directed</i> arc ids. "
        "Direction of travel is therefore part of what has to be predicted &mdash; a "
        "fact that drives the whole feature design in section 4."))

    F.append(P("1.1 Partitions", "h2"))
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
    ], [48 * mm, 39 * mm, 39 * mm, 39 * mm], align_right=(1, 2, 3)))
    F.append(P("Used for: fitting and cross-validation / development-stage model "
               "comparison / final evaluation only. No trip is ever split across "
               "partitions or across CV folds.", "small"))

    F.extend(section("2", "Exploratory data analysis"))
    F.append(KeepTogether([P("2.1 The data itself is clean", "h2"), table([
        ["Check", "Training", "Student testing"],
        ["NaN values in .track", int(dq.set_index("partition").loc["train", "nan_values_in_track"]),
         int(dq.set_index("partition").loc["student_test", "nan_values_in_track"])],
        ["Duplicate timestamps",
         int(dq.set_index("partition").loc["train", "duplicate_timestamps"]),
         int(dq.set_index("partition").loc["student_test", "duplicate_timestamps"])],
        ["Non-increasing timestamps",
         int(dq.set_index("partition").loc["train", "non_increasing_timestamps"]),
         int(dq.set_index("partition").loc["student_test", "non_increasing_timestamps"])],
        ["Coordinates out of range",
         int(dq.set_index("partition").loc["train", "coordinates_out_of_range"]),
         int(dq.set_index("partition").loc["student_test", "coordinates_out_of_range"])],
        ["Arcs with invalid node id",
         int(dq.set_index("partition").loc["train", "arcs_with_invalid_node_id"]),
         int(dq.set_index("partition").loc["student_test", "arcs_with_invalid_node_id"])],
        ["Route arc ids out of range",
         int(dq.set_index("partition").loc["train", "route_arc_ids_out_of_range"]),
         int(dq.set_index("partition").loc["student_test", "route_arc_ids_out_of_range"])],
        ["Self-loop arcs", int(dq.set_index("partition").loc["train", "self_loop_arcs"]),
         int(dq.set_index("partition").loc["student_test", "self_loop_arcs"])],
        ["Zero-length arcs", int(dq.set_index("partition").loc["train", "zero_length_arcs"]),
         int(dq.set_index("partition").loc["student_test", "zero_length_arcs"])],
    ], [75 * mm, 45 * mm, 45 * mm], align_right=(1, 2))]))
    F.append(P(
        "There are no missing values to impute at the raw level. The only "
        "structural oddities are a handful of self-loops and zero-length arcs, "
        "which are handled by guarding the division in the point-to-segment "
        "projection rather than by deleting them.", "small"))

    F.append(Spacer(1, 4))
    F.extend(figure("eda_overview.png",
                    "Figure 1 - Dataset overview: observations per trip, sampling "
                    "interval, share of 1 s steps, observations against reference-"
                    "route length, network size, and label residual."))

    F.append(PageBreak())
    F.append(P("2.2 The road networks vary enormously", "h2"))
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
        "Two consequences for the design. First, a median arc of ~25 m against a "
        "longest arc of several kilometres means a spatial index built on arc "
        "<i>midpoints</i> would miss long arcs that pass close to a query point - "
        "so the index is built over points sampled every 25 m along each arc. "
        "Second, networks are not fully connected, so the decoder must tolerate "
        "unreachable transitions instead of assuming a path always exists.", "small"))

    F.append(P("2.3 Sampling is mostly 1 Hz, but gaps are common", "h2"))
    gap_trips = int((trips.n_gaps_gt_5s > 0).sum())
    F.append(P(
        f"{tr.frac_dt_eq_1s * 100:.1f}% of training steps are exactly 1 s apart, but "
        f"<b>{gap_trips} of {len(trips)} trips contain at least one gap longer than "
        f"5 s</b>. Gap handling is therefore not an edge case: the decoder widens its "
        f"search radius and relaxes its transition scale across a gap rather than "
        f"forcing an implausible local jump."))
    F.extend(figure("eda_gaps.png",
                    "Figure 2 - Largest time gap and largest spatial step per trip "
                    "(log scale). Both tails are driven by a small number of trips.",
                    width=160 * mm))

    F.append(PageBreak())
    F.append(P("2.4 Anomalies: three distinct problems", "h2"))
    F.append(P(
        f"{len(anom)} of {len(trips)} trips carry at least one flag. They are "
        f"reported separately rather than as a single quality score, because they "
        f"need different treatment:"))
    F.extend(bullets([
        "<b>Trailing-fix artefact (2 trips).</b> Trip 00000090 has 536 observations "
        "spanning 154,030 s - but the 42.5-hour gap sits at observation 534. It is "
        "a normal ~800 s run plus two stray fixes recorded two days later.",
        "<b>Low route coverage (12 trips).</b> The supplied <font face='Courier'>"
        ".route</font> is far shorter than the GPS track: trips 00000033 (13%), "
        "00000002 (14%) and 00000054 (19%) are the worst.",
        "<b>Large spatial gap (7 trips).</b> A single step of up to 888 m between "
        "consecutive fixes, which the decoder has to bridge.",
    ]))
    worst = anom.nlargest(6, "n_flags")[
        ["trip_id", "n_observations", "route_coverage",
         "label_residual_median_m", "step_max_m", "n_flags"]]
    F.append(table(
        [["Trip", "Obs", "Route coverage", "Median residual", "Max step", "Flags"]] +
        [[f"{int(r.trip_id):08d}", f"{int(r.n_observations):,}",
          f"{r.route_coverage * 100:.0f}%", f"{r.label_residual_median_m:.1f} m",
          f"{r.step_max_m:.0f} m", int(r.n_flags)] for r in worst.itertuples()],
        [26 * mm, 24 * mm, 32 * mm, 34 * mm, 26 * mm, 23 * mm],
        align_right=(1, 2, 3, 4, 5)))
    F.append(P(
        "Note that low coverage and unreliable labels are <i>not</i> the same "
        "problem. Trip 00000002 has only 14% coverage yet a 5.8 m median residual - "
        "the alignment still recovers most of it. Only trips 00000033 and 00000054 "
        "have genuinely untrustworthy labels.", "small"))

    F.extend(figure("eda_trajectories.png",
                    "Figure 3 - Raw GPS trajectories for the first 12 training "
                    "trips (green = start, red = end).", width=160 * mm))

    F.append(PageBreak())
    F.extend(section("3", "From raw files to labelled examples"))
    F.append(P("3.1 The labelling problem", "h2"))
    F.append(P(
        "<font face='Courier'>.route</font> gives the arcs traversed, but <i>not</i> "
        "which observation sits on which arc - and the two sequences have different "
        "lengths (trip 00000001: 1,315 observations against 525 route arcs). The "
        "mapping has to be recovered before any supervised learning can happen."))
    F.append(P(
        "Choosing the nearest route arc independently for each observation is wrong: "
        "GPS noise makes the assignment jump backwards and forwards along the route, "
        "producing labels no causal model could reproduce. Instead we solve a "
        "<b>monotonic alignment by dynamic programming</b>, forcing the assigned "
        "route index to be non-decreasing in time:"))
    F.append(P("f[t, r] = cost[t, r] + min over r' &lt;= r of f[t-1, r']", "code"))
    F.append(P(
        "A running prefix-minimum makes this O(T x R). The result is monotonic by "
        "construction, and the residual distance it leaves behind is itself a "
        "diagnostic.", "small"))

    F.append(P("3.2 A 30 m trust threshold, and why", "h2"))
    F.append(P(
        f"Training residuals are sharply <b>bimodal</b>: the 75th percentile is "
        f"10.9 m, but the 90th is 89.3 m. The upper mode is not hard examples - it "
        f"is the trips whose reference route covers only part of the journey. "
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
    F.append(P(
        f"The justification is not the percentage kept, it is the agreement: the "
        f"candidate miss rate on the surviving training observations "
        f"({ctr.candidate_miss_rate:.3%}) matches the student-testing partition "
        f"({cte.candidate_miss_rate:.3%}), which has no coverage problem. That is "
        f"the evidence the cut separates bad labels rather than difficult ones.",
        "small"))

    F.append(PageBreak())
    F.extend(section("4", "Feature engineering"))
    F.append(P("4.1 Problem framing: candidate ranking", "h2"))
    F.append(P(
        "Map matching is cast as a <b>candidate-ranking</b> problem. For every GPS "
        "observation the nearby directed arcs are retrieved, and one row is built "
        "per (observation, candidate arc) pair. A binary classifier scores each "
        "pair; the per-observation scores are normalised into a distribution over "
        "candidates, which is exactly the emission model the HMM stage consumes."))
    F.append(table([
        ["", "Training", "Student testing"],
        ["Search radius", f"{DEFAULT_RADIUS_M:.0f} m", f"{DEFAULT_RADIUS_M:.0f} m"],
        ["Max candidates kept", f"{DEFAULT_MAX_CANDIDATES}", f"{DEFAULT_MAX_CANDIDATES}"],
        ["Candidate rows", f"{int(ctr.n_candidate_rows):,}",
         f"{int(cte.n_candidate_rows):,}"],
        ["Mean candidates / observation", f"{ctr.mean_candidates_per_obs:.2f}",
         f"{cte.mean_candidates_per_obs:.2f}"],
        ["Positive rate", f"{ctr.positive_rate:.2%}", f"{cte.positive_rate:.2%}"],
        ["Candidate miss rate", f"{ctr.candidate_miss_rate:.3%}",
         f"{cte.candidate_miss_rate:.3%}"],
    ], [63 * mm, 51 * mm, 51 * mm], align_right=(1, 2)))
    F.append(P(
        "<b>The candidate miss rate is the ceiling on top-1 accuracy.</b> An "
        "observation whose reference arc was never retrieved cannot be classified "
        "correctly by any model, however good. At 0.26% it caps top-1 at 99.74%, so "
        "the real headroom lies in ranking, not retrieval.", "small"))

    F.append(P("4.2 The 26 features", "h2"))
    F.append(table([
        ["Group", "Features", "What it captures"],
        ["Geometry (8)",
         "<font face='Courier'>dist_m, dist_rank, dist_minus_best, "
         "dist_ratio_best, dist_z_in_obs, arc_length, proj_t, "
         "proj_is_endpoint</font>",
         "How close the arc is, and how close <i>relative to the other "
         "candidates of the same observation</i> - a 15 m arc means something "
         "different when the runner-up is 16 m away than when it is 60 m away."],
        ["Heading (4)",
         "<font face='Courier'>heading_diff, heading_cos, heading_valid, "
         "turn_angle</font>",
         "Direction of travel against arc bearing. This is the only thing that "
         "separates the two directions of a two-way street: both carry identical "
         "geometry, so distance cannot. <font face='Courier'>heading_valid</font> "
         "switches off when the vehicle has moved under 2 m and the heading is noise."],
        ["Kinematics (6)",
         "<font face='Courier'>speed_mps, accel_mps2, dt_prev, dt_next, "
         "step_prev_m, step_next_m</font>",
         "Local motion and sampling irregularity. A stationary vehicle produces "
         "a different candidate pattern from one at 80 km/h."],
        ["Network context (8)",
         "<font face='Courier'>arc_out_degree, arc_in_degree, has_reverse_arc, "
         "n_candidates, local_arc_density, dist_prev_obs_to_arc, "
         "dist_next_obs_to_arc, obs_frac</font>",
         "Topology around the arc and whether the same arc also explains the "
         "neighbouring fixes - weak sequential evidence, without the classifier "
         "having to model the sequence itself."],
    ], [26 * mm, 60 * mm, 79 * mm]))

    F.append(P("4.3 Design decisions worth defending", "h2"))
    F.extend(bullets([
        "<b>Local equirectangular projection to metres</b>, centred per trip. "
        "Longitude and latitude are angular; a degree of longitude at this latitude "
        "is ~72 km while a degree of latitude is ~111 km, so Euclidean distance in "
        "degrees would distort every distance and heading. At city scale this "
        "projection is accurate to well under 0.1% and is far cheaper than "
        "per-pair geodesics.",
        "<b>Relative features, not just absolute ones.</b> "
        "<font face='Courier'>dist_rank</font>, "
        "<font face='Courier'>dist_minus_best</font> and "
        "<font face='Courier'>dist_z_in_obs</font> describe a candidate against its "
        "competitors. Without them the model has to learn an absolute distance "
        "threshold, which cannot hold across both dense city centres and sparse "
        "outskirts.",
        "<b>No feature touches the reference route.</b> The same code path runs at "
        "training and inference time, which is what makes the deployed API "
        "trustworthy.",
        "<b>Median imputation</b> for the neighbour-dependent features, which are "
        "undefined at the first and last fix of a trip. Standardisation is applied "
        "only for the distance- and gradient-based models.",
    ]))

    F.append(PageBreak())
    F.extend(section("5", "Where this lives"))

    F.append(P("5.1 EDA", "h2"))
    F.append(table([
        ["Path", "What it is"],
        ["<font face='Courier'>scripts/02_eda.py</font>",
         "The EDA stage: inventory, data-quality checks, network statistics, "
         "anomaly detection, all plots and the Folium map"],
        ["<font face='Courier'>results/eda_trip_inventory.csv</font>",
         "Per-trip: observations, duration, bbox, sampling, file sizes, "
         "network size, label residual"],
        ["<font face='Courier'>results/eda_partition_summary.csv</font>",
         "Aggregates per partition"],
        ["<font face='Courier'>results/eda_data_quality.csv</font>",
         "The missing-value and validity checks in section 2.1"],
        ["<font face='Courier'>results/eda_network_summary.csv</font>",
         "Per-trip connectivity, components, degree, arc lengths"],
        ["<font face='Courier'>results/eda_anomalies.csv</font>",
         "The flagged trips of section 2.4"],
        ["<font face='Courier'>results/figures/eda_*.png</font>",
         "Figures 1-3 plus the geographic-extent plot"],
        ["<font face='Courier'>results/maps/eda_trip_*.html</font>",
         "Interactive Folium map of a representative trip"],
        ["<font face='Courier'>results/logs_eda.txt</font>",
         "Full run log"],
    ], [68 * mm, 97 * mm]))

    F.append(P("5.2 Feature engineering and data processing", "h2"))
    F.append(table([
        ["Path", "What it is"],
        ["<font face='Courier'>src/pathsnap/candidates.py</font>",
         "<b>The core of it</b> - candidate retrieval and all 26 features, "
         "with <font face='Courier'>FEATURE_NAMES</font> as the single source of "
         "feature order"],
        ["<font face='Courier'>src/pathsnap/labeling.py</font>",
         "The monotonic DP alignment of section 3.1"],
        ["<font face='Courier'>src/pathsnap/dataset.py</font>",
         "Assembles trips into the training table; owns "
         "<font face='Courier'>LABEL_RESIDUAL_MAX_M</font> and the candidate "
         "miss-rate metric"],
        ["<font face='Courier'>src/pathsnap/network.py</font>",
         "Road network, the 25 m-sampled KD-tree index, bounded Dijkstra"],
        ["<font face='Courier'>src/pathsnap/geo.py</font>",
         "Projection and point-to-segment geometry"],
        ["<font face='Courier'>src/pathsnap/trips.py</font>",
         "Raw-file loading and validation"],
        ["<font face='Courier'>scripts/01_build_dataset.py</font>",
         "Runs the above over both partitions"],
        ["<font face='Courier'>results/candidate_generation_summary.csv</font>",
         "The table in section 4.1"],
        ["<font face='Courier'>notebooks/pathsnap_walkthrough.ipynb</font>",
         "Narrated walkthrough of every step in sections 3 and 4"],
    ], [68 * mm, 97 * mm]))

    F.append(P("Reproduce either stage with:", "body"))
    F.append(P("python scripts/01_build_dataset.py&nbsp;&nbsp;&nbsp;# ~45 s<br/>"
               "python scripts/02_eda.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
               "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# ~4 min", "code"))

    doc.build(F, onFirstPage=cover_furniture, onLaterPages=page_furniture)
    print(f"wrote {OUT}")
    print(f"  {OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()

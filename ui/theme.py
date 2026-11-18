"""Visual theme for the PathSnap evaluator interface.

Design intent: the map is the subject, so the chrome recedes. A deep navy
base with a satellite basemap, one saturated accent reserved for the
predicted route, and glass panels that let the map show through behind the
numbers rather than boxing it in.

Route colours are deliberately not the usual red/green pair: on satellite
imagery both disappear into terrain. Cyan and amber hold up against it, and
the two remain distinguishable in the most common forms of colour blindness.
"""
from __future__ import annotations

# --- palette ---------------------------------------------------------------
BG_DEEP = "#0A0E1A"
BG_PANEL = "#121829"
BG_RAISED = "#1A2238"
BORDER = "rgba(148, 170, 220, 0.16)"
BORDER_BRIGHT = "rgba(148, 170, 220, 0.30)"

TEXT = "#E8EDF9"
TEXT_DIM = "#8FA0C4"
TEXT_FAINT = "#5C6B8A"

ACCENT = "#6EE7F9"          # cyan - primary accent, the predicted route
ACCENT_DEEP = "#22D3EE"
VIOLET = "#A78BFA"
AMBER = "#FBBF24"           # classifier-only route
EMERALD = "#34D399"         # reference route
ROSE = "#FB7185"

# --- map layers ------------------------------------------------------------
MAP_COLOURS = {
    "network": "#64748B",
    "gps": "#93C5FD",
    "reference": EMERALD,
    "classifier": AMBER,
    "predicted": ACCENT,
}

#: free, no API key required, and far more legible than a flat street map
#: once a route is drawn over it
BASEMAPS = {
    "Satellite": "Esri.WorldImagery",
    "Light canvas": "Esri.WorldGrayCanvas",
    "Street": "OpenStreetMap.Mapnik",
    "Topographic": "Esri.WorldTopoMap",
}


def css() -> str:
    """The full stylesheet, injected once per run."""
    return f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {{
  --bg-deep: {BG_DEEP};
  --bg-panel: {BG_PANEL};
  --bg-raised: {BG_RAISED};
  --border: {BORDER};
  --border-bright: {BORDER_BRIGHT};
  --text: {TEXT};
  --text-dim: {TEXT_DIM};
  --text-faint: {TEXT_FAINT};
  --accent: {ACCENT};
  --accent-deep: {ACCENT_DEEP};
  --violet: {VIOLET};
  --amber: {AMBER};
  --emerald: {EMERALD};
  --rose: {ROSE};
}}

/* ---------- base ---------- */
html, body, [class*="css"] {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  -webkit-font-smoothing: antialiased;
}}

.stApp {{
  background:
    radial-gradient(1100px 520px at 18% -8%, rgba(110,231,249,.055), transparent 62%),
    linear-gradient(180deg, #0A0E1A 0%, #0A0F1C 60%, #090D18 100%);
  background-attachment: fixed;
}}

.block-container {{
  padding-top: 1.6rem;
  padding-bottom: 3rem;
  max-width: 1560px;
}}

/* hide Streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}

/* ---------- typography ---------- */
h1, h2, h3, h4 {{ font-family: 'Inter', sans-serif; letter-spacing: -0.02em; }}
.mono {{ font-family: 'JetBrains Mono', ui-monospace, monospace; }}

/* ---------- hero ---------- */
.ps-hero {{
  position: relative;
  border-radius: 20px;
  padding: 26px 30px;
  margin-bottom: 18px;
  background: rgba(18,24,41,.74);
  border: 1px solid var(--border);
  backdrop-filter: blur(18px) saturate(140%);
  -webkit-backdrop-filter: blur(18px) saturate(140%);
  overflow: hidden;
  box-shadow: 0 8px 28px rgba(0,0,0,.30), inset 0 1px 0 rgba(255,255,255,.045);
}}
.ps-hero h1 {{
  margin: 0;
  font-size: 2.0rem;
  font-weight: 800;
  line-height: 1.1;
  color: var(--text);
}}
.ps-hero .tag {{
  margin-top: 8px;
  color: var(--text-dim);
  font-size: .94rem;
  max-width: 70ch;
  line-height: 1.5;
}}
.ps-brandmark {{
  display: inline-flex; align-items: center; gap: 9px;
  font-size: .70rem; font-weight: 700; letter-spacing: .16em;
  text-transform: uppercase; color: var(--accent);
  margin-bottom: 12px;
}}
.ps-brandmark .dot {{
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 0 4px rgba(110,231,249,.16), 0 0 18px var(--accent);
  animation: ps-pulse 2.4s ease-in-out infinite;
}}
@keyframes ps-pulse {{
  0%, 100% {{ opacity: 1; transform: scale(1); }}
  50%      {{ opacity: .55; transform: scale(.82); }}
}}

/* ---------- metric cards ---------- */
.ps-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(168px, 1fr));
  gap: 13px;
  margin: 4px 0 20px 0;
}}
.ps-card {{
  position: relative;
  border-radius: 15px;
  padding: 15px 17px 14px 17px;
  background: rgba(22,29,48,.80);
  border: 1px solid var(--border);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  box-shadow: 0 4px 16px rgba(0,0,0,.24);
  transition: transform .22s cubic-bezier(.22,1,.36,1),
              box-shadow .22s ease, border-color .22s ease;
  overflow: hidden;
}}
.ps-card::before {{
  content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
  background: var(--spark, var(--accent));
  box-shadow: 0 0 16px var(--spark, var(--accent));
}}
.ps-card:hover {{
  transform: translateY(-2px);
  border-color: var(--border-bright);
  box-shadow: 0 10px 26px rgba(0,0,0,.34);
}}
.ps-card .label {{
  font-size: .685rem; font-weight: 600; letter-spacing: .09em;
  text-transform: uppercase; color: var(--text-faint);
  display: block; margin-bottom: 7px;
}}
.ps-card .value {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.62rem; font-weight: 600; line-height: 1.05;
  color: var(--text); display: block;
}}
.ps-card .value .unit {{ font-size: .92rem; color: var(--text-dim); margin-left: 1px; }}
.ps-card .delta {{
  display: inline-block; margin-top: 8px;
  font-size: .73rem; font-weight: 600;
  padding: 3px 9px; border-radius: 999px;
  font-family: 'JetBrains Mono', monospace;
}}
.delta-up   {{ background: rgba(52,211,153,.14); color: var(--emerald); }}
.delta-down {{ background: rgba(251,113,133,.14); color: var(--rose); }}
.delta-flat {{ background: rgba(143,160,196,.12); color: var(--text-dim); }}

/* ---------- panels ---------- */
.ps-panel {{
  border-radius: 17px;
  padding: 19px 21px;
  background: rgba(18,24,41,.66);
  border: 1px solid var(--border);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  box-shadow: 0 12px 32px rgba(0,0,0,.30);
  margin-bottom: 15px;
}}
.ps-panel-title {{
  font-size: .70rem; font-weight: 700; letter-spacing: .13em;
  text-transform: uppercase; color: var(--text-faint);
  margin-bottom: 13px; display: flex; align-items: center; gap: 9px;
}}
.ps-panel-title::after {{
  content: ""; flex: 1; height: 1px;
  background: linear-gradient(90deg, var(--border), transparent);
}}

/* ---------- legend ---------- */
.ps-legend {{ display: flex; flex-wrap: wrap; gap: 9px; margin: 2px 0 12px 0; }}
.ps-chip {{
  display: inline-flex; align-items: center; gap: 7px;
  padding: 6px 13px; border-radius: 999px;
  background: rgba(26,34,56,.78);
  border: 1px solid var(--border);
  font-size: .775rem; color: var(--text-dim); font-weight: 500;
  transition: border-color .18s ease, color .18s ease;
}}
.ps-chip:hover {{ border-color: var(--border-bright); color: var(--text); }}
.ps-chip .sw {{
  width: 15px; height: 3.5px; border-radius: 2px; display: inline-block;
}}
.ps-chip .sw.dotted {{
  background-image: repeating-linear-gradient(90deg,
    currentColor 0 4px, transparent 4px 7px);
}}

/* ---------- status pill ---------- */
.ps-status {{
  display: inline-flex; align-items: center; gap: 8px;
  padding: 6px 13px; border-radius: 999px;
  font-size: .775rem; font-weight: 600;
  font-family: 'JetBrains Mono', monospace;
}}
.ps-ok   {{ background: rgba(52,211,153,.13); color: var(--emerald);
            border: 1px solid rgba(52,211,153,.28); }}
.ps-err  {{ background: rgba(251,113,133,.13); color: var(--rose);
            border: 1px solid rgba(251,113,133,.28); }}
.ps-wait {{ background: rgba(251,191,36,.13); color: var(--amber);
            border: 1px solid rgba(251,191,36,.28); }}

/* ---------- streamlit widget restyling ---------- */
div[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, rgba(14,19,34,.96), rgba(10,14,26,.98));
  border-right: 1px solid var(--border);
}}
div[data-testid="stSidebar"] .block-container {{ padding-top: 1.6rem; }}

.stButton > button {{
  width: 100%;
  border-radius: 12px;
  border: 1px solid rgba(110,231,249,.34);
  background: rgba(110,231,249,.14);
  color: var(--text);
  font-weight: 650;
  letter-spacing: .01em;
  padding: .60rem 1rem;
  transition: transform .18s cubic-bezier(.22,1,.36,1),
              box-shadow .18s ease, border-color .18s ease;
}}
.stButton > button:hover {{
  border-color: var(--accent);
  background: rgba(110,231,249,.22);
  color: #fff;
}}
.stButton > button:active {{ transform: translateY(0); }}

div[data-baseweb="select"] > div {{
  background: rgba(26,34,56,.82) !important;
  border: 1px solid var(--border) !important;
  border-radius: 11px !important;
}}
div[data-baseweb="select"] > div:hover {{ border-color: var(--border-bright) !important; }}

.stTextInput input {{
  background: rgba(26,34,56,.82) !important;
  border: 1px solid var(--border) !important;
  border-radius: 11px !important;
  color: var(--text) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: .85rem !important;
}}
.stTextInput input:focus {{
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(110,231,249,.14) !important;
}}

.stTabs [data-baseweb="tab-list"] {{
  gap: 6px;
  background: rgba(18,24,41,.56);
  padding: 5px; border-radius: 13px;
  border: 1px solid var(--border);
}}
.stTabs [data-baseweb="tab"] {{
  border-radius: 9px; padding: 7px 17px;
  color: var(--text-dim); font-weight: 600; font-size: .86rem;
}}
.stTabs [aria-selected="true"] {{
  background: rgba(110,231,249,.16) !important;
  color: var(--text) !important;
}}

div[data-testid="stDataFrame"] {{
  border-radius: 13px; overflow: hidden; border: 1px solid var(--border);
}}

/* map container gets a rim so it reads as a panel, not a hole */
iframe[title="streamlit_folium.st_folium"] {{
  border-radius: 17px;
  border: 1px solid var(--border);
  box-shadow: 0 18px 46px rgba(0,0,0,.46);
}}

/* scrollbar */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{
  background: rgba(148,170,220,.20); border-radius: 6px;
  border: 2px solid transparent; background-clip: content-box;
}}
::-webkit-scrollbar-thumb:hover {{ background: rgba(148,170,220,.34); background-clip: content-box; }}

/* entrance animation */
@keyframes ps-rise {{
  from {{ opacity: 0; transform: translateY(9px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}
.ps-hero, .ps-grid {{ animation: ps-rise .28s ease-out both; }}
</style>
"""


def metric_card(label: str, value: str, unit: str = "", delta: str | None = None,
                delta_kind: str = "flat", spark: str = ACCENT) -> str:
    unit_html = f'<span class="unit">{unit}</span>' if unit else ""
    delta_html = (f'<span class="delta delta-{delta_kind}">{delta}</span>'
                  if delta else "")
    return (f'<div class="ps-card" style="--spark:{spark}">'
            f'<span class="label">{label}</span>'
            f'<span class="value">{value}{unit_html}</span>'
            f'{delta_html}</div>')


def legend(entries: list[tuple[str, str, bool]]) -> str:
    """entries: (label, colour, dashed)"""
    chips = "".join(
        f'<span class="ps-chip">'
        f'<span class="sw{" dotted" if dashed else ""}" '
        f'style="background:{"" if dashed else col};color:{col}"></span>'
        f'{label}</span>'
        for label, col, dashed in entries)
    return f'<div class="ps-legend">{chips}</div>'

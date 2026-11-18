"""Build a self-contained zip that runs the PathSnap API on Google Colab.

Colab needs the inference path only, so the bundle carries the library, the
API, the deployed model, the ten evaluation trips and the result CSVs the
/metrics endpoint serves. Training data, notebooks, figures and the saved
maps are left out - they are not read at inference time and would multiply
the upload size for nothing.

The bundle also contains a notebook that installs the dependencies, starts
uvicorn and opens a public tunnel, so the UI on another machine can reach it.

Run:  python scripts/make_colab_bundle.py
Out:  dist/PathSnap_Colab_Backend.zip
"""
from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
OUT = DIST / "PathSnap_Colab_Backend.zip"

REQUIREMENTS = """\
# PathSnap backend - inference only.
# Deliberately excludes streamlit, folium, matplotlib and optuna: the API
# never imports them, and they add minutes to a Colab install.
fastapi==0.163.2
uvicorn[standard]==0.54.0
numpy==2.5.5
scipy==1.18.1
pandas==3.1.1
scikit-learn==1.9.1
xgboost==3.4.1
joblib==1.6.0
pydantic==2.14.3
"""

README = """\
# PathSnap backend on Google Colab

Runs the inference API in Colab and exposes it on a public URL so the
Streamlit UI (on any machine) can call it.

## Quick start

1. Upload `PathSnap_Colab_Backend.zip` to Colab.
2. Open `PathSnap_Colab.ipynb`.
3. Runtime -> Run all.
4. Copy the public URL the tunnel cell prints.
5. Paste that URL into the UI sidebar's "API endpoint" box.

## What is in here

```
api/                    FastAPI backend
src/pathsnap/           inference library
artifacts/              deployed model + inference config
results/                hmm_config.json and the experiment CSVs (/metrics)
data/raw/student_test/  the ten evaluation trips
requirements.txt        inference-only dependencies
```

## Swapping in the evaluator trips

Upload them in the same layout and point the API at the directory:

```
data/raw/evaluator_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}
```

```python
os.environ["PATHSNAP_EVAL_DIR"] = "/content/pathsnap/data/raw/evaluator_test"
```

Then restart the API cell.

## Notes

- Colab runtimes are ephemeral. If the runtime disconnects, re-run the
  notebook; the tunnel URL changes each time.
- The free tunnel needs no account. If it fails, the notebook falls back
  to instructions for ngrok.
- Ground truth stays server-side: the reference route is read only inside
  /predict, after prediction, purely to score it.
"""


def notebook() -> dict:
    def code(*lines):
        return {"cell_type": "code", "execution_count": None, "metadata": {},
                "outputs": [], "source": [l + "\n" for l in lines]}

    def md(*lines):
        return {"cell_type": "markdown", "metadata": {},
                "source": [l + "\n" for l in lines]}

    cells = [
        md("# PathSnap — inference API on Colab",
           "",
           "Runs the map-matching backend and exposes it on a public URL.",
           "",
           "**Runtime → Run all**, then copy the URL printed by the tunnel cell",
           "into the Streamlit UI sidebar."),

        md("## 1. Unpack"),
        code("import os, zipfile, pathlib, glob",
             "",
             "TARGET = '/content/pathsnap'",
             "",
             "# find the uploaded zip whether it was dropped in /content or elsewhere",
             "zips = glob.glob('/content/**/PathSnap_Colab_Backend.zip', recursive=True)",
             "if not zips:",
             "    from google.colab import files",
             "    print('Upload PathSnap_Colab_Backend.zip:')",
             "    up = files.upload()",
             "    zips = [list(up.keys())[0]]",
             "",
             "with zipfile.ZipFile(zips[0]) as z:",
             "    z.extractall(TARGET)",
             "",
             "os.chdir(TARGET)",
             "print('working dir:', os.getcwd())",
             "for p in sorted(pathlib.Path('.').iterdir()):",
             "    print(' ', p.name)"),

        md("## 2. Install dependencies",
           "",
           "Inference-only, so this is far smaller than the full project."),
        code("!pip install -q -r requirements.txt 2>&1 | tail -3",
             "print('done')"),

        md("## 3. Sanity check",
           "",
           "Loads the model and scores one trip in-process, before any server",
           "is involved. If this cell passes, the pipeline itself is fine."),
        code("import sys, os",
             "sys.path.insert(0, '/content/pathsnap')",
             "sys.path.insert(0, '/content/pathsnap/src')",
             "",
             "# the bundle ships the compressed deployable model rather than",
             "# the raw per-experiment artifact named in hmm_config.json",
             "os.environ['PATHSNAP_MODEL'] = '/content/pathsnap/artifacts/pathsnap_model.joblib'",
             "",
             "import joblib, json",
             "from pathsnap.trips import load_trip, list_trips",
             "from pathsnap.network import build_network",
             "from pathsnap.pipeline import PathSnapMatcher, evaluate_against_reference",
             "from pathsnap.hmm import HMMConfig",
             "",
             "cfg = HMMConfig(**json.load(open('results/hmm_config.json'))['config'])",
             "model = joblib.load(os.environ['PATHSNAP_MODEL'])",
             "matcher = PathSnapMatcher(model, cfg)",
             "",
             "tid = sorted(list_trips('data/raw/student_test'))[0]",
             "trip = load_trip('data/raw/student_test', tid)",
             "net = build_network(trip)",
             "res = matcher.match(trip, net, run_hmm=True)",
             "ev = evaluate_against_reference(res, trip, net)",
             "",
             "print(f'trip {tid}: {trip.n_obs} observations')",
             "print(f\"  classifier top-1 : {ev['classifier_top1_accuracy']:.4f}\")",
             "print(f\"  HMM top-1        : {ev['hmm_top1_accuracy']:.4f}\")",
             "print(f\"  route continuity : {ev['hmm_route_continuity']:.4f}\")"),

        md("## 4. Start the API"),
        code("import subprocess, time, requests, os",
             "",
             "env = dict(os.environ)",
             "env['PATHSNAP_MODEL'] = '/content/pathsnap/artifacts/pathsnap_model.joblib'",
             "env['PYTHONPATH'] = '/content/pathsnap:/content/pathsnap/src'",
             "",
             "api = subprocess.Popen(",
             "    ['uvicorn', 'api.main:app', '--host', '0.0.0.0', '--port', '8000'],",
             "    cwd='/content/pathsnap', env=env,",
             "    stdout=open('/content/api.log', 'w'),",
             "    stderr=subprocess.STDOUT)",
             "",
             "for _ in range(40):",
             "    try:",
             "        h = requests.get('http://127.0.0.1:8000/health', timeout=2).json()",
             "        print('API up —', h['model'], '|', h['n_trips'], 'trips')",
             "        break",
             "    except Exception:",
             "        time.sleep(1)",
             "else:",
             "    print('API did not start; log below')",
             "    print(open('/content/api.log').read()[-2000:])"),

        md("## 5. Open a public tunnel",
           "",
           "Colab cannot be reached directly from another machine, so the API",
           "is published through a tunnel. This one needs no account."),
        code("import subprocess, time, re, os",
             "",
             "if not os.path.exists('/content/cloudflared'):",
             "    !wget -q -O /content/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
             "    !chmod +x /content/cloudflared",
             "",
             "tun = subprocess.Popen(",
             "    ['/content/cloudflared', 'tunnel', '--url', 'http://localhost:8000',",
             "     '--no-autoupdate'],",
             "    stdout=open('/content/tunnel.log', 'w'),",
             "    stderr=subprocess.STDOUT)",
             "",
             "public = None",
             "for _ in range(45):",
             "    time.sleep(1)",
             "    try:",
             "        log = open('/content/tunnel.log').read()",
             "    except FileNotFoundError:",
             "        continue",
             "    m = re.search(r'https://[-a-z0-9]+\\.trycloudflare\\.com', log)",
             "    if m:",
             "        public = m.group(0)",
             "        break",
             "",
             "if public:",
             "    print('=' * 62)",
             "    print('  PUBLIC API URL')",
             "    print(' ', public)",
             "    print()",
             "    print('  Paste that into the UI sidebar: \"API endpoint\"')",
             "    print('=' * 62)",
             "else:",
             "    print('Tunnel did not start. Log:')",
             "    print(open('/content/tunnel.log').read()[-1500:])",
             "    print()",
             "    print('Fallback — ngrok (needs a free authtoken):')",
             "    print('  !pip install -q pyngrok')",
             "    print('  from pyngrok import ngrok')",
             "    print('  ngrok.set_auth_token(\"YOUR_TOKEN\")')",
             "    print('  print(ngrok.connect(8000).public_url)')"),

        md("## 6. Verify the public URL end to end"),
        code("import requests",
             "",
             "base = public.rstrip('/')",
             "print('health :', requests.get(f'{base}/health', timeout=30).json())",
             "trips = requests.get(f'{base}/trips', timeout=30).json()",
             "print('trips  :', len(trips))",
             "",
             "small = min(trips, key=lambda t: t['n_observations'])['trip_id']",
             "r = requests.post(f'{base}/predict',",
             "                  json={'trip_id': small, 'include_gps': False},",
             "                  timeout=600).json()",
             "ev = r['evaluation']",
             "print(f\"predict {small}: classifier {ev['classifier_top1_accuracy']:.4f}\"",
             "      f\" -> HMM {ev['hmm_top1_accuracy']:.4f}\"",
             "      f\" ({ev['hmm_improvement_pp']:+.2f} pp)\")"),

        md("## 7. Live request log",
           "",
           "Re-run this whenever you want to show that requests are arriving."),
        code("print(open('/content/api.log').read()[-3000:])"),

        md("---",
           "### Using the evaluator trips",
           "",
           "Upload them as `data/raw/evaluator_test/<trip_id>/<trip_id>.*`, then:",
           "",
           "```python",
           "os.environ['PATHSNAP_EVAL_DIR'] = '/content/pathsnap/data/raw/evaluator_test'",
           "```",
           "",
           "and re-run the API cell.",
           "",
           "### Keep the runtime alive",
           "Colab disconnects idle runtimes. Keep the tab open during the demo;",
           "if it drops, re-run all cells — the tunnel URL will change."),
    ]

    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3",
                                        "language": "python", "name": "python3"},
                         "language_info": {"name": "python"},
                         "colab": {"provenance": [], "toc_visible": True}},
            "nbformat": 4, "nbformat_minor": 0}


def main() -> None:
    DIST.mkdir(exist_ok=True)

    model = ROOT / "artifacts" / "pathsnap_model.joblib"
    if not model.exists():
        sys.exit("artifacts/pathsnap_model.joblib missing - "
                 "run scripts/07_export_artifacts.py first")

    if OUT.exists():
        OUT.unlink()

    added = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        def add(path: Path, arc: str):
            nonlocal added
            z.write(path, arc)
            added += 1

        # library + API
        for p in sorted((ROOT / "src" / "pathsnap").glob("*.py")):
            add(p, f"src/pathsnap/{p.name}")
        for p in sorted((ROOT / "api").glob("*.py")):
            add(p, f"api/{p.name}")
        init = ROOT / "api" / "__init__.py"
        if not init.exists():
            z.writestr("api/__init__.py", "")
            added += 1

        # deployed model and config
        add(model, "artifacts/pathsnap_model.joblib")

        # Alternate classifiers, so /models and model switching work in Colab.
        # Capped by size: extra_trees is 422 MB and random_forest 350 MB, which
        # would dwarf everything else for a feature that is nice to have.
        MAX_ALT_MB = 25
        for alt in sorted((ROOT / "artifacts" / "models").glob("*.joblib")):
            mb = alt.stat().st_size / 1e6
            if mb <= MAX_ALT_MB:
                add(alt, f"artifacts/models/{alt.name}")
            else:
                print(f"  skipped {alt.name} ({mb:.0f} MB, over {MAX_ALT_MB} MB)")
        for name in ("inference_config.json", "manifest.json"):
            p = ROOT / "artifacts" / name
            if p.exists():
                add(p, f"artifacts/{name}")

        # hmm config + the CSVs behind /metrics
        for p in sorted((ROOT / "results").glob("*.csv")):
            add(p, f"results/{p.name}")
        for p in sorted((ROOT / "results").glob("*.json")):
            add(p, f"results/{p.name}")

        # the ten evaluation trips
        for p in sorted((ROOT / "data" / "raw" / "student_test").rglob("*")):
            if p.is_file():
                add(p, f"data/raw/student_test/{p.relative_to(ROOT / 'data' / 'raw' / 'student_test')}")

        z.writestr("requirements.txt", REQUIREMENTS)
        z.writestr("README.md", README)
        z.writestr("PathSnap_Colab.ipynb", json.dumps(notebook(), indent=1))
        added += 3

    size_mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT}")
    print(f"  {added} files, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()

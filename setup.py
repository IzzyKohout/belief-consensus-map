#!/usr/bin/env python3
"""
setup.py — single entry point for the Belief Consensus Map project.

Usage:
    python setup.py

What it does:
    1. Clones the Global Dialogues repo (or pulls if already cloned)
    2. Runs the preprocessing script to build app/data.json from GD8 data
    3. Validates the output
    4. Prints instructions for running locally and deploying to GitHub Pages
"""

import subprocess
import sys
import os
import json
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

GD_REPO_URL  = "https://github.com/collect-intel/global-dialogues.git"
GD_REPO_DIR  = Path(__file__).parent / "global-dialogues"
PROJECT_ROOT = Path(__file__).parent
ROUND        = 8

# ── Helpers ───────────────────────────────────────────────────────────────────

def run(cmd, cwd=None):
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n❌ Command failed:\n{result.stderr}")
        sys.exit(1)
    return result.stdout.strip()

def step(n, msg):
    print(f"\n{'─'*60}")
    print(f"  Step {n}: {msg}")
    print(f"{'─'*60}")

# ── Steps ─────────────────────────────────────────────────────────────────────

def step1_clone_or_pull():
    step(1, "Fetching Global Dialogues data")
    if GD_REPO_DIR.exists():
        print(f"  Repo already exists at {GD_REPO_DIR} — pulling latest...")
        run(["git", "pull"], cwd=GD_REPO_DIR)
    else:
        print(f"  Cloning {GD_REPO_URL}...")
        run(["git", "clone", GD_REPO_URL, str(GD_REPO_DIR)])
    print("  ✓ Data repo ready")

def step2_preprocess():
    step(2, f"Preprocessing GD{ROUND} data → app/data.json")
    preprocess_script = PROJECT_ROOT / "preprocess" / "build_data.py"
    run([
        sys.executable, str(preprocess_script),
        "--gd-path",  str(GD_REPO_DIR),
        "--round",    str(ROUND),
        "--output",   str(PROJECT_ROOT / "app" / "data.json"),
    ])
    print("  ✓ data.json built")

def step3_validate():
    step(3, "Validating output")
    data_path = PROJECT_ROOT / "app" / "data.json"
    if not data_path.exists():
        print("  ❌ app/data.json not found")
        sys.exit(1)

    with open(data_path) as f:
        data = json.load(f)

    n_questions = len(data["questions"])
    n_countries = max(len(q["countries"]) for q in data["questions"])
    size_kb     = data_path.stat().st_size // 1024

    print(f"  ✓ {n_questions} questions")
    print(f"  ✓ up to {n_countries} countries per question")
    print(f"  ✓ {size_kb} KB")

def step4_instructions():
    step(4, "Done! Next steps")
    print("""
  ── Run locally ──────────────────────────────────────────
  cd app
  python -m http.server 8000
  Open: http://localhost:8000

  ── Deploy to GitHub Pages ───────────────────────────────
  1. Create a GitHub repo for this project
  2. Push everything:
       git init
       git add .
       git commit -m "Initial commit"
       git remote add origin https://github.com/YOUR_USERNAME/belief-consensus-map.git
       git push -u origin main

  3. Deploy the app/ folder to gh-pages:
       git subtree push --prefix app origin gh-pages

  4. Enable GitHub Pages in repo Settings → Pages → Branch: gh-pages
  Live at: https://YOUR_USERNAME.github.io/belief-consensus-map/

  ── Update to a new round later ──────────────────────────
  python setup.py --round 9
    """)

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Allow --round flag override
    if "--round" in sys.argv:
        idx = sys.argv.index("--round")
        ROUND = int(sys.argv[idx + 1])

    print("\n╔══════════════════════════════════════════╗")
    print(  "║   Belief Consensus Map — Setup & Build  ║")
    print(  "╚══════════════════════════════════════════╝\n")
    print(f"  Target round: GD{ROUND}")

    step1_clone_or_pull()
    step2_preprocess()
    step3_validate()
    step4_instructions()
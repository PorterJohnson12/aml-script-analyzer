"""Dataset paths and constants.

Paths are resolved relative to the repo root so the package works from a notebook,
a script, or an import — no hard-coded absolute paths.
"""
from __future__ import annotations
from pathlib import Path

# src/outlier/config.py -> parents[2] == repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

# --- TRIPOD (text + gold turning-point labels) ---
TRIPOD_DIR = DATA_DIR / "TRIPOD-master"
SCREENPLAYS_DIR = TRIPOD_DIR / "Screenplays_and_imdb_meta"
GOLD_SCREENPLAYS_CSV = TRIPOD_DIR / "Synopses_and_annotations" / "TRIPOD_screenplays_test.csv"
SYNOPSES_TRAIN_CSV = TRIPOD_DIR / "Synopses_and_annotations" / "TRIPOD_synopses_train.csv"
SYNOPSES_TEST_CSV = TRIPOD_DIR / "Synopses_and_annotations" / "TRIPOD_synopses_test.csv"

# --- SUMMER (silver train labels, projected synopsis->scene) ---
SILVER_LABELS_PICKLE = DATA_DIR / "SUMMER-master" / "dataset" / "labels_train_TRIPOD_silver.pickle"

# --- ScriptBase (full text corpus, for the secondary principles later) ---
SCRIPTBASE_DIR = DATA_DIR / "scriptbase-master"

# --- turning-point definitions (Syd-Field-style, as used by TRIPOD) ---
N_TPS = 5
TP_NAMES = [
    "Opportunity",         # tp1: introductory event after the setup
    "Change of Plans",     # tp2: the story's main goal is defined
    "Point of No Return",  # tp3: protagonist fully commits
    "Major Setback",       # tp4: everything falls apart
    "Climax",              # tp5: final event of the main story
]

# scene-heading markers used to segment a screenplay (must match TRIPOD's segmenter)
SLUGLINE_MARKERS = ("INT.", "EXT.", "INT/EXT.", "EXT./INT.", "INT./EXT.")

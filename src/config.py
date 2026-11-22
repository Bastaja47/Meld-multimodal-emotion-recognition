"""
Sve putanje projekta na jednom mestu.

Sirovi MELD skup (video klipovi + CSV fajlovi) NIJE u repozitorijumu.
Putanju do njega zadaje promenljiva okruzenja MELD_ROOT, npr.

    PowerShell:  $env:MELD_ROOT = "D:\\podaci\\MELD.Raw"
    Linux/Mac:   export MELD_ROOT=/putanja/do/MELD.Raw

Ako promenljiva nije postavljena, trazi se folder MELD.Raw u korenu projekta.

Izvedeni podaci (audio, lica, obelezja, manifesti) se cuvaju u MELD_ROOT,
a checkpoint-i i rezultati u folderima projekta.
"""

import os
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

# --- ulazni podaci (MELD) ---------------------------------------------
MELD_ROOT = Path(os.environ.get("MELD_ROOT", PROJECT_DIR / "MELD.Raw"))

# split -> (CSV sa labelama, folder sa .mp4 klipovima), kako je u MELD arhivi
SPLITS = {
    "train": ("train_sent_emo.csv", "train_splits"),
    "dev":   ("dev_sent_emo.csv", "dev_splits_complete"),        # dev = validacioni skup
    "test":  ("test_sent_emo.csv", "output_repeated_splits_test"),
}

# --- izvedeni podaci (nisu u git-u) -----------------------------------
DATA_DIR = MELD_ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"                    # 6s, 16kHz, mono .wav
FACES_DIR = DATA_DIR / "faces"                    # isecena lica, (8, 3, 160, 160) uint8
AUDIO_FEAT_DIR = DATA_DIR / "audio_features"      # wav2vec2, (T, 768)
VIDEO_FEAT_DIR = DATA_DIR / "video_features"      # InceptionResnetV1, (8, 512)

MANIFEST = MELD_ROOT / "manifest.csv"
MANIFEST_FINAL = MELD_ROOT / "manifest_final.csv"

# --- izlazi projekta --------------------------------------------------
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"      # nije u git-u
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_CSV = RESULTS_DIR / "results.csv"
CONF_MAT_DIR = RESULTS_DIR / "confusion_matrices"

"""
Pokrece sve treninge i kasnu fuziju redom, jednom komandom (iz korena projekta):

    python src/run_all.py

Redosled:
  1) audio (base), audio (tp_cw)
  2) video (base), video (tp_cw)
  3) early_fusion (base), early_fusion (tp_cw)
  4) late fusion (base), late fusion (tp_cw) - koristi modele iz koraka 1 i 2

Modeli ciji checkpoint vec postoji se preskacu (obrisi checkpoint da bi se
model ponovo trenirao). Rezultati idu u results/results.csv, matrice konfuzije
u results/confusion_matrices/, checkpoint-i u checkpoints/.
"""

import subprocess
import sys

from config import SRC_DIR, CHECKPOINT_DIR, RESULTS_CSV, CONF_MAT_DIR

TRAIN_RUNS = [
    ("audio", "base"),
    ("audio", "tp_cw"),
    ("video", "base"),
    ("video", "tp_cw"),
    ("early_fusion", "base"),
    ("early_fusion", "tp_cw"),
]

LATE_FUSION_VARIANTS = ["base", "tp_cw"]


def main():
    for model, variant in TRAIN_RUNS:
        ckpt_path = CHECKPOINT_DIR / f"{model}_{variant}.pt"
        if ckpt_path.exists():
            print(f"[PRESKACEM] {model} ({variant}) - checkpoint vec postoji: {ckpt_path}")
            continue

        print("=" * 90)
        print(f"TRENING: model={model} varijanta={variant}")
        print("=" * 90)
        subprocess.run(
            [sys.executable, str(SRC_DIR / "train.py"), "--model", model, "--variant", variant],
            check=True,
        )

    for variant in LATE_FUSION_VARIANTS:
        print("=" * 90)
        print(f"KASNA FUZIJA: varijanta={variant}")
        print("=" * 90)
        subprocess.run(
            [sys.executable, str(SRC_DIR / "late_fusion_eval.py"), "--variant", variant],
            check=True,
        )

    print("\nSVE ZAVRSENO.")
    print(f"Rezultati: {RESULTS_CSV}")
    print(f"Matrice konfuzije: {CONF_MAT_DIR}")


if __name__ == "__main__":
    main()

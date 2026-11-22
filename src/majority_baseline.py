"""
Referentni (trivijalni) model: uvek predvidja najcescu klasu iz train skupa.

MELD je jako neuravnotezen (klasa neutral je najbrojnija), pa tacnost samo
sama po sebi moze da zavara. Ovaj red u results.csv sluzi da se vidi koliko
istrenirani modeli stvarno dodaju u odnosu na pogadjanje najcesce klase.

Primer:  python src/majority_baseline.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from common import append_result_row
from config import MANIFEST_FINAL, RESULTS_CSV


def main():
    manifest = pd.read_csv(MANIFEST_FINAL)
    train_df = manifest[manifest["split"] == "train"]
    test_df = manifest[manifest["split"] == "test"]

    majority = train_df["emotion"].value_counts().idxmax()
    preds = [majority] * len(test_df)

    acc = accuracy_score(test_df["emotion"], preds)
    f1 = f1_score(test_df["emotion"], preds, average="weighted", zero_division=0)

    print(f"Najcesca klasa u train skupu: {majority}")
    print("\nDistribucija test skupa:")
    print(test_df["emotion"].value_counts().to_string())
    print(f"\nREFERENTNI MODEL ('{majority}') | test acc={acc:.4f} | weighted F1={f1:.4f}")

    append_result_row(RESULTS_CSV, {
        "model": "majority_class",
        "variant": "-",
        "test_accuracy": acc,
        "test_f1": f1,
        "best_val_f1": np.nan,
    })
    print(f"Rezultat upisan u: {RESULTS_CSV}")


if __name__ == "__main__":
    main()

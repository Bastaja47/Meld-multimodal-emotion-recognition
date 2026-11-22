"""
Korak 8: kasna fuzija iz vec istreniranih audio i video modela
(za varijantu base ili tp_cw):

    p_final = alpha * p_audio + (1 - alpha) * p_video,   alpha = 0.5

Primer:  python src/late_fusion_eval.py --variant base
"""

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from common import MELDFeatureDataset, UnimodalMLP, NUM_CLASSES, plot_confusion_matrix, append_result_row
from config import MANIFEST_FINAL, CHECKPOINT_DIR, RESULTS_CSV, CONF_MAT_DIR

ALPHA = 0.5   # jednaka tezina audio i video grani
BATCH_SIZE = 32


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = UnimodalMLP(input_dim=ckpt["input_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def get_probs(model, loader, device):
    """Vraca softmax verovatnoce (N, 7) i listu pravih labela."""
    all_probs, all_labels = [], []
    with torch.no_grad():
        for feats, labels in loader:
            feats = feats.to(device)
            probs = F.softmax(model(feats), dim=1)
            all_probs.append(probs.cpu())
            all_labels.extend(labels.tolist())
    return torch.cat(all_probs, dim=0).numpy(), all_labels


def main():
    parser = argparse.ArgumentParser()
    # base: bez temporal pooling-a; tp_cw: temporal pooling + ponderisanje klasa
    parser.add_argument("--variant", required=True, choices=["base", "tp_cw"])
    args = parser.parse_args()

    pooling = "tp" if args.variant == "tp_cw" else "none"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    audio_ckpt = CHECKPOINT_DIR / f"audio_{args.variant}.pt"
    video_ckpt = CHECKPOINT_DIR / f"video_{args.variant}.pt"
    if not audio_ckpt.exists() or not video_ckpt.exists():
        raise FileNotFoundError(
            f"Nedostaje checkpoint ({audio_ckpt} ili {video_ckpt}). "
            f"Prvo istreniraj audio i video model za varijantu '{args.variant}'."
        )

    manifest = pd.read_csv(MANIFEST_FINAL)
    test_df = manifest[manifest["split"] == "test"]

    audio_loader = DataLoader(MELDFeatureDataset(test_df, "audio", pooling), batch_size=BATCH_SIZE, shuffle=False)
    video_loader = DataLoader(MELDFeatureDataset(test_df, "video", pooling), batch_size=BATCH_SIZE, shuffle=False)

    audio_model = load_model(audio_ckpt, device)
    video_model = load_model(video_ckpt, device)

    audio_probs, labels_a = get_probs(audio_model, audio_loader, device)
    video_probs, labels_v = get_probs(video_model, video_loader, device)

    assert labels_a == labels_v, "Redosled test uzoraka mora biti isti za audio i video!"

    final_probs = ALPHA * audio_probs + (1 - ALPHA) * video_probs
    preds = final_probs.argmax(axis=1)

    acc = accuracy_score(labels_a, preds)
    f1 = f1_score(labels_a, preds, average="weighted", zero_division=0)
    print(f"KASNA FUZIJA ({args.variant}) | test acc={acc:.4f} | weighted F1={f1:.4f}")

    cm = confusion_matrix(labels_a, preds, labels=list(range(NUM_CLASSES)))
    CONF_MAT_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        cm, title=f"late_fusion ({args.variant})",
        out_path=CONF_MAT_DIR / f"late_fusion_{args.variant}.png",
    )
    np.save(CONF_MAT_DIR / f"late_fusion_{args.variant}.npy", cm)

    append_result_row(RESULTS_CSV, {
        "model": "late_fusion",
        "variant": args.variant,
        "test_accuracy": acc,
        "test_f1": f1,
        "best_val_f1": np.nan,
    })
    print(f"Rezultat upisan u: {RESULTS_CSV}")


if __name__ == "__main__":
    main()

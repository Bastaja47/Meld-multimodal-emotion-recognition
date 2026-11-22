"""
Korak 7: trening jednog modela (audio / video / early_fusion).

Varijante:
  base   - bez temporal pooling-a (puna vremenska sekvenca, flatten),
           bez ponderisanja klasa
  tp_cw  - sa temporal pooling-om (mean+std) I ponderisanjem klasa

Napomena: tp_cw menja dve stvari odjednom (pooling i pondere klasa), pa
razlika izmedju varijanti ne moze da se pripise samo jednoj od njih.

Parametri treninga:
  Adam, lr=1e-4, batch=32, max 30 epoha, ReduceLROnPlateau (patience=2),
  rano zaustavljanje (patience=5, kriterijum = weighted F1 na dev skupu),
  unakrsna entropija (sa/bez ponderisanja klasa).

Najbolji model (po dev F1) se cuva u checkpoints/{model}_{variant}.pt i
jednom se evaluira na test skupu. Rezultat ide u results/results.csv, a
matrica konfuzije u results/confusion_matrices/.

Primer:  python src/train.py --model audio --variant base
"""

import argparse
import copy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from common import (
    MELDFeatureDataset, UnimodalMLP, EarlyFusionMLP, LABEL2IDX,
    NUM_CLASSES, plot_confusion_matrix, append_result_row,
)
from config import MANIFEST_FINAL, CHECKPOINT_DIR, RESULTS_CSV, CONF_MAT_DIR

BATCH_SIZE = 32
LR = 1e-4
MAX_EPOCHS = 30
EARLY_STOP_PATIENCE = 5
LR_PATIENCE = 2
SEED = 42   # fiksira nasumicnost; prvobitni rezultati su dobijeni bez fiksnog seed-a


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_modality(model_name):
    return {"audio": "audio", "video": "video", "early_fusion": "both"}[model_name]


def build_model(model_name, input_dim):
    if model_name == "early_fusion":
        return EarlyFusionMLP(input_dim=input_dim)
    return UnimodalMLP(input_dim=input_dim)


def run_epoch(model, loader, criterion, device, optimizer=None, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(train):
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(feats)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * feats.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return avg_loss, acc, f1, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["audio", "video", "early_fusion"])
    parser.add_argument("--variant", required=True, choices=["base", "tp_cw"])
    args = parser.parse_args()

    set_seed(SEED)

    pooling = "tp" if args.variant == "tp_cw" else "none"
    use_class_weight = args.variant == "tp_cw"
    modality = get_modality(args.model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model: {args.model} | Varijanta: {args.variant} | Device: {device}")

    manifest = pd.read_csv(MANIFEST_FINAL)
    train_df = manifest[manifest["split"] == "train"]
    dev_df = manifest[manifest["split"] == "dev"]
    test_df = manifest[manifest["split"] == "test"]

    train_ds = MELDFeatureDataset(train_df, modality, pooling)
    dev_ds = MELDFeatureDataset(dev_df, modality, pooling)
    test_ds = MELDFeatureDataset(test_df, modality, pooling)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    sample_feat, _ = train_ds[0]
    input_dim = sample_feat.shape[0]
    print(f"Dimenzija ulaznog vektora: {input_dim}")

    model = build_model(args.model, input_dim).to(device)

    if use_class_weight:
        train_label_idx = train_df["emotion"].map(LABEL2IDX).tolist()
        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.arange(NUM_CLASSES), y=train_label_idx,
        )
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
        print(f"Class weights: {np.round(class_weights, 3)}")
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=LR_PATIENCE
    )

    best_f1 = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc, train_f1, _, _ = run_epoch(
            model, train_loader, criterion, device, optimizer, train=True
        )
        val_loss, val_acc, val_f1, _, _ = run_epoch(
            model, dev_loader, criterion, device, train=False
        )
        scheduler.step(val_f1)

        print(
            f"Epoha {epoch:2d} | train_loss={train_loss:.4f} train_f1={train_f1:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"Rano zaustavljanje posle {epoch} epoha (bez poboljsanja {EARLY_STOP_PATIENCE} epoha).")
            break

    model.load_state_dict(best_state)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"{args.model}_{args.variant}.pt"
    torch.save({"state_dict": best_state, "input_dim": input_dim}, ckpt_path)
    print(f"Najbolji model sacuvan: {ckpt_path}")

    # test skup se koristi samo jednom, na kraju
    test_loss, test_acc, test_f1, test_preds, test_labels = run_epoch(
        model, test_loader, criterion, device, train=False
    )
    print(f"\nTEST | acc={test_acc:.4f} | weighted F1={test_f1:.4f}")

    CONF_MAT_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(test_labels, test_preds, labels=list(range(NUM_CLASSES)))
    plot_confusion_matrix(
        cm, title=f"{args.model} ({args.variant})",
        out_path=CONF_MAT_DIR / f"{args.model}_{args.variant}.png",
    )
    np.save(CONF_MAT_DIR / f"{args.model}_{args.variant}.npy", cm)

    append_result_row(RESULTS_CSV, {
        "model": args.model,
        "variant": args.variant,
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "best_val_f1": best_f1,
    })
    print(f"Rezultat upisan u: {RESULTS_CSV}")


if __name__ == "__main__":
    main()

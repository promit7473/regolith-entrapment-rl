"""
Phase 1 — Train the CNN-GRU sinkage detector.

Usage:
    python phase1_detection/scripts/train_detector.py \
        --data_dir phase1_detection/data \
        --out_dir  phase1_detection/models/saved \
        --epochs 50 --batch_size 256

Loads all .npz files from data_dir, trains SinkageDetector, saves best checkpoint.
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import classification_report, confusion_matrix

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from phase1_detection.models.cnn_gru import SinkageDetector, LABEL_NAMES


def load_dataset(data_dir: str):
    files = sorted(glob.glob(os.path.join(data_dir, "sequences_*.npz")))
    if not files:
        raise FileNotFoundError(f"No sequence files found in {data_dir}. "
                                f"Run scripts/collect_data.py first.")
    Xs, ys = [], []
    for f in files:
        d = np.load(f)
        Xs.append(d["X"])
        ys.append(d["y"])
        print(f"  Loaded {len(d['X'])} windows from {os.path.basename(f)}")
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return torch.from_numpy(X), torch.from_numpy(y)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str,
                        default=os.path.join(REPO_ROOT, "phase1_detection", "data"))
    parser.add_argument("--out_dir",    type=str,
                        default=os.path.join(REPO_ROOT, "phase1_detection", "models", "saved"))
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--val_split",  type=float, default=0.15)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nLoading dataset from {args.data_dir} ...")
    X, y = load_dataset(args.data_dir)

    # Class weights to handle imbalance
    counts = torch.bincount(y, minlength=3).float()
    weights = (counts.sum() / (3 * counts)).to(device)
    print(f"  Samples — normal: {int(counts[0])}  sinking: {int(counts[1])}  "
          f"entrapped: {int(counts[2])}")
    print(f"  Class weights: {weights.tolist()}")

    # Train/val split
    dataset = TensorDataset(X, y)
    n_val   = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    model     = SinkageDetector().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = 0.0
    print(f"\nTraining for {args.epochs} epochs on {device} ...\n")

    for epoch in range(1, args.epochs + 1):
        # ── Train ────────────────────────────────────────────────────────
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct    += (logits.argmax(1) == yb).sum().item()
            total      += len(yb)
        train_acc  = correct / total
        train_loss = total_loss / total

        # ── Validate ─────────────────────────────────────────────────────
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred   = model(xb).argmax(1)
                all_pred.append(pred.cpu())
                all_true.append(yb.cpu())

        all_pred = torch.cat(all_pred).numpy()
        all_true = torch.cat(all_true).numpy()
        from sklearn.metrics import f1_score
        entrapped_f1 = f1_score(all_true, all_pred, labels=[2], average="macro",
                                zero_division=0)
        macro_f1     = f1_score(all_true, all_pred, average="macro", zero_division=0)

        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val_macro_f1={macro_f1:.3f} val_entrapped_f1={entrapped_f1:.3f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            ckpt = os.path.join(args.out_dir, "best_detector.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_macro_f1": macro_f1,
                "val_entrapped_f1": entrapped_f1,
            }, ckpt)
            print(f"  → Saved best model (f1={macro_f1:.3f})")

    # ── Final report ──────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Best val macro F1: {best_f1:.3f}")
    print(f"  Checkpoint: {ckpt}")
    print(f"\nClassification report (final epoch):")
    print(classification_report(all_true, all_pred, target_names=LABEL_NAMES,
                                zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(all_true, all_pred))


if __name__ == "__main__":
    train()

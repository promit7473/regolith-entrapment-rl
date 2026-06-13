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

from detection.models.cnn_gru import SinkageDetector, LABEL_NAMES


def load_dataset(data_dir: str):
    files = sorted(glob.glob(os.path.join(data_dir, "sequences_*.npz")))
    if not files:
        raise FileNotFoundError(
            f"No sequence files found in {data_dir}. "
            f"Generate training data using eval.py --save-data or create a custom "
            f"data collection script to produce sequences_*.npz files."
        )
    Xs, ys, file_ids = [], [], []
    for fi, f in enumerate(files):
        d = np.load(f)
        Xs.append(d["X"])
        ys.append(d["y"])
        file_ids.append(np.full(len(d["X"]), fi, dtype=np.int64))
        print(f"  Loaded {len(d['X'])} windows from {os.path.basename(f)}")
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    fid = np.concatenate(file_ids, axis=0)
    return torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(fid)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str,
                        default=os.path.join(REPO_ROOT, "detection", "data"))
    parser.add_argument("--out_dir",    type=str,
                        default=os.path.join(REPO_ROOT, "detection", "models", "saved"))
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
    X, y, fid = load_dataset(args.data_dir)


    counts = torch.bincount(y, minlength=3).float()
    weights = (counts.sum() / (3 * counts)).to(device)
    print(f"  Samples — normal: {int(counts[0])}  sinking: {int(counts[1])}  "
          f"entrapped: {int(counts[2])}")
    print(f"  Class weights: {weights.tolist()}")

    # Split by SOURCE FILE (collection run), not by window. Sliding windows
    # from the same episode are temporally correlated — a random window-level
    # split leaks near-duplicates of every training window into the val set
    # and inflates F1. With file-level splitting, val episodes are unseen.
    n_files = int(fid.max().item()) + 1
    if n_files < 2:
        print("  WARNING: only one sequence file — falling back to window-level "
              "split. Val metrics will be optimistically biased; collect more "
              "runs (one file per eval run) for honest numbers.")
        dataset = TensorDataset(X, y)
        n_val   = int(len(dataset) * args.val_split)
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val],
                                          generator=torch.Generator().manual_seed(args.seed))
    else:
        g = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(n_files, generator=g)
        n_val_files = max(1, int(round(n_files * args.val_split)))
        val_files = set(perm[:n_val_files].tolist())
        val_mask  = torch.tensor([int(f) in val_files for f in fid])
        train_set = TensorDataset(X[~val_mask], y[~val_mask])
        val_set   = TensorDataset(X[val_mask],  y[val_mask])
        print(f"  File-level split: {n_files - n_val_files} train / "
              f"{n_val_files} val files ({len(train_set)}/{len(val_set)} windows)")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    model     = SinkageDetector().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = 0.0
    ckpt = None
    print(f"\nTraining for {args.epochs} epochs on {device} ...\n")

    for epoch in range(1, args.epochs + 1):

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


    print(f"\n{'='*55}")
    print(f"  Best val macro F1: {best_f1:.3f}")
    if ckpt:
        print(f"  Checkpoint: {ckpt}")
    else:
        print("  No checkpoint saved (no improvement during training)")
    print(f"\nClassification report (final epoch):")
    print(classification_report(all_true, all_pred, target_names=LABEL_NAMES,
                                zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(all_true, all_pred))


if __name__ == "__main__":
    train()

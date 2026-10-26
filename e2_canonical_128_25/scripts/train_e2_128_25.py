import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.thesis_eval import (
    count_trainable_params,
    metrics_dict,
    per_class_metrics,
    save_confusion_matrix,
    save_predictions,
)

FEATURE_DIR = PROJECT_ROOT / "e2_multimodal_fusion" / "features"
RESULTS_DIR = EXPERIMENT_ROOT / "results"
CHECKPOINT_DIR = EXPERIMENT_ROOT / "checkpoints"
EXPECTED_SPLIT_COUNTS = {"train": 46713, "val": 5193, "test": 5767}


class E2Canonical128To25Head(nn.Module):
    def __init__(self, input_dim=1344, num_classes=25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Train E2 1344->128->25 fusion head")
    parser.add_argument("--feature-dir", default=str(FEATURE_DIR))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--checkpoint-dir", default=str(CHECKPOINT_DIR))
    parser.add_argument("--run-name", default="e2_canonical_128_25")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_embedding_manifest(feature_dir):
    with open(feature_dir / "e2_embedding_manifest.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_split(feature_dir, split_name):
    path = feature_dir / f"{split_name}_embeddings.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run extract_e2_embeddings.py first")
    data = np.load(path, allow_pickle=False)
    image_embedding = data["image_embedding"].astype(np.float32)
    iot_embedding = data["iot_embedding"].astype(np.float32)
    x = np.concatenate([image_embedding, iot_embedding], axis=1).astype(np.float32)
    y = data["y"].astype(np.int64)
    meta = pd.DataFrame(
        {
            "image": data["image_id"].astype(str),
            "label": data["label"].astype(str),
            "stage": data["stage"].astype(str),
            "label_idx": y,
        }
    )

    expected = EXPECTED_SPLIT_COUNTS[split_name]
    if len(y) != expected:
        raise RuntimeError(f"{split_name} row count {len(y)} != expected {expected}")
    if x.shape[1] != 1344:
        raise RuntimeError(f"{split_name} input dimension {x.shape[1]} != expected 1344")
    if np.isnan(x).any():
        raise RuntimeError(f"{split_name} embeddings contain NaN values")

    return x, y, meta


def make_loader(x, y, batch_size, shuffle, num_workers):
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        total_count += len(y)
    return total_loss / max(total_count, 1)


def predict(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_targets = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            total_loss += loss.item() * len(y)
            total_count += len(y)
            all_targets.append(y.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    return {
        "loss": total_loss / max(total_count, 1),
        "targets": np.concatenate(all_targets),
        "preds": np.concatenate(all_preds),
        "probs": np.concatenate(all_probs),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    feature_dir = Path(args.feature_dir)
    results_dir = Path(args.results_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "confusion_matrices").mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_embedding_manifest(feature_dir)
    classes = manifest["classes"]
    if manifest["image_embedding_dim"] != 1280 or manifest["iot_embedding_dim"] != 64:
        raise RuntimeError("Embedding manifest dimensions do not match E2 interface")

    x_train, y_train, _ = load_split(feature_dir, "train")
    x_val, y_val, val_meta = load_split(feature_dir, "val")
    x_test, y_test, test_meta = load_split(feature_dir, "test")

    class_counts = np.bincount(y_train, minlength=len(classes))
    class_weights = len(y_train) / (len(classes) * np.maximum(class_counts, 1))
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader = make_loader(x_train, y_train, args.batch_size, True, args.num_workers)
    val_loader = make_loader(x_val, y_val, args.batch_size, False, args.num_workers)
    test_loader = make_loader(x_test, y_test, args.batch_size, False, args.num_workers)

    model = E2Canonical128To25Head(input_dim=x_train.shape[1], num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )

    checkpoint_path = checkpoint_dir / f"best_{args.run_name}.pt"
    best_val_macro_f1 = -1.0
    best_epoch = 0
    patience_count = 0
    history = []

    print(f"Training device: {device}")
    print(f"Train/val/test rows: {len(y_train)}/{len(y_val)}/{len(y_test)}")
    print(f"Trainable parameters: {count_trainable_params(model)}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_output = predict(model, val_loader, criterion, device)
        val_metrics = metrics_dict(val_output, classes)

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_early_stage_recall": val_metrics["early_stage_recall"],
            "val_ece": val_metrics["ece"],
        }
        history.append(row)
        print(
            "epoch %03d train_loss=%.4f val_loss=%.4f val_acc=%.4f val_macro_f1=%.4f val_macro_recall=%.4f val_early=%.4f val_ece=%.5f"
            % (
                epoch,
                train_loss,
                val_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["macro_f1"],
                val_metrics["macro_recall"],
                val_metrics["early_stage_recall"],
                val_metrics["ece"],
            )
        )

        scheduler.step(val_metrics["macro_f1"])
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_count = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_count += 1
        if patience_count >= args.patience:
            print(f"early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    val_output = predict(model, val_loader, criterion, device)
    test_output = predict(model, test_loader, criterion, device)
    val_metrics = metrics_dict(val_output, classes)
    test_metrics = metrics_dict(test_output, classes)
    test_metrics["per_class"] = per_class_metrics(test_output, classes)

    metrics = {
        "architecture": "e2_canonical_128_25",
        "seed": args.seed,
        "run_name": args.run_name,
        "best_epoch": best_epoch,
        "device": str(device),
        "input_dim": x_train.shape[1],
        "image_embedding_dim": manifest["image_embedding_dim"],
        "iot_embedding_dim": manifest["iot_embedding_dim"],
        "num_classes": len(classes),
        "fusion_head": "Linear(1344,128)-BatchNorm-ReLU-Dropout(0.1)-Linear(128,25)",
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "optimizer": "Adam",
        "loss": "weighted_cross_entropy",
        "scheduler": "ReduceLROnPlateau",
        "scheduler_monitor": "validation_macro_f1",
        "train_rows": len(y_train),
        "val_rows": len(y_val),
        "test_rows": len(y_test),
        "trainable_params": count_trainable_params(model),
        "e0_checkpoint": manifest["e0_checkpoint"],
        "e1_checkpoint": manifest["e1_checkpoint"],
        "val": val_metrics,
        "test": test_metrics,
        "classes": classes,
    }

    pd.DataFrame(history).to_csv(results_dir / f"{args.run_name}_history.csv", index=False)
    save_predictions(results_dir / f"{args.run_name}_val_predictions.csv", val_meta, val_output, classes)
    save_predictions(results_dir / f"{args.run_name}_test_predictions.csv", test_meta, test_output, classes)
    save_confusion_matrix(
        results_dir / "confusion_matrices" / f"{args.run_name}_test_confusion_matrix.csv",
        test_output,
        classes,
    )
    with open(results_dir / f"{args.run_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"saved metrics to {results_dir / f'{args.run_name}_metrics.json'}")


if __name__ == "__main__":
    main()

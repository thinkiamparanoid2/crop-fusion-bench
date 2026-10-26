import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

E1_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = E1_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.models import E1MLP
from common.thesis_eval import (
    count_trainable_params,
    metrics_dict,
    save_confusion_matrix,
)

FEATURE_DIR = E1_ROOT / "features"
SPLIT_DIR = PROJECT_ROOT / "Data"
RESULTS_DIR = E1_ROOT / "results"
CHECKPOINT_DIR = E1_ROOT / "checkpoints"

META_COLS = ["image", "label", "crop", "disease", "stage", "n_rows"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", default=str(FEATURE_DIR))
    parser.add_argument("--split-dir", default=str(SPLIT_DIR))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--checkpoint-dir", default=str(CHECKPOINT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=32)
    parser.add_argument("--dropout1", type=float, default=0.2)
    parser.add_argument("--dropout2", type=float, default=0.1)
    parser.add_argument("--scheduler", action="store_true")
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


def load_csv(path):
    return pd.read_csv(path, dtype={"image": str, "label": str})


def select_by_split(features, split_manifest, split_name):
    image_order = split_manifest["image"].astype(str).tolist()
    selected = features.set_index("image").loc[image_order].reset_index()

    expected = split_manifest["label"].astype(str).tolist()
    actual = selected["label"].astype(str).tolist()
    if actual != expected:
        raise RuntimeError("%s labels do not match Abrar split CSV" % split_name)

    return selected


def make_loader(x, y, batch_size, shuffle, num_workers):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(x_tensor, y_tensor)
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
        x = x.to(device)
        y = y.to(device)

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
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            total_loss += loss.item() * len(y)
            total_count += len(y)
            all_targets.extend(y.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    return {
        "loss": total_loss / max(total_count, 1),
        "targets": np.array(all_targets),
        "preds": np.array(all_preds),
        "probs": np.array(all_probs),
    }


def save_predictions(path, split_df, eval_output, label_encoder):
    out = split_df[["image", "label", "crop", "disease", "stage"]].copy()
    out["pred_label"] = label_encoder.inverse_transform(eval_output["preds"])
    out["correct"] = out["label"] == out["pred_label"]
    out["confidence"] = eval_output["probs"].max(axis=1)
    out.to_csv(path, index=False)


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.run_name is None:
        args.run_name = "seed_%d_h%d_%d_bs%d_lr%s" % (
            args.seed,
            args.hidden1,
            args.hidden2,
            args.batch_size,
            str(args.lr).replace(".", "p"),
        )

    feature_dir = Path(args.feature_dir)
    split_dir = Path(args.split_dir)
    results_dir = Path(args.results_dir)
    checkpoint_dir = Path(args.checkpoint_dir)

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "confusion_matrices").mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    all_train_features = load_csv(feature_dir / "train_iot_features.csv")
    all_test_features = load_csv(feature_dir / "test_iot_features.csv")

    abrar_train = load_csv(split_dir / "train.csv")
    abrar_val = load_csv(split_dir / "val.csv")
    abrar_test = load_csv(split_dir / "test.csv")

    train_df = select_by_split(all_train_features, abrar_train, "train")
    val_df = select_by_split(all_train_features, abrar_val, "val")
    test_df = select_by_split(all_test_features, abrar_test, "test")

    feature_cols = [c for c in train_df.columns if c not in META_COLS]

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["label"])

    if set(val_df["label"]) - set(label_encoder.classes_):
        raise RuntimeError("Validation split contains labels not present in train split")
    if set(test_df["label"]) - set(label_encoder.classes_):
        raise RuntimeError("Test split contains labels not present in train split")

    classes = label_encoder.classes_.tolist()

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(train_df[feature_cols])
    x_val = scaler.transform(val_df[feature_cols])
    x_test = scaler.transform(test_df[feature_cols])

    y_train = label_encoder.transform(train_df["label"])
    y_val = label_encoder.transform(val_df["label"])
    y_test = label_encoder.transform(test_df["label"])

    class_counts = np.bincount(y_train, minlength=len(label_encoder.classes_))
    class_weights = len(y_train) / (len(label_encoder.classes_) * np.maximum(class_counts, 1))
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_loader = make_loader(x_train, y_train, args.batch_size, True, args.num_workers)
    val_loader = make_loader(x_val, y_val, args.batch_size, False, args.num_workers)
    test_loader = make_loader(x_test, y_test, args.batch_size, False, args.num_workers)

    model = E1MLP(
        input_dim=len(feature_cols),
        num_classes=len(label_encoder.classes_),
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        dropout1=args.dropout1,
        dropout2=args.dropout2,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
        )

    best_val_macro_f1 = -1.0
    best_epoch = 0
    patience_count = 0
    history = []
    checkpoint_path = checkpoint_dir / ("best_%s.pt" % args.run_name)

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
            "epoch %03d train_loss=%.4f val_loss=%.4f val_acc=%.4f val_macro_f1=%.4f"
            % (epoch, train_loss, val_metrics["loss"], val_metrics["accuracy"], val_metrics["macro_f1"])
        )

        if scheduler is not None:
            scheduler.step(val_metrics["macro_f1"])

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_count = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_count += 1

        if patience_count >= args.patience:
            print("early stopping at epoch %d" % epoch)
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    val_output = predict(model, val_loader, criterion, device)
    test_output = predict(model, test_loader, criterion, device)
    val_metrics = metrics_dict(val_output, classes)
    test_metrics = metrics_dict(test_output, classes)

    metrics = {
        "seed": args.seed,
        "run_name": args.run_name,
        "best_epoch": best_epoch,
        "device": str(device),
        "input_dim": len(feature_cols),
        "num_classes": len(label_encoder.classes_),
        "hidden1": args.hidden1,
        "hidden2": args.hidden2,
        "dropout1": args.dropout1,
        "dropout2": args.dropout2,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": bool(args.scheduler),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "trainable_params": count_trainable_params(model),
        "val": val_metrics,
        "test": test_metrics,
        "classes": label_encoder.classes_.tolist(),
    }

    pd.DataFrame(history).to_csv(results_dir / ("%s_history.csv" % args.run_name), index=False)
    save_predictions(results_dir / ("%s_val_predictions.csv" % args.run_name), val_df, val_output, label_encoder)
    save_predictions(results_dir / ("%s_test_predictions.csv" % args.run_name), test_df, test_output, label_encoder)

    save_confusion_matrix(
        results_dir / "confusion_matrices" / ("%s_test_confusion_matrix.csv" % args.run_name),
        test_output,
        classes,
    )

    with open(results_dir / ("%s_metrics.json" % args.run_name), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("saved metrics to %s" % (results_dir / ("%s_metrics.json" % args.run_name)))


if __name__ == "__main__":
    main()

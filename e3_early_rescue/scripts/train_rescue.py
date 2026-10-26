"""Early-stage recall rescue experiments for E2-matched / E3-4.

This script intentionally writes to e3_early_rescue/ so the completed baseline
E3 artifacts remain untouched.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

RESCUE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RESCUE_ROOT.parent
E2_SCRIPTS = PROJECT_ROOT / "e2_multimodal_fusion" / "scripts"
E3_ROOT = PROJECT_ROOT / "e3_quantum_fusion"
for _p in (PROJECT_ROOT, E2_SCRIPTS, E3_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common.thesis_eval import metrics_dict, per_class_metrics, save_confusion_matrix, save_predictions  # noqa: E402
from models import BOTTLENECK, NUM_CLASSES, E2MatchedHead, E3HybridHead, count_param_split  # noqa: E402
from train_e2_fusion import load_embedding_manifest, load_split, make_loader, predict, set_seed  # noqa: E402

FEATURE_DIR = PROJECT_ROOT / "e2_multimodal_fusion" / "features"
RESULTS_DIR = RESCUE_ROOT / "results"
CHECKPOINT_DIR = RESCUE_ROOT / "checkpoints"

ARCH_DESC = {
    "e2_matched": "Linear(1344,128)-BN-ReLU-Dropout(0.1)-Linear(128,4)-tanh-Linear(4,4,bias=False)-tanh-Linear(4,25)",
    "e3_4": "Linear(1344,128)-BN-ReLU-Dropout(0.1)-Linear(128,4)-(tanh*pi)-4qubitVQC-Linear(4,25)",
}


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def parse_args():
    parser = argparse.ArgumentParser(description="Early-recall rescue training")
    parser.add_argument("--model", choices=["e2_matched", "e3_4"], required=True)
    parser.add_argument("--loss", choices=["weighted_ce", "focal", "weighted_focal"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--checkpoint-metric", choices=["val_early_stage_recall", "val_macro_f1"], default="val_early_stage_recall")
    parser.add_argument("--sim", default="lightning.qubit", help="PennyLane device for E3")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def make_criterion(loss_name, class_weights, device, gamma):
    weights = class_weights.to(device)
    if loss_name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=weights)
    if loss_name == "focal":
        return FocalLoss(gamma=gamma, weight=None)
    if loss_name == "weighted_focal":
        return FocalLoss(gamma=gamma, weight=weights)
    raise ValueError(loss_name)


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


def metric_value(metrics, checkpoint_metric):
    if checkpoint_metric == "val_early_stage_recall":
        return metrics["early_stage_recall"]
    if checkpoint_metric == "val_macro_f1":
        return metrics["macro_f1"]
    raise ValueError(checkpoint_metric)


def main():
    args = parse_args()
    set_seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "confusion_matrices").mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if args.model == "e3_4" or args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    manifest = load_embedding_manifest(FEATURE_DIR)
    classes = manifest["classes"]
    if manifest["image_embedding_dim"] != 1280 or manifest["iot_embedding_dim"] != 64:
        raise RuntimeError("Embedding manifest dimensions do not match the E2/E3 interface")
    if len(classes) != NUM_CLASSES:
        raise RuntimeError(f"Expected {NUM_CLASSES} classes, manifest has {len(classes)}")

    x_train, y_train, train_meta = load_split(FEATURE_DIR, "train")
    x_val, y_val, val_meta = load_split(FEATURE_DIR, "val")
    x_test, y_test, test_meta = load_split(FEATURE_DIR, "test")

    class_counts = np.bincount(y_train, minlength=len(classes))
    class_weights = len(y_train) / (len(classes) * np.maximum(class_counts, 1))
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    train_loader = make_loader(x_train, y_train, args.batch_size, True, 0)
    val_loader = make_loader(x_val, y_val, args.batch_size, False, 0)
    test_loader = make_loader(x_test, y_test, args.batch_size, False, 0)

    if args.model == "e2_matched":
        model = E2MatchedHead(num_classes=len(classes))
        qubits = None
    else:
        model = E3HybridHead(num_classes=len(classes), sim=args.sim)
        qubits = model.vqc.n_qubits
    model = model.to(device)

    criterion = make_criterion(args.loss, class_weights, device, args.gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )

    total_params, post_params = count_param_split(model)
    checkpoint_path = CHECKPOINT_DIR / f"best_{args.run_name}.pt"
    best_score = -1.0
    best_epoch = 0
    patience_count = 0
    history = []

    print(f"Model: {args.model}  loss: {args.loss}  device: {device}")
    print(f"Checkpoint metric: {args.checkpoint_metric}")
    print(f"Train/val/test rows: {len(y_train)}/{len(y_val)}/{len(y_test)}")
    print(f"Trainable params: total={total_params} post_bottleneck={post_params}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_output = predict(model, val_loader, criterion, device)
        val_metrics = metrics_dict(val_output, classes)
        score = metric_value(val_metrics, args.checkpoint_metric)

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_early_stage_recall": val_metrics["early_stage_recall"],
            "val_ece": val_metrics["ece"],
            "checkpoint_score": score,
        })
        print(
            "epoch %03d train_loss=%.4f val_loss=%.4f val_acc=%.4f val_macro_f1=%.4f val_macro_recall=%.4f val_early=%.4f val_ece=%.5f score=%.4f"
            % (
                epoch,
                train_loss,
                val_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["macro_f1"],
                val_metrics["macro_recall"],
                val_metrics["early_stage_recall"],
                val_metrics["ece"],
                score,
            )
        )

        scheduler.step(score)
        if score > best_score:
            best_score = score
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
        "experiment_group": "e3_early_rescue",
        "seed": args.seed,
        "run_name": args.run_name,
        "model": args.model,
        "architecture": ARCH_DESC[args.model],
        "qubits": qubits,
        "bottleneck_dim": BOTTLENECK,
        "best_epoch": best_epoch,
        "best_checkpoint_score": best_score,
        "checkpoint_metric": args.checkpoint_metric,
        "device": str(device),
        "sim": args.sim if args.model == "e3_4" else None,
        "input_dim": x_train.shape[1],
        "image_embedding_dim": manifest["image_embedding_dim"],
        "iot_embedding_dim": manifest["iot_embedding_dim"],
        "num_classes": len(classes),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "optimizer": "Adam",
        "loss": args.loss,
        "focal_gamma": args.gamma if args.loss in {"focal", "weighted_focal"} else None,
        "scheduler": "ReduceLROnPlateau",
        "scheduler_monitor": args.checkpoint_metric,
        "train_rows": len(y_train),
        "val_rows": len(y_val),
        "test_rows": len(y_test),
        "total_trainable_params": total_params,
        "post_bottleneck_trainable_params": post_params,
        "e0_checkpoint": manifest["e0_checkpoint"],
        "e1_checkpoint": manifest["e1_checkpoint"],
        "val": val_metrics,
        "test": test_metrics,
        "classes": classes,
    }

    pd.DataFrame(history).to_csv(RESULTS_DIR / f"{args.run_name}_history.csv", index=False)
    save_predictions(RESULTS_DIR / f"{args.run_name}_val_predictions.csv", val_meta, val_output, classes)
    save_predictions(RESULTS_DIR / f"{args.run_name}_test_predictions.csv", test_meta, test_output, classes)
    save_confusion_matrix(
        RESULTS_DIR / "confusion_matrices" / f"{args.run_name}_test_confusion_matrix.csv",
        test_output,
        classes,
    )
    with open(RESULTS_DIR / f"{args.run_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"saved metrics to {RESULTS_DIR / f'{args.run_name}_metrics.json'}")
    print(
        "best_epoch=%d val_early=%.4f val_macro_f1=%.4f test_early=%.4f test_macro_f1=%.4f test_acc=%.4f"
        % (
            best_epoch,
            val_metrics["early_stage_recall"],
            val_metrics["macro_f1"],
            test_metrics["early_stage_recall"],
            test_metrics["macro_f1"],
            test_metrics["accuracy"],
        )
    )


if __name__ == "__main__":
    main()


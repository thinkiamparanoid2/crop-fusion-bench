"""Train bottleneck-k E2-matched and E3 VQC heads on cached E2 embeddings."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SWEEP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SWEEP_ROOT.parent
E2_SCRIPTS = PROJECT_ROOT / "e2_multimodal_fusion" / "scripts"
E3_ROOT = PROJECT_ROOT / "e3_quantum_fusion"
for _p in (PROJECT_ROOT, E2_SCRIPTS, E3_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common.thesis_eval import metrics_dict, per_class_metrics, save_confusion_matrix, save_predictions  # noqa: E402
from quantum_layer import VQCLayer  # noqa: E402
from train_e2_fusion import load_embedding_manifest, load_split, make_loader, predict, set_seed, train_one_epoch  # noqa: E402

FUSED_DIM = 1344
HIDDEN = 128
NUM_CLASSES = 25
ANGLE_SCALE = 3.141592653589793
FEATURE_DIR = PROJECT_ROOT / "e2_multimodal_fusion" / "features"
RESULTS_DIR = SWEEP_ROOT / "results"
CHECKPOINT_DIR = SWEEP_ROOT / "checkpoints"


class FusionProjectionK(nn.Module):
    def __init__(self, bottleneck, fused_dim=FUSED_DIM, hidden=HIDDEN, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, bottleneck),
        )

    def forward(self, x):
        return self.net(x)


class E2MatchedK(nn.Module):
    def __init__(self, bottleneck, num_classes=NUM_CLASSES):
        super().__init__()
        self.bottleneck = bottleneck
        self.projection = FusionProjectionK(bottleneck)
        self.middle = nn.Linear(bottleneck, bottleneck, bias=False)
        self.readout = nn.Linear(bottleneck, num_classes)

    def forward(self, x):
        z = torch.tanh(self.projection(x))
        z = torch.tanh(self.middle(z))
        return self.readout(z)

    def post_bottleneck_parameters(self):
        return self.middle.parameters()


class E3VQCK(nn.Module):
    def __init__(self, bottleneck, vqc_layers, num_classes=NUM_CLASSES, sim="lightning.qubit"):
        super().__init__()
        self.bottleneck = bottleneck
        self.vqc_layers = vqc_layers
        self.projection = FusionProjectionK(bottleneck)
        self.vqc = VQCLayer(n_qubits=bottleneck, n_layers=vqc_layers, sim=sim)
        self.readout = nn.Linear(bottleneck, num_classes)

    def forward(self, x):
        z = ANGLE_SCALE * torch.tanh(self.projection(x))
        z = self.vqc(z)
        return self.readout(z)

    def post_bottleneck_parameters(self):
        return self.vqc.parameters()


def count_param_split(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    post = sum(p.numel() for p in model.post_bottleneck_parameters() if p.requires_grad)
    return total, post


def parse_args():
    parser = argparse.ArgumentParser(description="Train bottleneck-k E2-matched / E3 VQC")
    parser.add_argument("--model", choices=["e2_matched", "e3_vqc"], required=True)
    parser.add_argument("--bottleneck", type=int, default=8)
    parser.add_argument("--vqc-layers", type=int, default=None)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--sim", default="lightning.qubit")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def default_vqc_layers_for_match(bottleneck):
    # Match Linear(k,k,bias=False): k*k params. VQC has layers*k*2 params.
    # layers = k/2 gives exact match for even k.
    return max(1, bottleneck // 2)


def main():
    args = parse_args()
    set_seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "confusion_matrices").mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    vqc_layers = args.vqc_layers or default_vqc_layers_for_match(args.bottleneck)
    if args.model == "e3_vqc" or args.cpu or not torch.cuda.is_available():
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
        model = E2MatchedK(args.bottleneck, num_classes=len(classes))
        qubits = None
        architecture = f"Linear(1344,128)-BN-ReLU-Dropout(0.1)-Linear(128,{args.bottleneck})-tanh-Linear({args.bottleneck},{args.bottleneck},bias=False)-tanh-Linear({args.bottleneck},25)"
        sim = None
    else:
        model = E3VQCK(args.bottleneck, vqc_layers, num_classes=len(classes), sim=args.sim)
        qubits = args.bottleneck
        architecture = f"Linear(1344,128)-BN-ReLU-Dropout(0.1)-Linear(128,{args.bottleneck})-(tanh*pi)-{args.bottleneck}qubitVQC({vqc_layers} layers)-Linear({args.bottleneck},25)"
        sim = args.sim
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )

    total_params, post_params = count_param_split(model)
    checkpoint_path = CHECKPOINT_DIR / f"best_{args.run_name}.pt"
    best_val_macro_f1 = -1.0
    best_epoch = 0
    patience_count = 0
    history = []

    print(f"Model: {args.model}  bottleneck: {args.bottleneck}  device: {device}")
    print(f"VQC layers: {vqc_layers if args.model == 'e3_vqc' else 'n/a'}")
    print(f"Train/val/test rows: {len(y_train)}/{len(y_val)}/{len(y_test)}")
    print(f"Trainable params: total={total_params} post_bottleneck={post_params}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_output = predict(model, val_loader, criterion, device)
        val_metrics = metrics_dict(val_output, classes)

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_early_stage_recall": val_metrics["early_stage_recall"],
            "val_mid_stage_recall": val_metrics["per_stage_recall"]["2"],
            "val_severe_stage_recall": val_metrics["per_stage_recall"]["3"],
            "val_ece": val_metrics["ece"],
        })
        print(
            "epoch %03d train_loss=%.4f val_loss=%.4f val_acc=%.4f val_macro_f1=%.4f val_macro_recall=%.4f val_early=%.4f val_mid=%.4f val_severe=%.4f val_ece=%.5f"
            % (
                epoch,
                train_loss,
                val_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["macro_f1"],
                val_metrics["macro_recall"],
                val_metrics["early_stage_recall"],
                val_metrics["per_stage_recall"]["2"],
                val_metrics["per_stage_recall"]["3"],
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
        "experiment_group": "e3_bottleneck_sweep",
        "seed": args.seed,
        "run_name": args.run_name,
        "model": args.model,
        "architecture": architecture,
        "qubits": qubits,
        "vqc_layers": vqc_layers if args.model == "e3_vqc" else None,
        "bottleneck_dim": args.bottleneck,
        "best_epoch": best_epoch,
        "device": str(device),
        "sim": sim,
        "input_dim": x_train.shape[1],
        "image_embedding_dim": manifest["image_embedding_dim"],
        "iot_embedding_dim": manifest["iot_embedding_dim"],
        "num_classes": len(classes),
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
        "best_epoch=%d val_macro_f1=%.4f test_macro_f1=%.4f test_acc=%.4f test_early=%.4f test_mid=%.4f test_severe=%.4f"
        % (
            best_epoch,
            val_metrics["macro_f1"],
            test_metrics["macro_f1"],
            test_metrics["accuracy"],
            test_metrics["early_stage_recall"],
            test_metrics["per_stage_recall"]["2"],
            test_metrics["per_stage_recall"]["3"],
        )
    )


if __name__ == "__main__":
    main()


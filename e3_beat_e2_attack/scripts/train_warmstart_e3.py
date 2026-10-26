"""Warm-start E3-8 from E2-matched-8 and train the VQC block first.

This is Phase 1 of the E3 beat-E2 attack plan:
1. Copy E2 projection/readout weights into E3.
2. Train only the VQC block.
3. Joint fine-tune projection/readout and VQC with split learning rates.
4. Save dual checkpoints: best validation Macro F1 and best validation mid recall.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ATTACK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ATTACK_ROOT.parent
SWEEP_ROOT = PROJECT_ROOT / "e3_bottleneck_sweep"
E2_SCRIPTS = PROJECT_ROOT / "e2_multimodal_fusion" / "scripts"
E3_ROOT = PROJECT_ROOT / "e3_quantum_fusion"
for _p in (PROJECT_ROOT, SWEEP_ROOT / "scripts", E2_SCRIPTS, E3_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common.thesis_eval import metrics_dict, per_class_metrics, save_confusion_matrix, save_predictions  # noqa: E402
from train_bottleneck import E2MatchedK, E3VQCK, count_param_split  # noqa: E402
from train_e2_fusion import load_embedding_manifest, load_split, make_loader, predict, set_seed  # noqa: E402

FEATURE_DIR = PROJECT_ROOT / "e2_multimodal_fusion" / "features"
SOURCE_CHECKPOINT_DIR = SWEEP_ROOT / "checkpoints"
RESULTS_DIR = ATTACK_ROOT / "results"
CHECKPOINT_DIR = ATTACK_ROOT / "checkpoints"
BOTTLENECK = 8
VQC_LAYERS = 4
NUM_CLASSES = 25
E2_MACRO_F1_BAR = 0.973356
ANGLE_SCALE = 3.141592653589793


def parse_args():
    parser = argparse.ArgumentParser(description="Warm-start E2/E3 attack heads from E2-matched-8")
    parser.add_argument("--model", choices=["e3_vqc", "e2_matched"], default="e3_vqc")
    parser.add_argument("--run-name", default="e3_8_warm_s42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--vqc-only-epochs", type=int, default=15)
    parser.add_argument("--joint-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--vqc-only-lr", type=float, default=5e-3)
    parser.add_argument("--joint-vqc-lr", type=float, default=1e-3)
    parser.add_argument("--joint-head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--sim", default="lightning.qubit")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--min-patience", type=int, default=20)
    parser.add_argument("--early-stop-warmup", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--progress-batches", type=int, default=25)
    parser.add_argument("--pair-loss", action="store_true")
    parser.add_argument("--pair-true-label", default="3_a9_2")
    parser.add_argument("--pair-confuser-label", default="3_a9_1")
    parser.add_argument("--pair-lambda", type=float, default=2.0)
    parser.add_argument("--pair-margin-lambda", type=float, default=1.0)
    parser.add_argument("--pair-margin", type=float, default=0.5)
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--alpha-init", type=float, default=0.1)
    parser.add_argument("--alpha-bounded", action="store_true")
    parser.add_argument("--alpha-max", type=float, default=1.0)
    return parser.parse_args()


def alpha_parameter_init(alpha_init, alpha_bounded, alpha_max):
    if not alpha_bounded:
        return float(alpha_init)
    if alpha_max <= 0:
        raise ValueError("--alpha-max must be positive when --alpha-bounded is enabled")
    ratio = max(min(float(alpha_init) / float(alpha_max), 0.999), -0.999)
    return math.atanh(ratio)


def effective_alpha(alpha_param, alpha_bounded, alpha_max):
    if alpha_bounded:
        return float(alpha_max) * torch.tanh(alpha_param)
    return alpha_param


class E2MatchedKResidual(nn.Module):
    def __init__(
        self,
        bottleneck,
        num_classes=NUM_CLASSES,
        alpha_init=0.1,
        alpha_bounded=False,
        alpha_max=1.0,
    ):
        super().__init__()
        base = E2MatchedK(bottleneck, num_classes=num_classes)
        self.bottleneck = bottleneck
        self.projection = base.projection
        self.middle = base.middle
        self.readout = base.readout
        self.alpha_bounded = bool(alpha_bounded)
        self.alpha_max = float(alpha_max)
        init = alpha_parameter_init(alpha_init, self.alpha_bounded, self.alpha_max)
        self.alpha = nn.Parameter(torch.tensor(init, dtype=torch.float32))

    def effective_alpha(self):
        return effective_alpha(self.alpha, self.alpha_bounded, self.alpha_max)

    def forward(self, x):
        z = torch.tanh(self.projection(x))
        delta = self.middle(z)
        z = z + self.effective_alpha() * delta
        return self.readout(z)

    def post_bottleneck_parameters(self):
        yield from self.middle.parameters()
        yield self.alpha


class E3VQCKResidual(nn.Module):
    def __init__(
        self,
        bottleneck,
        vqc_layers,
        num_classes=NUM_CLASSES,
        sim="lightning.qubit",
        alpha_init=0.1,
        alpha_bounded=False,
        alpha_max=1.0,
    ):
        super().__init__()
        base = E3VQCK(bottleneck, vqc_layers, num_classes=num_classes, sim=sim)
        self.bottleneck = bottleneck
        self.vqc_layers = vqc_layers
        self.projection = base.projection
        self.vqc = base.vqc
        self.readout = base.readout
        self.alpha_bounded = bool(alpha_bounded)
        self.alpha_max = float(alpha_max)
        init = alpha_parameter_init(alpha_init, self.alpha_bounded, self.alpha_max)
        self.alpha = nn.Parameter(torch.tensor(init, dtype=torch.float32))

    def effective_alpha(self):
        return effective_alpha(self.alpha, self.alpha_bounded, self.alpha_max)

    def forward(self, x):
        z = torch.tanh(self.projection(x))
        q = self.vqc(ANGLE_SCALE * z)
        z = z + self.effective_alpha() * q
        return self.readout(z)

    def post_bottleneck_parameters(self):
        yield from self.vqc.parameters()
        yield self.alpha


class PairTargetedLoss(nn.Module):
    def __init__(
        self,
        class_weights,
        pair_true_idx=None,
        pair_confuser_idx=None,
        pair_lambda=0.0,
        pair_margin_lambda=0.0,
        pair_margin=0.5,
    ):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.pair_true_idx = pair_true_idx
        self.pair_confuser_idx = pair_confuser_idx
        self.pair_lambda = pair_lambda
        self.pair_margin_lambda = pair_margin_lambda
        self.pair_margin = pair_margin

    def forward(self, logits, targets):
        loss = F.cross_entropy(logits, targets, weight=self.class_weights)
        if self.pair_true_idx is None or self.pair_confuser_idx is None:
            return loss
        pair_mask = targets == self.pair_true_idx
        if not pair_mask.any():
            return loss
        pair_logits = logits[pair_mask]
        pair_targets = targets[pair_mask]
        pair_ce = F.cross_entropy(pair_logits, pair_targets, weight=self.class_weights)
        true_logits = pair_logits[:, self.pair_true_idx]
        confuser_logits = pair_logits[:, self.pair_confuser_idx]
        pair_margin = F.relu(confuser_logits - true_logits + self.pair_margin).mean()
        return loss + self.pair_lambda * pair_ce + self.pair_margin_lambda * pair_margin


def set_requires_grad(module, enabled):
    for param in module.parameters():
        param.requires_grad = enabled


def transplant_projection_readout(model):
    e2 = E2MatchedK(BOTTLENECK, num_classes=NUM_CLASSES)
    e2_state = torch.load(SOURCE_CHECKPOINT_DIR / "best_e2_matched_8.pt", map_location="cpu")
    e2.load_state_dict(e2_state)
    model.projection.load_state_dict(e2.projection.state_dict())
    model.readout.load_state_dict(e2.readout.state_dict())


def build_model(args, num_classes):
    if args.model == "e2_matched":
        if args.residual:
            return E2MatchedKResidual(
                BOTTLENECK,
                num_classes=num_classes,
                alpha_init=args.alpha_init,
                alpha_bounded=args.alpha_bounded,
                alpha_max=args.alpha_max,
            )
        return E2MatchedK(BOTTLENECK, num_classes=num_classes)
    if args.residual:
        return E3VQCKResidual(
            BOTTLENECK,
            VQC_LAYERS,
            num_classes=num_classes,
            sim=args.sim,
            alpha_init=args.alpha_init,
            alpha_bounded=args.alpha_bounded,
            alpha_max=args.alpha_max,
        )
    return E3VQCK(BOTTLENECK, VQC_LAYERS, num_classes=num_classes, sim=args.sim)


def middle_module(model):
    if hasattr(model, "vqc"):
        return model.vqc
    return model.middle


def middle_parameters(model):
    params = list(middle_module(model).parameters())
    if hasattr(model, "alpha"):
        params.append(model.alpha)
    return params


def alpha_value(model):
    if not hasattr(model, "alpha"):
        return None
    if hasattr(model, "effective_alpha"):
        return float(model.effective_alpha().detach().cpu().item())
    return float(model.alpha.detach().cpu().item())


def alpha_raw_value(model):
    if not hasattr(model, "alpha"):
        return None
    return float(model.alpha.detach().cpu().item())


def train_one_epoch_attack(
    model,
    loader,
    criterion,
    optimizer,
    device,
    freeze_projection_bn=False,
    max_batches=None,
    progress_batches=25,
    phase="train",
    epoch=None,
):
    model.train()
    if freeze_projection_bn:
        model.projection.eval()
        model.readout.eval()
    total_loss = 0.0
    total_count = 0
    total_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        total_count += len(y)
        if progress_batches and ((batch_idx + 1) % progress_batches == 0 or (batch_idx + 1) == total_batches):
            prefix = f"{phase} epoch={epoch:03d}" if epoch is not None else phase
            print(
                f"{prefix} batch={batch_idx + 1}/{total_batches} running_loss={total_loss / max(total_count, 1):.4f}",
                flush=True,
            )
    return total_loss / max(total_count, 1)


def predict_limited(model, loader, criterion, device, max_batches=None, progress_batches=0, phase="eval", epoch=None):
    if max_batches is None:
        if not progress_batches:
            return predict(model, loader, criterion, device)
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_targets = []
    all_preds = []
    all_probs = []
    total_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
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
            if progress_batches and ((batch_idx + 1) % progress_batches == 0 or (batch_idx + 1) == total_batches):
                prefix = f"{phase} epoch={epoch:03d}" if epoch is not None else phase
                print(
                    f"{prefix} batch={batch_idx + 1}/{total_batches} running_loss={total_loss / max(total_count, 1):.4f}",
                    flush=True,
                )
    return {
        "loss": total_loss / max(total_count, 1),
        "targets": np.concatenate(all_targets),
        "preds": np.concatenate(all_preds),
        "probs": np.concatenate(all_probs),
    }


def save_checkpoint(path, model, meta):
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint_state(path, device):
    obj = torch.load(path, map_location=device)
    return obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj


def append_history(history, phase, epoch, train_loss, val_metrics, optimizer):
    history.append({
        "phase": phase,
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


def metric_float(value):
    return float("nan") if value is None else float(value)


def evaluate_and_save(run_name, checkpoint_kind, model, checkpoint_path, loaders, metas, criterion, device, classes, args, history):
    model.load_state_dict(load_checkpoint_state(checkpoint_path, device))
    val_output = predict_limited(model, loaders["val"], criterion.to(device), device, args.max_eval_batches)
    test_output = predict_limited(model, loaders["test"], criterion.to(device), device, args.max_eval_batches)
    val_meta = metas["val"].iloc[: len(val_output["targets"])].copy()
    test_meta = metas["test"].iloc[: len(test_output["targets"])].copy()
    val_metrics = metrics_dict(val_output, classes)
    test_metrics = metrics_dict(test_output, classes)
    test_metrics["per_class"] = per_class_metrics(test_output, classes)
    total_params, post_params = count_param_split(model)

    result_name = f"{run_name}_{checkpoint_kind}"
    save_predictions(RESULTS_DIR / f"{result_name}_val_predictions.csv", val_meta, val_output, classes)
    save_predictions(RESULTS_DIR / f"{result_name}_test_predictions.csv", test_meta, test_output, classes)
    save_confusion_matrix(RESULTS_DIR / "confusion_matrices" / f"{result_name}_test_confusion_matrix.csv", test_output, classes)

    metrics = {
        "experiment_group": "e3_beat_e2_attack",
        "run_name": run_name,
        "checkpoint_kind": checkpoint_kind,
        "seed": args.seed,
        "model": args.model,
        "residual": args.residual,
        "alpha_init": args.alpha_init if args.residual else None,
        "alpha_bounded": args.alpha_bounded if args.residual else None,
        "alpha_max": args.alpha_max if args.residual else None,
        "alpha_final": alpha_value(model),
        "alpha_raw_final": alpha_raw_value(model),
        "warm_start_source": str(SOURCE_CHECKPOINT_DIR / "best_e2_matched_8.pt"),
        "transplanted_modules": ["projection", "readout"],
        "bottleneck_dim": BOTTLENECK,
        "qubits": BOTTLENECK if args.model == "e3_vqc" else None,
        "vqc_layers": VQC_LAYERS if args.model == "e3_vqc" else None,
        "sim": args.sim if args.model == "e3_vqc" else None,
        "device": str(device),
        "batch_size": args.batch_size,
        "vqc_only_epochs": args.vqc_only_epochs,
        "joint_epochs": args.joint_epochs,
        "patience": args.patience,
        "effective_patience": max(args.patience, args.min_patience),
        "early_stop_warmup": args.early_stop_warmup,
        "vqc_only_lr": args.vqc_only_lr,
        "joint_vqc_lr": args.joint_vqc_lr,
        "joint_head_lr": args.joint_head_lr,
        "weight_decay": args.weight_decay,
        "optimizer": "AdamW",
        "loss": "weighted_cross_entropy_pair_targeted" if args.pair_loss else "weighted_cross_entropy",
        "pair_loss": args.pair_loss,
        "pair_true_label": args.pair_true_label if args.pair_loss else None,
        "pair_confuser_label": args.pair_confuser_label if args.pair_loss else None,
        "pair_lambda": args.pair_lambda if args.pair_loss else None,
        "pair_margin_lambda": args.pair_margin_lambda if args.pair_loss else None,
        "pair_margin": args.pair_margin if args.pair_loss else None,
        "dual_checkpoint_rule": "best_val_macro_f1 and best_val_mid_stage_recall",
        "e2_matched_8_test_macro_f1_bar": E2_MACRO_F1_BAR,
        "beats_e2_macro_f1_bar": bool(test_metrics["macro_f1"] > E2_MACRO_F1_BAR),
        "total_trainable_params": total_params,
        "post_bottleneck_trainable_params": post_params,
        "val": val_metrics,
        "test": test_metrics,
        "classes": classes,
    }
    with open(RESULTS_DIR / f"{result_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "confusion_matrices").mkdir(parents=True, exist_ok=True)

    if args.model == "e3_vqc" or args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    effective_patience = max(args.patience, args.min_patience)
    print(f"Run: {args.run_name}", flush=True)
    print(f"Model: {args.model}  residual={args.residual}", flush=True)
    print(f"Device: {device}  sim={args.sim if args.model == 'e3_vqc' else 'n/a'}", flush=True)
    print(
        f"Early stop: requested_patience={args.patience} effective_patience={effective_patience} "
        f"warmup_global_epochs={args.early_stop_warmup}",
        flush=True,
    )
    print("Loading frozen E2 embeddings...", flush=True)
    manifest = load_embedding_manifest(FEATURE_DIR)
    classes = manifest["classes"]
    if manifest["image_embedding_dim"] != 1280 or manifest["iot_embedding_dim"] != 64:
        raise RuntimeError("Embedding manifest dimensions do not match the E2/E3 interface")
    if len(classes) != NUM_CLASSES:
        raise RuntimeError(f"Expected {NUM_CLASSES} classes, found {len(classes)}")

    x_train, y_train, train_meta = load_split(FEATURE_DIR, "train")
    x_val, y_val, val_meta = load_split(FEATURE_DIR, "val")
    x_test, y_test, test_meta = load_split(FEATURE_DIR, "test")
    print(f"Loaded train/val/test rows: {len(y_train)}/{len(y_val)}/{len(y_test)}", flush=True)
    class_counts = np.bincount(y_train, minlength=len(classes))
    class_weights = len(y_train) / (len(classes) * np.maximum(class_counts, 1))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    if args.pair_loss:
        if args.pair_true_label not in classes:
            raise RuntimeError(f"Pair true label {args.pair_true_label!r} not found in class list")
        if args.pair_confuser_label not in classes:
            raise RuntimeError(f"Pair confuser label {args.pair_confuser_label!r} not found in class list")
        pair_true_idx = classes.index(args.pair_true_label)
        pair_confuser_idx = classes.index(args.pair_confuser_label)
        criterion = PairTargetedLoss(
            class_weights_tensor,
            pair_true_idx=pair_true_idx,
            pair_confuser_idx=pair_confuser_idx,
            pair_lambda=args.pair_lambda,
            pair_margin_lambda=args.pair_margin_lambda,
            pair_margin=args.pair_margin,
        )
        print(
            "Using pair-targeted loss: true=%s confuser=%s pair_lambda=%.3f margin_lambda=%.3f margin=%.3f"
            % (
                args.pair_true_label,
                args.pair_confuser_label,
                args.pair_lambda,
                args.pair_margin_lambda,
                args.pair_margin,
            ),
            flush=True,
        )
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    loaders = {
        "train": make_loader(x_train, y_train, args.batch_size, True, 0),
        "val": make_loader(x_val, y_val, args.batch_size, False, 0),
        "test": make_loader(x_test, y_test, args.batch_size, False, 0),
    }
    metas = {"val": val_meta, "test": test_meta}

    print("Building attack model...", flush=True)
    model = build_model(args, len(classes)).to(device)
    print("Transplanting E2 projection/readout weights...", flush=True)
    transplant_projection_readout(model)

    total_params, post_params = count_param_split(model)
    print(f"Params: total={total_params} post_bottleneck={post_params}", flush=True)
    print("Warm start: copied projection and readout from best_e2_matched_8.pt", flush=True)
    if args.residual:
        print(
            f"Residual alpha init={args.alpha_init} bounded={args.alpha_bounded} alpha_max={args.alpha_max}",
            flush=True,
        )

    best_macro = {"score": -1.0, "epoch": 0, "phase": None}
    best_mid = {"score": -1.0, "epoch": 0, "phase": None}
    best_macro_path = CHECKPOINT_DIR / f"best_{args.run_name}_val_macro_f1.pt"
    best_mid_path = CHECKPOINT_DIR / f"best_{args.run_name}_val_mid_recall.pt"
    history = []

    set_requires_grad(model.projection, False)
    set_requires_grad(model.readout, False)
    set_requires_grad(middle_module(model), True)
    if hasattr(model, "alpha"):
        model.alpha.requires_grad = True
    optimizer = torch.optim.AdamW(middle_parameters(model), lr=args.vqc_only_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.scheduler_factor, patience=args.scheduler_patience
    )

    patience_count = 0
    global_epoch = 0
    for epoch in range(1, args.vqc_only_epochs + 1):
        global_epoch += 1
        train_loss = train_one_epoch_attack(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            freeze_projection_bn=True,
            max_batches=args.max_train_batches,
            progress_batches=args.progress_batches,
            phase="vqc_only_train",
            epoch=global_epoch,
        )
        val_output = predict_limited(
            model,
            loaders["val"],
            criterion,
            device,
            args.max_eval_batches,
            progress_batches=args.progress_batches,
            phase="vqc_only_val",
            epoch=global_epoch,
        )
        val_metrics = metrics_dict(val_output, classes)
        append_history(history, "vqc_only", global_epoch, train_loss, val_metrics, optimizer)
        scheduler.step(val_metrics["macro_f1"])
        mid = metric_float(val_metrics["per_stage_recall"]["2"])
        improved = False
        if val_metrics["macro_f1"] > best_macro["score"]:
            best_macro = {"score": val_metrics["macro_f1"], "epoch": global_epoch, "phase": "vqc_only"}
            save_checkpoint(best_macro_path, model, best_macro)
            improved = True
        if not np.isnan(mid) and mid > best_mid["score"]:
            best_mid = {"score": mid, "epoch": global_epoch, "phase": "vqc_only"}
            save_checkpoint(best_mid_path, model, best_mid)
        patience_count = 0 if improved else patience_count + 1
        print(
            "phase=vqc_only epoch=%03d train_loss=%.4f val_macro_f1=%.4f val_mid=%.4f val_early=%.4f val_severe=%.4f"
            % (
                global_epoch,
                train_loss,
                val_metrics["macro_f1"],
                mid,
                metric_float(val_metrics["early_stage_recall"]),
                metric_float(val_metrics["per_stage_recall"]["3"]),
            ),
            flush=True,
        )

    set_requires_grad(model.projection, True)
    set_requires_grad(model.readout, True)
    set_requires_grad(middle_module(model), True)
    if hasattr(model, "alpha"):
        model.alpha.requires_grad = True
    optimizer = torch.optim.AdamW(
        [
            {"params": model.projection.parameters(), "lr": args.joint_head_lr},
            {"params": model.readout.parameters(), "lr": args.joint_head_lr},
            {"params": middle_parameters(model), "lr": args.joint_vqc_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.scheduler_factor, patience=args.scheduler_patience
    )

    patience_count = 0
    for epoch in range(1, args.joint_epochs + 1):
        global_epoch += 1
        train_loss = train_one_epoch_attack(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            max_batches=args.max_train_batches,
            progress_batches=args.progress_batches,
            phase="joint_train",
            epoch=global_epoch,
        )
        val_output = predict_limited(
            model,
            loaders["val"],
            criterion,
            device,
            args.max_eval_batches,
            progress_batches=args.progress_batches,
            phase="joint_val",
            epoch=global_epoch,
        )
        val_metrics = metrics_dict(val_output, classes)
        append_history(history, "joint", global_epoch, train_loss, val_metrics, optimizer)
        scheduler.step(val_metrics["macro_f1"])
        mid = metric_float(val_metrics["per_stage_recall"]["2"])
        improved = False
        if val_metrics["macro_f1"] > best_macro["score"]:
            best_macro = {"score": val_metrics["macro_f1"], "epoch": global_epoch, "phase": "joint"}
            save_checkpoint(best_macro_path, model, best_macro)
            improved = True
        if not np.isnan(mid) and mid > best_mid["score"]:
            best_mid = {"score": mid, "epoch": global_epoch, "phase": "joint"}
            save_checkpoint(best_mid_path, model, best_mid)
        patience_count = 0 if improved else patience_count + 1
        print(
            "phase=joint epoch=%03d train_loss=%.4f val_macro_f1=%.4f val_mid=%.4f val_early=%.4f val_severe=%.4f patience=%d"
            % (
                global_epoch,
                train_loss,
                val_metrics["macro_f1"],
                mid,
                metric_float(val_metrics["early_stage_recall"]),
                metric_float(val_metrics["per_stage_recall"]["3"]),
                patience_count,
            ),
            flush=True,
        )
        if global_epoch >= args.early_stop_warmup and patience_count >= effective_patience:
            print(f"early stopping at global epoch {global_epoch}", flush=True)
            break

    pd.DataFrame(history).to_csv(RESULTS_DIR / f"{args.run_name}_history.csv", index=False)
    if not best_mid_path.exists():
        best_mid = {"score": best_macro["score"], "epoch": best_macro["epoch"], "phase": best_macro["phase"], "fallback": "best_macro"}
        model.load_state_dict(load_checkpoint_state(best_macro_path, device))
        save_checkpoint(best_mid_path, model, best_mid)
    macro_metrics = evaluate_and_save(
        args.run_name, "best_macro", model, best_macro_path, loaders, metas, criterion, device, classes, args, history
    )
    mid_metrics = evaluate_and_save(
        args.run_name, "best_mid", model, best_mid_path, loaders, metas, criterion, device, classes, args, history
    )

    summary_rows = []
    for kind, metrics in [("best_macro", macro_metrics), ("best_mid", mid_metrics)]:
        summary_rows.append({
            "run_name": args.run_name,
            "checkpoint_kind": kind,
            "seed": args.seed,
            "test_accuracy": metrics["test"]["accuracy"],
            "test_macro_f1": metrics["test"]["macro_f1"],
            "test_macro_recall": metrics["test"]["macro_recall"],
            "test_early_recall": metrics["test"]["early_stage_recall"],
            "test_mid_recall": metrics["test"]["per_stage_recall"]["2"],
            "test_severe_recall": metrics["test"]["per_stage_recall"]["3"],
            "test_ece": metrics["test"]["ece"],
            "beats_e2_macro_f1_bar": metrics["beats_e2_macro_f1_bar"],
        })
    summary_path = RESULTS_DIR / f"{args.run_name}_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"saved history and metrics to {RESULTS_DIR}", flush=True)
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

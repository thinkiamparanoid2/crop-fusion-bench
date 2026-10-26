"""Train the E0 image-only MobileNetV3 baseline.

This version mirrors the E1 comparison policy where it matters:
fixed train/val/test split, train-only class weights, validation Macro F1
checkpointing, patience-based early stopping, and final test evaluation from the
best validation checkpoint.
"""

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import InterpolationMode

from common.thesis_eval import (
    count_trainable_params,
    metrics_dict,
    per_class_metrics,
    save_confusion_matrix,
)
from config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_DIR,
    LEARNING_RATE,
    MODEL_DIR,
    NUM_WORKERS,
    TEST_CSV,
    TEST_DIR,
    TRAIN_CSV,
    TRAIN_DIR,
    TRAIN_IMAGE_SUFFIX,
    VAL_CSV,
    VAL_DIR,
    WEIGHT_DECAY,
)
from model.mobilenetv3 import MobileNetV3Classifier


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train E0 image-only MobileNetV3 baseline")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--unfreeze-last-blocks", type=int, default=2)
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-channels-last", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    return parser.parse_args()


def set_seed(seed: int, cudnn_benchmark: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = not cudnn_benchmark
    torch.backends.cudnn.benchmark = cudnn_benchmark


def configure_cuda_runtime(allow_tf32: bool) -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    persistent_workers: bool,
    prefetch_factor: int,
) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def format_progress(current: int, total: int, width: int = 30) -> str:
    ratio = current / total if total else 0
    filled = int(ratio * width)
    bar = "[" + "#" * filled + "-" * (width - filled) + "]"
    return f"{bar} {current}/{total}"


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_label_map(csv_path: Path) -> Dict[str, int]:
    labels = sorted({row["label"].strip() for row in read_manifest(csv_path)})
    return {label: idx for idx, label in enumerate(labels)}


def label_to_stage(label: str) -> str:
    return label.split("_")[2]


def find_image_path(root_dir: Path, image_id: str, suffix: str) -> Path:
    folder = root_dir / image_id
    preferred = folder / f"{image_id}{suffix}.jpg"
    if preferred.exists():
        return preferred

    fallback = folder / f"{image_id}.jpg"
    if fallback.exists():
        return fallback

    candidates = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No image found for {image_id} in {folder}")


class SplitCSVImageDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        root_dir: Path,
        label_map: Dict[str, int],
        suffix: str = "",
    ):
        self.label_map = label_map
        self.rows = read_manifest(csv_path)
        self.samples: List[Tuple[Path, str, int]] = []

        for row in self.rows:
            image_id = row["image"].strip()
            label_name = row["label"].strip()
            self.samples.append((
                find_image_path(root_dir, image_id, suffix),
                label_name,
                label_map[label_name],
            ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, _, label = self.samples[idx]
        image = read_image(str(path), ImageReadMode.RGB)
        image = TF.resize(image, [224, 224], interpolation=InterpolationMode.BILINEAR)
        return image, label


def prepare_batch(images: torch.Tensor, device: torch.device, channels_last: bool) -> torch.Tensor:
    if channels_last:
        images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
    else:
        images = images.to(device, non_blocking=True)
    return images.float().div_(255.0)


def apply_gpu_augmentation(images: torch.Tensor) -> torch.Tensor:
    if torch.rand(1, device=images.device).item() < 0.5:
        images = torch.flip(images, dims=[3])
    brightness = 1.0 + (torch.rand(images.size(0), 1, 1, 1, device=images.device) - 0.5) * 0.2
    contrast = 1.0 + (torch.rand(images.size(0), 1, 1, 1, device=images.device) - 0.5) * 0.2
    images = torch.clamp(images * brightness, 0.0, 1.0)
    mean = images.mean(dim=(2, 3), keepdim=True)
    images = torch.clamp((images - mean) * contrast + mean, 0.0, 1.0)
    return images


def normalize_batch(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def class_weights_from_rows(rows: List[Dict[str, str]], label_map: Dict[str, int]) -> torch.Tensor:
    encoded = [label_map[row["label"].strip()] for row in rows]
    counts = np.bincount(encoded, minlength=len(label_map))
    weights = len(encoded) / (len(label_map) * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def count_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def unfreeze_last_feature_blocks(model: nn.Module, n_blocks: int) -> int:
    if n_blocks <= 0 or not hasattr(model, "features"):
        return 0
    blocks = list(model.features.children())
    n_blocks = min(n_blocks, len(blocks))
    for block in blocks[-n_blocks:]:
        for param in block.parameters():
            param.requires_grad = True
    return n_blocks


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler,
    use_amp: bool,
    channels_last: bool,
    augment: bool,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0
    data_len = len(dataloader)

    for batch_idx, (images, labels) in enumerate(dataloader, start=1):
        images = prepare_batch(images, device, channels_last)
        if augment:
            images = apply_gpu_augmentation(images)
        images = normalize_batch(images)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += images.size(0)
        print(f"  Training {format_progress(batch_idx, data_len)} | loss={loss.item():.4f}", end="\r", flush=True)

    print(" " * 120, end="\r")
    return running_loss / total


def predict(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    channels_last: bool,
) -> Dict[str, np.ndarray]:
    model.eval()
    running_loss = 0.0
    total = 0
    all_probs = []
    all_preds = []
    all_targets = []
    data_len = len(dataloader)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader, start=1):
            images = normalize_batch(prepare_batch(images, device, channels_last))
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)

            running_loss += loss.item() * images.size(0)
            total += images.size(0)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(probs.argmax(dim=1).cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            print(f"  Evaluation {format_progress(batch_idx, data_len)} | loss={loss.item():.4f}", end="\r", flush=True)

    print(" " * 120, end="\r")
    return {
        "loss": running_loss / total,
        "probs": np.concatenate(all_probs),
        "preds": np.concatenate(all_preds),
        "targets": np.concatenate(all_targets),
    }


def save_predictions(path: Path, dataset: SplitCSVImageDataset, eval_output: Dict[str, np.ndarray], idx_to_label: Dict[int, str]) -> None:
    probs = eval_output["probs"]
    preds = eval_output["preds"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "label", "stage", "pred_label", "pred_stage", "correct", "confidence"],
        )
        writer.writeheader()
        for row, pred_idx, prob in zip(dataset.rows, preds, probs):
            pred_label = idx_to_label[int(pred_idx)]
            writer.writerow({
                "image": row["image"],
                "label": row["label"],
                "stage": label_to_stage(row["label"]),
                "pred_label": pred_label,
                "pred_stage": label_to_stage(pred_label),
                "correct": row["label"] == pred_label,
                "confidence": float(prob.max()),
            })


def write_csv_rows(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def count_missing_images(rows: List[Dict[str, str]], root_dir: Path, suffix: str) -> int:
    missing = 0
    for row in rows:
        try:
            find_image_path(root_dir, row["image"].strip(), suffix)
        except FileNotFoundError:
            missing += 1
    return missing


def save_data_audit(path: Path, train_rows: List[Dict[str, str]], val_rows: List[Dict[str, str]], test_rows: List[Dict[str, str]]) -> None:
    def summarize(rows):
        labels = Counter(row["label"] for row in rows)
        stages = Counter(label_to_stage(row["label"]) for row in rows)
        return {
            "rows": len(rows),
            "num_labels": len(labels),
            "label_counts": dict(sorted(labels.items())),
            "stage_counts": dict(sorted(stages.items())),
        }

    train_ids = {row["image"] for row in train_rows}
    val_ids = {row["image"] for row in val_rows}
    test_ids = {row["image"] for row in test_rows}
    audit = {
        "train": summarize(train_rows),
        "val": summarize(val_rows),
        "test": summarize(test_rows),
        "missing_images": {
            "train": count_missing_images(train_rows, TRAIN_DIR, TRAIN_IMAGE_SUFFIX),
            "val": count_missing_images(val_rows, VAL_DIR, TRAIN_IMAGE_SUFFIX),
            "test": count_missing_images(test_rows, TEST_DIR, TRAIN_IMAGE_SUFFIX),
        },
        "overlap": {
            "train_val": len(train_ids & val_ids),
            "train_test": len(train_ids & test_ids),
            "val_test": len(val_ids & test_ids),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)


def main() -> None:
    args = parse_args()
    set_seed(args.seed, args.cudnn_benchmark)
    configure_cuda_runtime(args.allow_tf32)

    if args.run_name is None:
        args.run_name = f"e0_seed_{args.seed}_mobilenetv3_large_bs{args.batch_size}_lr{args.lr:g}"

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Training device: {device}")
    print(f"Run name: {args.run_name}")

    results_dir = LEARNING_DIR / "results"
    confusion_dir = results_dir / "confusion_matrices"
    results_dir.mkdir(parents=True, exist_ok=True)
    confusion_dir.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_rows = read_manifest(TRAIN_CSV)
    val_rows = read_manifest(VAL_CSV)
    test_rows = read_manifest(TEST_CSV)
    save_data_audit(results_dir / f"{args.run_name}_data_audit.json", train_rows, val_rows, test_rows)

    label_map = build_label_map(TRAIN_CSV)
    classes = [label for label, _ in sorted(label_map.items(), key=lambda item: item[1])]
    idx_to_label = {idx: label for label, idx in label_map.items()}
    print(f"Found {len(label_map)} classes.")

    for split_name, rows in [("validation", val_rows), ("test", test_rows)]:
        missing = set(row["label"] for row in rows) - set(label_map)
        if missing:
            raise RuntimeError(f"{split_name} split contains labels not present in train split: {sorted(missing)}")

    with open(results_dir / f"{args.run_name}_class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    run_config = {
        "seed": args.seed,
        "run_name": args.run_name,
        "epochs": args.epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
        "persistent_workers": args.persistent_workers if args.num_workers > 0 else False,
        "optimizer": "Adam",
        "loss": "weighted_cross_entropy",
        "scheduler": "ReduceLROnPlateau",
        "scheduler_monitor": "validation_macro_f1",
        "scheduler_mode": "max",
        "scheduler_patience": args.scheduler_patience,
        "scheduler_factor": args.scheduler_factor,
        "checkpoint_metric": "validation_macro_f1",
        "unfreeze_last_blocks": args.unfreeze_last_blocks,
        "amp": not args.no_amp,
        "channels_last": not args.no_channels_last,
        "cudnn_benchmark": args.cudnn_benchmark,
        "allow_tf32": args.allow_tf32,
        "image_size": 224,
        "image_suffix": TRAIN_IMAGE_SUFFIX,
        "augmentation": not args.no_augment,
        "train_csv": str(TRAIN_CSV),
        "val_csv": str(VAL_CSV),
        "test_csv": str(TEST_CSV),
    }
    with open(results_dir / f"{args.run_name}_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    train_dataset = SplitCSVImageDataset(TRAIN_CSV, TRAIN_DIR, label_map, suffix=TRAIN_IMAGE_SUFFIX)
    val_dataset = SplitCSVImageDataset(VAL_CSV, VAL_DIR, label_map, suffix=TRAIN_IMAGE_SUFFIX)
    test_dataset = SplitCSVImageDataset(TEST_CSV, TEST_DIR, label_map, suffix=TRAIN_IMAGE_SUFFIX)

    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        True,
        args.num_workers,
        device,
        args.persistent_workers,
        args.prefetch_factor,
    )
    val_loader = make_loader(
        val_dataset,
        args.batch_size,
        False,
        args.num_workers,
        device,
        args.persistent_workers,
        args.prefetch_factor,
    )
    test_loader = make_loader(
        test_dataset,
        args.batch_size,
        False,
        args.num_workers,
        device,
        args.persistent_workers,
        args.prefetch_factor,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    print(f"Batch size: {args.batch_size}, epochs: {args.epochs}, patience: {args.patience}")
    print(
        "Loader/runtime: workers=%d persistent=%s prefetch=%s cudnn_benchmark=%s allow_tf32=%s"
        % (
            args.num_workers,
            args.persistent_workers if args.num_workers > 0 else False,
            args.prefetch_factor if args.num_workers > 0 else None,
            args.cudnn_benchmark,
            args.allow_tf32,
        )
    )

    builder = MobileNetV3Classifier(num_classes=len(label_map), device=str(device))
    model = builder.build()
    channels_last = device.type == "cuda" and not args.no_channels_last
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    unfrozen_blocks = unfreeze_last_feature_blocks(model, args.unfreeze_last_blocks)
    if unfrozen_blocks:
        print(f"Unfroze last {unfrozen_blocks} MobileNetV3 feature blocks for fine-tuning.")

    class_weights = class_weights_from_rows(train_rows, label_map)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_macro_f1 = -1.0
    best_epoch = 0
    patience_count = 0
    history: List[Dict] = []
    checkpoint_path = MODEL_DIR / f"best_{args.run_name}.pth"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        epoch_start = time.monotonic()
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            use_amp,
            channels_last,
            not args.no_augment,
        )
        val_output = predict(model, val_loader, criterion, device, channels_last)
        val_metrics = metrics_dict(val_output, classes)
        epoch_time = time.monotonic() - epoch_start
        lr = float(optimizer.param_groups[0]["lr"])

        improved = val_metrics["macro_f1"] > best_val_macro_f1 + args.min_delta
        if improved:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_count = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_count += 1

        scheduler.step(val_metrics["macro_f1"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": lr,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_early_stage_recall": val_metrics["early_stage_recall"],
            "val_stage0_recall": val_metrics["per_stage_recall"]["0"],
            "val_stage1_recall": val_metrics["per_stage_recall"]["1"],
            "val_stage2_recall": val_metrics["per_stage_recall"]["2"],
            "val_stage3_recall": val_metrics["per_stage_recall"]["3"],
            "val_ece": val_metrics["ece"],
            "epoch_time_sec": epoch_time,
            "is_best": improved,
        }
        history.append(row)

        print(
            "epoch %03d train_loss=%.4f val_loss=%.4f val_acc=%.4f val_macro_f1=%.4f "
            "val_early_recall=%.4f patience=%d/%d time=%.1fs"
            % (
                epoch,
                train_loss,
                val_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["macro_f1"],
                val_metrics["early_stage_recall"],
                patience_count,
                args.patience,
                epoch_time,
            )
        )

        if patience_count >= args.patience:
            print(f"early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    val_output = predict(model, val_loader, criterion, device, channels_last)
    test_output = predict(model, test_loader, criterion, device, channels_last)
    val_metrics = metrics_dict(val_output, classes)
    test_metrics = metrics_dict(test_output, classes)
    test_metrics["per_class"] = per_class_metrics(test_output, classes)

    metrics = {
        "seed": args.seed,
        "run_name": args.run_name,
        "best_epoch": best_epoch,
        "checkpoint_metric": "validation_macro_f1",
        "device": str(device),
        "model": "MobileNetV3-Large image-only",
        "num_classes": len(classes),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
        "persistent_workers": args.persistent_workers if args.num_workers > 0 else False,
        "epochs_requested": args.epochs,
        "patience": args.patience,
        "scheduler": True,
        "scheduler_patience": args.scheduler_patience,
        "scheduler_factor": args.scheduler_factor,
        "augmentation": not args.no_augment,
        "amp": use_amp,
        "channels_last": channels_last,
        "cudnn_benchmark": args.cudnn_benchmark,
        "allow_tf32": args.allow_tf32,
        "unfreeze_last_blocks": unfrozen_blocks,
        "train_rows": len(train_dataset),
        "val_rows": len(val_dataset),
        "test_rows": len(test_dataset),
        "total_params": count_total_params(model),
        "trainable_params": count_trainable_params(model),
        "val": val_metrics,
        "test": test_metrics,
        "classes": classes,
    }

    write_csv_rows(results_dir / f"{args.run_name}_history.csv", history)
    save_predictions(results_dir / f"{args.run_name}_val_predictions.csv", val_dataset, val_output, idx_to_label)
    save_predictions(results_dir / f"{args.run_name}_test_predictions.csv", test_dataset, test_output, idx_to_label)
    save_confusion_matrix(confusion_dir / f"{args.run_name}_test_confusion_matrix.csv", test_output, classes)

    with open(results_dir / f"{args.run_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"saved metrics to {results_dir / f'{args.run_name}_metrics.json'}")
    print(f"best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()

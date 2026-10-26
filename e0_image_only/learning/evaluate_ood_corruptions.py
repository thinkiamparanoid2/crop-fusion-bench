"""Evaluate E0 checkpoint robustness under image corruptions.

This script does not retrain and does not modify image files. It loads the saved
E0 checkpoint, applies each corruption in memory to the test split, and writes a
compact CSV/JSON summary.
"""

import argparse
import csv
import json
import sys
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
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import InterpolationMode

from config import MODEL_DIR, TEST_CSV, TEST_DIR, TRAIN_IMAGE_SUFFIX
from model.mobilenetv3 import MobileNetV3Classifier

from common.thesis_eval import expected_calibration_error, stage_recall


CANONICAL_E0_RUN = "e0_bs768_mobilenetv3large"
CANONICAL_E0_DIR = REPO_ROOT / "canonical"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate E0 OOD corruptions")
    parser.add_argument("--run-name", default=CANONICAL_E0_RUN)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
    raise FileNotFoundError(f"No image found for {image_id}")


class TestImageDataset(Dataset):
    def __init__(self, rows: List[Dict[str, str]], label_map: Dict[str, int]):
        self.rows = rows
        self.label_map = label_map
        self.samples: List[Tuple[Path, int]] = []
        for row in rows:
            image_id = row["image"].strip()
            label = row["label"].strip()
            self.samples.append((find_image_path(TEST_DIR, image_id, TRAIN_IMAGE_SUFFIX), label_map[label]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = read_image(str(path), ImageReadMode.RGB)
        image = TF.resize(image, [224, 224], interpolation=InterpolationMode.BILINEAR)
        return image, label


def prepare(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = images.to(device, memory_format=torch.channels_last, non_blocking=True).float().div_(255.0)
    return images


def normalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def corrupt(images: torch.Tensor, name: str) -> torch.Tensor:
    if name == "clean":
        return images
    if name == "brightness_low":
        return torch.clamp(images * 0.75, 0.0, 1.0)
    if name == "brightness_high":
        return torch.clamp(images * 1.25, 0.0, 1.0)
    if name == "contrast_low":
        mean = images.mean(dim=(2, 3), keepdim=True)
        return torch.clamp((images - mean) * 0.65 + mean, 0.0, 1.0)
    if name == "gaussian_noise":
        return torch.clamp(images + torch.randn_like(images) * 0.08, 0.0, 1.0)
    if name == "blur":
        return TF.gaussian_blur(images, kernel_size=[7, 7], sigma=[1.5, 1.5])
    if name == "combined":
        out = torch.clamp(images * 0.8, 0.0, 1.0)
        out = torch.clamp(out + torch.randn_like(out) * 0.06, 0.0, 1.0)
        out = TF.gaussian_blur(out, kernel_size=[5, 5], sigma=[1.0, 1.0])
        return out
    raise ValueError(f"Unknown corruption: {name}")


def evaluate(model, loader, device, idx_to_label, condition: str) -> Dict:
    model.eval()
    probs_all = []
    preds_all = []
    targets_all = []
    with torch.no_grad():
        for images, labels in loader:
            images = corrupt(prepare(images, device), condition)
            logits = model(normalize(images))
            probs = torch.softmax(logits, dim=1)
            probs_all.append(probs.cpu().numpy())
            preds_all.append(probs.argmax(dim=1).cpu().numpy())
            targets_all.append(labels.numpy())

    probs = np.concatenate(probs_all)
    preds = np.concatenate(preds_all)
    targets = np.concatenate(targets_all)
    true_labels = np.array([idx_to_label[int(i)] for i in targets])
    pred_labels = np.array([idx_to_label[int(i)] for i in preds])
    per_stage = stage_recall(true_labels, pred_labels)
    return {
        "condition": condition,
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(targets, preds, average="macro", zero_division=0)),
        "early_stage_recall": per_stage["1"],
        "stage0_recall": per_stage["0"],
        "stage1_recall": per_stage["1"],
        "stage2_recall": per_stage["2"],
        "stage3_recall": per_stage["3"],
        "ece": expected_calibration_error(probs, targets),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    results_dir = REPO_ROOT / "learning" / "results"
    if args.run_name == CANONICAL_E0_RUN:
        mapping_path = CANONICAL_E0_DIR / f"{args.run_name}_class_mapping.json"
        checkpoint_path = CANONICAL_E0_DIR / f"best_{args.run_name}.pth"
    else:
        mapping_path = results_dir / f"{args.run_name}_class_mapping.json"
        checkpoint_path = MODEL_DIR / f"best_{args.run_name}.pth"

    with open(mapping_path, encoding="utf-8") as f:
        label_map = json.load(f)
    idx_to_label = {idx: label for label, idx in label_map.items()}
    rows = read_manifest(TEST_CSV)
    loader = DataLoader(TestImageDataset(rows, label_map), batch_size=args.batch_size, shuffle=False, num_workers=0)

    builder = MobileNetV3Classifier(num_classes=len(label_map), device=str(device))
    model = builder.build().to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    conditions = ["clean", "brightness_low", "brightness_high", "contrast_low", "gaussian_noise", "blur", "combined"]
    results = [evaluate(model, loader, device, idx_to_label, condition) for condition in conditions]

    json_path = results_dir / f"{args.run_name}_ood_corruptions.json"
    csv_path = results_dir / f"{args.run_name}_ood_corruptions.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()

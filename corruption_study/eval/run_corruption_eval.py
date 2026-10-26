"""Evaluate clean/corrupted conditions for corruption robustness.

This script performs inference only. It does not train or fine-tune models.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.io import ImageReadMode, read_image
from torchvision.models import mobilenet_v3_large
from torchvision.transforms import InterpolationMode


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
for _path in (
    PROJECT_ROOT / "e3_bottleneck_sweep" / "scripts",
    PROJECT_ROOT / "e3_quantum_fusion",
    PROJECT_ROOT / "e3_beat_e2_attack" / "scripts",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common.models import E1MLP  # noqa: E402
from common.thesis_eval import metrics_dict  # noqa: E402
from e2_multimodal_fusion.scripts.train_e2_fusion import E2FusionHead  # noqa: E402
from train_bottleneck import E3VQCK  # noqa: E402
from train_warmstart_e3 import E3VQCKResidual  # noqa: E402


DATA_ROOT = PROJECT_ROOT / "Data"
RESULTS_DIR = PROJECT_ROOT / "corruption_study" / "results"
E0_CHECKPOINT = PROJECT_ROOT / "e0_image_only" / "canonical" / "best_e0_bs768_mobilenetv3large.pth"
E0_CLASS_MAPPING = PROJECT_ROOT / "e0_image_only" / "canonical" / "e0_bs768_mobilenetv3large_class_mapping.json"
E1_CHECKPOINT = PROJECT_ROOT / "e1_iot_only" / "canonical" / "best_e1_bs512_mlp128x64.pt"
E1_METRICS = PROJECT_ROOT / "e1_iot_only" / "canonical" / "e1_bs512_mlp128x64_metrics.json"
E1_TRAIN_FEATURES = PROJECT_ROOT / "e1_iot_only" / "features" / "train_iot_features.csv"
E1_TEST_FEATURES = PROJECT_ROOT / "e1_iot_only" / "features" / "test_iot_features.csv"
E2_CANONICAL_CHECKPOINT = PROJECT_ROOT / "e2_multimodal_fusion" / "canonical" / "best_e2_bs512_fusionmlp.pt"
E2_CLEAN_FEATURE_DIR = PROJECT_ROOT / "e2_multimodal_fusion" / "features"
E2_8_CHECKPOINT = PROJECT_ROOT / "e3_bottleneck_sweep" / "checkpoints" / "best_e2_matched_8.pt"
E2_128_25_CHECKPOINT = PROJECT_ROOT / "e2_canonical_128_25" / "checkpoints" / "best_e2_canonical_128_25.pt"
E3_8_VQC_CHECKPOINT = PROJECT_ROOT / "e3_bottleneck_sweep" / "checkpoints" / "best_e3_8_vqc.pt"
E3_8_ATTACK_CHECKPOINT = PROJECT_ROOT / "e3_beat_e2_attack" / "checkpoints" / "best_e3_8_warm_residual_bounded_pair_diag_s42_val_macro_f1.pt"
META_COLS = ["image", "label", "crop", "disease", "stage", "n_rows"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CHECKPOINTS = {
    "e0": E0_CHECKPOINT,
    "e1": E1_CHECKPOINT,
    "e2_canonical": E2_CANONICAL_CHECKPOINT,
    "e2_8": E2_8_CHECKPOINT,
    "e2_128_25": E2_128_25_CHECKPOINT,
    "e3_8_vqc": E3_8_VQC_CHECKPOINT,
    "e3_8_attack_best": E3_8_ATTACK_CHECKPOINT,
}


class FusionProjectionK(nn.Module):
    def __init__(self, bottleneck, fused_dim=1344, hidden=128, dropout=0.1):
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
    def __init__(self, bottleneck, num_classes=25):
        super().__init__()
        self.projection = FusionProjectionK(bottleneck)
        self.middle = nn.Linear(bottleneck, bottleneck, bias=False)
        self.readout = nn.Linear(bottleneck, num_classes)

    def forward(self, x):
        z = torch.tanh(self.projection(x))
        z = torch.tanh(self.middle(z))
        return self.readout(z)


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
    parser = argparse.ArgumentParser(description="Run corruption-study inference")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["e0", "e1", "e2_canonical", "e2_8", "e2_128_25", "e3_8_vqc", "e3_8_attack_best"],
        required=True,
    )
    parser.add_argument("--condition", required=True)
    parser.add_argument("--severity", type=int, required=True)
    parser.add_argument("--image-corruption", default=None)
    parser.add_argument("--iot-corruption", default=None)
    parser.add_argument("--corruption-seed", type=int, default=None)
    parser.add_argument("--image-root", default=str(DATA_ROOT / "test"))
    parser.add_argument("--iot-features", default=str(E1_TEST_FEATURES))
    parser.add_argument("--embedding-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_classes():
    with open(E0_CLASS_MAPPING, "r", encoding="utf-8") as f:
        e0_map = json.load(f)
    classes = [label for label, _ in sorted(e0_map.items(), key=lambda item: item[1])]
    with open(E1_METRICS, "r", encoding="utf-8") as f:
        e1_classes = json.load(f)["classes"]
    if classes != e1_classes:
        raise RuntimeError("E0 and E1 class lists differ")
    return classes


def find_image(root, image_id):
    folder = Path(root) / image_id
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = folder / f"{image_id}{suffix}"
        if candidate.exists():
            return candidate
    candidates = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not candidates:
        raise FileNotFoundError(f"No image found for {image_id} in {folder}")
    return candidates[0]


class ImageDataset(Dataset):
    def __init__(self, manifest, image_root, label_to_idx):
        self.manifest = manifest.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        image_id = str(row["image"])
        image = read_image(str(find_image(self.image_root, image_id)), ImageReadMode.RGB)
        if tuple(image.shape[-2:]) != (224, 224):
            image = TF.resize(image, [224, 224], interpolation=InterpolationMode.BILINEAR)
        label = self.label_to_idx[str(row["label"])]
        return image, label, image_id


def build_e0(num_classes, device):
    model = mobilenet_v3_large(weights=None)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1280),
        nn.Hardswish(),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, num_classes),
    )
    model.load_state_dict(torch.load(E0_CHECKPOINT, map_location=device))
    return model.to(device).eval()


def normalize_images(images, device):
    images = images.to(device, non_blocking=True).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (images - mean) / std


def predict_e0(args, classes, device):
    manifest = pd.read_csv(DATA_ROOT / "test.csv", dtype={"image": str, "label": str})
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    loader = DataLoader(ImageDataset(manifest, args.image_root, label_to_idx), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = build_e0(len(classes), device)
    targets, preds, probs, sample_ids = [], [], [], []
    with torch.no_grad():
        for images, y, ids in loader:
            logits = model(normalize_images(images, device))
            p = torch.softmax(logits, dim=1)
            targets.append(y.numpy())
            preds.append(p.argmax(dim=1).cpu().numpy())
            probs.append(p.cpu().numpy())
            sample_ids.extend(list(ids))
    return sample_ids, np.concatenate(targets), np.concatenate(preds), np.concatenate(probs)


def align_features(features, manifest, split_name):
    selected = features.set_index("image").loc[manifest["image"].astype(str).tolist()].reset_index()
    if selected["label"].astype(str).tolist() != manifest["label"].astype(str).tolist():
        raise RuntimeError(f"{split_name} feature labels do not match manifest")
    return selected


def predict_e1(args, classes, device):
    train_manifest = pd.read_csv(DATA_ROOT / "train.csv", dtype={"image": str, "label": str})
    test_manifest = pd.read_csv(DATA_ROOT / "test.csv", dtype={"image": str, "label": str})
    train_features = pd.read_csv(E1_TRAIN_FEATURES, dtype={"image": str, "label": str})
    test_features = pd.read_csv(args.iot_features, dtype={"image": str, "label": str})
    train_df = align_features(train_features, train_manifest, "train")
    test_df = align_features(test_features, test_manifest, "test")
    feature_cols = [c for c in train_df.columns if c not in META_COLS]
    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols])
    x = scaler.transform(test_df[feature_cols]).astype(np.float32)
    y = np.array([{label: i for i, label in enumerate(classes)}[label] for label in test_df["label"].astype(str)], dtype=np.int64)

    model = E1MLP(input_dim=len(feature_cols), num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(E1_CHECKPOINT, map_location=device))
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(x), torch.tensor(y)), batch_size=4096, shuffle=False)
    preds, probs = [], []
    with torch.no_grad():
        for xb, _yb in loader:
            logits = model(xb.to(device))
            p = torch.softmax(logits, dim=1)
            preds.append(p.argmax(dim=1).cpu().numpy())
            probs.append(p.cpu().numpy())
    return test_df["image"].astype(str).tolist(), y, np.concatenate(preds), np.concatenate(probs)


def load_embeddings(embedding_dir):
    feature_dir = Path(embedding_dir) if embedding_dir else E2_CLEAN_FEATURE_DIR
    data = np.load(feature_dir / "test_embeddings.npz", allow_pickle=True)
    x = np.concatenate([data["image_embedding"].astype(np.float32), data["iot_embedding"].astype(np.float32)], axis=1)
    y = data["y"].astype(np.int64)
    sample_ids = data["image_id"].astype(str).tolist()
    return sample_ids, x, y


def predict_e2(args, classes, device, model_name):
    sample_ids, x, y = load_embeddings(args.embedding_dir)
    if model_name == "e2_canonical":
        model = E2FusionHead(input_dim=x.shape[1], num_classes=len(classes)).to(device)
        checkpoint = E2_CANONICAL_CHECKPOINT
    elif model_name == "e2_8":
        model = E2MatchedK(8, num_classes=len(classes)).to(device)
        checkpoint = E2_8_CHECKPOINT
    elif model_name == "e2_128_25":
        model = E2Canonical128To25Head(input_dim=x.shape[1], num_classes=len(classes)).to(device)
        checkpoint = E2_128_25_CHECKPOINT
    else:
        raise ValueError(f"Unsupported E2 model: {model_name}")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(x), torch.tensor(y)), batch_size=args.batch_size, shuffle=False)
    preds, probs = [], []
    with torch.no_grad():
        for xb, _yb in loader:
            logits = model(xb.to(device))
            p = torch.softmax(logits, dim=1)
            preds.append(p.argmax(dim=1).cpu().numpy())
            probs.append(p.cpu().numpy())
    return sample_ids, y, np.concatenate(preds), np.concatenate(probs)



def load_checkpoint_state(path, device):
    obj = torch.load(path, map_location=device)
    return obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj


def predict_e3(args, classes, device, model_name):
    sample_ids, x, y = load_embeddings(args.embedding_dir)
    if model_name == "e3_8_vqc":
        model = E3VQCK(8, 4, num_classes=len(classes), sim="lightning.qubit").to(device)
        checkpoint = E3_8_VQC_CHECKPOINT
    elif model_name == "e3_8_attack_best":
        model = E3VQCKResidual(
            8,
            4,
            num_classes=len(classes),
            sim="lightning.qubit",
            alpha_init=0.1,
            alpha_bounded=True,
            alpha_max=1.0,
        ).to(device)
        checkpoint = E3_8_ATTACK_CHECKPOINT
    else:
        raise ValueError(f"Unsupported E3 model: {model_name}")
    model.load_state_dict(load_checkpoint_state(checkpoint, device))
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(x), torch.tensor(y)), batch_size=args.batch_size, shuffle=False)
    preds, probs = [], []
    with torch.no_grad():
        for xb, _yb in loader:
            logits = model(xb.to(device))
            p = torch.softmax(logits, dim=1)
            preds.append(p.argmax(dim=1).cpu().numpy())
            probs.append(p.cpu().numpy())
    return sample_ids, y, np.concatenate(preds), np.concatenate(probs)

def run_name(args, model):
    parts = [model, args.condition]
    if args.image_corruption:
        parts.append(f"img-{args.image_corruption}" if args.condition == "both_corrupted" else args.image_corruption)
    if args.iot_corruption:
        parts.append(f"iot-{args.iot_corruption}" if args.condition == "both_corrupted" else args.iot_corruption)
    parts.append(f"sev{args.severity}")
    if args.corruption_seed is not None:
        parts.append(f"cseed{args.corruption_seed}")
    return "__".join(parts)


def save_outputs(args, model, classes, sample_ids, y, pred, probs):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "predictions").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "confusion").mkdir(parents=True, exist_ok=True)
    name = run_name(args, model)
    output = {"loss": 0.0, "targets": y, "preds": pred, "probs": probs}
    metrics = metrics_dict(output, classes)
    macro_precision, _macro_recall, _macro_f1, _macro_support = precision_recall_fscore_support(
        y,
        pred,
        average="macro",
        zero_division=0,
    )
    _weighted_precision, _weighted_recall, weighted_f1, _weighted_support = precision_recall_fscore_support(
        y,
        pred,
        average="weighted",
        zero_division=0,
    )
    metrics.update({
        "model": model,
        "condition": args.condition,
        "image_corruption_type": args.image_corruption,
        "iot_corruption_type": args.iot_corruption,
        "corruption_type": args.image_corruption or args.iot_corruption,
        "severity": args.severity,
        "corruption_seed": args.corruption_seed,
        "image_root_used": args.image_root,
        "iot_csv_used": args.iot_features,
        "embedding_dir_used": args.embedding_dir,
        "num_samples": int(len(y)),
        "macro_precision": float(macro_precision),
        "weighted_f1": float(weighted_f1),
        "training_seed": None,
        "checkpoint_path": str(CHECKPOINTS[model].relative_to(PROJECT_ROOT)),
        "config_path": None,
        "seed": args.seed,
        "timestamp": pd.Timestamp.now().isoformat(),
        "note": "Inference only; no training.",
    })

    with open(RESULTS_DIR / f"{name}__metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(RESULTS_DIR / "predictions" / f"{name}__predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "y_true", "y_pred", "label_true", "label_pred"])
        for sid, yt, yp in zip(sample_ids, y, pred):
            writer.writerow([sid, int(yt), int(yp), classes[int(yt)], classes[int(yp)]])

    cm = confusion_matrix(y, pred, labels=list(range(len(classes))))
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(RESULTS_DIR / "confusion" / f"{name}__confusion.csv")
    print(f"{name}: accuracy={metrics['accuracy']:.6f} macro_f1={metrics['macro_f1']:.6f}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    classes = load_classes()

    for model in args.models:
        if model == "e0":
            sample_ids, y, pred, probs = predict_e0(args, classes, device)
        elif model == "e1":
            sample_ids, y, pred, probs = predict_e1(args, classes, device)
        elif model in {"e2_canonical", "e2_8", "e2_128_25"}:
            sample_ids, y, pred, probs = predict_e2(args, classes, device, model)
        else:
            sample_ids, y, pred, probs = predict_e3(args, classes, device, model)
        save_outputs(args, model, classes, sample_ids, y, pred, probs)


if __name__ == "__main__":
    main()

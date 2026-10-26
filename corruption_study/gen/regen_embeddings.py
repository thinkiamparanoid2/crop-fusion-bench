"""Regenerate E2/E2-8 TEST embeddings from clean or corrupted TEST inputs.

This is frozen-encoder inference only. It does not train or modify checkpoints.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.models import mobilenet_v3_large
from torchvision.transforms import InterpolationMode


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.models import E1MLP  # noqa: E402


DATA_ROOT = PROJECT_ROOT / "Data"
E0_CHECKPOINT = PROJECT_ROOT / "e0_image_only" / "canonical" / "best_e0_bs768_mobilenetv3large.pth"
E0_CLASS_MAPPING = PROJECT_ROOT / "e0_image_only" / "canonical" / "e0_bs768_mobilenetv3large_class_mapping.json"
E1_CHECKPOINT = PROJECT_ROOT / "e1_iot_only" / "canonical" / "best_e1_bs512_mlp128x64.pt"
E1_METRICS = PROJECT_ROOT / "e1_iot_only" / "canonical" / "e1_bs512_mlp128x64_metrics.json"
E1_TRAIN_FEATURES = PROJECT_ROOT / "e1_iot_only" / "features" / "train_iot_features.csv"
META_COLS = ["image", "label", "crop", "disease", "stage", "n_rows"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate TEST embeddings through frozen E0/E1 encoders")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--iot-features", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", default=str(DATA_ROOT / "test.csv"))
    parser.add_argument("--train-manifest", default=str(DATA_ROOT / "train.csv"))
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
    e0_classes = [label for label, _ in sorted(e0_map.items(), key=lambda item: item[1])]
    with open(E1_METRICS, "r", encoding="utf-8") as f:
        e1_classes = json.load(f)["classes"]
    if e0_classes != e1_classes:
        raise RuntimeError("E0 and E1 class lists differ")
    return e0_classes


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


class TestImageDataset(Dataset):
    def __init__(self, manifest, image_root):
        self.manifest = manifest.reset_index(drop=True)
        self.image_root = Path(image_root)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        image_id = str(self.manifest.iloc[idx]["image"])
        path = find_image(self.image_root, image_id)
        image = read_image(str(path), ImageReadMode.RGB)
        if tuple(image.shape[-2:]) != (224, 224):
            image = TF.resize(image, [224, 224], interpolation=InterpolationMode.BILINEAR)
        return image, idx


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
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def e0_embedding(model, images):
    x = model.features(images)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    x = model.classifier[0](x)
    x = model.classifier[1](x)
    return x


def normalize_images(images, device):
    images = images.to(device, non_blocking=True).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (images - mean) / std


def extract_image_embeddings(model, manifest, image_root, batch_size, num_workers, device):
    dataset = TestImageDataset(manifest, image_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    out = np.zeros((len(dataset), 1280), dtype=np.float32)
    with torch.no_grad():
        for images, indices in loader:
            images = normalize_images(images, device)
            emb = e0_embedding(model, images).detach().cpu().numpy().astype(np.float32)
            out[indices.numpy()] = emb
    return out


def align_features(features, manifest, split_name):
    selected = features.set_index("image").loc[manifest["image"].astype(str).tolist()].reset_index()
    if selected["label"].astype(str).tolist() != manifest["label"].astype(str).tolist():
        raise RuntimeError(f"{split_name} feature labels do not match manifest")
    return selected


def extract_iot_embeddings(classes, test_features_path, train_manifest, test_manifest, device):
    train_features = pd.read_csv(E1_TRAIN_FEATURES, dtype={"image": str, "label": str})
    test_features = pd.read_csv(test_features_path, dtype={"image": str, "label": str})
    train_df = align_features(train_features, train_manifest, "train")
    test_df = align_features(test_features, test_manifest, "test")
    feature_cols = [c for c in train_df.columns if c not in META_COLS]

    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols])
    x_test = scaler.transform(test_df[feature_cols])

    model = E1MLP(input_dim=len(feature_cols), num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(E1_CHECKPOINT, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    x_tensor = torch.tensor(x_test, dtype=torch.float32, device=device)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x_tensor), 4096):
            chunks.append(model.embedding(x_tensor[start:start + 4096]).cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0), feature_cols


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    classes = load_classes()
    manifest = pd.read_csv(args.manifest, dtype={"image": str, "label": str})
    train_manifest = pd.read_csv(args.train_manifest, dtype={"image": str, "label": str})

    e0 = build_e0(len(classes), device)
    image_embeddings = extract_image_embeddings(e0, manifest, args.image_root, args.batch_size, args.num_workers, device)
    iot_embeddings, feature_cols = extract_iot_embeddings(classes, args.iot_features, train_manifest, manifest, device)

    labels = manifest["label"].astype(str).tolist()
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    y = np.array([label_to_idx[label] for label in labels], dtype=np.int64)
    stages = np.array([label.split("_")[2] for label in labels], dtype=str)

    pd.DataFrame({
        "image": manifest["image"].astype(str),
        "label": labels,
        "stage": stages,
        "label_idx": y,
    }).to_csv(output_dir / "test_metadata.csv", index=False)

    np.savez_compressed(
        output_dir / "test_embeddings.npz",
        image_id=manifest["image"].astype(str).to_numpy(),
        label=np.array(labels, dtype=str),
        stage=stages,
        y=y,
        image_embedding=image_embeddings.astype(np.float32),
        iot_embedding=iot_embeddings.astype(np.float32),
    )
    with open(output_dir / "e2_embedding_manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "condition": args.condition,
            "classes": classes,
            "image_embedding_dim": 1280,
            "iot_embedding_dim": 64,
            "fusion_input_dim": 1344,
            "image_root": str(Path(args.image_root)),
            "iot_features": str(Path(args.iot_features)),
            "feature_cols": feature_cols,
            "note": "Generated by frozen E0/E1 encoder inference only; no training.",
        }, f, indent=2)
    print(f"saved {output_dir / 'test_embeddings.npz'} image={image_embeddings.shape} iot={iot_embeddings.shape}")


if __name__ == "__main__":
    main()


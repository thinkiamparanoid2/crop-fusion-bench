import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode
from torchvision.io import read_image
from torchvision.models import mobilenet_v3_large
from torchvision.transforms import InterpolationMode


E2_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = E2_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.models import E1MLP

DATA_ROOT = PROJECT_ROOT / "Data"

E0_CHECKPOINT = PROJECT_ROOT / "e0_image_only" / "canonical" / "best_e0_bs768_mobilenetv3large.pth"
E0_CLASS_MAPPING = PROJECT_ROOT / "e0_image_only" / "canonical" / "e0_bs768_mobilenetv3large_class_mapping.json"

E1_CHECKPOINT = PROJECT_ROOT / "e1_iot_only" / "canonical" / "best_e1_bs512_mlp128x64.pt"
E1_METRICS = PROJECT_ROOT / "e1_iot_only" / "canonical" / "e1_bs512_mlp128x64_metrics.json"
E1_FEATURE_DIR = PROJECT_ROOT / "e1_iot_only" / "features"

OUTPUT_DIR = E2_ROOT / "features"

META_COLS = ["image", "label", "crop", "disease", "stage", "n_rows"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
EXPECTED_SPLIT_COUNTS = {"train": 46713, "val": 5193, "test": 5767}


class SplitImageDataset(Dataset):
    def __init__(self, manifest, split_name):
        self.manifest = manifest.reset_index(drop=True)
        self.root_dir = DATA_ROOT / split_name

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        image_id = str(row["image"])
        image_path = find_image_path(self.root_dir, image_id)
        image = read_image(str(image_path), ImageReadMode.RGB)
        image = TF.resize(image, [224, 224], interpolation=InterpolationMode.BILINEAR)
        return image, idx


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frozen E0/E1 embeddings for E2 fusion")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_manifest(split):
    return pd.read_csv(DATA_ROOT / f"{split}.csv", dtype={"image": str, "label": str})


def load_classes():
    with open(E0_CLASS_MAPPING, "r", encoding="utf-8") as f:
        e0_map = json.load(f)
    e0_classes = [label for label, _ in sorted(e0_map.items(), key=lambda item: item[1])]

    with open(E1_METRICS, "r", encoding="utf-8") as f:
        e1_metrics = json.load(f)
    e1_classes = e1_metrics["classes"]

    if e0_classes != e1_classes:
        raise RuntimeError("E0 and E1 class lists are not identical")
    return e0_classes


def find_image_path(root_dir, image_id):
    folder = root_dir / image_id
    preferred = folder / f"{image_id}.jpg"
    if preferred.exists():
        return preferred
    candidates = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No image found for {image_id} in {folder}")


def build_e0_model(num_classes, device):
    model = mobilenet_v3_large(weights=None)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1280),
        nn.Hardswish(),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, num_classes),
    )
    state_dict = torch.load(E0_CHECKPOINT, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def e0_embedding(model, images):
    x = model.features(images)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    x = model.classifier[0](x)
    x = model.classifier[1](x)
    return x


def prepare_images(images, device):
    images = images.to(device, non_blocking=True).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (images - mean) / std


def extract_image_embeddings(model, manifest, split_name, batch_size, num_workers, device):
    dataset = SplitImageDataset(manifest, split_name)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    out = np.zeros((len(dataset), 1280), dtype=np.float32)
    with torch.no_grad():
        for batch_idx, (images, indices) in enumerate(loader, start=1):
            images = prepare_images(images, device)
            embeddings = e0_embedding(model, images).detach().cpu().numpy().astype(np.float32)
            out[indices.numpy()] = embeddings
            if batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == len(loader):
                print(f"[{split_name}] image embeddings {batch_idx}/{len(loader)}")
    return out


def select_iot_features(feature_df, manifest, split_name):
    selected = feature_df.set_index("image").loc[manifest["image"].astype(str).tolist()].reset_index()
    expected_labels = manifest["label"].astype(str).tolist()
    actual_labels = selected["label"].astype(str).tolist()
    if actual_labels != expected_labels:
        raise RuntimeError(f"{split_name} labels do not match split CSV order")
    return selected


def extract_iot_embeddings(classes, device):
    train_manifest = load_manifest("train")
    val_manifest = load_manifest("val")
    test_manifest = load_manifest("test")

    trainval_features = pd.read_csv(E1_FEATURE_DIR / "train_iot_features.csv", dtype={"image": str, "label": str})
    test_features = pd.read_csv(E1_FEATURE_DIR / "test_iot_features.csv", dtype={"image": str, "label": str})

    train_df = select_iot_features(trainval_features, train_manifest, "train")
    val_df = select_iot_features(trainval_features, val_manifest, "val")
    test_df = select_iot_features(test_features, test_manifest, "test")

    feature_cols = [col for col in train_df.columns if col not in META_COLS]
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(train_df[feature_cols])
    x_val = scaler.transform(val_df[feature_cols])
    x_test = scaler.transform(test_df[feature_cols])

    model = E1MLP(input_dim=len(feature_cols), num_classes=len(classes)).to(device)
    state_dict = torch.load(E1_CHECKPOINT, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    out = {}
    for split_name, x in [("train", x_train), ("val", x_val), ("test", x_test)]:
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        chunks = []
        with torch.no_grad():
            for start in range(0, len(x_tensor), 4096):
                chunks.append(model.embedding(x_tensor[start:start + 4096]).cpu().numpy().astype(np.float32))
        out[split_name] = np.concatenate(chunks, axis=0)
        print(f"[{split_name}] iot embeddings {out[split_name].shape}")

    return out, feature_cols


def validate_manifest_counts(manifests):
    for split_name, manifest in manifests.items():
        expected = EXPECTED_SPLIT_COUNTS[split_name]
        if len(manifest) != expected:
            raise RuntimeError(f"{split_name} row count {len(manifest)} != expected {expected}")


def save_split(output_dir, split_name, manifest, image_embeddings, iot_embeddings, classes):
    if len(manifest) != len(image_embeddings) or len(manifest) != len(iot_embeddings):
        raise RuntimeError(f"{split_name} embedding row count mismatch")
    if np.isnan(image_embeddings).any() or np.isnan(iot_embeddings).any():
        raise RuntimeError(f"{split_name} embeddings contain NaN values")

    labels = manifest["label"].astype(str).tolist()
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    y = np.array([label_to_idx[label] for label in labels], dtype=np.int64)
    stages = np.array([label.split("_")[2] for label in labels])

    meta = pd.DataFrame({
        "image": manifest["image"].astype(str),
        "label": labels,
        "stage": stages,
        "label_idx": y,
    })
    meta.to_csv(output_dir / f"{split_name}_metadata.csv", index=False)
    np.savez_compressed(
        output_dir / f"{split_name}_embeddings.npz",
        image_id=meta["image"].to_numpy(dtype=str),
        label=np.array(labels, dtype=str),
        stage=stages.astype(str),
        y=y,
        image_embedding=image_embeddings.astype(np.float32),
        iot_embedding=iot_embeddings.astype(np.float32),
    )
    print(f"[{split_name}] saved embeddings: image={image_embeddings.shape}, iot={iot_embeddings.shape}")


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Extraction device: {device}")

    classes = load_classes()
    manifests = {split: load_manifest(split) for split in ["train", "val", "test"]}
    validate_manifest_counts(manifests)

    e0_model = build_e0_model(num_classes=len(classes), device=device)
    image_embeddings = {
        split: extract_image_embeddings(
            e0_model,
            manifest,
            split,
            args.batch_size,
            args.num_workers,
            device,
        )
        for split, manifest in manifests.items()
    }
    iot_embeddings, feature_cols = extract_iot_embeddings(classes, device)

    for split in ["train", "val", "test"]:
        save_split(output_dir, split, manifests[split], image_embeddings[split], iot_embeddings[split], classes)

    with open(output_dir / "e2_embedding_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "e0_checkpoint": E0_CHECKPOINT.relative_to(PROJECT_ROOT).as_posix(),
                "e1_checkpoint": E1_CHECKPOINT.relative_to(PROJECT_ROOT).as_posix(),
                "classes": classes,
                "image_embedding_dim": 1280,
                "iot_embedding_dim": 64,
                "iot_feature_cols": feature_cols,
                "split_counts": {split: len(manifest) for split, manifest in manifests.items()},
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()

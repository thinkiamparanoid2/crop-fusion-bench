"""Materialize corrupted TEST images for the corruption study.

This script only writes corrupted TEST inputs. It does not train or evaluate models.
"""

import argparse
import csv
import io
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_CORRUPTIONS = [
    "gaussian_noise",
    "motion_blur",
    "defocus_blur",
    "brightness_increase",
    "brightness_decrease",
    "contrast",
    "jpeg_compression",
]
STOCHASTIC = {"gaussian_noise"}

SCHEDULES = {
    "gaussian_noise": [0.08, 0.12, 0.18, 0.26, 0.38],
    "motion_blur": [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)],
    "defocus_blur": [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)],
    "brightness_increase": [0.1, 0.2, 0.3, 0.4, 0.5],
    "brightness_decrease": [-0.1, -0.2, -0.3, -0.4, -0.5],
    "contrast": [0.4, 0.3, 0.2, 0.1, 0.05],
    "jpeg_compression": [25, 18, 15, 10, 7],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate materialized corrupted TEST images")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--corruptions", nargs="+", default=DEFAULT_CORRUPTIONS)
    parser.add_argument("--severities", nargs="+", type=int, default=[2, 3, 5])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--passthrough", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_image(root, image_id):
    folder = Path(root) / image_id
    preferred = folder / f"{image_id}.jpg"
    if preferred.exists():
        return preferred
    candidates = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not candidates:
        raise FileNotFoundError(f"No image found for {image_id} in {folder}")
    return candidates[0]


def resize_clean_geometry(path, image_size):
    image = Image.open(path).convert("RGB")
    return image.resize((image_size, image_size), resample=Image.BILINEAR)


def motion_blur(image, severity):
    radius, _sigma = SCHEDULES["motion_blur"][severity - 1]
    try:
        import cv2

        arr = np.asarray(image).astype(np.float32)
        kernel = np.zeros((radius, radius), dtype=np.float32)
        kernel[radius // 2, :] = 1.0 / radius
        blurred = cv2.filter2D(arr, -1, kernel)
        return Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8), mode="RGB")
    except Exception:
        return image.filter(ImageFilter.GaussianBlur(radius=max(1, radius // 6)))


def apply_corruption(image, corruption, severity, rng):
    if corruption == "gaussian_noise":
        std = SCHEDULES[corruption][severity - 1]
        arr = np.asarray(image).astype(np.float32) / 255.0
        arr = np.clip(arr + rng.normal(0.0, std, size=arr.shape), 0.0, 1.0)
        return Image.fromarray((arr * 255.0).round().astype(np.uint8), mode="RGB")
    if corruption == "motion_blur":
        return motion_blur(image, severity)
    if corruption == "defocus_blur":
        radius, _alias = SCHEDULES[corruption][severity - 1]
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    if corruption in {"brightness_increase", "brightness_decrease"}:
        delta = SCHEDULES[corruption][severity - 1]
        return ImageEnhance.Brightness(image).enhance(1.0 + delta)
    if corruption == "contrast":
        factor = SCHEDULES[corruption][severity - 1]
        return ImageEnhance.Contrast(image).enhance(factor)
    if corruption == "jpeg_compression":
        quality = SCHEDULES[corruption][severity - 1]
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    raise ValueError(f"Unsupported corruption: {corruption}")


def output_path(output_root, image_id, corruption=None, severity=None, seed=None, passthrough=False):
    if passthrough:
        return Path(output_root) / "passthrough" / image_id / f"{image_id}.png"
    base = Path(output_root) / corruption / f"sev_{severity}"
    if corruption in STOCHASTIC:
        base = base / f"cseed_{seed}"
    return base / image_id / f"{image_id}.png"


def save_image(image, path, overwrite):
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return True


def main():
    args = parse_args()
    random.seed(args.seeds[0])
    np.random.seed(args.seeds[0])
    rows = read_manifest(args.manifest)
    if args.max_samples:
        rows = rows[: args.max_samples]

    if args.passthrough:
        written = 0
        for row in rows:
            image_id = row["image"].strip()
            clean = resize_clean_geometry(find_image(args.input_root, image_id), args.image_size)
            path = output_path(args.output_root, image_id, passthrough=True)
            written += int(save_image(clean, path, args.overwrite))
        print(f"passthrough complete: rows={len(rows)} written={written}")
        return

    for corruption in args.corruptions:
        if corruption not in SCHEDULES:
            raise ValueError(f"Unsupported corruption: {corruption}")
        seeds = args.seeds if corruption in STOCHASTIC else [args.seeds[0]]
        for severity in args.severities:
            if severity < 1 or severity > 5:
                raise ValueError("Severity must be in 1..5")
            for seed in seeds:
                rng = np.random.default_rng(seed)
                written = 0
                for row in rows:
                    image_id = row["image"].strip()
                    clean = resize_clean_geometry(find_image(args.input_root, image_id), args.image_size)
                    corrupted = apply_corruption(clean, corruption, severity, rng)
                    path = output_path(args.output_root, image_id, corruption, severity, seed)
                    written += int(save_image(corrupted, path, args.overwrite))
                print(f"{corruption} sev_{severity} seed={seed}: rows={len(rows)} written={written}")


if __name__ == "__main__":
    main()

"""Augment transformed train split images in-place.

Scans the train split for images whose stem ends with a transformed suffix
(e.g. `_224`) and replaces each file with an augmented version.

Usage:
  python augment_train_images.py
  python augment_train_images.py --train-dir path/to/train --suffixes _224
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
    import torchvision.io as io
    import torchvision.transforms.functional as TF
    from torchvision.io import ImageReadMode
    from torchvision.transforms import InterpolationMode
except ModuleNotFoundError as exc:
    missing = exc.name
    raise SystemExit(
        f"{missing!r} is not installed in the current Python environment. "
        "Run this script with the project virtual environment Python."
    ) from exc

from config import TRAIN_DIR

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
DEFAULT_SUFFIXES = ['_224']
DEFAULT_SIZE = 224

# --- Change this variable to limit the number of files processed ---
# Set to an integer (e.g., 5) for a test run, or None to process everything
PROCESS_LIMIT = None


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in SUPPORTED_EXTENSIONS


def has_transformed_suffix(p: Path, suffixes: list[str]) -> bool:
    return any(p.stem.endswith(s.lstrip('_')) for s in suffixes)


def augment_image(img: torch.Tensor, size: int) -> torch.Tensor:
    # Resize directly on uint8 tensor (Bilinear works correctly here)
    img = TF.resize(img, [size, size], interpolation=InterpolationMode.BILINEAR)
    
    # Random horizontal flip with 50% probability
    if torch.rand(1).item() < 0.5:
        img = TF.hflip(img)
        
    # Rotate using NEAREST interpolation to prevent blurry black artifacts on edges
    angle = (torch.rand(1).item() * 20.0) - 10.0
    img = TF.rotate(img, angle, interpolation=InterpolationMode.NEAREST)
    
    # Color adjustments natively support uint8 data types without crashing/blacking out
    img = TF.adjust_brightness(img, brightness_factor=1.0 + (torch.rand(1).item() - 0.5) * 0.2)
    img = TF.adjust_contrast(img, contrast_factor=1.0 + (torch.rand(1).item() - 0.5) * 0.2)
    img = TF.adjust_saturation(img, saturation_factor=1.0 + (torch.rand(1).item() - 0.5) * 0.2)
    img = TF.adjust_hue(img, hue_factor=(torch.rand(1).item() - 0.5) * 0.1)
    return img


def write_image(tensor: torch.Tensor, dst: Path) -> None:
    s = dst.suffix.lower()
    if s in {'.jpg', '.jpeg'}:
        io.write_jpeg(tensor, str(dst), quality=95)
    elif s == '.png':
        io.write_png(tensor, str(dst))
    else:
        # For other formats, write as PNG to preserve content
        io.write_png(tensor, str(dst.with_suffix('.png')))


def process_image(path: Path, size: int) -> None:
    tmp = path.parent / f"{path.stem}.tmp{path.suffix}"
    img = io.read_image(str(path), ImageReadMode.RGB)
    
    # Process completely inside uint8 logic to prevent math scaling overflow issues
    aug = augment_image(img, size)
    
    write_image(aug, tmp)
    if not tmp.exists():
        raise FileNotFoundError(f'Temporary file was not created: {tmp}')
    tmp.replace(path)


def collect_images(root: Path, suffixes: list[str]) -> list[Path]:
    out = []
    for p in root.rglob('*'):
        if p.is_file() and is_image_file(p) and has_transformed_suffix(p, suffixes):
            out.append(p)
    return out


def parse_suffixes(s: str) -> list[str]:
    return [x if x.startswith('_') else f'_{x}' for x in s.split(',') if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description='In-place augment transformed train images')
    parser.add_argument('--train-dir', default=str(TRAIN_DIR), help='Train split root directory')
    parser.add_argument('--suffixes', default=','.join(DEFAULT_SUFFIXES), help='Comma-separated transformed suffixes (default: _224)')
    parser.add_argument('--size', type=int, default=DEFAULT_SIZE, help='Resize size before augmentation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    suffixes = parse_suffixes(args.suffixes)
    root = Path(args.train_dir)

    if not root.exists():
        raise SystemExit(f'Train directory not found: {root}')

    imgs = collect_images(root, suffixes)
    total_found = len(imgs)
    
    # Apply the hardcoded limit variable
    if PROCESS_LIMIT is not None and PROCESS_LIMIT > 0:
        imgs = imgs[:PROCESS_LIMIT]
        print(f'Found {total_found} transformed images. Limiting run to {len(imgs)} images.')
    else:
        print(f'Found {total_found} transformed images to augment in: {root}')

    ok = 0
    err = 0
    
    # Added enumerate to track the loop count (starting at 1)
    for i, p in enumerate(imgs, 1):
        try:
            process_image(p, args.size)
            ok += 1
        except Exception as e:
            print(f'Failed: {p} -> {e}')
            err += 1
            
        # Print progress every 500 images
        if i % 500 == 0:
            print(f'Processed {i} / {len(imgs)} images...')

    print(f'Completed: {ok} succeeded, {err} failed.')


if __name__ == '__main__':
    main()

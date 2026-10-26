"""Resize dataset images to 224x224 with PyTorch and save transformed copies alongside originals."""

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
        "Activate the project virtual environment or run the script with the venv Python:\n"
        "  .venv-e1\\Scripts\\python.exe e0_image_only\\legacy\\image_process\\transform_images.py\n"
    ) from exc

from config import TRAIN_DIR, VAL_DIR, TEST_DIR

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
DEFAULT_SUFFIX = '_224'
DEFAULT_SIZE = 224


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def output_path(src_path: Path, suffix: str) -> Path:
    return src_path.with_name(f"{src_path.stem}{suffix}{src_path.suffix}")


def write_image(tensor: torch.Tensor, dst_path: Path) -> None:
    suffix = dst_path.suffix.lower()
    if suffix in {'.jpg', '.jpeg'}:
        io.write_jpeg(tensor, str(dst_path), quality=95)
    elif suffix == '.png':
        io.write_png(tensor, str(dst_path))
    elif suffix in {'.bmp', '.tiff', '.tif'}:
        io.write_png(tensor, str(dst_path.with_suffix('.png')))
    else:
        io.write_png(tensor, str(dst_path.with_suffix('.png')))


def process_image(src_path: Path, size: int, suffix: str, force: bool) -> bool:
    dst_path = output_path(src_path, suffix)
    if dst_path.exists() and not force:
        return False

    image = io.read_image(str(src_path), ImageReadMode.RGB)
    image = image.to(torch.float32)
    image = TF.resize(image, [size, size], interpolation=InterpolationMode.BICUBIC)
    image = image.to(torch.uint8)

    write_image(image, dst_path)
    return True


def collect_images(root_dir: Path, suffix: str) -> list[Path]:
    if not root_dir.exists():
        return []
    files = []
    for path in root_dir.rglob('*'):
        if path.is_file() and is_image_file(path):
            if path.stem.endswith(suffix.lstrip('_')):
                continue
            files.append(path)
    return files


def main():
    parser = argparse.ArgumentParser(description="Resize dataset images to 224x224 with PyTorch and save transformed copies.")
    parser.add_argument('--size', type=int, default=DEFAULT_SIZE, help='Output image size. Default: 224')
    parser.add_argument('--suffix', default=DEFAULT_SUFFIX, help='Suffix added to modified filenames. Default: _224')
    parser.add_argument('--force', action='store_true', help='Overwrite existing transformed images')
    parser.add_argument('--train-dir', default=str(TRAIN_DIR), help='Training image directory')
    parser.add_argument('--val-dir', default=str(VAL_DIR), help='Validation image directory')
    parser.add_argument('--test-dir', default=str(TEST_DIR), help='Test image directory')
    args = parser.parse_args()

    roots = [Path(args.train_dir), Path(args.val_dir), Path(args.test_dir)]
    total = converted = skipped = errors = 0

    for root in roots:
        files = collect_images(root, args.suffix)
        print(f"Processing {len(files)} images in {root}")
        for src_path in files:
            total += 1
            try:
                changed = process_image(src_path, args.size, args.suffix, args.force)
                if changed:
                    converted += 1
                else:
                    skipped += 1
            except Exception as exc:
                print(f"Failed: {src_path} -> {exc}")
                errors += 1

    print(f"\nDone. Total files scanned: {total}")
    print(f"Converted: {converted}")
    print(f"Skipped (already exists): {skipped}")
    print(f"Errors: {errors}")


if __name__ == '__main__':
    main()

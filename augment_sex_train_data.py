#!/usr/bin/env python3
"""Create augmented training data for shell-image sex classification from an ID-level split CSV.

Input:
- sex_train.csv (or any CSV with id, view, image_path, sex, life_stage)

Output:
- Augmented images saved to disk
- A new CSV pointing to those augmented image paths
- A small preview montage for quick visual artifact checks
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


REQUIRED_COLUMNS = ["id", "view", "image_path", "sex", "life_stage"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment training images for sex classification")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("sex_train.csv"),
        help="Input train CSV from split_sex_data.py",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("sex_train_augmented.csv"),
        help="Output CSV with augmented image paths",
    )
    parser.add_argument(
        "--output-image-dir",
        type=Path,
        default=Path("sex_train_augmented_images"),
        help="Directory to store augmented images",
    )
    parser.add_argument(
        "--copies-per-image",
        type=int,
        default=1,
        help="How many augmented copies to create per input image",
    )
    parser.add_argument(
        "--max-rotation-deg",
        type=float,
        default=20.0,
        help="Maximum absolute rotation angle in degrees",
    )
    parser.add_argument(
        "--crop-scale-min",
        type=float,
        default=0.92,
        help="Minimum random crop scale before resize (light crop)",
    )
    parser.add_argument(
        "--crop-scale-max",
        type=float,
        default=1.0,
        help="Maximum random crop scale before resize",
    )
    parser.add_argument(
        "--brightness",
        type=float,
        default=0.15,
        help="Color jitter brightness range",
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=0.15,
        help="Color jitter contrast range",
    )
    parser.add_argument(
        "--saturation",
        type=float,
        default=0.10,
        help="Color jitter saturation range",
    )
    parser.add_argument(
        "--hue",
        type=float,
        default=0.02,
        help="Color jitter hue range",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=6,
        help="Number of original/augmented pairs to include in preview",
    )
    parser.add_argument(
        "--preview-out",
        type=Path,
        default=Path("sex_train_augmentation_preview.jpg"),
        help="Preview montage output image path",
    )
    return parser.parse_args()


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Input CSV missing required columns: {', '.join(missing)}")
        rows = list(reader)

    if not rows:
        raise ValueError("Input CSV is empty.")
    return rows


def random_resized_crop_light(img: Image.Image, rng: random.Random, scale_min: float, scale_max: float) -> Image.Image:
    w, h = img.size
    scale = rng.uniform(scale_min, scale_max)
    crop_w = max(1, int(round(w * scale)))
    crop_h = max(1, int(round(h * scale)))

    if crop_w >= w or crop_h >= h:
        return img

    left = rng.randint(0, w - crop_w)
    top = rng.randint(0, h - crop_h)
    return TF.resized_crop(
        img,
        top=top,
        left=left,
        height=crop_h,
        width=crop_w,
        size=[h, w],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )


def largest_inner_rect_after_rotation(width: int, height: int, angle_deg: float) -> Tuple[int, int]:
    """Return largest axis-aligned rect inside a rotated width/height rectangle."""
    if width <= 0 or height <= 0:
        return 1, 1

    angle = math.radians(abs(angle_deg) % 180.0)
    if angle > math.pi / 2:
        angle = math.pi - angle

    if angle == 0:
        return width, height

    sin_a = abs(math.sin(angle))
    cos_a = abs(math.cos(angle))

    width_is_longer = width >= height
    side_long, side_short = (width, height) if width_is_longer else (height, width)

    if side_short <= 2.0 * sin_a * cos_a * side_long:
        x = 0.5 * side_short
        if width_is_longer:
            inner_w = x / sin_a
            inner_h = x / cos_a
        else:
            inner_w = x / cos_a
            inner_h = x / sin_a
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        inner_w = (width * cos_a - height * sin_a) / cos_2a
        inner_h = (height * cos_a - width * sin_a) / cos_2a

    inner_w_i = max(1, min(int(inner_w), width))
    inner_h_i = max(1, min(int(inner_h), height))
    return inner_w_i, inner_h_i


def center_crop(img: Image.Image, crop_w: int, crop_h: int) -> Image.Image:
    w, h = img.size
    crop_w = max(1, min(crop_w, w))
    crop_h = max(1, min(crop_h, h))
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    return TF.crop(img, top=top, left=left, height=crop_h, width=crop_w)


def augment_image(
    img: Image.Image,
    rng: random.Random,
    max_rotation_deg: float,
    crop_scale_min: float,
    crop_scale_max: float,
    color_jitter: ColorJitter,
) -> Image.Image:
    original_w, original_h = img.size

    # Rotate with expand=True so shell content is not clipped by the original canvas.
    angle = rng.uniform(-max_rotation_deg, max_rotation_deg)
    rotated = TF.rotate(img, angle=angle, interpolation=InterpolationMode.BILINEAR, expand=True)

    # Source-fill crop: remove black wedges by taking the largest valid inner region.
    clean_w, clean_h = largest_inner_rect_after_rotation(original_w, original_h, angle)
    source_filled = center_crop(rotated, crop_w=clean_w, crop_h=clean_h)

    # Resize back to original dimensions before additional light crop jitter.
    resized = TF.resize(
        source_filled,
        [original_h, original_w],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )

    # Light crop + resize back to original size.
    cropped = random_resized_crop_light(resized, rng, crop_scale_min, crop_scale_max)

    # Mild color perturbation to reduce color/lighting shortcut learning.
    return color_jitter(cropped)


def safe_name(text: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    return "".join(ch if ch in allowed else "_" for ch in text)


def save_preview_montage(pairs: Sequence[Tuple[Image.Image, Image.Image]], output_path: Path) -> None:
    if not pairs:
        return

    thumb_w = 220
    thumb_h = 220
    gap = 12
    border = 12

    rows = len(pairs)
    canvas_w = border * 2 + thumb_w * 2 + gap
    canvas_h = border * 2 + rows * thumb_h + max(0, rows - 1) * gap

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(245, 245, 245))

    y = border
    for original, augmented in pairs:
        o = TF.resize(original, [thumb_h, thumb_w], interpolation=InterpolationMode.BILINEAR, antialias=True)
        a = TF.resize(augmented, [thumb_h, thumb_w], interpolation=InterpolationMode.BILINEAR, antialias=True)
        canvas.paste(o, (border, y))
        canvas.paste(a, (border + thumb_w + gap, y))
        y += thumb_h + gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=95)


def main() -> None:
    args = parse_args()

    if args.copies_per_image < 1:
        raise ValueError("--copies-per-image must be >= 1")
    if not (0.0 < args.crop_scale_min <= args.crop_scale_max <= 1.0):
        raise ValueError("Crop scales must satisfy 0 < min <= max <= 1")

    rng = random.Random(args.seed)

    rows = read_rows(args.input_csv)
    args.output_image_dir.mkdir(parents=True, exist_ok=True)

    color_jitter = ColorJitter(
        brightness=args.brightness,
        contrast=args.contrast,
        saturation=args.saturation,
        hue=args.hue,
    )

    out_rows: List[Dict[str, str]] = []
    preview_pairs: List[Tuple[Image.Image, Image.Image]] = []
    missing_source_images = 0

    for idx, row in enumerate(rows):
        src_path = Path((row.get("image_path") or "").strip())
        if not src_path.exists():
            missing_source_images += 1
            continue

        original = Image.open(src_path).convert("RGB")

        for copy_idx in range(args.copies_per_image):
            augmented = augment_image(
                img=original,
                rng=rng,
                max_rotation_deg=args.max_rotation_deg,
                crop_scale_min=args.crop_scale_min,
                crop_scale_max=args.crop_scale_max,
                color_jitter=color_jitter,
            )

            stem = safe_name(src_path.stem)
            id_token = safe_name((row.get("id") or "unknown").strip())
            out_name = f"{id_token}_{stem}_aug{copy_idx + 1}.jpg"
            out_path = args.output_image_dir / out_name
            augmented.save(out_path, format="JPEG", quality=95)

            out_rows.append(
                {
                    "id": row["id"],
                    "view": row["view"],
                    "image_path": str(out_path.resolve()),
                    "sex": row["sex"],
                    "life_stage": row["life_stage"],
                }
            )

            if len(preview_pairs) < args.preview_count:
                preview_pairs.append((original.copy(), augmented.copy()))

        if (idx + 1) % 50 == 0:
            print(f"  processed source rows: {idx + 1}/{len(rows)}")

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(out_rows)

    save_preview_montage(preview_pairs, args.preview_out)

    print("Augmentation complete:")
    print(f"  input_csv: {args.input_csv}")
    print(f"  output_csv: {args.output_csv}")
    print(f"  output_image_dir: {args.output_image_dir}")
    print(f"  preview_out: {args.preview_out}")
    print(f"  source_rows: {len(rows)}")
    print(f"  missing_source_images: {missing_source_images}")
    print(f"  augmented_rows_written: {len(out_rows)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""ConvNeXt-Base feature extraction scaffold for sex classification (no training).

Behavior:
1. Read image datapoints from CSV.
2. Convert each image to a tensor.
3. Run frozen ConvNeXt-Base backbone to extract image features.

This is a feature-only script; no classifier training is performed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import ConvNeXt_Base_Weights, convnext_base


REQUIRED_COLUMNS = ["id", "view", "image_path", "sex", "life_stage"]


class ConvNeXtBaseFeatureExtractor(torch.nn.Module):
    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_base(weights=weights)
        self.features = model.features
        self.avgpool = model.avgpool

        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ConvNeXt-Base feature extraction for sex classification")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("sex_train.csv"),
        help="Input CSV with datapoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device used for feature extraction",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Resize images to square size before feature extraction",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="How many datapoints to process (default: all)",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained ConvNeXt-Base weights",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("sex_train_features_convnextb.csv"),
        help="Output CSV file for extracted train features",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=None,
        help="Optional test CSV with datapoints",
    )
    parser.add_argument(
        "--test-output-csv",
        type=Path,
        default=Path("sex_test_features_convnextb.csv"),
        help="Output CSV file for extracted test features",
    )
    return parser.parse_args()


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def read_datapoints(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Input CSV missing required columns: {', '.join(missing)}")
        rows = list(reader)

    if not rows:
        raise ValueError("Input CSV has no datapoints.")

    return rows


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_image_tensor(image_path: Path, transform: transforms.Compose) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    x = transform(image)
    return x.unsqueeze(0)  # [1, 3, H, W]


def image_to_convnext_features(
    image_path: Path,
    feature_extractor: torch.nn.Module,
    transform: transforms.Compose,
    device: torch.device,
) -> torch.Tensor:
    """Transform one datapoint image into a 1D ConvNeXt-Base feature vector."""
    batch = load_image_tensor(image_path, transform).to(device)
    with torch.no_grad():
        features = feature_extractor(batch)  # [1, 1024]
    return features.squeeze(0).cpu()


def feature_header(feature_dim: int) -> List[str]:
    base = ["id", "view", "image_path", "sex", "life_stage"]
    feature_cols = [f"feature_{i}" for i in range(feature_dim)]
    return base + feature_cols


def write_feature_row(
    writer: csv.writer,
    row: Dict[str, str],
    image_path: Path,
    feature_vec: torch.Tensor,
) -> None:
    writer.writerow(
        [
            row["id"],
            row["view"],
            str(image_path),
            row["sex"],
            row["life_stage"],
            *feature_vec.tolist(),
        ]
    )


def extract_features_to_csv(
    rows: List[Dict[str, str]],
    output_csv: Path,
    feature_extractor: torch.nn.Module,
    transform: transforms.Compose,
    device: torch.device,
    max_samples: Optional[int],
    split_name: str,
) -> Dict[str, int]:
    processed = 0
    skipped_missing_image = 0
    first_feature_dim: Optional[int] = None

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        for row in rows:
            if max_samples is not None and processed >= max_samples:
                break

            image_path = Path((row.get("image_path") or "").strip())
            if not image_path.exists():
                skipped_missing_image += 1
                continue

            feature_vec = image_to_convnext_features(
                image_path=image_path,
                feature_extractor=feature_extractor,
                transform=transform,
                device=device,
            )

            if first_feature_dim is None:
                first_feature_dim = int(feature_vec.shape[0])
                writer.writerow(feature_header(first_feature_dim))

            write_feature_row(writer, row, image_path, feature_vec)
            processed += 1

            if processed <= 3:
                print(
                    f"  {split_name}_sample_{processed}: id={row['id']} view={row['view']} "
                    f"sex={row['sex']} feature_shape={tuple(feature_vec.shape)}"
                )

    if processed == 0:
        raise RuntimeError(f"No existing image files found to process for split '{split_name}'.")

    print(f"  {split_name}_output_csv: {output_csv}")
    print(f"  {split_name}_skipped_missing_image: {skipped_missing_image}")
    print(f"Processed {processed} {split_name} datapoint image(s).")

    return {
        "processed": processed,
        "skipped_missing_image": skipped_missing_image,
    }


def main() -> None:
    args = parse_args()

    device = pick_device(args.device)
    rows = read_datapoints(args.input_csv)
    transform = build_transform(args.image_size)
    use_pretrained = not args.no_pretrained
    feature_extractor = ConvNeXtBaseFeatureExtractor(pretrained=use_pretrained).to(device)

    print("ConvNeXt-Base feature extraction for sex classification")
    print(f"  device: {device}")
    print(f"  train_input_csv: {args.input_csv}")
    print(f"  pretrained: {use_pretrained}")

    extract_features_to_csv(
        rows=rows,
        output_csv=args.output_csv,
        feature_extractor=feature_extractor,
        transform=transform,
        device=device,
        max_samples=args.max_samples,
        split_name="train",
    )

    if args.test_csv is not None:
        test_rows = read_datapoints(args.test_csv)
        print(f"  test_input_csv: {args.test_csv}")
        extract_features_to_csv(
            rows=test_rows,
            output_csv=args.test_output_csv,
            feature_extractor=feature_extractor,
            transform=transform,
            device=device,
            max_samples=args.max_samples,
            split_name="test",
        )


if __name__ == "__main__":
    main()

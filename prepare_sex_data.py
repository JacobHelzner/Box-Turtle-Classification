#!/usr/bin/env python3
"""Prepare box turtle datapoints for sex classification from the 2025 CSV.

This script converts each turtle row into per-image records shaped as:
[[ID, plastron_or_carapace, image_path], [sex, life_stage]]

No ML libraries are used so the output is easy to inspect and trace.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


VIEW_COLUMNS = {
    "carapace": "carapace_image_paths",
    "plastron": "plastron_image_paths",
}


def split_image_paths(raw: str) -> List[str]:
    """Split a comma-separated image path field into clean paths."""
    if raw is None:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def normalize_label(value: str) -> str:
    """Trim and uppercase labels for consistency."""
    if value is None:
        return ""
    return value.strip().upper()


def build_datapoints(
    csv_path: Path,
    exclude_missing_labels: bool = True,
    require_existing_images: bool = False,
    exclude_juvenile_sex: bool = False,
) -> Tuple[List[List[List[str]]], Dict[str, int]]:
    """Build per-image datapoints and return stats for quick sanity checks."""
    datapoints: List[List[List[str]]] = []

    stats = {
        "rows_total": 0,
        "rows_skipped_missing_label": 0,
        "rows_skipped_juvenile_sex": 0,
        "image_records_total": 0,
        "image_records_skipped_missing_file": 0,
        "carapace_records": 0,
        "plastron_records": 0,
    }

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required_columns = {"Id", "sex", "life stage", *VIEW_COLUMNS.values()}
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"CSV is missing required columns: {missing_str}")

        base_dir = csv_path.parent

        for row in reader:
            stats["rows_total"] += 1

            turtle_id = (row.get("Id") or "").strip()
            sex = normalize_label(row.get("sex") or "")
            life_stage = normalize_label(row.get("life stage") or "")

            if exclude_missing_labels and (not sex or not life_stage):
                stats["rows_skipped_missing_label"] += 1
                continue

            if exclude_juvenile_sex and sex == "J":
                stats["rows_skipped_juvenile_sex"] += 1
                continue

            for view_name, column_name in VIEW_COLUMNS.items():
                image_paths = split_image_paths(row.get(column_name) or "")

                for rel_path in image_paths:
                    image_path = str((base_dir / rel_path).resolve())

                    if require_existing_images and not Path(image_path).exists():
                        stats["image_records_skipped_missing_file"] += 1
                        continue

                    datapoint = [
                        [turtle_id, view_name, image_path],
                        [sex, life_stage],
                    ]
                    datapoints.append(datapoint)

                    stats["image_records_total"] += 1
                    if view_name == "carapace":
                        stats["carapace_records"] += 1
                    else:
                        stats["plastron_records"] += 1

    return datapoints, stats


def write_jsonl(datapoints: Iterable[List[List[str]]], output_path: Path) -> None:
    """Write each datapoint on one line for easy downstream loading."""
    with output_path.open("w", encoding="utf-8") as f:
        for d in datapoints:
            f.write(json.dumps(d) + "\n")


def write_csv(datapoints: Iterable[List[List[str]]], output_path: Path) -> None:
    """Write a flat CSV view for spreadsheet inspection."""
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "view", "image_path", "sex", "life_stage"])
        for d in datapoints:
            features, labels = d
            turtle_id, view_name, image_path = features
            sex, life_stage = labels
            writer.writerow([turtle_id, view_name, image_path, sex, life_stage])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare turtle datapoints for sex classification")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("Data_AllSites_2025_minimal.csv"),
        help="Path to input CSV",
    )
    parser.add_argument(
        "--jsonl-out",
        type=Path,
        default=Path("sex_datapoints.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("sex_datapoints_flat.csv"),
        help="Output flat CSV path",
    )
    parser.add_argument(
        "--keep-missing-labels",
        action="store_true",
        help="Keep rows even when sex or life stage is missing",
    )
    parser.add_argument(
        "--require-existing-images",
        action="store_true",
        help="Drop image records whose files are missing on disk",
    )
    parser.add_argument(
        "--exclude-juvenile-sex",
        action="store_true",
        help="Drop rows where sex is J (juvenile/unknown sex)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    datapoints, stats = build_datapoints(
        csv_path=args.csv,
        exclude_missing_labels=not args.keep_missing_labels,
        require_existing_images=args.require_existing_images,
        exclude_juvenile_sex=args.exclude_juvenile_sex,
    )

    write_jsonl(datapoints, args.jsonl_out)
    write_csv(datapoints, args.csv_out)

    print("Wrote datapoints:")
    print(f"  JSONL: {args.jsonl_out}")
    print(f"  CSV:   {args.csv_out}")
    print("Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

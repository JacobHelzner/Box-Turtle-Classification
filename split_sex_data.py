#!/usr/bin/env python3
"""Split datapoints into 80/20 train/test by turtle ID for sex classification.

This script reads the flat datapoint CSV and creates train/test CSV files while
ensuring all images from the same turtle ID stay in one split.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


REQUIRED_COLUMNS = ["id", "view", "image_path", "sex", "life_stage"]


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Input CSV missing required columns: {', '.join(missing)}")
        return list(reader)


def group_by_id(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        turtle_id = (row.get("id") or "").strip()
        if not turtle_id:
            continue
        groups[turtle_id].append(row)
    return groups


def split_ids(ids: List[str], test_size: float, seed: int) -> Tuple[List[str], List[str]]:
    if not ids:
        raise ValueError("No turtle IDs found in input data.")

    shuffled_ids = ids[:]
    random.Random(seed).shuffle(shuffled_ids)

    split_index = int(len(shuffled_ids) * (1.0 - test_size))

    # Keep both splits non-empty when there is more than one ID.
    if len(shuffled_ids) > 1:
        split_index = max(1, min(split_index, len(shuffled_ids) - 1))

    train_ids = shuffled_ids[:split_index]
    test_ids = shuffled_ids[split_index:]
    return train_ids, test_ids


def write_rows(rows: List[Dict[str, str]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create ID-level 80/20 train/test split")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("sex_datapoints_flat.csv"),
        help="Input flat datapoint CSV",
    )
    parser.add_argument(
        "--train-out",
        type=Path,
        default=Path("sex_train.csv"),
        help="Output CSV for training rows",
    )
    parser.add_argument(
        "--test-out",
        type=Path,
        default=Path("sex_test.csv"),
        help="Output CSV for test rows",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test fraction by unique turtle ID (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not (0.0 < args.test_size < 1.0):
        raise ValueError("--test-size must be between 0 and 1.")

    rows = read_rows(args.input)
    groups = group_by_id(rows)

    all_ids = sorted(groups.keys())
    train_ids, test_ids = split_ids(all_ids, test_size=args.test_size, seed=args.seed)

    train_set = set(train_ids)
    train_rows: List[Dict[str, str]] = []
    test_rows: List[Dict[str, str]] = []

    for turtle_id in all_ids:
        target = train_rows if turtle_id in train_set else test_rows
        target.extend(groups[turtle_id])

    write_rows(train_rows, args.train_out)
    write_rows(test_rows, args.test_out)

    print("Split complete:")
    print(f"  Input rows: {len(rows)}")
    print(f"  Unique IDs: {len(all_ids)}")
    print(f"  Train IDs:  {len(train_ids)}")
    print(f"  Test IDs:   {len(test_ids)}")
    print(f"  Train rows: {len(train_rows)}")
    print(f"  Test rows:  {len(test_rows)}")
    print(f"  Train file: {args.train_out}")
    print(f"  Test file:  {args.test_out}")


if __name__ == "__main__":
    main()

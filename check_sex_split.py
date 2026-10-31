#!/usr/bin/env python3
"""Check train/test split integrity and label balance for sex classification.

Checks performed:
1. No turtle ID overlap between train and test.
2. Basic sex/life-stage label distribution in each split.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set


REQUIRED_COLUMNS = ["id", "view", "image_path", "sex", "life_stage"]


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {', '.join(missing)}")
        return list(reader)


def unique_ids(rows: List[Dict[str, str]]) -> Set[str]:
    return {(r.get("id") or "").strip() for r in rows if (r.get("id") or "").strip()}


def label_counts(rows: List[Dict[str, str]], key: str) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        value = (row.get(key) or "").strip().upper() or "<MISSING>"
        counts[value] += 1
    return counts


def print_counter(title: str, counts: Counter) -> None:
    print(title)
    for label, n in sorted(counts.items()):
        print(f"  {label}: {n}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check sex-classification split quality")
    parser.add_argument("--train", type=Path, default=Path("sex_train.csv"))
    parser.add_argument("--test", type=Path, default=Path("sex_test.csv"))
    args = parser.parse_args()

    train_rows = read_rows(args.train)
    test_rows = read_rows(args.test)

    train_ids = unique_ids(train_rows)
    test_ids = unique_ids(test_rows)
    overlap = train_ids.intersection(test_ids)

    print("Split check summary:")
    print(f"  Train rows: {len(train_rows)}")
    print(f"  Test rows: {len(test_rows)}")
    print(f"  Train IDs: {len(train_ids)}")
    print(f"  Test IDs: {len(test_ids)}")
    print(f"  Overlap IDs: {len(overlap)}")

    if overlap:
        print("  ERROR: overlapping IDs detected")
        for turtle_id in sorted(overlap):
            print(f"    {turtle_id}")
        raise SystemExit(1)

    print("  OK: no ID overlap between train and test")

    print_counter("Train sex distribution:", label_counts(train_rows, "sex"))
    print_counter("Test sex distribution:", label_counts(test_rows, "sex"))

    print_counter("Train life-stage distribution:", label_counts(train_rows, "life_stage"))
    print_counter("Test life-stage distribution:", label_counts(test_rows, "life_stage"))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Sex-classification head training on extracted ResNet-101 features.

Upgrades in this version:
1. Normalizes ResNet feature dimensions with train-set mean/std.
2. Concatenates one-hot view encoding (carapace/plastron) to features.
3. Uses a richer MLP head: 2050 -> 512 -> 128 -> 1 (for default settings).
4. Adds an optional turtle-level loss term to better match evaluation objective.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEX_TO_LABEL = {"F": 1.0, "M": 0.0}
VIEW_TO_ONEHOT = {
    "carapace": [1.0, 0.0],
    "plastron": [0.0, 1.0],
}


@dataclass
class RawSplit:
    feature_rows: List[List[float]]
    labels: List[float]
    ids: List[str]
    views: List[str]


@dataclass
class SplitData:
    features: torch.Tensor
    labels: torch.Tensor
    ids: List[str]
    id_indices: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the sex-classification FC head on feature CSVs")
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("sex_train_features_resnet101.csv"),
        help="Training feature CSV",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path("sex_test_features_resnet101.csv"),
        help="Test feature CSV",
    )
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.35, help="Dropout probability")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay")
    parser.add_argument("--hidden-dim1", type=int, default=512, help="First hidden layer size")
    parser.add_argument("--hidden-dim2", type=int, default=128, help="Second hidden layer size")
    parser.add_argument(
        "--turtle-loss-weight",
        type=float,
        default=0.2,
        help="Weight for turtle-level BCE loss term (set 0.0 to disable)",
    )
    parser.add_argument(
        "--disable-feature-normalization",
        action="store_true",
        help="Disable feature standardization ablation (use raw backbone features)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Training device",
    )
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=Path("sex_head_best.pt"),
        help="Path to save best-performing head checkpoint",
    )
    return parser.parse_args()


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_view(raw_view: str) -> str:
    return raw_view.strip().lower()


def read_feature_rows(csv_path: Path) -> RawSplit:
    if not csv_path.exists():
        raise FileNotFoundError(f"Feature CSV not found: {csv_path}")

    ids: List[str] = []
    views: List[str] = []
    features: List[List[float]] = []
    labels: List[float] = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        feature_cols = [c for c in fieldnames if c.startswith("feature_")]

        if not feature_cols:
            raise ValueError(f"No feature_* columns found in {csv_path}")
        required = {"id", "sex", "view"}
        if not required.issubset(set(fieldnames)):
            raise ValueError(f"CSV must include id, sex, and view columns: {csv_path}")

        for row in reader:
            sex = (row.get("sex") or "").strip().upper()
            view = normalize_view(row.get("view") or "")
            if sex not in SEX_TO_LABEL:
                continue
            if view not in VIEW_TO_ONEHOT:
                continue

            feat = [float(row[c]) for c in feature_cols]
            ids.append((row.get("id") or "").strip())
            views.append(view)
            features.append(feat)
            labels.append(SEX_TO_LABEL[sex])

    if not features:
        raise ValueError(f"No usable F/M rows found in {csv_path}")

    return RawSplit(feature_rows=features, labels=labels, ids=ids, views=views)


def compute_feature_normalizer(train_feature_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = train_feature_matrix.mean(dim=0)
    std = train_feature_matrix.std(dim=0, unbiased=False)
    std = torch.clamp(std, min=1e-6)
    return mean, std


def ids_to_index_tensor(ids: Sequence[str]) -> torch.Tensor:
    id_to_idx: Dict[str, int] = {}
    idxs: List[int] = []
    for turtle_id in ids:
        if turtle_id not in id_to_idx:
            id_to_idx[turtle_id] = len(id_to_idx)
        idxs.append(id_to_idx[turtle_id])
    return torch.tensor(idxs, dtype=torch.long)


def build_input_tensor(
    raw_features: List[List[float]],
    raw_views: List[str],
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> torch.Tensor:
    feat = torch.tensor(raw_features, dtype=torch.float32)
    feat_norm = (feat - feature_mean) / feature_std

    view = torch.tensor([VIEW_TO_ONEHOT[v] for v in raw_views], dtype=torch.float32)
    return torch.cat([feat_norm, view], dim=1)


def build_split(raw: RawSplit, feature_mean: torch.Tensor, feature_std: torch.Tensor) -> SplitData:
    x = build_input_tensor(raw.feature_rows, raw.views, feature_mean, feature_std)
    y = torch.tensor(raw.labels, dtype=torch.float32).unsqueeze(1)
    id_idx = ids_to_index_tensor(raw.ids)
    return SplitData(features=x, labels=y, ids=raw.ids, id_indices=id_idx)


class HeadClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim1: int, hidden_dim2: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LayerNorm(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def batch_turtle_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    id_indices: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    unique_ids = torch.unique(id_indices)
    grouped_logits: List[torch.Tensor] = []
    grouped_labels: List[torch.Tensor] = []

    for uid in unique_ids:
        mask = id_indices == uid
        grouped_logits.append(logits[mask].mean(dim=0, keepdim=True))
        grouped_labels.append(labels[mask][0:1])

    turtle_logits = torch.cat(grouped_logits, dim=0)
    turtle_labels = torch.cat(grouped_labels, dim=0)
    return criterion(turtle_logits, turtle_labels)


@torch.no_grad()
def evaluate_image_level(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    logits = model(x.to(device))
    loss = criterion(logits, y.to(device)).item()
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    acc = (preds == y.to(device)).float().mean().item()
    return {"loss": loss, "accuracy": acc}


@torch.no_grad()
def evaluate_turtle_level(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    ids: Sequence[str],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    logits = model(x.to(device)).squeeze(1).cpu()
    labels = y.squeeze(1).cpu()

    grouped_logits: Dict[str, List[float]] = {}
    grouped_labels: Dict[str, float] = {}
    for idx, turtle_id in enumerate(ids):
        grouped_logits.setdefault(turtle_id, []).append(float(logits[idx]))
        grouped_labels[turtle_id] = float(labels[idx])

    correct = 0
    total = 0
    for turtle_id, logit_list in grouped_logits.items():
        mean_logit = sum(logit_list) / len(logit_list)
        pred = 1.0 if torch.sigmoid(torch.tensor(mean_logit)).item() >= 0.5 else 0.0
        if pred == grouped_labels[turtle_id]:
            correct += 1
        total += 1

    return {"accuracy": correct / total if total else 0.0, "n_turtles": float(total)}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    train_raw = read_feature_rows(args.train_csv)
    test_raw = read_feature_rows(args.test_csv)

    train_feat_matrix = torch.tensor(train_raw.feature_rows, dtype=torch.float32)
    if args.disable_feature_normalization:
        feature_mean = torch.zeros(train_feat_matrix.shape[1], dtype=torch.float32)
        feature_std = torch.ones(train_feat_matrix.shape[1], dtype=torch.float32)
    else:
        feature_mean, feature_std = compute_feature_normalizer(train_feat_matrix)

    train = build_split(train_raw, feature_mean=feature_mean, feature_std=feature_std)
    test = build_split(test_raw, feature_mean=feature_mean, feature_std=feature_std)

    input_dim = train.features.shape[1]
    model = HeadClassifier(
        input_dim=input_dim,
        hidden_dim1=args.hidden_dim1,
        hidden_dim2=args.hidden_dim2,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loader = DataLoader(
        TensorDataset(train.features, train.labels, train.id_indices),
        batch_size=args.batch_size,
        shuffle=True,
    )

    print("Sex-classification head training")
    print(f"  device: {device}")
    print(f"  train_rows: {len(train.ids)}")
    print(f"  test_rows: {len(test.ids)}")
    print(f"  input_dim_with_view: {input_dim}")
    print(f"  epochs: {args.epochs}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  lr: {args.lr}")
    print(f"  dropout: {args.dropout}")
    print(f"  weight_decay: {args.weight_decay}")
    print(f"  hidden_dim1: {args.hidden_dim1}")
    print(f"  hidden_dim2: {args.hidden_dim2}")
    print(f"  turtle_loss_weight: {args.turtle_loss_weight}")
    print(f"  feature_normalization_enabled: {not args.disable_feature_normalization}")

    best_turtle_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for xb, yb, idb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            idb = idb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)

            image_loss = criterion(logits, yb)
            if args.turtle_loss_weight > 0.0:
                turtle_loss = batch_turtle_bce_loss(logits, yb, idb, criterion)
                loss = image_loss + args.turtle_loss_weight * turtle_loss
            else:
                loss = image_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train.ids)

        test_image = evaluate_image_level(model, test.features, test.labels, criterion, device)
        test_turtle = evaluate_turtle_level(model, test.features, test.labels, test.ids, device)

        if test_turtle["accuracy"] > best_turtle_acc:
            best_turtle_acc = test_turtle["accuracy"]
            args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "hidden_dim1": args.hidden_dim1,
                    "hidden_dim2": args.hidden_dim2,
                    "dropout": args.dropout,
                    "sex_to_label": SEX_TO_LABEL,
                    "use_view_onehot": True,
                    "view_to_onehot": VIEW_TO_ONEHOT,
                    "feature_mean": feature_mean.tolist(),
                    "feature_std": feature_std.tolist(),
                    "feature_dim_raw": int(feature_mean.shape[0]),
                    "feature_normalization_enabled": (not args.disable_feature_normalization),
                },
                args.checkpoint_out,
            )

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"test_image_loss={test_image['loss']:.4f} "
            f"test_image_acc={test_image['accuracy']:.4f} "
            f"test_turtle_acc={test_turtle['accuracy']:.4f}"
        )

    print("Training complete")
    print(f"  best_test_turtle_acc: {best_turtle_acc:.4f}")
    print(f"  best_checkpoint: {args.checkpoint_out}")


if __name__ == "__main__":
    main()

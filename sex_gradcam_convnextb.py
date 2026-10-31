#!/usr/bin/env python3
"""Grad-CAM visualization for sex classification using pretrained ConvNeXt-Base + trained head.

This script rebuilds the full image model:
image -> pretrained ConvNeXt-Base backbone -> trained FC head
and writes Grad-CAM outputs for selected images.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torchvision import transforms
from torchvision.models import ConvNeXt_Base_Weights, convnext_base


REQUIRED_COLUMNS = ["id", "view", "image_path", "sex", "life_stage"]
DEFAULT_VIEW_TO_ONEHOT = {
    "carapace": [1.0, 0.0],
    "plastron": [0.0, 1.0],
}


class HeadClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int = 512,
        hidden_dim2: int = 128,
        dropout: float = 0.35,
    ) -> None:
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


class ConvNeXtBackboneWithMap(nn.Module):
    """ConvNeXt-Base backbone that returns both map and pooled features."""

    def __init__(self) -> None:
        super().__init__()
        model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        self.features = model.features
        self.avgpool = model.avgpool

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fmap = self.features(x)
        pooled = self.avgpool(fmap)
        flat = torch.flatten(pooled, 1)
        return fmap, flat


class FullSexModel(nn.Module):
    def __init__(
        self,
        head: nn.Module,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        use_view_onehot: bool,
    ) -> None:
        super().__init__()
        self.backbone = ConvNeXtBackboneWithMap()
        self.head = head
        self.use_view_onehot = use_view_onehot

        self.register_buffer("feature_mean", feature_mean)
        self.register_buffer("feature_std", feature_std)

    def forward(self, x: torch.Tensor, view_onehot: Optional[torch.Tensor] = None) -> torch.Tensor:
        _, features = self.backbone(x)
        features = (features - self.feature_mean) / self.feature_std

        if self.use_view_onehot:
            if view_onehot is None:
                raise ValueError("view_onehot is required for this checkpoint.")
            features = torch.cat([features, view_onehot], dim=1)

        return self.head(features)


class GradCAM:
    def __init__(self, model: FullSexModel, target_module: nn.Module) -> None:
        self.model = model
        self.target_module = target_module
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._fwd_handle = target_module.register_forward_hook(self._save_activation)
        self._bwd_handle = target_module.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module: nn.Module, inp: Tuple[torch.Tensor], out: torch.Tensor) -> None:
        self.activations = out

    def _save_gradient(
        self,
        module: nn.Module,
        grad_input: Tuple[torch.Tensor],
        grad_output: Tuple[torch.Tensor],
    ) -> None:
        self.gradients = grad_output[0]

    def generate(self, score: torch.Tensor, input_size: Tuple[int, int]) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        grads = self.gradients
        acts = self.activations

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_size, mode="bilinear", align_corners=False)

        cam = cam.squeeze(0).squeeze(0)
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.detach().cpu()

    def close(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ConvNeXt-Base Grad-CAM generation for sex classification")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("sex_test.csv"),
        help="CSV containing source images for Grad-CAM",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("sex_head_best_convnextb.pt"),
        help="Head checkpoint from sex_head_training_loop.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sex_gradcam_outputs_convnextb"),
        help="Directory to save CAM maps and overlays",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device for Grad-CAM",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Model input image size",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=12,
        help="How many CSV rows to process",
    )
    parser.add_argument(
        "--target-id",
        type=str,
        default=None,
        help="Optional turtle ID filter",
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


def normalize_view(raw_view: str) -> str:
    return raw_view.strip().lower()


def view_to_onehot_tensor(view: str, mapping: Dict[str, List[float]]) -> torch.Tensor:
    key = normalize_view(view)
    if key not in mapping:
        raise ValueError(f"Unknown view '{view}'. Expected one of: {', '.join(sorted(mapping.keys()))}")
    return torch.tensor([mapping[key]], dtype=torch.float32)


def read_rows(csv_path: Path, target_id: Optional[str]) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    rows: List[Dict[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Input CSV missing required columns: {', '.join(missing)}")

        for row in reader:
            if target_id is not None and (row.get("id") or "").strip() != target_id:
                continue
            rows.append(row)

    if not rows:
        raise ValueError("No matching rows found for Grad-CAM.")

    return rows


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_image_for_model(image_path: Path, transform: transforms.Compose) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    x = transform(img)
    return x.unsqueeze(0)


def load_image_for_overlay(image_path: Path, image_size: int) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def render_overlay(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = np.zeros_like(image_rgb)
    heat[..., 0] = cam
    heat[..., 1] = 0.7 * cam
    overlay = (1.0 - alpha) * image_rgb + alpha * heat
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay


def render_directional_overlay(image_rgb: np.ndarray, signed_cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    pos = np.clip(signed_cam, 0.0, 1.0)
    neg = np.clip(-signed_cam, 0.0, 1.0)

    heat = np.zeros_like(image_rgb)
    heat[..., 0] = pos
    heat[..., 2] = neg

    overlay = (1.0 - alpha) * image_rgb + alpha * heat
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay


def make_directional_legend_image(
    overlay_rgb: np.ndarray,
    true_label: str,
    predicted_label: str,
    prob_female: float,
) -> Image.Image:
    overlay_img = Image.fromarray((overlay_rgb * 255.0).astype(np.uint8), mode="RGB")
    w, h = overlay_img.size
    legend_h = 72

    canvas = Image.new("RGB", (w, h + legend_h), color=(248, 248, 248))
    canvas.paste(overlay_img, (0, legend_h))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    meta = f"True: {true_label}    Pred: {predicted_label}    P(F): {prob_female:.3f}"
    draw.text((12, 10), meta, fill=(20, 20, 20), font=font)

    box_y = 36
    draw.rectangle((12, box_y, 28, box_y + 16), fill=(220, 30, 30))
    draw.text((34, box_y + 1), "F-support", fill=(20, 20, 20), font=font)

    draw.rectangle((128, box_y, 144, box_y + 16), fill=(30, 30, 220))
    draw.text((150, box_y + 1), "M-support", fill=(20, 20, 20), font=font)

    return canvas


def safe_token(text: str) -> str:
    keep = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    return "".join(ch if ch in keep else "_" for ch in text)


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    img = Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def save_pil(path: Path, img: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def save_gray(path: Path, gray: np.ndarray) -> None:
    img = Image.fromarray((gray * 255.0).astype(np.uint8), mode="L")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def build_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> Tuple[FullSexModel, Dict[str, List[float]]]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    input_dim = int(checkpoint.get("input_dim", 1026))
    hidden_dim1 = int(checkpoint.get("hidden_dim1", checkpoint.get("hidden_dim", 256)))
    hidden_dim2 = int(checkpoint.get("hidden_dim2", 128))
    dropout = float(checkpoint.get("dropout", 0.35))
    use_view_onehot = bool(checkpoint.get("use_view_onehot", False))

    feature_mean_list = checkpoint.get("feature_mean")
    feature_std_list = checkpoint.get("feature_std")

    if feature_mean_list is None or feature_std_list is None:
        raw_dim = input_dim - 2 if use_view_onehot else input_dim
        feature_mean = torch.zeros((1, raw_dim), dtype=torch.float32, device=device)
        feature_std = torch.ones((1, raw_dim), dtype=torch.float32, device=device)
    else:
        feature_mean = torch.tensor(feature_mean_list, dtype=torch.float32, device=device).unsqueeze(0)
        feature_std = torch.tensor(feature_std_list, dtype=torch.float32, device=device).unsqueeze(0)

    head = HeadClassifier(
        input_dim=input_dim,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        dropout=dropout,
    )
    head.load_state_dict(checkpoint["model_state_dict"])

    model = FullSexModel(
        head=head,
        feature_mean=feature_mean,
        feature_std=feature_std,
        use_view_onehot=use_view_onehot,
    )
    model.to(device)
    model.eval()

    mapping = checkpoint.get("view_to_onehot", DEFAULT_VIEW_TO_ONEHOT)
    return model, mapping


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    rows = read_rows(args.input_csv, args.target_id)

    transform = build_transform(args.image_size)
    model, view_to_onehot = build_model_from_checkpoint(args.checkpoint, device)

    # Last CNBlock depthwise conv is a reasonable ConvNeXt Grad-CAM target.
    target_module = model.backbone.features[-1][-1].block[0]
    cam_engine = GradCAM(model=model, target_module=target_module)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Sex-classification ConvNeXt-Base Grad-CAM")
    print(f"  device: {device}")
    print(f"  input_csv: {args.input_csv}")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  output_dir: {args.output_dir}")

    processed = 0
    for row in rows:
        if processed >= args.max_samples:
            break

        turtle_id = (row.get("id") or "").strip()
        view = (row.get("view") or "").strip()
        sex = (row.get("sex") or "").strip().upper()
        image_path = Path((row.get("image_path") or "").strip())

        if not image_path.exists():
            continue

        x = load_image_for_model(image_path, transform).to(device)
        rgb = load_image_for_overlay(image_path, args.image_size)
        view_onehot = view_to_onehot_tensor(view, view_to_onehot).to(device)

        model.zero_grad(set_to_none=True)
        logits = model(x, view_onehot=view_onehot)
        prob_female = torch.sigmoid(logits).item()

        score = logits[0, 0] if prob_female >= 0.5 else -logits[0, 0]
        cam = cam_engine.generate(score=score, input_size=(args.image_size, args.image_size)).numpy()

        cam_f = cam_engine.generate(score=logits[0, 0], input_size=(args.image_size, args.image_size)).numpy()
        cam_m = cam_engine.generate(score=-logits[0, 0], input_size=(args.image_size, args.image_size)).numpy()
        directional_signed = np.clip(cam_f - cam_m, -1.0, 1.0)

        overlay = render_overlay(rgb, cam)
        directional_overlay = render_directional_overlay(rgb, directional_signed)
        predicted = "F" if prob_female >= 0.5 else "M"
        directional_with_legend = make_directional_legend_image(
            overlay_rgb=directional_overlay,
            true_label=sex,
            predicted_label=predicted,
            prob_female=prob_female,
        )

        stem = safe_token(image_path.stem)
        prefix = f"{safe_token(turtle_id)}_{safe_token(view)}_{processed + 1:03d}_{stem}"

        cam_path = args.output_dir / f"{prefix}_cam.png"
        overlay_path = args.output_dir / f"{prefix}_overlay.png"
        cam_f_path = args.output_dir / f"{prefix}_cam_female.png"
        cam_m_path = args.output_dir / f"{prefix}_cam_male.png"
        directional_path = args.output_dir / f"{prefix}_directional_overlay.png"
        directional_legend_path = args.output_dir / f"{prefix}_directional_overlay_legend.png"

        save_gray(cam_path, cam)
        save_rgb(overlay_path, overlay)
        save_gray(cam_f_path, cam_f)
        save_gray(cam_m_path, cam_m)
        save_rgb(directional_path, directional_overlay)
        save_pil(directional_legend_path, directional_with_legend)

        print(
            f"  sample_{processed + 1}: id={turtle_id} view={view} true={sex} "
            f"pred={predicted} p_female={prob_female:.3f}"
        )

        processed += 1

    cam_engine.close()

    if processed == 0:
        raise RuntimeError("No images were processed for Grad-CAM.")

    print(f"Wrote Grad-CAM outputs for {processed} sample(s).")


if __name__ == "__main__":
    main()

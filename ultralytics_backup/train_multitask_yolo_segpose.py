#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multitask YOLOv8 (Segmentation + Wrist Pose)
============================================

Pipeline:
  Stage 1  → Pose fine-tuning (2 wrist keypoints) starting from yolov8m-pose.pt (industrial wrist dataset)
  Stage 2  → Segmentation fine-tuning (gloves / no_gloves) using the Pose backbone
  Inference→ Unified single-pass with a shared backbone → (seg masks + wrist keypoints)

Notes
-----
- No CLI arguments required; edit the CONFIG section below if needed.
- Uses only Ultralytics public API (`YOLO`) and a local wrapper class for unified inference.
- UTF-8-safe file I/O (fixes cp949/UnicodeDecodeError on Windows).
- Works with Ultralytics >= 8.x.
"""

from __future__ import annotations

# ===== Standard Library =====
import os
import time
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

# ===== Third-Party =====
import torch
import torch.nn as nn

# OpenCV is only used for FPS tests and image I/O; remove if not needed.
import cv2

# Ultralytics (YOLOv8)
from ultralytics import YOLO


# =============================================================================
# ============================== CONFIGURATION ================================
# =============================================================================

# --- Data YAMLs (edit for your environment) ---
# Pose dataset: industrial wrist dataset with only 2 keypoints (wrist left/right).
POSE_DATA_YAML: str = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\data_wrist\data.yaml"

# Segmentation dataset: two classes → gloves / no_gloves
SEG_DATA_YAML: str = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\data\gloves.yaml"

# --- Model cfgs (not strictly used when we start from yolov8m-pose.pt) ---
POSE_CFG: str = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\ultralytics\ultralytics\cfg\models\v8\yolov8m-pose.yaml"
SEG_CFG:  str = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\ultralytics\ultralytics\cfg\models\v8\yolov8m-seg.yaml"

# --- Training schedule ---
EPOCHS_POSE: int = 100
EPOCHS_SEG:  int = 100
BATCH_SIZE:  int = 8
DEVICE:      str = "0"      # "cpu" or CUDA index string like "0"

# --- Output save directory ---
SAVE_DIR: Path = Path("runs/multitask_segpose_adv")

# --- Keypoints (wrist-only: 2 keypoints) ---
KPT_SHAPE: Tuple[int, int] = (2, 3)  # (num_keypoints, dims) dims=3 → (x,y,conf)
KEYPOINT_NAMES: List[str] = ["left_wrist", "right_wrist"]

# --- Backbone split index (empirically valid for yolov8m* models) ---
# You can adjust to control how much backbone is shared vs kept in task heads.
BACKBONE_END_IDX: int = 15

# --- FPS test settings (optional) ---
FPS_TEST_IMGS: int = 50   # number of images for FPS test (0 = skip)
IMG_SIZE: int    = 640
CONF_THRES: float = 0.25

# --- Misc options ---
# If you want deterministic behavior during training, you may set torch.backends flags here.


# =============================================================================
# ============================ UNIFIED DUAL-HEAD ===============================
# =============================================================================

class DualHeadSegPose(nn.Module):
    """
    Shared-backbone dual head model for unified inference:

        shared backbone (from pose model) → segmentation head (from seg model)
                                         → pose head (from pose model)

    Motivation
    ----------
    - The pose model trained/fine-tuned in Stage 1 provides robust wrist features.
    - We share the *backbone* of the pose model and graft the segmentation head from the seg model.
    - During unified inference, a single backbone forward pass feeds both heads.

    Notes
    -----
    - The split index (BACKBONE_END_IDX) determines where backbone ends and heads begin.
    - The pose model acts as the backbone provider (empirically better when Stage 1 is pose).
    """

    def __init__(self, seg_model: YOLO, pose_model: YOLO, backbone_end_idx: int = 15):
        super().__init__()

        # Extract raw torch modules
        seg_raw  = seg_model.model
        pose_raw = pose_model.model

        # Build shared backbone from pose model (pretrained/fine-tuned)
        self.backbone = nn.Sequential(*list(pose_raw.model)[:backbone_end_idx])

        # Build heads (start right after the shared backbone split)
        self.seg_head  = nn.Sequential(*list(seg_raw.model)[backbone_end_idx:])
        self.pose_head = nn.Sequential(*list(pose_raw.model)[backbone_end_idx:])

        # Convenience: track device from the pose model params
        self.device = next(pose_raw.parameters()).device

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        Forward with shared backbone. Returns:
            seg_out, pose_out (raw head outputs)
        """
        feats   = self.backbone(x)
        seg_out = self.seg_head(feats)
        pose_out= self.pose_head(feats)
        return seg_out, pose_out

    def predict(self, x: torch.Tensor):
        """Alias for forward (for consistency with Ultralytics style)"""
        return self.forward(x)


# =============================================================================
# ============================= HELPER UTILITIES ===============================
# =============================================================================

def _read_yaml(yaml_path: str | Path) -> Dict[str, Any]:
    """UTF-8 safe YAML reader."""
    yaml_path = str(yaml_path)
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(obj: Dict[str, Any], out_path: str | Path) -> None:
    """UTF-8 safe YAML writer with directory creation."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def _ensure_dir(p: str | Path) -> None:
    """Ensure directory exists."""
    Path(p).mkdir(parents=True, exist_ok=True)


def _transfer_backbone_from_pose_to_seg(seg_model: YOLO, pose_weights_path: Path) -> None:
    """
    Load pose weights, copy matching (same shape) non-head weights into the segmentation model.
    Head layers are skipped via a simple heuristic (indices 23/24 are commonly pose-head parts in v8).
    """
    print("\n📥 Transferring backbone/neck from Pose → Seg ...")
    ckpt = torch.load(pose_weights_path, map_location="cpu")

    # Ultralytics checkpoints: ckpt['model'] is a nn.Module with .state_dict()
    if "model" in ckpt and hasattr(ckpt["model"], "state_dict"):
        pose_sd = ckpt["model"].state_dict()
    else:
        # Fallback for raw state_dict
        pose_sd = ckpt.get("state_dict", ckpt)

    seg_sd = seg_model.model.state_dict()
    transferred, skipped = 0, 0

    for name, p in pose_sd.items():
        # Skip pose head weights by index heuristic
        # This is robust enough for typical YOLOv8 pose architectures.
        if ".23." in name or ".24." in name or name.endswith(".23") or name.endswith(".24"):
            skipped += 1
            continue
        if name in seg_sd and seg_sd[name].shape == p.shape:
            seg_sd[name] = p
            transferred += 1
        else:
            skipped += 1

    seg_model.model.load_state_dict(seg_sd, strict=False)
    print(f"   ✅ transferred={transferred}, skipped={skipped}")


def _freeze_backbone(model: YOLO, head_keep: int = 6) -> None:
    """
    Freeze all but the last `head_keep` layers in the internal .model.model sequence.
    Effectively trains segmentation head-only (and maybe last part of neck).
    """
    print("🔒 Freezing backbone/neck (train head only)")
    layers = list(model.model.model)
    for lyr in layers[:-head_keep]:
        for prm in lyr.parameters():
            prm.requires_grad = False


def _measure_unified_fps(
    unified: DualHeadSegPose,
    dataset_yaml: str,
    n_imgs: int = 50,
    imgsz: int = 640
) -> Optional[float]:
    """
    Measures FPS of unified shared-backbone model using real dataset images.
    This is forward-only time (no postprocessing).
    """
    try:
        ds = _read_yaml(dataset_yaml)

        # Find test or val images directory
        base = Path(ds.get("path", "."))
        rel  = ds.get("test", ds.get("val", "images/val"))
        img_dir = base / rel

        if not img_dir.exists():
            print(f"   ⚠ No images folder at: {img_dir}, skip FPS test.")
            return None

        # Collect images
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        imgs: List[Path] = []
        for e in exts:
            imgs.extend(list(img_dir.glob(e)))

        if not imgs:
            print("   ⚠ No images found, skip FPS test.")
            return None

        if n_imgs and n_imgs > 0:
            imgs = imgs[:min(n_imgs, len(imgs))]

        device = next(unified.parameters()).device

        # Warmup
        warm = cv2.imread(str(imgs[0]))
        if warm is not None:
            warm = cv2.cvtColor(warm, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(warm).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            t = torch.nn.functional.interpolate(t, (imgsz, imgsz))
            t = t.to(device)
            for _ in range(3):
                with torch.no_grad():
                    unified.predict(t)

        # FPS timing
        total, ok = 0.0, 0
        for p in imgs:
            im = cv2.imread(str(p))
            if im is None:
                continue
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(im).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            t = torch.nn.functional.interpolate(t, (imgsz, imgsz))
            t = t.to(device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.time()
            with torch.no_grad():
                _ = unified.predict(t)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            total += (time.time() - start)
            ok += 1

        if ok == 0:
            return None

        fps = ok / total
        return fps

    except Exception as e:
        print(f"   ⚠ Unified FPS failed: {e}")
        return None


# =============================================================================
# ============================== TRAINING PIPELINE =============================
# =============================================================================

class MultiTaskSegPose:
    """
    Training/Testing driver for:
      - Stage 1: Pose fine-tuning on industrial wrist dataset (2 keypoints) from yolov8m-pose.pt
      - Stage 2: Segmentation fine-tuning (gloves / no_gloves) using Stage-1 backbone
      - Unified test: single-pass shared-backbone inference + optional FPS measure
    """

    @staticmethod
    def train_pipeline() -> Dict[str, Any]:
        """
        Stage 1: Pose fine-tuning (yolov8m-pose.pt → industrial wrist 2-kpt)
        Stage 2: Seg fine-tuning (glove/no_glove) with pose backbone
        Returns:
            info dict with paths to best weights and metadata
        """
        _ensure_dir(SAVE_DIR)

        # ------------------ Stage 1: Pose Fine-Tuning ------------------
        # Save into a dedicated folder to avoid conflicts with previous runs.
        pose_dir  = SAVE_DIR / "1_pose_pretrain_industrial"
        pose_best = pose_dir / "weights" / "best.pt"

        if pose_best.exists():
            print(f"⚡ Using existing Stage 1 pose weights: {pose_best}")
        else:
            print("\n=== Stage 1/2: Pose Fine-Tuning from Pretrained YOLOv8m-Pose (2-kpt wrist) ===")
            # Load the official pretrained pose checkpoint (17 kpts) and adapt to your 2-kpt dataset via fine-tuning.
            pose_model = YOLO("yolov8m-pose.pt")
            pose_model.train(
                data=POSE_DATA_YAML,
                epochs=EPOCHS_POSE,
                batch=BATCH_SIZE,
                device=DEVICE,
                project=str(SAVE_DIR),
                name="1_pose_pretrain_industrial",
                exist_ok=True,
                # FT hyperparams tuned for stable adaptation from 17→2 kpts:
                lr0=0.005,           # slightly lower LR than scratch
                warmup_epochs=3,
                freeze=10,           # freeze early backbone layers for stability
                imgsz=640,
                hsv_s=0.30, hsv_v=0.20,
                mosaic=0.50, mixup=0.00, erasing=0.20
            )
            print(f"✅ Pose fine-tuning done → {pose_best}")

        # ------------------ Stage 2: Segmentation Fine-Tuning ------------------
        print("\n=== Stage 2/2: Segmentation Fine-Tuning (glove / no_glove) ===")

        # Build the segmentation model from cfg (fresh head)
        seg_model = YOLO(SEG_CFG)

        # Transfer backbone from Stage-1 pose checkpoint into segmentation model
        _transfer_backbone_from_pose_to_seg(seg_model, pose_best)

        # Freeze backbone/neck and train only the segmentation head
        _freeze_backbone(seg_model, head_keep=6)

        seg_model.train(
            data=SEG_DATA_YAML,
            epochs=EPOCHS_SEG,
            batch=BATCH_SIZE,
            device=DEVICE,
            project=str(SAVE_DIR),
            name="2_seg_finetune",
            exist_ok=True,
            lr0=0.001,
            warmup_epochs=5
        )

        seg_best = SAVE_DIR / "2_seg_finetune" / "weights" / "best.pt"
        print(f"✅ Seg fine-tuning done → {seg_best}")

        # ------------------ Save training summary ------------------
        info: Dict[str, Any] = {
            "pose_best": str(pose_best),
            "seg_best":  str(seg_best),
            "kpt_shape": list(KPT_SHAPE),
            "seg_nc": 2,
            "seg_classes": ["gloves", "no_gloves"],
            "save_dir": str(SAVE_DIR),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        _write_yaml(info, SAVE_DIR / "training_info.yaml")

        print("\n🎉 Training pipeline complete.")
        return info

    # -------------------------------------------------------------------------
    # Unified testing: shared-backbone model build + optional FPS measurement
    # -------------------------------------------------------------------------
    @staticmethod
    def unified_test(
        seg_weights: str | Path = None,
        pose_weights: str | Path = None,
        imgsz: int = IMG_SIZE,
        conf: float = CONF_THRES,
        fps_imgs: int = FPS_TEST_IMGS
    ) -> None:
        """
        Load Stage 1 & Stage 2 weights, build the shared-backbone model in memory,
        run optional val() for both individual models, and measure unified FPS
        (one forward through the backbone for both tasks).

        Args
        ----
        seg_weights: path to Stage-2 best.pt (segmentation)
        pose_weights: path to Stage-1 best.pt (pose)
        imgsz: inference size for tests
        conf: val() confidence threshold
        fps_imgs: how many images to use when measuring unified FPS (0=skip)
        """
        seg_weights = Path(seg_weights or (SAVE_DIR / "2_seg_finetune" / "weights" / "best.pt"))
        pose_weights = Path(pose_weights or (SAVE_DIR / "1_pose_pretrain_industrial" / "weights" / "best.pt"))

        if not seg_weights.exists() or not pose_weights.exists():
            print("❌ Missing weights. Run training pipeline first.")
            print(f"   seg_weights exists?  {seg_weights.exists()}  → {seg_weights}")
            print(f"   pose_weights exists? {pose_weights.exists()} → {pose_weights}")
            return

        # Load trained models (Ultralytics will load architecture from checkpoint)
        seg_model  = YOLO(str(seg_weights))
        pose_model = YOLO(str(pose_weights))

        # Build unified dual-head model with shared backbone (from pose)
        print("\n🧩 Building unified dual-head model (shared backbone)")
        unified = DualHeadSegPose(seg_model, pose_model, backbone_end_idx=BACKBONE_END_IDX)
        device = f"cuda:{DEVICE}" if DEVICE != "cpu" and torch.cuda.is_available() else "cpu"
        unified = unified.to(device).eval()
        print(f"   ✅ Unified model ready on device: {device}")

        # Optional: individual validation runs for reference
        try:
            print("\n📈 Reference validation (individual models)")
            seg_res = seg_model.val(
                data=SEG_DATA_YAML,
                device=DEVICE,
                conf=conf,
                imgsz=imgsz,
                verbose=False
            )
            if hasattr(seg_res, "box"):
                print(f"   Seg: mAP50={seg_res.box.map50:.4f}, mAP50-95={seg_res.box.map:.4f}")
        except Exception as e:
            print(f"   (skip seg val) {e}")

        try:
            pose_res = pose_model.val(
                data=POSE_DATA_YAML,
                device=DEVICE,
                conf=conf,
                imgsz=imgsz,
                verbose=False
            )
            if hasattr(pose_res, "pose"):
                print(f"   Pose: mAP50={pose_res.pose.map50:.4f}, mAP50-95={pose_res.pose.map:.4f}")
        except Exception as e:
            print(f"   (skip pose val) {e}")

        # Unified FPS (shared backbone, single pass)
        if fps_imgs and fps_imgs > 0:
            print("\n⚡ Measuring unified FPS (shared backbone, single pass)...")
            fps = _measure_unified_fps(unified, SEG_DATA_YAML, n_imgs=fps_imgs, imgsz=imgsz)
            if fps:
                print(f"   ✅ Unified FPS ≈ {fps:.2f}  (batch=1, {fps_imgs} imgs)")
                print(f"   Avg time ≈ {1000.0 / fps:.2f} ms")
            else:
                print("   ⚠ Could not measure FPS (no images or error).")

        print("\n✅ Unified test complete.")


# =============================================================================
# ================================== MAIN =====================================
# =============================================================================

if __name__ == "__main__":
    """
    Entry point. No CLI required.
    1) Runs the training pipeline (Stage 1 → Stage 2).
    2) Builds the unified shared-backbone model and evaluates (optional FPS).
    """
    # 1) Train pipeline (Stage 1 pose → Stage 2 segmentation)
    results = MultiTaskSegPose.train_pipeline()

    # 2) Unified test (builds shared-backbone model + optional FPS)
    MultiTaskSegPose.unified_test(
        seg_weights=results["seg_best"],
        pose_weights=results["pose_best"],
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        fps_imgs=FPS_TEST_IMGS
    )

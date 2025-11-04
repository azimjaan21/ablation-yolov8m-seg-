#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resume Stage 2 only: Segmentation fine-tuning (LKALite P3)
Uses pretrained Pose backbone from Stage 1 (best.pt)
"""

from __future__ import annotations
import os, time, yaml
from pathlib import Path
from datetime import datetime
import torch, torch.nn as nn
import cv2
from ultralytics import YOLO

# =============================================================================
# ============================== CONFIGURATION ================================
# =============================================================================

POSE_CFG = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\ultralytics\ultralytics\cfg\models\v8\LKALite_pose.yaml"
SEG_CFG  = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\ultralytics\ultralytics\cfg\models\v8\LKALite_p3.yaml"

POSE_DATA_YAML = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\data_wrist\data.yaml"
SEG_DATA_YAML  = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\data\gloves.yaml"

EPOCHS_SEG = 100
BATCH_SIZE, DEVICE = 8, "0"
SAVE_DIR = Path("runs/multitask_lkalite_segpose")
BACKBONE_END_IDX = 7
KPT_SHAPE = (2, 3)
CONF_THRES, IMG_SIZE = 0.25, 640
FPS_TEST_IMGS = 50

# =============================================================================
# ============================== HELPERS ======================================
# =============================================================================

def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def write_yaml(obj, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True)

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def transfer_backbone(seg_model, pose_wts):
    """Transfer Pose backbone → Seg."""
    print("\n📥 Transferring Pose backbone → Seg ...")
    ckpt = torch.load(pose_wts, map_location="cpu")
    pose_sd = ckpt["model"].state_dict() if "model" in ckpt else ckpt
    seg_sd = seg_model.model.state_dict()
    tr, sk = 0, 0
    for k, v in pose_sd.items():
        if "pose" in k or "kpt" in k or "head" in k:
            sk += 1
            continue
        if k in seg_sd and seg_sd[k].shape == v.shape:
            seg_sd[k] = v; tr += 1
        else:
            sk += 1
    seg_model.model.load_state_dict(seg_sd, strict=False)
    print(f"   ✅ transferred={tr}, skipped={sk}")

def freeze_backbone(model, keep=3):
    print("🔒 Freezing backbone (train head only)")
    layers = list(model.model.model)
    for lyr in layers[:-keep]:
        for p in lyr.parameters():
            p.requires_grad = False

# =============================================================================
# ============================== TRAIN STAGE 2 ================================
# =============================================================================

if __name__ == "__main__":
    ensure_dir(SAVE_DIR)
    pose_best = SAVE_DIR / "1_pose_industrial" / "weights" / "best.pt"
    seg_dir = SAVE_DIR / "2_seg_finetune"

    if not pose_best.exists():
        raise FileNotFoundError(f"❌ Pose weights not found: {pose_best}\nRun Stage 1 first!")

    print("\n=== Resuming Stage 2: Segmentation Fine-Tuning (LKALite P3) ===")
    seg_model = YOLO(SEG_CFG)
    transfer_backbone(seg_model, pose_best)
    freeze_backbone(seg_model, keep=3)

    seg_model.train(
        data=SEG_DATA_YAML,
        epochs=EPOCHS_SEG,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(SAVE_DIR),
        name="2_seg_finetune",
        exist_ok=True,
        imgsz=640,
        lr0=0.001,
        warmup_epochs=5
    )

    seg_best = seg_dir / "weights" / "best.pt"

    info = {
        "pose_best": str(pose_best),
        "seg_best": str(seg_best),
        "pose_cfg": str(POSE_CFG),
        "seg_cfg": str(SEG_CFG),
        "save_dir": str(SAVE_DIR),
        "kpt_shape": list(KPT_SHAPE),
        "classes": ["gloves", "no_gloves"],
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    write_yaml(info, SAVE_DIR / "training_info.yaml")

    print("\n🎉 Stage 2 resumed and completed successfully!")

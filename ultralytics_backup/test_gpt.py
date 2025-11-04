#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multitask LKALite YOLO (Segmentation + Wrist Pose)
==================================================

Stage 1 → Pose fine-tuning (2 wrist keypoints) using LKALite backbone
Stage 2 → Segmentation fine-tuning (gloves / no_gloves) sharing the pose backbone
Inference → Unified single-pass backbone → (seg masks + wrist keypoints)
"""

from __future__ import annotations
import os, time, yaml
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple
import torch, torch.nn as nn
import cv2
from ultralytics import YOLO

# =============================================================================
# ============================== CONFIGURATION ================================
# =============================================================================

# --- Custom model YAMLs ---
POSE_CFG = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\ultralytics\ultralytics\cfg\models\v8\LKALite_pose.yaml"
SEG_CFG  = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\ultralytics\ultralytics\cfg\models\v8\LKALite_p3.yaml"

# --- Data paths ---
POSE_DATA_YAML = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\data_wrist\data.yaml"
SEG_DATA_YAML  = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\data\gloves.yaml"

# --- Training ---
EPOCHS_POSE, EPOCHS_SEG = 100, 100
BATCH_SIZE, DEVICE = 8, "0"

SAVE_DIR = Path("runs/multitask_lkalite_segpose")
BACKBONE_END_IDX = 7   # based on your YAML backbone length
KPT_SHAPE = (2, 3)
CONF_THRES, IMG_SIZE = 0.25, 640
FPS_TEST_IMGS = 50

# =============================================================================
# ============================== MODEL WRAPPER ================================
# =============================================================================

class DualHeadSegPose(nn.Module):
    """Shared-backbone dual-head (Seg + Pose) model."""

    def __init__(self, seg_model: YOLO, pose_model: YOLO, backbone_end_idx: int = 7):
        super().__init__()
        seg_raw, pose_raw = seg_model.model, pose_model.model

        # Shared backbone from pose model
        self.backbone = nn.Sequential(*list(pose_raw.model)[:backbone_end_idx])
        self.seg_head  = nn.Sequential(*list(seg_raw.model)[backbone_end_idx:])
        self.pose_head = nn.Sequential(*list(pose_raw.model)[backbone_end_idx:])
        self.device = next(pose_raw.parameters()).device

    @torch.no_grad()
    def forward(self, x):
        feats = self.backbone(x)
        return self.seg_head(feats), self.pose_head(feats)

# =============================================================================
# ============================== HELPER FUNCTIONS =============================
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
    """Transfer shared backbone weights Pose → Seg (skip pose heads)."""
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
            seg_sd[k] = v
            tr += 1
        else:
            sk += 1
    seg_model.model.load_state_dict(seg_sd, strict=False)
    print(f"   ✅ transferred={tr}, skipped={sk}")

def freeze_backbone(model, keep=3):
    """Freeze backbone & train only the final head layers."""
    print("🔒 Freezing backbone (train head only)")
    layers = list(model.model.model)
    for lyr in layers[:-keep]:
        for p in lyr.parameters():
            p.requires_grad = False

def measure_fps(unified, dataset_yaml, n=50, imgsz=640):
    try:
        ds = read_yaml(dataset_yaml)
        base = Path(ds.get("path", ".")); rel = ds.get("test", ds.get("val", "images/val"))
        img_dir = base / rel
        imgs = [p for e in ["*.jpg", "*.png", "*.jpeg"] for p in img_dir.glob(e)]
        if not imgs: return None
        imgs = imgs[:min(n, len(imgs))]
        dev = next(unified.parameters()).device
        warm = cv2.imread(str(imgs[0]))
        t = torch.from_numpy(cv2.cvtColor(warm, cv2.COLOR_BGR2RGB)).permute(2,0,1).float().unsqueeze(0)/255
        t = torch.nn.functional.interpolate(t,(imgsz,imgsz)).to(dev)
        for _ in range(3): unified(t)
        total=0; ok=0
        for p in imgs:
            im = cv2.imread(str(p))
            if im is None: continue
            t = torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).permute(2,0,1).float().unsqueeze(0)/255
            t = torch.nn.functional.interpolate(t,(imgsz,imgsz)).to(dev)
            torch.cuda.synchronize()
            s=time.time(); unified(t); torch.cuda.synchronize()
            total+=time.time()-s; ok+=1
        return ok/total if ok>0 else None
    except Exception as e:
        print("FPS error:", e)
        return None

# =============================================================================
# ============================== TRAINING PIPELINE ============================
# =============================================================================

class MultiTaskSegPose:
    """Stage 1: Pose fine-tuning, Stage 2: Seg fine-tuning, Unified inference."""

    @staticmethod
    def train_pipeline():
        ensure_dir(SAVE_DIR)
        pose_dir, seg_dir = SAVE_DIR/"1_pose_industrial", SAVE_DIR/"2_seg_finetune"
        pose_best = pose_dir/"weights"/"best.pt"

        # ---------------- Stage 1: Pose ----------------
        if pose_best.exists():
            print(f"⚡ Using existing Pose weights: {pose_best}")
        else:
            print("\n=== Stage 1: Pose Fine-Tuning (LKALite Pose) ===")
            pose_model = YOLO(POSE_CFG)
            pose_model.train(
                data=POSE_DATA_YAML,
                epochs=EPOCHS_POSE,
                batch=BATCH_SIZE,
                device=DEVICE,
                project=str(SAVE_DIR),
                name="1_pose_industrial",
                exist_ok=True,
                imgsz=640,
                lr0=0.005,
                warmup_epochs=3,
                freeze=5,
                hsv_s=0.3, hsv_v=0.2,
                mosaic=0.5, mixup=0.0, erasing=0.2
            )
            print(f"✅ Pose fine-tuned → {pose_best}")

        # ---------------- Stage 2: Segmentation ----------------
        print("\n=== Stage 2: Segmentation Fine-Tuning (LKALite P3) ===")
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
        seg_best = seg_dir/"weights"/"best.pt"

        # Save info
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
        write_yaml(info, SAVE_DIR/"training_info.yaml")
        print("\n🎉 Training pipeline complete.")
        return info

    # ---------------- Unified Test ----------------
    @staticmethod
    def unified_test(seg_w=None, pose_w=None, imgsz=640, conf=0.25, fps_imgs=50):
        seg_w = Path(seg_w or SAVE_DIR/"2_seg_finetune"/"weights"/"best.pt")
        pose_w = Path(pose_w or SAVE_DIR/"1_pose_industrial"/"weights"/"best.pt")

        if not seg_w.exists() or not pose_w.exists():
            print("❌ Missing weights. Run training first.")
            return

        seg_model, pose_model = YOLO(str(seg_w)), YOLO(str(pose_w))
        print("\n🧩 Building Unified LKALite Dual-Head Model")
        unified = DualHeadSegPose(seg_model, pose_model, backbone_end_idx=BACKBONE_END_IDX)
        dev = f"cuda:{DEVICE}" if DEVICE!="cpu" and torch.cuda.is_available() else "cpu"
        unified = unified.to(dev).eval()
        print(f"   ✅ Unified model ready on {dev}")

        try:
            seg_res = seg_model.val(data=SEG_DATA_YAML, device=DEVICE, conf=conf, imgsz=imgsz, verbose=False)
            print(f"   Seg: mAP50={seg_res.box.map50:.4f}, mAP50-95={seg_res.box.map:.4f}")
        except: pass

        try:
            pose_res = pose_model.val(data=POSE_DATA_YAML, device=DEVICE, conf=conf, imgsz=imgsz, verbose=False)
            print(f"   Pose: mAP50={pose_res.pose.map50:.4f}, mAP50-95={pose_res.pose.map:.4f}")
        except: pass

        if fps_imgs>0:
            print("\n⚡ Measuring unified FPS...")
            fps = measure_fps(unified, SEG_DATA_YAML, n=fps_imgs, imgsz=imgsz)
            if fps: print(f"   ✅ Unified FPS ≈ {fps:.2f}")
            else: print("   ⚠ FPS test skipped.")

        print("\n✅ Unified LKALite test complete.")

# =============================================================================
# ================================== MAIN =====================================
# =============================================================================

if __name__ == "__main__":
    results = MultiTaskSegPose.train_pipeline()
    MultiTaskSegPose.unified_test(
        seg_w=results["seg_best"],
        pose_w=results["pose_best"],
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        fps_imgs=FPS_TEST_IMGS
    )

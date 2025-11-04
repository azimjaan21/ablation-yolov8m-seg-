#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Multitask Inference (Seg + Pose) — Shared backbone, robust masks draw, proper FPS.
"""
import os, time, cv2, torch, numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch.nn as nn

# -----------------
# CONFIG
# -----------------
SEG_WEIGHTS  = r"runs\multitask_segpose_adv\2_seg_finetune\weights\best.pt"
POSE_WEIGHTS = r"runs\multitask_segpose_adv\1_pose_pretrain_industrial\weights\best.pt"
BACKBONE_END_IDX = 15
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

IMG_PATH = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\ppe.jpg"
OUT_DIR = Path("runs/multitask_segpose_debug_fps"); OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE   = 640
CONF_THRES = 0.25      # lower if you don’t see masks, e.g. 0.15
MASK_ALPHA = 0.45
WARMUP     = 10        # warmup iterations for fair timing
RUNS       = 50        # measured iterations for FPS

# -----------------
# Shared-backbone dual head
# -----------------
class DualHeadSegPose(nn.Module):
    def __init__(self, seg_model: YOLO, pose_model: YOLO, backbone_end_idx=15):
        super().__init__()
        seg_raw, pose_raw = seg_model.model, pose_model.model
        self.backbone_layers = nn.ModuleList(list(pose_raw.model[:backbone_end_idx]))
        self.seg_head_layers = nn.ModuleList(list(seg_raw.model[backbone_end_idx:]))
        self.pose_head_layers = nn.ModuleList(list(pose_raw.model[backbone_end_idx:]))

    @torch.no_grad()
    def forward(self, x):
        feats = []
        z = x
        # backbone
        for m in self.backbone_layers:
            fi = getattr(m, "f", -1)
            if isinstance(fi, int):
                xin = z if fi == -1 else feats[fi]
            else:
                xin = [feats[j] if j != -1 else z for j in fi]
            z = m(xin) if isinstance(xin, list) else m(xin)
            feats.append(z)

        # seg head
        y_seg = z
        seg_feats = feats.copy()
        for m in self.seg_head_layers:
            fi = getattr(m, "f", -1)
            if isinstance(fi, int):
                xin = y_seg if fi == -1 else seg_feats[fi]
            else:
                xin = [seg_feats[j] if j != -1 else y_seg for j in fi]
            y_seg = m(xin) if isinstance(xin, list) else m(xin)
            seg_feats.append(y_seg)

        # pose head
        y_pose = z
        pose_feats = feats.copy()
        for m in self.pose_head_layers:
            fi = getattr(m, "f", -1)
            if isinstance(fi, int):
                xin = y_pose if fi == -1 else pose_feats[fi]
            else:
                xin = [pose_feats[j] if j != -1 else y_pose for j in fi]
            y_pose = m(xin) if isinstance(xin, list) else m(xin)
            pose_feats.append(y_pose)

        return y_seg, y_pose

# -----------------
# Utils
# -----------------
def preprocess(img, size=640):
    im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (size, size))
    t = torch.from_numpy(im).float().permute(2, 0, 1) / 255.0
    return t.unsqueeze(0)

def draw_masks(image, masks, alpha=0.45, color=(0,255,0)):
    if masks is None or len(masks) == 0:
        return image
    overlay = image.copy()
    for m in masks:
        m_bool = m.astype(bool)
        overlay[m_bool] = color
    return cv2.addWeighted(overlay, alpha, image, 1-alpha, 0)

def draw_keypoints(image, keypoints, color=(255,0,0)):
    if keypoints is None: return image
    for kp in keypoints:
        # kp shape: (K,3) or (K,2) — handle both
        for j in range(len(kp)):
            if kp.shape[1] == 3:
                x, y, v = kp[j]
                if v <= 0: continue
            else:
                x, y = kp[j]; v = 1
            cv2.circle(image, (int(x), int(y)), 4, color, -1)
    return image

def robust_mask_list(result, target_hw):
    """
    Get list[np.ndarray HxW] masks from a Ultralytics segmentation result,
    resized to target image size if needed.
    """
    if result is None or result.masks is None:
        return []
    m = result.masks
    # v8 provides m.data with shape [N, Hm, Wm] (already upsampled to input)
    data = getattr(m, "data", None)
    if data is None:
        return []
    arr = data.cpu().numpy()  # [N, H, W]
    masks = []
    H, W = target_hw
    for i in range(arr.shape[0]):
        mi = arr[i]
        if mi.ndim == 3:  # rare: [1,H,W]
            mi = mi.squeeze(0)
        if mi.shape != (H, W):
            mi = cv2.resize(mi.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        masks.append((mi > 0.5).astype(np.uint8))
    return masks

# -----------------
# Main
# -----------------
def main():
    seg_model  = YOLO(SEG_WEIGHTS)
    pose_model = YOLO(POSE_WEIGHTS)

    print("🧩 Building shared-backbone dual-head model ...")
    unified = DualHeadSegPose(seg_model, pose_model, BACKBONE_END_IDX).to(DEVICE).eval()

    img = cv2.imread(IMG_PATH)
    assert img is not None, f"❌ Image not found: {IMG_PATH}"
    inp = preprocess(img, IMG_SIZE).to(DEVICE)

    # ---- Warmup (no timing)
    for _ in range(WARMUP):
        with torch.no_grad():
            _ = unified(inp)
    if torch.cuda.is_available(): torch.cuda.synchronize()

    # ---- Timed runs (shared backbone only)
    t0 = time.time()
    for _ in range(RUNS):
        with torch.no_grad():
            seg_out, pose_out = unified(inp)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t1 = time.time()
    avg_ms = (t1 - t0) * 1000.0 / RUNS
    fps = 1000.0 / avg_ms
    print(f"⚡ Unified forward avg: {avg_ms:.2f} ms  ({fps:.2f} FPS)  [{RUNS} runs, warmup {WARMUP}]")

    # ---- Decode for visualization (no extra backbone timing)
    # We use single-image .predict() to get masks/keypoints tensors (draw-only).
    # This does re-run full model internally, but we DO NOT include it in FPS.
    seg_res  = seg_model.predict(source=img, conf=CONF_THRES, verbose=False, device=DEVICE, imgsz=IMG_SIZE)
    pose_res = pose_model.predict(source=img, conf=CONF_THRES, verbose=False, device=DEVICE, imgsz=IMG_SIZE)

    H, W = img.shape[:2]
    masks = robust_mask_list(seg_res[0] if len(seg_res) else None, (H, W))
    kpts  = pose_res[0].keypoints.data.cpu().numpy() if (len(pose_res) and pose_res[0].keypoints is not None) else None

    print(f"🧪 Instances — masks: {len(masks)}; keypoints batches: {0 if kpts is None else kpts.shape[0]}")

    vis = img.copy()
    vis = draw_masks(vis, masks, MASK_ALPHA)
    vis = draw_keypoints(vis, kpts)

    cv2.putText(vis, f"Unified FPS (backbone+heads only): {fps:.2f}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    out_path = OUT_DIR / Path(IMG_PATH).name
    cv2.imwrite(str(out_path), vis)
    print(f"✅ Saved → {out_path}")

if __name__ == "__main__":
    main()

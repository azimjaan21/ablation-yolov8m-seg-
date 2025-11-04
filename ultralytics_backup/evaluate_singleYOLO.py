#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single YOLO-Segmentation Evaluation (for Glove Detection)
==========================================================

Evaluates a single YOLOv8-Seg model on a validation set using:
- IoU-based Precision / Recall / F1 metrics
- Per-class and macro F1 scores
- Visual overlay results (TP, FP, FN)
- Optional Ultralytics mAP@50-95 benchmark
- CSV logging for comparison tables
"""

import os, cv2, csv
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path

# ===============================
# CONFIGURATION
# ===============================
MODEL_PATH = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\runs\yolov8m-seg\yolov8m-seg\weights\best.pt"
DATA_DIR   = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\data\valid"
OUT_DIR    = Path("exp_results/single_yolo_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRES = 0.25
IOU_THRESHOLD = 0.5
IMG_EXTS = (".jpg", ".jpeg", ".png")

# ===============================
# Helper Functions
# ===============================
def parse_yolo_seg_label(label_path, img_w, img_h):
    """Parse YOLO polygon .txt file → pixel coordinates."""
    gt_gloves, gt_nogloves = [], []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                cid = int(parts[0])
                pts = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
                pts[:, 0] *= img_w
                pts[:, 1] *= img_h
                pts = pts.astype(np.int32)
                if cid == 0: gt_gloves.append(pts)
                else: gt_nogloves.append(pts)
    return gt_gloves, gt_nogloves


def calculate_iou(mask1, mask2, h, w):
    """Compute IoU between two polygon masks."""
    blank = np.zeros((h, w), dtype=np.uint8)
    m1 = cv2.fillPoly(blank.copy(), [mask1], 1)
    m2 = cv2.fillPoly(blank.copy(), [mask2], 1)
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return inter / union if union > 0 else 0.0


def draw_results(img, gt_g, gt_ng, pred_g, pred_ng, fps):
    """Draw ground truths, predictions, and FPS."""
    vis = img.copy()
    # GT outlines (thin)
    for gt in gt_g:  cv2.polylines(vis, [gt], True, (0, 200, 0), 1)
    for gt in gt_ng: cv2.polylines(vis, [gt], True, (0, 0, 200), 1)
    # Predictions (bold)
    for m in pred_g:  cv2.polylines(vis, [m], True, (0, 255, 0), 2)
    for m in pred_ng: cv2.polylines(vis, [m], True, (0, 0, 255), 2)
    cv2.putText(vis, f"FPS:{fps:.1f}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    return vis


# ===============================
# Evaluation Logic
# ===============================
def run_single_yolo_eval(model_path, data_dir, out_dir):
    model = YOLO(model_path)
    img_dir = Path(data_dir) / "images"
    lbl_dir = Path(data_dir) / "labels"
    vis_dir = Path(out_dir) / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    TP, FP, FN = {0: 0, 1: 0}, {0: 0, 1: 0}, {0: 0, 1: 0}

    # Measure inference FPS
    dummy = torch.randn(1, 3, 640, 640).to(DEVICE)
    for _ in range(5): model(dummy)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    import time
    t0 = time.time()
    for _ in range(50): model(dummy)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.time()
    fps = 50 / (t1 - t0)
    print(f"⚡ Avg Inference Speed: {(t1 - t0)/50*1000:.2f} ms  ({fps:.2f} FPS)")

    # Dataset loop
    for img_file in os.listdir(img_dir):
        if not img_file.lower().endswith(IMG_EXTS): continue

        img_path = img_dir / img_file
        lbl_path = lbl_dir / (Path(img_file).stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]

        gt_g, gt_ng = parse_yolo_seg_label(lbl_path, w, h)
        match_g, match_ng = [False]*len(gt_g), [False]*len(gt_ng)

        # Prediction
        results = model.predict(img, conf=CONF_THRES, device=DEVICE, verbose=False)
        pred_g, pred_ng = [], []
        if len(results) and results[0].masks is not None:
            for mask, cls in zip(results[0].masks.xy, results[0].boxes.cls):
                mask = mask.astype(np.int32)
                if int(cls) == 0: pred_g.append(mask)
                else: pred_ng.append(mask)

        # Gloves evaluation
        for m in pred_g:
            matched = False
            for i, gt in enumerate(gt_g):
                if not match_g[i] and calculate_iou(m, gt, h, w) > IOU_THRESHOLD:
                    TP[0]+=1; match_g[i]=True; matched=True; break
            if not matched: FP[0]+=1
        FN[0]+=match_g.count(False)

        # No-gloves evaluation
        for m in pred_ng:
            matched = False
            for i, gt in enumerate(gt_ng):
                if not match_ng[i] and calculate_iou(m, gt, h, w) > IOU_THRESHOLD:
                    TP[1]+=1; match_ng[i]=True; matched=True; break
            if not matched: FP[1]+=1
        FN[1]+=match_ng.count(False)

        # Visualization
        vis = draw_results(img, gt_g, gt_ng, pred_g, pred_ng, fps)
        cv2.imwrite(str(vis_dir / img_file), vis)

    # Metrics
    def calc(cls):
        p = TP[cls]/(TP[cls]+FP[cls]+1e-6)
        r = TP[cls]/(TP[cls]+FN[cls]+1e-6)
        f1 = 2*p*r/(p+r+1e-6)
        return p, r, f1
    p0,r0,f0 = calc(0)
    p1,r1,f1 = calc(1)
    macro_f1 = (f0 + f1) / 2

    print("\n========== RESULTS ==========")
    print(f"Gloves:    P={p0:.3f} R={r0:.3f} F1={f0:.3f}")
    print(f"No-Gloves: P={p1:.3f} R={r1:.3f} F1={f1:.3f}")
    print(f"Macro-F1:  {macro_f1:.3f}")
    print("=============================\n")

    # Save CSV
    csv_path = Path(out_dir) / "results.csv"
    with open(csv_path, "w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["Class","TP","FP","FN","Precision","Recall","F1","FPS"])
        wri.writerow(["Gloves",TP[0],FP[0],FN[0],p0,r0,f0,fps])
        wri.writerow(["NoGloves",TP[1],FP[1],FN[1],p1,r1,f1,fps])
        wri.writerow(["MacroAvg","","","", "","",macro_f1,fps])
    print(f"✅ Results saved to {csv_path}")

    # Optional: mAP@50-95 benchmark
    print("\n📊 Running Ultralytics built-in mAP evaluation...")
    try:
        model.val(data=os.path.join(DATA_DIR, "..", "gloves.yaml"), task="segment", device=DEVICE)
    except Exception as e:
        print(f"⚠️ mAP evaluation skipped: {e}")


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    run_single_yolo_eval(MODEL_PATH, DATA_DIR, OUT_DIR)

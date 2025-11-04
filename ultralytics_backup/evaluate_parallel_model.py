#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel Multitask Evaluation (Segmentation + Wrist Pose)
=========================================================

- Runs two separate YOLO models (pose + segmentation)
- Adaptive wrist–mask fusion
- IoU-based evaluation (Precision / Recall / F1)
- FPS measurement for combined inference
- GFLOPs + Params for both models (Ultralytics-native)
- Overlay visualization + CSV results
"""

import os, time, cv2, csv, io, sys, re
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path

# ==============================
# CONFIGURATION
# ==============================
SEG_MODEL_PATH  = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\runs\yolov8m-seg\yolov8m-seg\weights\best.pt"
POSE_MODEL_PATH = r"runs\multitask_segpose_adv\1_pose_pretrain_industrial\weights\best.pt"
DATA_DIR        = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\data\valid"
OUT_DIR         = Path("exp_results/parallel_fusion_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRES = 0.25
IOU_THRESHOLD = 0.5
IMG_SIZE = 640
MASK_ALPHA = 0.45

WRIST_INDEXES = [0, 1]
SCALE_ALPHA   = 0.5
MIN_THRESHOLD = 40
MAX_THRESHOLD = 60


# ==============================
# Helper functions
# ==============================
def calculate_iou(mask1, mask2, h, w):
    blank = np.zeros((h, w), dtype=np.uint8)
    m1 = cv2.fillPoly(blank.copy(), [mask1], 1)
    m2 = cv2.fillPoly(blank.copy(), [mask2], 1)
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return inter / union if union > 0 else 0.0


def parse_yolo_seg_label(label_path, img_w, img_h):
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


def get_adaptive_threshold(mask_pts):
    x, y = zip(*mask_pts)
    bw, bh = max(x)-min(x), max(y)-min(y)
    thr = SCALE_ALPHA * np.sqrt(bw * bh)
    return np.clip(thr, MIN_THRESHOLD, MAX_THRESHOLD)


def point_to_line_distance(p, a, b):
    a, b, p = np.array(a), np.array(b), np.array(p)
    v = b - a
    if np.dot(v, v) == 0:
        return np.linalg.norm(p - a)
    t = max(0, min(1, np.dot(p - a, v) / np.dot(v, v)))
    proj = a + t * v
    return np.linalg.norm(p - proj)


def is_wrist_near_mask(wrist, mask_pts):
    thr = get_adaptive_threshold(mask_pts)
    if cv2.pointPolygonTest(mask_pts, wrist, False) >= 0:
        return True
    for i in range(len(mask_pts) - 1):
        if point_to_line_distance(wrist, mask_pts[i], mask_pts[i + 1]) < thr:
            return True
    return False


# ==========================================
#   Ultralytics-native Params + GFLOPs
# ==========================================
def measure_model_complexity_yolo(model, name="Model", img_size=640):
    """Extract true GFLOPs and Params from Ultralytics YOLO summary reliably (v8.3+ compatible)."""
    import re
    from contextlib import redirect_stdout
    import io

    print(f"\n📊 {name} summary:")

    # Capture Ultralytics printed output
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        model.info(detailed=False, verbose=True)
    output = buffer.getvalue()

    # --- NEW: fallback capture using model.info_dict if available ---
    gflops = 0.0
    params = 0.0

    # Regex parse GFLOPs
    match_flops = re.search(r"([0-9]+\.?[0-9]*)\s*GFLOPs", output)
    if match_flops:
        gflops = float(match_flops.group(1))

    # Regex parse parameters
    match_params = re.search(r"([0-9,]+)\s*parameters", output)
    if match_params:
        params = float(match_params.group(1).replace(",", "")) / 1e6
    else:
        params = sum(p.numel() for p in model.model.parameters()) / 1e6

    # Fallback if regex missed but info_dict exists
    if hasattr(model.model, "info_dict"):
        info = model.model.info_dict
        if info.get("params"): params = info["params"] / 1e6
        if info.get("flops"): gflops = info["flops"] / 1e9

    print(output.strip())
    print(f"✅ {name}: Params={params:.3f}M | GFLOPs={gflops:.2f}")
    return gflops, params


# ==============================
# Evaluation Logic
# ==============================
def run_parallel_experiment(seg_path, pose_path, data_dir, out_dir):
    seg_model  = YOLO(seg_path)
    pose_model = YOLO(pose_path)

    # === Measure model complexities ===
    print("\n🔍 Measuring Model Complexities...")

    seg_flops, seg_params = measure_model_complexity_yolo(seg_model, "Segmentation", IMG_SIZE)
    pose_flops, pose_params = measure_model_complexity_yolo(pose_model, "Pose", IMG_SIZE)
    total_flops = seg_flops + pose_flops
    total_params = seg_params + pose_params
    print(f"\n📈 Combined Parallel Model — Params: {total_params:.3f}M | GFLOPs: {total_flops:.2f}\n")

    img_dir = Path(data_dir) / "images"
    lbl_dir = Path(data_dir) / "labels"
    vis_dir = Path(out_dir) / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    TP, FP, FN = {0:0, 1:0}, {0:0, 1:0}, {0:0, 1:0}

    # Warm-up & FPS
    print("⏱️ Measuring parallel inference FPS ...")
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    for _ in range(5):
        _ = pose_model(dummy)
        _ = seg_model(dummy)
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    t0 = time.time()
    for _ in range(50):
        _ = pose_model(dummy)
        _ = seg_model(dummy)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.time()
    fps = 50 / (t1 - t0)
    print(f"⚡ Parallel average inference: {(t1 - t0)/50*1000:.2f} ms  ({fps:.2f} FPS)")

    # === Dataset loop ===
    for img_file in os.listdir(img_dir):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = img_dir / img_file
        lbl_path = lbl_dir / (Path(img_file).stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]

        # 1️⃣ Pose
        pose_res = pose_model.predict(img, conf=CONF_THRES, device=DEVICE, verbose=False)
        wrists = []
        if len(pose_res) and pose_res[0].keypoints is not None:
            kpts = pose_res[0].keypoints.data.cpu().numpy()
            for p in kpts:
                for wi in WRIST_INDEXES:
                    wrists.append((int(p[wi, 0]), int(p[wi, 1])))

        # 2️⃣ Segmentation
        seg_res = seg_model.predict(img, conf=CONF_THRES, device=DEVICE, verbose=False)
        gloves, nogloves = [], []
        if len(seg_res) and seg_res[0].masks is not None:
            for mask, cls in zip(seg_res[0].masks.xy, seg_res[0].boxes.cls):
                if int(cls) == 0: gloves.append(mask.astype(np.int32))
                else: nogloves.append(mask.astype(np.int32))

        # 3️⃣ Adaptive Fusion
        valid_g, valid_ng = [], []
        for m in gloves:
            if any(is_wrist_near_mask(wr, m) for wr in wrists): valid_g.append(m)
        for m in nogloves:
            if any(is_wrist_near_mask(wr, m) for wr in wrists): valid_ng.append(m)

        # 4️⃣ Ground truth
        gt_g, gt_ng = parse_yolo_seg_label(lbl_path, w, h)
        match_g = [False]*len(gt_g)
        match_ng = [False]*len(gt_ng)

        # Evaluate
        for m in valid_g:
            matched = False
            for i, gt in enumerate(gt_g):
                if not match_g[i] and calculate_iou(m, gt, h, w) > IOU_THRESHOLD:
                    TP[0]+=1; match_g[i]=True; matched=True; break
            if not matched: FP[0]+=1
        FN[0]+=match_g.count(False)

        for m in valid_ng:
            matched = False
            for i, gt in enumerate(gt_ng):
                if not match_ng[i] and calculate_iou(m, gt, h, w) > IOU_THRESHOLD:
                    TP[1]+=1; match_ng[i]=True; matched=True; break
            if not matched: FP[1]+=1
        FN[1]+=match_ng.count(False)

        # Visualization
        vis = img.copy()
        for gt in gt_g: cv2.polylines(vis, [gt], True, (0,200,0), 1)
        for gt in gt_ng: cv2.polylines(vis, [gt], True, (0,0,200), 1)
        for m in valid_g: cv2.polylines(vis, [m], True, (0,255,0), 2)
        for m in valid_ng: cv2.polylines(vis, [m], True, (0,0,255), 2)
        cv2.putText(vis, f"FPS:{fps:.1f}", (15,35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        cv2.imwrite(str(vis_dir / img_file), vis)

    # === Metrics ===
    def calc(cls):
        p = TP[cls]/(TP[cls]+FP[cls]+1e-6)
        r = TP[cls]/(TP[cls]+FN[cls]+1e-6)
        f1= 2*p*r/(p+r+1e-6)
        return p,r,f1

    p0,r0,f0 = calc(0)
    p1,r1,f1 = calc(1)
    macro_f1 = (f0+f1)/2

    print("\n========== RESULTS (Parallel) ==========")
    print(f"Gloves:    P={p0:.3f} R={r0:.3f} F1={f0:.3f}")
    print(f"No-Gloves: P={p1:.3f} R={r1:.3f} F1={f1:.3f}")
    print(f"Macro-F1:  {macro_f1:.3f}")
    print(f"GFLOPs: Seg={seg_flops:.2f}, Pose={pose_flops:.2f}, Total={total_flops:.2f}")
    print(f"Params:  Seg={seg_params:.3f}, Pose={pose_params:.3f}, Total={total_params:.3f}")
    print("========================================\n")

    # === Save CSV ===
    csv_path = Path(out_dir) / "results.csv"
    with open(csv_path, "w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["Class","TP","FP","FN","Precision","Recall","F1","FPS",
                      "Seg_GFLOPs","Pose_GFLOPs","Total_GFLOPs","Seg_Params","Pose_Params","Total_Params"])
        wri.writerow(["Gloves",TP[0],FP[0],FN[0],p0,r0,f0,fps,seg_flops,pose_flops,total_flops,seg_params,pose_params,total_params])
        wri.writerow(["NoGloves",TP[1],FP[1],FN[1],p1,r1,f1,fps,seg_flops,pose_flops,total_flops,seg_params,pose_params,total_params])
        wri.writerow(["MacroAvg","","","", "","",macro_f1,fps,seg_flops,pose_flops,total_flops,seg_params,pose_params,total_params])
    print(f"✅ Results saved to {csv_path}")


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    run_parallel_experiment(SEG_MODEL_PATH, POSE_MODEL_PATH, DATA_DIR, OUT_DIR)

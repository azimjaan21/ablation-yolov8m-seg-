#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Multitask Evaluation (Segmentation + Wrist Pose)
========================================================

- Shared backbone dual-head model (1 backbone, 2 heads)
- Adaptive wrist–mask fusion
- IoU-based evaluation (Precision / Recall / F1)
- FPS measurement on unified forward
- GFLOPs + Params measurement
- Overlay visualization + CSV results
"""

import os, cv2, time, csv
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch.nn as nn
from thop import profile, clever_format  # <-- NEW

# ==============================
# ===== CONFIGURATION ==========
# ==============================
SEG_WEIGHTS  = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\runs\multitask_lkalite_segpose\2_seg_finetune\weights\best.pt"
POSE_WEIGHTS = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\runs\multitask_lkalite_segpose\1_pose_industrial\weights\best.pt"
DATA_DIR     = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\data\valid"

OUT_DIR      = Path("exp_results/multitask_fusion_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 640
CONF_THRES = 0.25
BACKBONE_END_IDX = 15
IOU_THRESHOLD = 0.5
MASK_ALPHA = 0.45

# Adaptive wrist–mask fusion params
WRIST_INDEXES = [0, 1]
SCALE_ALPHA = 0.5
MIN_THRESHOLD = 40
MAX_THRESHOLD = 60


# =======================================
# ======= Dual-Head Shared Backbone =====
# =======================================
class DualHeadSegPose(nn.Module):
    """YOLOv8 shared backbone feeding two task heads (seg + pose)."""
    def __init__(self, seg_model: YOLO, pose_model: YOLO, backbone_end_idx=15):
        super().__init__()
        seg_raw, pose_raw = seg_model.model, pose_model.model
        self.backbone_layers = nn.ModuleList(list(pose_raw.model[:backbone_end_idx]))
        self.seg_head_layers = nn.ModuleList(list(seg_raw.model[backbone_end_idx:]))
        self.pose_head_layers = nn.ModuleList(list(pose_raw.model[backbone_end_idx:]))

    @torch.no_grad()
    def forward(self, x):
        outputs = []
        for m in self.backbone_layers:
            fi = getattr(m, "f", -1)
            if isinstance(fi, int):
                x_in = x if fi == -1 else outputs[fi]
            else:
                x_in = [outputs[j] if j != -1 else x for j in fi]
            x = m(x_in) if isinstance(x_in, list) else m(x_in)
            outputs.append(x)
        # Seg head
        y_seg = x; seg_feats = outputs.copy()
        for m in self.seg_head_layers:
            fi = getattr(m, "f", -1)
            x_in = y_seg if fi == -1 else (seg_feats[fi] if isinstance(fi, int)
                      else [seg_feats[j] if j != -1 else y_seg for j in fi])
            y_seg = m(x_in) if isinstance(x_in, list) else m(x_in)
            seg_feats.append(y_seg)
        # Pose head
        y_pose = x; pose_feats = outputs.copy()
        for m in self.pose_head_layers:
            fi = getattr(m, "f", -1)
            x_in = y_pose if fi == -1 else (pose_feats[fi] if isinstance(fi, int)
                      else [pose_feats[j] if j != -1 else y_pose for j in fi])
            y_pose = m(x_in) if isinstance(x_in, list) else m(x_in)
            pose_feats.append(y_pose)
        return y_seg, y_pose


# =======================================
# ======= Helper Functions ==============
# =======================================
def calculate_iou(mask1, mask2, h, w):
    blank = np.zeros((h, w), dtype=np.uint8)
    m1 = cv2.fillPoly(blank.copy(), [mask1], 1)
    m2 = cv2.fillPoly(blank.copy(), [mask2], 1)
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return intersection / union if union > 0 else 0.0

def parse_yolo_seg_label(label_path, img_w, img_h):
    gt_gloves, gt_no_gloves = [], []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                d = line.strip().split()
                cid = int(d[0])
                pts = np.array([float(x) for x in d[1:]]).reshape(-1, 2)
                pts[:, 0] *= img_w; pts[:, 1] *= img_h
                polygon = pts.astype(np.int32)
                if cid == 0: gt_gloves.append(polygon)
                elif cid == 1: gt_no_gloves.append(polygon)
    return gt_gloves, gt_no_gloves

def get_adaptive_threshold(mask_pts):
    x, y = zip(*mask_pts)
    bw, bh = max(x) - min(x), max(y) - min(y)
    thr = SCALE_ALPHA * np.sqrt(bw * bh)
    return np.clip(thr, MIN_THRESHOLD, MAX_THRESHOLD)

def point_to_line_distance(p, a, b):
    a, b, p = np.array(a), np.array(b), np.array(p)
    v = b - a
    if np.dot(v, v) == 0: return np.linalg.norm(p - a)
    t = max(0, min(1, np.dot(p - a, v) / np.dot(v, v)))
    proj = a + t * v
    return np.linalg.norm(p - proj)

def is_wrist_near_mask(wrist, mask_pts):
    thr = get_adaptive_threshold(mask_pts)
    if cv2.pointPolygonTest(mask_pts, wrist, False) >= 0:
        return True
    for i in range(len(mask_pts)-1):
        if point_to_line_distance(wrist, mask_pts[i], mask_pts[i+1]) < thr:
            return True
    return False


# =======================================
# ======= Measure Model Complexity ======
# =======================================

def measure_model_complexity(model, img_size=640):
    """
    Compute GFLOPs and Params for the unified YOLO dual-head model using THOP.
    This works even if the model isn't a native YOLO class.
    """
    from thop import profile
    print("\n📊 Measuring Unified Model Complexity...")

    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)

    # THOP FLOPs & Params
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    gflops = flops / 1e9
    params_m = params / 1e6

    print(f"✅ Unified Model Summary: Params={params_m:.3f}M | GFLOPs={gflops:.2f}\n")
    return gflops, params_m




# =======================================
# ======= Unified Experiment =============
# =======================================
def run_unified_experiment(seg_weights, pose_weights, data_dir, output_dir):
    seg_model = YOLO(seg_weights)
    pose_model = YOLO(pose_weights)
    unified = DualHeadSegPose(seg_model, pose_model, backbone_end_idx=BACKBONE_END_IDX).to(DEVICE).eval()

    # Compute GFLOPs and Params
    flops, params = measure_model_complexity(unified, IMG_SIZE)

    img_dir = Path(data_dir) / "images"
    lbl_dir = Path(data_dir) / "labels"
    out_vis = Path(output_dir) / "vis"
    out_vis.mkdir(parents=True, exist_ok=True)

    TP, FP, FN = {0:0, 1:0}, {0:0, 1:0}, {0:0, 1:0}

    # Warm-up + FPS
    sample = torch.randn(1,3,IMG_SIZE,IMG_SIZE).to(DEVICE)
    for _ in range(5): unified(sample)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    for _ in range(50): unified(sample)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.time()
    fps = 50 / (t1 - t0)
    print(f"⚡ Unified forward avg: {(t1 - t0)/50*1000:.2f} ms  ({fps:.2f} FPS)")

    # Evaluation Loop
    for img_file in os.listdir(img_dir):
        if not img_file.lower().endswith(('.jpg','.jpeg','.png')): continue
        img_path = img_dir / img_file
        label_path = lbl_dir / (Path(img_file).stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]

        pose_res = pose_model.predict(img, conf=CONF_THRES, device=DEVICE, verbose=False)
        wrists = []
        if len(pose_res) and pose_res[0].keypoints is not None:
            kpts = pose_res[0].keypoints.data.cpu().numpy()
            for p in kpts:
                for wi in WRIST_INDEXES:
                    wrists.append((int(p[wi,0]), int(p[wi,1])))

        seg_res = seg_model.predict(img, conf=CONF_THRES, device=DEVICE, verbose=False)
        gloves, nogloves = [], []
        if len(seg_res) and seg_res[0].masks is not None:
            for mask, cls in zip(seg_res[0].masks.xy, seg_res[0].boxes.cls):
                if int(cls)==0: gloves.append(mask.astype(np.int32))
                else: nogloves.append(mask.astype(np.int32))

        valid_g, valid_ng = [], []
        for m in gloves:
            if any(is_wrist_near_mask(wr, m) for wr in wrists): valid_g.append(m)
        for m in nogloves:
            if any(is_wrist_near_mask(wr, m) for wr in wrists): valid_ng.append(m)

        gt_gloves, gt_nogloves = parse_yolo_seg_label(label_path, w, h)
        match_g, match_ng = [False]*len(gt_gloves), [False]*len(gt_nogloves)

        for m in valid_g:
            matched=False
            for i,g in enumerate(gt_gloves):
                if not match_g[i] and calculate_iou(m,g,h,w)>IOU_THRESHOLD:
                    TP[0]+=1; match_g[i]=True; matched=True; break
            if not matched: FP[0]+=1
        FN[0]+=match_g.count(False)

        for m in valid_ng:
            matched=False
            for i,g in enumerate(gt_nogloves):
                if not match_ng[i] and calculate_iou(m,g,h,w)>IOU_THRESHOLD:
                    TP[1]+=1; match_ng[i]=True; matched=True; break
            if not matched: FP[1]+=1
        FN[1]+=match_ng.count(False)

        vis = img.copy()
        for gt in gt_gloves: cv2.polylines(vis,[gt],True,(0,200,0),1)
        for gt in gt_nogloves: cv2.polylines(vis,[gt],True,(0,0,200),1)
        for m in valid_g: cv2.polylines(vis,[m],True,(0,255,0),2)
        for m in valid_ng: cv2.polylines(vis,[m],True,(0,0,255),2)
        cv2.putText(vis,f"FPS:{fps:.1f}",(15,35),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
        cv2.imwrite(str(out_vis / img_file), vis)

    def calc(cls):
        p = TP[cls]/(TP[cls]+FP[cls]+1e-6)
        r = TP[cls]/(TP[cls]+FN[cls]+1e-6)
        f1= 2*p*r/(p+r+1e-6)
        return p,r,f1
    p0,r0,f0 = calc(0); p1,r1,f1 = calc(1)
    macro_f1 = (f0+f1)/2

    print("\n========== RESULTS ==========")
    print(f"Gloves:    P={p0:.3f} R={r0:.3f} F1={f0:.3f}")
    print(f"No-Gloves: P={p1:.3f} R={r1:.3f} F1={f1:.3f}")
    print(f"Macro-F1:  {macro_f1:.3f}")
    print(f"GFLOPs:    {flops} | Params: {params}")
    print("=============================\n")

    # Save CSV
    with open(Path(output_dir)/"results.csv","w",newline="") as f:
        wri=csv.writer(f)
        wri.writerow(["Class","TP","FP","FN","Precision","Recall","F1","FPS","GFLOPs","Params"])
        wri.writerow(["Gloves",TP[0],FP[0],FN[0],p0,r0,f0,fps,flops,params])
        wri.writerow(["NoGloves",TP[1],FP[1],FN[1],p1,r1,f1,fps,flops,params])
        wri.writerow(["MacroAvg","","","", "","",macro_f1,fps,flops,params])

    print(f"✅ Results saved to {output_dir}")


# =======================================
# ========== MAIN =======================
# =======================================
if __name__ == "__main__":
    run_unified_experiment(SEG_WEIGHTS, POSE_WEIGHTS, DATA_DIR, OUT_DIR)

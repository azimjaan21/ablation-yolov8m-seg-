import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

def parse_yolo_seg_label(label_path, img_width, img_height):
    polygons = []
    with open(label_path, 'r') as f:
        for line in f:
            data = line.strip().split()
            coords = np.array([float(x) for x in data[1:]]).reshape(-1, 2)
            coords[:, 0] *= img_width
            coords[:, 1] *= img_height
            polygon = coords.astype(np.int32)
            polygons.append(polygon)
    return polygons

def mask_from_polygons(polygons, img_shape):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for poly in polygons:
        cv2.fillPoly(mask, [poly], 1)
    return mask

def calculate_iou(mask_gt, mask_pred):
    intersection = np.logical_and(mask_gt, mask_pred).sum()
    union = np.logical_or(mask_gt, mask_pred).sum()
    iou = intersection / union if union > 0 else 0.0
    return iou

def apply_mask_with_opacity(image, mask, color, alpha=0.5):
    overlay = image.copy()
    color_arr = np.array(color, dtype=np.uint8)
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = (1 - alpha) * image[mask_bool] + alpha * color_arr
    return overlay.astype(np.uint8)

def draw_polygons_edges(image, polygons, color=(0, 0, 0), thickness=1):
    image_with_edges = image.copy()
    for poly in polygons:
        cv2.polylines(image_with_edges, [poly], isClosed=True, color=color, thickness=thickness)
    return image_with_edges

def highlight_tp_fp_fn(image, mask_gt, mask_pred, alpha=0.5):
    TP = np.logical_and(mask_gt, mask_pred)
    FP = np.logical_and(~mask_gt.astype(bool), mask_pred.astype(bool))
    FN = np.logical_and(mask_gt.astype(bool), ~mask_pred.astype(bool))
    overlay = image.copy()
    overlay[TP] = (192, 192, 192)    # TP: gray
    overlay[FP] = (0, 255, 255)      # FP: cyan
    overlay[FN] = (255, 0, 0)        # FN: red
    blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    return blended

# --- Set paths ---
image_path = 'example.jpg'
gt_label_path = 'example.txt'
model_path = 'yolov8m_seg.pt'

# --- Read image and prepare masks ---
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w = image.shape[:2]
gt_polygons = parse_yolo_seg_label(gt_label_path, w, h)
mask_gt = mask_from_polygons(gt_polygons, image.shape)

# --- YOLOv8m-seg - Prediction ---
model = YOLO(model_path)
results = model.predict(image, task="segment")

pred_polygons = []
if results[0].masks is not None:
    for mask in results[0].masks.xy:
        poly = mask.astype(np.int32)
        pred_polygons.append(poly)
mask_pred = mask_from_polygons(pred_polygons, image.shape)

iou_value = calculate_iou(mask_gt, mask_pred)
print(f"IoU: {iou_value:.4f}")

# --- Overlays and overlap with borders only on overlap panel ---
gt_overlay = apply_mask_with_opacity(image, mask_gt, (255, 0, 0), alpha=0.5)
pred_overlay = apply_mask_with_opacity(image, mask_pred, (0, 255, 255), alpha=0.5)
overlap_img = highlight_tp_fp_fn(image, mask_gt, mask_pred, alpha=0.5)
overlap_img_edges = draw_polygons_edges(overlap_img, gt_polygons, color=(0, 0, 0), thickness=1)

# --- Plotting the results ---
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].imshow(gt_overlay)
axs[0].set_title("Ground Truth Mask")
axs[1].imshow(pred_overlay)
axs[1].set_title("Predicted Mask")
axs[2].imshow(overlap_img_edges)
axs[2].set_title(f"Overlap (IoU = {iou_value:.3f})\nTP (gray) | FP (cyan) | FN (red)\nGT border: thin black line")
for ax in axs:
    ax.axis('off')
plt.tight_layout()
plt.show()
plt.savefig('mask_comparison_tp_fp_fn_edges.png')

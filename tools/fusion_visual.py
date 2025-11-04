import cv2
import numpy as np
from ultralytics import YOLO

# ------------------ Paths ------------------ #
img_path = "ppe.jpg"
pose_model_path = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\wrist_kpt\runs\pose\#trans_[ab1]\weights\best.pt"
seg_model_path = r"C:\Users\dalab\Desktop\azimjaan21\my_PAPERS\AAAI 25 Summer Symposium\Development\YOLOstarNET\experiment_models\p3head.pt"

# ------------------ Load models ------------------ #
pose_model = YOLO(pose_model_path)
seg_model = YOLO(seg_model_path)

# ------------------ Load image ------------------ #
image = cv2.imread(img_path)
height, width, _ = image.shape

# ------------------ Predict ------------------ #
pose_results = pose_model.predict(img_path, task="pose", conf=0.25)[0]
seg_results = seg_model.predict(img_path, task="segment", conf=0.25)[0]

# ------------------ Wrist keypoints ------------------ #
wrist_keypoints = []
if pose_results.keypoints is not None:
    for kp in pose_results.keypoints.xy:
        wrist_keypoints.append((int(kp[0][0]), int(kp[0][1])))
        wrist_keypoints.append((int(kp[1][0]), int(kp[1][1])))

# ------------------ Segmentation masks ------------------ #
glove_masks = []
glove_confs = []
if seg_results.masks is not None:
    for mask, cls, conf in zip(seg_results.masks.xy, seg_results.boxes.cls, seg_results.boxes.conf):
        if int(cls) == 0:  # glove class
            glove_masks.append(mask.astype(np.int32))
            glove_confs.append(conf.item())  # confidence as float

# ------------------ Overlay masks ------------------ #
overlay = image.copy()

# Draw glove masks: green fill with 50% opacity
mask_overlay = np.zeros_like(image)
for mask in glove_masks:
    cv2.fillPoly(mask_overlay, [mask], color=(0, 255, 0))  # green fill
overlay = cv2.addWeighted(overlay, 1.0, mask_overlay, 0.5, 0)

# Draw polygon borders: yellow
for mask in glove_masks:
    cv2.polylines(overlay, [mask], True, (0, 255, 255), 1)  # yellow border

# Draw labels with confidence above each mask (fixed background size)
for mask, conf in zip(glove_masks, glove_confs):
    x, y, w_box, h_box = cv2.boundingRect(mask)
    label = f"Gloves {conf:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    # # Draw rectangle background exactly around text
    # cv2.rectangle(overlay, (x, y - text_h - baseline - 2), (x + text_w + 4, y), (0, 255, 255), -1)
    # # Draw text
    # cv2.putText(overlay, label, (x + 2, y - 4), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

# # Draw wrist keypoints: red dots
# for wrist in wrist_keypoints:
#     cv2.circle(overlay, wrist, 2, (0, 0, 255), -1)  # red dot

# ------------------ Show and save ------------------ #
cv2.imwrite("wrist_mask_with_conf_fixed.png", overlay)
cv2.imshow("Wrist + Mask Visualization", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()

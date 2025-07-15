import os
import random
import shutil

# === Paths ===
root = 'C:/Users/dalab/Desktop/azimjaan21/RESEARCH/ablation_yolov8m_seg/data'
images_dir = os.path.join(root, 'all/images')
labels_dir = os.path.join(root, 'all/labels')
train_img_dir = os.path.join(root, 'train/images')
train_lbl_dir = os.path.join(root, 'train/labels')
val_img_dir = os.path.join(root, 'valid/images')
val_lbl_dir = os.path.join(root, 'valid/labels')

# === Create folders ===
for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
    os.makedirs(d, exist_ok=True)

# === Parameters ===
val_split = 0.2
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
random.shuffle(image_files)
val_count = int(len(image_files) * val_split)

# === Split and copy ===
for i, file in enumerate(image_files):
    base = os.path.splitext(file)[0]
    src_img = os.path.join(images_dir, file)
    src_lbl = os.path.join(labels_dir, base + '.txt')

    if i < val_count:
        dst_img = os.path.join(val_img_dir, file)
        dst_lbl = os.path.join(val_lbl_dir, base + '.txt')
    else:
        dst_img = os.path.join(train_img_dir, file)
        dst_lbl = os.path.join(train_lbl_dir, base + '.txt')

    shutil.copy(src_img, dst_img)
    if os.path.exists(src_lbl):
        shutil.copy(src_lbl, dst_lbl)

print(f"✅ Split complete: {len(image_files) - val_count} train / {val_count} val")

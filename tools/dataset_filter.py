import os
import shutil

original_images_dir = r'C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation[yolov8m-seg]\dataset\valid\images'
original_labels_dir = r'C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation[yolov8m-seg]\dataset\valid\labels'
filtered_images_dir = r'data/valid/images'
filtered_labels_dir = r'data/valid/labels'
target_class_id_to_remove = '1'

# === Create output directories ===
os.makedirs(filtered_images_dir, exist_ok=True)
os.makedirs(filtered_labels_dir, exist_ok=True)

# === Filter label files ===
label_files = [f for f in os.listdir(original_labels_dir) if f.endswith('.txt')]
copied_count = 0
skipped_count = 0

for label_file in label_files:
    label_path = os.path.join(original_labels_dir, label_file)

    # Read lines from the label file
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Remove lines that contain class_id = 1
    filtered_lines = [line for line in lines if not line.startswith(target_class_id_to_remove + ' ')]

    # Skip if no class_id = 0 remains
    if not filtered_lines:
        skipped_count += 1
        continue

    # Save filtered label
    new_label_path = os.path.join(filtered_labels_dir, label_file)
    with open(new_label_path, 'w') as f:
        f.writelines(filtered_lines)

    # Copy corresponding image
    base_name = label_file.replace('.txt', '')
    found = False
    for ext in ['.jpg', '.png', '.jpeg']:
        image_file = base_name + ext
        image_path = os.path.join(original_images_dir, image_file)
        if os.path.exists(image_path):
            shutil.copy(image_path, os.path.join(filtered_images_dir, image_file))
            copied_count += 1
            found = True
            break

    if not found:
        print(f"⚠️ Warning: Image not found for label {label_file}")

print("\n✅ Done.")
print(f"📝 Copied {copied_count} image/label pairs with class_id = 0 (gloves).")
print(f"🗑️ Skipped {skipped_count} files with only class_id = 1 (no-gloves).")

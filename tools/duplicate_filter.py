import os

images_dir = r'C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation[yolov8m-seg]\data\train\images'
labels_dir = r'C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation[yolov8m-seg]\data\train\labels'

image_extensions = ['.jpg', '.png', '.jpeg']
image_basenames = set()

for filename in os.listdir(images_dir):
    base, ext = os.path.splitext(filename)
    if ext.lower() in image_extensions:
        image_basenames.add(base)


deleted_count = 0
label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

for label_file in label_files:
    base = os.path.splitext(label_file)[0]
    if base not in image_basenames:
        label_path = os.path.join(labels_dir, label_file)
        os.remove(label_path)
        deleted_count += 1
        print(f"🗑️ Deleted: {label_file} (no matching image)")

print(f"\n✅ Done. {deleted_count} label files were deleted because they had no matching image.")

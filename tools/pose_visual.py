from ultralytics import YOLO
import cv2

# Load the pretrained YOLOv8 pose model
model = YOLO(r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\wrist_kpt\runs\pose\#trans_[ab2]\weights\best.pt")

# Run prediction
results = model("kpt.jpg")  # Replace with your image path

# Save the annotated image
for i, r in enumerate(results):
    annotated_img = r.plot()  # returns image with keypoints drawn
    cv2.imwrite(f"pose_result_{i}.jpg", annotated_img)

print("Image saved successfully.")

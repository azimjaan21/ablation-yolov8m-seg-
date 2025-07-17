import torch
import time
import cv2
from ultralytics import YOLO

# Load model
model = YOLO(r'C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\runs\segment\x2_rch\weights\best.pt')
model.to('cuda').eval()

# Load video
video_path = r'gloves.mp4'  
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise Exception("❌ Error opening video file")

# FPS calculation setup
frame_count = 0
total_time = 0.0

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize and normalize
        frame_resized = cv2.resize(frame, (640, 640))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0).cuda()

        # Run model and measure time
        start = time.time()
        _ = model(frame_tensor)
        torch.cuda.synchronize()
        end = time.time()

        total_time += (end - start)
        frame_count += 1

cap.release()

# Final FPS result
fps = frame_count / total_time if total_time > 0 else 0.0
print(f"\n🎥 Inference FPS on TITAN RTX from video (640x640): {fps:.2f} FPS")

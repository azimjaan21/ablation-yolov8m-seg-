import torch
import time
from ultralytics import YOLO  

# Load model
model = YOLO(r'C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\runs\segment\p3_only_head\weights\best.pt')  
model.to('cuda').eval()

# Warm-up with normalized dummy input
dummy = torch.rand(1, 3, 640, 640).cuda()  # [0, 1] range
for _ in range(10):
    _ = model(dummy)

# Benchmark
num_runs = 100
total_time = 0.0

with torch.no_grad():
    for _ in range(num_runs):
        dummy = torch.rand(1, 3, 640, 640).cuda()  # Re-randomize each time
        start = time.time()
        _ = model(dummy)
        torch.cuda.synchronize()
        end = time.time()
        total_time += (end - start)

fps = num_runs / total_time
print(f"FPS on TITAN RTX (batch=1, 640×640): {fps:.2f}")

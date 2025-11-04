import time
import torch
from ultralytics import YOLO

model = YOLO(r"C:\Users\dalab\Desktop\azimjaan21\my_PAPERS\AAAI 25 Summer Symposium\Development\YOLOstarNET\compare_models\LKA_p3.pt", task="segment")

dummy = torch.randn(1, 3, 640, 640).cuda()

# Warmup
for _ in range(10):
    model.predict(dummy, imgsz=640, verbose=False)

# Measure
t0 = time.perf_counter()
for _ in range(100):
    model.predict(dummy, imgsz=640, verbose=False)
t1 = time.perf_counter()

avg_time = (t1 - t0) / 100
fps = 1 / avg_time
print(f"Pure inference FPS: {fps:.2f}")

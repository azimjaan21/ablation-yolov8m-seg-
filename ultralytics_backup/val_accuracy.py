import time
import torch
import sys, os
import pandas as pd

# ============================================================
# Setup paths and import local Ultralytics repo
# ============================================================
sys.path.insert(0, os.path.abspath(".")) 
print("Using ultralytics from:", os.path.abspath("ultralytics"))

from ultralytics.models.yolo.model import YOLO


# ============================================================
# MAIN FUNCTION
# ============================================================
def main():
    DATA = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\data_wrist\data.yaml"
    DEVICE = 0
    IMG_SIZE = 640

    # ============================================================
    # List of Models to Evaluate
    # ============================================================
    models = {
        "LKALiteSHBB_pose": r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\runs\multitask_lkalite_segpose\1_pose_industrial\weights\best.pt",
        # Add more models if needed:
        # "YOLOv11m-Seg": "compare_models/yolo11m_seg.pt",
        # "LKA-YOLO": "compare_models/LKA_p3.pt",
    }

    results = []

    # ============================================================
    # Evaluation Loop
    # ============================================================
    for name, path in models.items():
        print(f"\n=== Evaluating {name} ===")
        model = YOLO(path)

        # --------------------------
        # Run Validation
        # --------------------------
        metrics = model.val(
            data=DATA,
            imgsz=IMG_SIZE,
            batch=16,
            device=DEVICE,
            verbose=True
        )

        # --------------------------
        # Measure FPS
        # --------------------------
        dummy = torch.rand(1, 3, IMG_SIZE, IMG_SIZE).to(f"cuda:{DEVICE}")
        with torch.no_grad():
            # warm-up
            for _ in range(10):
                model.predict(dummy, imgsz=IMG_SIZE, device=DEVICE, verbose=False)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(100):
                model.predict(dummy, imgsz=IMG_SIZE, device=DEVICE, verbose=False)
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        fps = 1 / ((t1 - t0) / 100)

        # --------------------------
        # Extract Metrics Safely
        # --------------------------
        def safe_metric(obj, attr):
            try:
                return getattr(obj, attr)
            except Exception:
                return None

        mAP50_box = safe_metric(metrics.box, "map50") if hasattr(metrics, "box") else None
        mAP50_seg = safe_metric(metrics.seg, "map50") if hasattr(metrics, "seg") else None
        mAP50_pose = safe_metric(metrics.keypoints, "map50") if hasattr(metrics, "keypoints") else None
        precision = safe_metric(metrics.box, "mp") if hasattr(metrics, "box") else None
        recall = safe_metric(metrics.box, "mr") if hasattr(metrics, "box") else None

        results.append({
            "Model": name,
            "FPS": fps,
            "mAP50_bbox": mAP50_box,
            "mAP50_mask": mAP50_seg,
            "mAP50_pose": mAP50_pose,
            "Precision": precision,
            "Recall": recall
        })

        # --------------------------
        # Print Summary per Model
        # --------------------------
        print(f"\n{name} Results:")
        print(f"  FPS: {fps:.2f}")
        if mAP50_box is not None:
            print(f"  mAP50 (bbox): {mAP50_box:.4f}")
        if mAP50_seg is not None:
            print(f"  mAP50 (mask): {mAP50_seg:.4f}")
        if mAP50_pose is not None:
            print(f"  mAP50 (pose): {mAP50_pose:.4f}")
        if precision is not None:
            print(f"  Precision: {precision:.4f}")
        if recall is not None:
            print(f"  Recall: {recall:.4f}")

    # ============================================================
    # Save All Results to CSV
    # ============================================================
    df = pd.DataFrame(results)
    df.to_csv("evaluation_summary.csv", index=False)
    print("\n✅ Evaluation complete! Results saved to evaluation_summary.csv")
    print(df)


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    main()

from ultralytics import YOLO

def main():

    model = YOLO(r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\runs\multi_LKA\multi_LKA\weights\best.pt",
                 task="segment")  

    # Run validation
    metrics = model.val(
        data=r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\ultralytics\ultralytics\cfg\datasets\gloves.yaml",
        imgsz=640,
        batch=1,
        visualize=True,
        project="val_results/multi_LKA",
        name="multi_LKA"
    )

    # Speed metrics
    print("Speed metrics (ms per image):", metrics.speed)
    fps = 1000 / sum(metrics.speed.values())
    print(f"Official FPS: {fps:.2f}")

if __name__ == "__main__":
    main()

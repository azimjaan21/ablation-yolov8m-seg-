from ultralytics import YOLO
import torch

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")

    COCO_PRETRAINED_MODEL = 'yolo11s-pose.pt'  # Official pretrained pose weights
    DATA_YAML = r'C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\data_wrist\data.yaml'
    IMG_SIZE = 640
    EPOCHS = 100
    BATCH_SIZE = 16
    WORKERS = 5
    PATIENCE = 15
    PROJECT_DIR = 'runs/pose/'
    EXPERIMENT_NAME = 'yolo11s_pose_wrist_only'

    print("[INFO] Loading pretrained YOLOv8s-pose weights...")
    model = YOLO(COCO_PRETRAINED_MODEL)

    print("[INFO] Modifying pose head for wrist-only training...")
    model.model.kpt_shape = [2, 3]  # Wrist-left and wrist-right keypoints

    # Optional: Freeze backbone and neck for transfer learning
    print("[INFO] Freezing backbone and neck layers...")
    for name, param in model.model.named_parameters():
        if 'backbone' in name or 'neck' in name:
            param.requires_grad = False

    print("[INFO] Starting training...")
    model.train(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        project=PROJECT_DIR,
        name=EXPERIMENT_NAME,
        device=device,
        workers=WORKERS,
        visualize=True,
        patience=PATIENCE,
        task="pose"
    )

if __name__ == "__main__":
    main()

from ultralytics import YOLO
import torch

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    MODEL_YAML = r'C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\ultralytics\ultralytics\cfg\models\v8\LKALite_withP5.yaml'  
    DATA_YAML = r'C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\data_ppe\data.yaml'                           
    IMG_SIZE = 640
    EPOCHS = 100
    BATCH_SIZE = 16
    WORKERS=0
    PROJECT_DIR = 'runs/PPE_LKALite_SPPF/'
    EXPERIMENT_NAME = 'PPE_LKALite_SPPF_experiment'

    print("[INFO] Loading ########Model######## architecture...")
    model = YOLO(MODEL_YAML)

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
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.02,
        momentum=0.937,
        cache=True
    )

if __name__ == "__main__":
    main()

from ultralytics import YOLO

def main():
    model = YOLO(r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\runs\segment\no_P5_sppf\weights\best.pt")  

    # Run validation (val dataset and set batch=1 for true FPS)
    metrics = model.val(data= r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\ultralytics\ultralytics\cfg\datasets\gloves.yaml",
                        imgsz=640, 
                        batch=1,
                        project='val_results',
                        name='no_P5')  
    # Set batch=1 for single-image FPS

    # speed metrics
    print("Speed metrics (ms per image):", metrics.speed)
    fps = 1000 / sum(metrics.speed.values())
    print(f"Official FPS: {fps:.2f}")

if __name__ == "__main__":
    main()
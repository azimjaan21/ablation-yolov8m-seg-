from ultralytics import YOLO


model = YOLO(r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\runs\segment\mask_64\weights\best.pt")  

image_folder = r'C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\data\valid\images/'  

results = model.predict(
    source=image_folder,
    imgsz=640,              
    save=True,              
    show=False,            
    project='visual_results/',
    name='yolov8m_seg',   
    exist_ok=True          
)

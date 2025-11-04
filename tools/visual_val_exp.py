from ultralytics import YOLO


model = YOLO(r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\wrist_kpt\runs\pose\#trans_[ab4]\weights\best.pt")  

image_folder = r'C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\data\valid\images/'  

results = model.predict(
    source=image_folder,
    imgsz=640,              
    save=True,              
    show=False,            
    project='wrist_results/',
    name='ab4',   
    exist_ok=True          
)

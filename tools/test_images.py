from pathlib import Path
INPUT_DIR = r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\data\valid\images"
INPUT_DIR = Path(INPUT_DIR).resolve()
print("Resolved:", INPUT_DIR)
print("Found:", len(list(INPUT_DIR.rglob('*.jpg'))))

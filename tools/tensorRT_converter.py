from ultralytics import YOLO

# # Convert pose model
# pose_model = YOLO("yolo11s-pose.pt")
# pose_model.export(format="engine")  # Produces 'yolo11s-pose.engine'

# Convert segmentation model
seg_model = YOLO("weights/star_p3head.pt")
seg_model.export(format="engine")   # Produces 'p3head.engine'

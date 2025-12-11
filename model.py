from ultralytics import YOLO

def create_yolov8_model():
  model = model = YOLO("yolov8n.pt")
  return model

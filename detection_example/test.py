from ultralytics import YOLO

model = YOLO('yolov8n.pt')   # model name


detection_predicts = model.predict(source="0", show=True, save=False,  show_labels = True)
# detection_predicts = model.predict(source="0", show=True, save=False, show_labels = True)



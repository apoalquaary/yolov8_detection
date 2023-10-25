from ultralytics import YOLO

model = YOLO('plant_detection.pt')   # model name


detection_predicts = model.predict(source="imgs", show=False, save=True,  show_labels = True)
# detection_predicts = model.predict(source="0", show=True, save=False, show_labels = True)



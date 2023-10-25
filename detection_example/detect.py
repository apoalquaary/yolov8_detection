from ultralytics import YOLO
import cv2 
import numpy as np
import os 


# call the model
model = YOLO('yolov8n.pt')

# classes names
names = model.names

# boxes design
rect_color = (0, 255, 0)
thickness = 2
fontScale = 2
font = cv2.FONT_HERSHEY_SIMPLEX
font_color = (0,0,255)

# webcam detection
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Set desired video capture properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # Set height
cap.set(cv2.CAP_PROP_FPS, 30)            # Set frames per second (FPS)

while True:
    ret, frame = cap.read()
    frame = model.predict(source=frame, show=False, save=False, show_conf=False, show_labels = False)

    img = frame[0].orig_img.copy()
    for box in frame[0].boxes:
        box_vec = box.xyxy.cpu().detach().numpy().copy()
        box_vec = np.squeeze(box_vec)
        box_vec = (np.rint(box_vec)).astype(int)
        cls_lbl = int(box.cls.cpu().detach().numpy().copy())
        point1 = (box_vec[0], box_vec[1])
        point2 = (box_vec[2], box_vec[3])

        img = cv2.rectangle(img, point1, point2, rect_color, thickness)
        img = cv2.putText(img, names[cls_lbl], (box_vec[0]+15, box_vec[1]-15), font, 
                   fontScale, font_color, thickness, cv2.LINE_AA)
        # print(point1, point2, names[cls_lbl])

        if (cls_lbl == 39):
            print("wow")

    cv2.imshow('Input', img)


    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


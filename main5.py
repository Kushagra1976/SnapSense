import cv2
from ultralytics import YOLO

model = YOLO('helmet_detection/runs/detect/train2/weights/best.pt')

results = model.predict(source=0, show=True)

from ultralytics import YOLO

model = YOLO('helmet_detection/modelLite/best_saved_model/best_float32.tflite')

results = model(source='helmet_detection/videoSample.mp4', show=True)
import cv2
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO("helmet_detection/runs/detect/train2/weights/best.pt")

# Open the video stream from the IP webcam
video_url = "helmet_detection/videoSample.mp4"  # URL of the IP webcam
cap = cv2.VideoCapture(video_url)

# Loop to read frames from the webcam stream
while True:
    ret, frame = cap.read()  # Read a frame from the stream
    if not ret:
        break  # Break the loop if there are no more frames

    # Predict objects in the frame using YOLO
    results = model.predict(frame, show=True)

    # Display the frame with detected objects
    cv2.imshow('frame', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()

from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error reading frame")
            break

        # Run detection
        results = model(frame, verbose=False)

        # Draw bounding boxes
        annotated_frame = results[0].plot()

        # Show frame
        cv2.imshow("YOLOv8 Object Detection", annotated_frame)

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

# Initialize the Picamera2

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

while True:
    # Capture frame-by-frame
    frame = picam2.capture_array()
    # Run YOLO11 inference on the frame
    results = model(frame)
    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    
    # Display the resulting frame
    
    detect_objects = []
    for result in results:
        for box in result.boxes:
            class_id=int(box.cls[0])
            name = result.names[class_id]
            detect_objects.append(f"{name}")
    print(detect_objects)
    
    cv2.imwrite("t3.jpg",annotated_frame)
    # cv2.imshow("Camera", annotated_frame)
    
    # Break the loop if 'q' is pressed
    # if cv2.waitKey(1) == ord("q"):
    break

# Release resources and close windows    source .venv/bin/activate
cv2.destroyAllWindows()


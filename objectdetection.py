import cv2
from ultralytics import YOLO

# Loading the YOLOv8 Nano model
model = YOLO('yolov8n.pt')

# Initialize the webcam (0 for default camera/ 1 for external camera)
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,500)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLOv8 inference on the frame
    # stream=True is more efficient for real-time video
    results = model(frame, stream=True)

    target_box = None
    max_area = 0
    target_label = ""

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get coordinates: [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Calculate Area (Width * Height) as a proxy for distance
            width = x1 - x2
            height = y1 - y2
            area = abs(width * height)

            # 4. Logic: Keep only the object with the largest area (closest)
            if area > max_area:
                max_area = area
                target_box = [int(x1), int(y1), int(x2), int(y2)]
                
                # Get class name and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                target_label = f"{model.names[cls]} {conf:.2f}"

    # 5. Visualize only the single "closest" object
    if target_box:
        x1, y1, x2, y2 = target_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, target_label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("YOLOv8 Single Object Detection", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
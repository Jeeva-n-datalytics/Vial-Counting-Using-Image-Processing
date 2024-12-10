from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO(r"C:\Users\Dell\Desktop\Project_200\11_Deployment code\train_100_Epoch_L_V5/best.pt")


# Path to your input video file
video_path = r"C:/Users/Dell/Desktop/Project_200/Videos All trained/Vials-20240703T030609Z-001/VID_20240618_164405.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video writer initialized to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = 'output_video.mp4'
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)

    # Draw bounding boxes and labels on the frame
    for result in results:
        boxes = result.boxes.xyxy  # Get bounding box coordinates
        confidences = result.boxes.conf  # Get confidence scores
        class_ids = result.boxes.cls  # Get class indices

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)  # Extract coordinates directly from box
            label = model.names[int(class_id)] if hasattr(model, 'names') else 'unknown'  # Assuming model.names exists
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame into the output video
    out.write(frame)

    # Display the frame
    cv2.imshow('YOLOv8 Video Inference', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

# Print the output video path
print(f"Output video saved at: {output_path}")

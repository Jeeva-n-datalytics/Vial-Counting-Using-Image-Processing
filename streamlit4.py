import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# Load YOLOv8 model
@st.cache_resource
def load_model():
    model = YOLO('best.pt')
    return model

model = load_model()

def detect_objects(image, model, confidence, class_name):
    # Convert PIL image to numpy array
    image_np = np.array(image)

    # Perform inference
    results = model(image_np)

    # Extract detections
    detections = results[0].boxes.data.cpu().numpy()
    class_map = {"syringe": 0, "vial": 1}
    class_id = class_map[class_name]

    # Filter results by confidence and class
    filtered_detections = [det for det in detections if det[4] > confidence and int(det[5]) == class_id]

    # Draw boxes on image
    for det in filtered_detections:
        x1, y1, x2, y2, conf, cls = det
        cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(image_np, f"{class_name} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    return Image.fromarray(image_np), len(filtered_detections)

# Streamlit app
st.title("Object Detection with YOLOv8")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Select class
class_name = st.selectbox("Select class", ["syringe", "vial"])

# Select confidence score
confidence = st.slider("Select confidence score", 0.0, 1.0, 0.5)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Run detection
    if st.button('Detect Objects'):
        output_image, count = detect_objects(image, model, confidence, class_name)
        
        # Display the output image
        st.image(output_image, caption='Processed Image.', use_column_width=True)
        
        # Display the count of detected objects
        st.write(f"Number of {class_name}s detected: {count}")

import streamlit as st
import cv2
import numpy as np
import tempfile
import torch
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt').to('cuda' if torch.cuda.is_available() else 'cpu')

st.title("Vehicle Detection and Counting")

# File uploader for input video
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

# If a video is uploaded
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)  # Create temp file to store video
    tfile.write(uploaded_file.read())
    
    # Initialize video capture
    cap = cv2.VideoCapture(tfile.name)
    
    # Get video dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define a line position for vehicle counting
    line_position = frame_height * 2 // 3

    # Vehicle counting variables
    entry_count_cars = 0
    exit_count_cars = 0
    entry_count_lorries = 0
    exit_count_lorries = 0
    
    # Vehicle classes (from COCO dataset)
    car_class = 2
    lorry_class = 7

    # Vehicle tracker to maintain object information across frames
    vehicle_tracker = {}
    next_vehicle_id = 0

    def is_car_or_lorry(class_id):
        return int(class_id) in [car_class, lorry_class]

    def process_detections(results, frame):
        global entry_count_cars, exit_count_cars, entry_count_lorries, exit_count_lorries
        global next_vehicle_id, vehicle_tracker

        detections = results[0].boxes.data.cpu().numpy()
        new_centroids = []

        for detection in detections:
            if len(detection) >= 6:
                x1, y1, x2, y2, confidence, class_id = detection[:6]
                if is_car_or_lorry(class_id):
                    vehicle_type = "Car" if int(class_id) == car_class else "Lorry"
                    box_color = (0, 255, 0) if vehicle_type == "Car" else (255, 0, 0)

                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
                    
                    # Calculate centroid
                    x_center = int((x1 + x2) / 2)
                    y_center = int((y1 + y2) / 2)
                    new_centroids.append((x_center, y_center, class_id))

        updated_tracker = {}
        for vehicle_id, (prev_centroid, counted, class_id) in vehicle_tracker.items():
            distances = [np.linalg.norm(np.array(prev_centroid[:2]) - np.array(new_c[:2])) for new_c in new_centroids]
            if distances:
                min_dist_idx = np.argmin(distances)
                min_dist = distances[min_dist_idx]
                new_centroid = new_centroids.pop(min_dist_idx)

                if min_dist < 50:  # Threshold for tracking
                    updated_tracker[vehicle_id] = (new_centroid, counted, class_id)
                    prev_y = prev_centroid[1]
                    new_y = new_centroid[1]

                    if not counted and prev_y < line_position <= new_y:
                        if class_id == car_class:
                            exit_count_cars += 1
                        elif class_id == lorry_class:
                            exit_count_lorries += 1
                        updated_tracker[vehicle_id] = (new_centroid, True, class_id)

                    elif not counted and prev_y > line_position >= new_y:
                        if class_id == car_class:
                            entry_count_cars += 1
                        elif class_id == lorry_class:
                            entry_count_lorries += 1
                        updated_tracker[vehicle_id] = (new_centroid, True, class_id)

        for new_centroid in new_centroids:
            updated_tracker[next_vehicle_id] = (new_centroid, False, new_centroid[2])
            next_vehicle_id += 1

        vehicle_tracker = updated_tracker
        return frame

    # Process video frames
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        results = model(frame)
        
        # Draw the counting line
        cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 0, 255), 2)

        # Process detections and update frame
        frame = process_detections(results, frame)

        # Show the processed video in Streamlit
        st.image(frame, channels="BGR")

        if st.button('Stop'):
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()
else:
    st.write("Upload a video to begin")

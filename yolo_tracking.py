import cv2
import numpy as np
import pandas as pd
import joblib
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from trajectory_prediction import VehicleTracker

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize DeepSORT Tracker
tracker = DeepSort(max_age=30)

# Load trained ML anomaly detection model & scaler
anomaly_model = joblib.load("anomaly_model.pkl")
scaler = joblib.load("scaler.pkl")

# Store detected anomalies
anomalies = []

def detect_anomalies(speed, acceleration, steering_angle, lane_deviation):
    """Use trained ML model to detect anomalies in vehicle movement."""
    features = np.array([[speed, acceleration, steering_angle, lane_deviation]])
    features = scaler.transform(features)  # Normalize features
    prediction = anomaly_model.predict(features)[0]
    return "Anomaly" if prediction == -1 else "Normal"

def detect_and_track(video_source):
    cap = cv2.VideoCapture(video_source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = []

        for r in results:
            for box in r.boxes.data:
                x1, y1, x2, y2, score, class_id = box.tolist()
                if int(class_id) in [2, 3, 5, 7]:
                    detections.append(([x1, y1, x2, y2], score, class_id))

        if anomalies:
            df_anomalies = pd.DataFrame(anomalies)
            df_anomalies.to_csv("detected_anomalies.csv", index=False)

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

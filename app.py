from flask import Flask, request, render_template, jsonify, Response
import os
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from flask_sqlalchemy import SQLAlchemy
from sort import Sort  # SORT Tracker
from gps_module import get_gps_data  # ✅ Import GPS module
from vehicle_tracker import VehicleTracker  # ✅ Kalman Filter for prediction
from motion_detection import detect_motion_changes  # ✅ Motion anomaly detection

app = Flask(__name__)

# ✅ Set Upload Folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ✅ Setup SQLite Database
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///events.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ✅ Define Database Model
class EventLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    vehicle_id = db.Column(db.Integer, nullable=False)
    event_type = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    x1 = db.Column(db.Integer)
    y1 = db.Column(db.Integer)
    x2 = db.Column(db.Integer)
    y2 = db.Column(db.Integer)
    ttc = db.Column(db.Float, nullable=True)
    latitude = db.Column(db.Float, nullable=True)  # ✅ GPS Data
    longitude = db.Column(db.Float, nullable=True)  # ✅ GPS Data

# ✅ Initialize Database
with app.app_context():
    db.create_all()

# ✅ Load YOLOv8 Model
model = YOLO("yolov8n.pt")

# ✅ Initialize SORT Tracker
tracker = Sort()

# ✅ Initialize Kalman Filter Tracker
kalman_tracker = VehicleTracker()

def calculate_ttc(ego_speed, frontier_speed, distance):
    """Calculates Time-to-Collision (TTC) using speeds in km/h."""
    if frontier_speed <= ego_speed or distance <= 0:
        return float('inf')  # No collision risk if ego vehicle is not approaching
    relative_speed = (frontier_speed - ego_speed) / 3.6  # Convert km/h to m/s
    return round(distance / relative_speed, 2) if relative_speed > 0 else float('inf')

def estimate_frontier_speed(track_id):
    """Simulated function to estimate frontier vehicle speed."""
    return np.random.uniform(20, 80)  # Simulated speed between 20-80 km/h

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)
    return jsonify({"filename": file.filename})

@app.route("/video_feed/<filename>")
def video_feed(filename):
    return Response(process_video(os.path.join(app.config["UPLOAD_FOLDER"], filename)),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    prev_frame = None

    with app.app_context():
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            height, width, _ = frame.shape
            center_x = width // 2  # Middle of the frame

            # ✅ Get GPS data
            gps_data = get_gps_data()
            ego_speed = gps_data["speed"]  # Speed in km/h

            # ✅ Motion detection (Sudden Braking)
            motion_status = "✅ Normal Motion"
            if prev_frame is not None:
                motion_status = detect_motion_changes(prev_frame, frame)

            prev_frame = frame.copy()

            # ✅ Run YOLOv8 Detection
            results = model(frame)[0]
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                if class_id in [2, 3, 5, 7]:  # ✅ Vehicles
                    detections.append([x1, y1, x2, y2])

            tracked_objects = tracker.update(np.array(detections) if detections else np.empty((0, 4)))

            # ✅ Frontier Vehicle Identification
            frontier_vehicle = None
            min_y = height
            for track in tracked_objects:
                if len(track) < 5:
                    continue
                x1, y1, x2, y2, track_id = map(int, track)
                obj_center_x = (x1 + x2) // 2
                obj_center_y = y2  # Bottom Y-coordinate

                if abs(obj_center_x - center_x) < width * 0.2 and obj_center_y < min_y:
                    min_y = obj_center_y
                    frontier_vehicle = track

            for track in tracked_objects:
                if len(track) < 5:
                    continue
                x1, y1, x2, y2, track_id = map(int, track)
                color = (255, 0, 0)
                event_type = "Tracked"
                ttc = None

                if np.array_equal(track, frontier_vehicle):
                    color = (0, 0, 255)
                    event_type = "Frontier"
                    frontier_speed = estimate_frontier_speed(track_id)
                    distance = height - y2  # Approximate distance
                    ttc = calculate_ttc(ego_speed, frontier_speed, distance)
                    if ttc < 2:
                        event_type = "Near Collision"

                    cv2.putText(frame, f"TTC: {ttc}s", (x1, y1 - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                event = EventLog(vehicle_id=track_id, event_type=event_type,
                                 x1=x1, y1=y1, x2=x2, y2=y2, ttc=ttc,
                                 latitude=gps_data["latitude"], longitude=gps_data["longitude"])
                db.session.add(event)

            db.session.commit()  # ✅ Commit once per frame for efficiency

            _, buffer = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    cap.release()

@app.route("/events", methods=["GET"])
def get_events():
    events = EventLog.query.order_by(EventLog.timestamp.desc()).limit(10).all()
    data = [{
        "id": e.id,
        "vehicle_id": e.vehicle_id,
        "event_type": e.event_type,
        "timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "x1": e.x1, "y1": e.y1, "x2": e.x2, "y2": e.y2,
        "ttc": e.ttc if e.ttc is not None else "N/A",
        "latitude": e.latitude,
        "longitude": e.longitude
    } for e in events]
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)

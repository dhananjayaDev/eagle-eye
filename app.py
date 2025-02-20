from flask import Flask, request, render_template, jsonify, Response
import os
import cv2
import numpy as np
import logging
from datetime import datetime
from ultralytics import YOLO
from flask_sqlalchemy import SQLAlchemy
from sort import Sort  # SORT Tracker

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
    event_type = db.Column(db.String(50), nullable=False)  # "Tracked" or "Frontier"
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    x1 = db.Column(db.Integer)
    y1 = db.Column(db.Integer)
    x2 = db.Column(db.Integer)
    y2 = db.Column(db.Integer)


# ✅ Initialize Database
with app.app_context():
    db.create_all()

# ✅ Load YOLOv8 Model
model = YOLO("yolov8n.pt")

# ✅ Initialize SORT Tracker
tracker = Sort()

# ✅ Main Page Route
@app.route("/")
def index():
    return render_template("index.html")

# ✅ File Upload Endpoint
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

# ✅ Video Streaming Endpoint
@app.route("/video_feed/<filename>")
def video_feed(filename):
    return Response(process_video(os.path.join(app.config["UPLOAD_FOLDER"], filename)),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ✅ Process Uploaded Video & Track Vehicles
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    # ✅ Ensure Application Context is Set
    with app.app_context():
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            height, width, _ = frame.shape
            center_x = width // 2  # Middle of the frame

            # ✅ Run YOLOv8 Detection on Frame
            results = model(frame)[0]

            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                if class_id in [2, 3, 5, 7]:  # ✅ Vehicle detection only
                    detections.append([x1, y1, x2, y2])

            tracked_objects = tracker.update(np.array(detections) if detections else np.empty((0, 4)))

            # ✅ Frontier Vehicle Identification
            frontier_vehicle = None
            min_y = height  # Start with maximum Y position (lower y2 value = closer vehicle)

            for track in tracked_objects:
                if len(track) < 5:
                    continue
                x1, y1, x2, y2, track_id = map(int, track)

                obj_center_x = (x1 + x2) // 2
                obj_center_y = y2  # Bottom Y-coordinate

                if abs(obj_center_x - center_x) < width * 0.2:  # ✅ Vehicle in center lane
                    if obj_center_y < min_y:  # ✅ Closest to the camera (lower y2 value)
                        min_y = obj_center_y
                        frontier_vehicle = track

            for track in tracked_objects:
                if len(track) < 5:
                    continue
                x1, y1, x2, y2, track_id = map(int, track)

                color = (255, 0, 0)  # Default: Blue
                event_type = "Tracked"

                # ✅ Highlight the Frontier Vehicle in RED
                if np.array_equal(track, frontier_vehicle):
                    color = (0, 0, 255)  # Red for frontier vehicle
                    event_type = "Frontier"
                    cv2.putText(frame, f"Frontier Vehicle ID: {track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # ✅ Store Event in Database
                event = EventLog(vehicle_id=track_id, event_type=event_type,
                                 x1=x1, y1=y1, x2=x2, y2=y2)
                db.session.add(event)
                db.session.commit()

            _, buffer = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    cap.release()


@app.route("/frontier_vehicles", methods=["GET"])
def get_frontier_vehicles():
    events = EventLog.query.filter_by(event_type="Frontier").order_by(EventLog.timestamp.desc()).limit(10).all()
    data = [{"vehicle_id": e.vehicle_id, "event_type": e.event_type,
             "timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
             "x1": e.x1, "y1": e.y1, "x2": e.x2, "y2": e.y2} for e in events]
    return jsonify(data)



# ✅ Real-Time Events API
@app.route("/events", methods=["GET"])
def get_events():
    events = EventLog.query.order_by(EventLog.timestamp.desc()).limit(10).all()
    data = [{"vehicle_id": e.vehicle_id, "event_type": e.event_type,
             "timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
             "x1": e.x1, "y1": e.y1, "x2": e.x2, "y2": e.y2} for e in events]
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, render_template, jsonify, Response, send_file
import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from flask_sqlalchemy import SQLAlchemy
from sort import Sort
from gps_module import get_gps_data
from vehicle_tracker import VehicleTracker
from motion_detection import detect_motion_changes
import joblib
import torch
import io
import atexit

from blockchain import blockchain_bp  # Import the new blueprint

app = Flask(__name__)

# Register the blockchain blueprint
app.register_blueprint(blockchain_bp, url_prefix='/blockchain')

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///events.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# Use torch.cuda.is_available() instead of is_connected
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

PIXELS_PER_METER = 0.1  # Placeholder; calibrate later
vehicle_history = {}
collided_vehicles = set()  # Track vehicles involved in collisions
collision_cooldown = {}  # Track cooldown for highlighting (frame-based)

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
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    motion_status = db.Column(db.String(50), nullable=True)

with app.app_context():
    db.create_all()

model = YOLO("yolov8n.pt").to(device)  # Can switch to yolov8s.pt for lighter model if needed
tracker = Sort()
kalman_tracker = VehicleTracker()
anomaly_model = joblib.load("frontier_anomaly_model.pkl")
scaler = joblib.load("scaler.pkl")

def calculate_ttc(ego_speed, frontier_speed, distance):
    if frontier_speed <= ego_speed or distance <= 0:
        return float('inf')
    relative_speed = (ego_speed - frontier_speed) / 3.6
    return round(distance / relative_speed, 2) if relative_speed > 0 else float('inf')

def estimate_frontier_speed(track_id, y_center, frame_count, frame_time):
    if track_id not in vehicle_history:
        vehicle_history[track_id] = {"last_y": y_center, "last_frame": frame_count, "speed": 40.0}
        return 40.0
    last_y = vehicle_history[track_id]["last_y"]
    last_frame = vehicle_history[track_id]["last_frame"]
    time_diff = (frame_count - last_frame) * frame_time
    if time_diff > 0:
        displacement = last_y - y_center
        speed_pixels_per_sec = displacement / time_diff
        speed_mps = speed_pixels_per_sec * PIXELS_PER_METER
        speed_kmh = speed_mps * 3.6
        alpha = 0.7
        new_speed = max(0, min(120, speed_kmh))
        smoothed_speed = alpha * new_speed + (1 - alpha) * vehicle_history[track_id]["speed"]
        vehicle_history[track_id]["speed"] = smoothed_speed
    vehicle_history[track_id]["last_y"] = y_center
    vehicle_history[track_id]["last_frame"] = frame_count
    return vehicle_history[track_id]["speed"]

def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 200)
    height, width = frame.shape[:2]
    mask = np.zeros_like(edges)
    roi_vertices = np.array([
        [0, height * 0.6], [width, height * 0.6], [width, height], [0, height]
    ], np.int32)
    cv2.fillPoly(mask, [roi_vertices], 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    masked_edges = cv2.bitwise_and(masked_edges, masked_edges, mask=white_mask)
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=100, minLineLength=80, maxLineGap=30)
    lane_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            lane_lines.append((x1, y1, x2, y2))
    return lane_lines

def get_ego_lane_bounds(lane_lines, width, height):
    if not lane_lines:
        return 0, width
    left_lane_x = width
    right_lane_x = 0
    left_lines = []
    right_lines = []
    for x1, y1, x2, y2 in lane_lines:
        if y1 != y2:
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
            x_bottom = x1 + (x2 - x1) * (height - y1) / (y2 - y1)
            if slope > 0.5:
                right_lines.append(x_bottom)
            elif slope < -0.5:
                left_lines.append(x_bottom)
    if left_lines:
        left_lane_x = max(0, min(left_lines) - 50)
    if right_lines:
        right_lane_x = min(width, max(right_lines) + 50)
    return int(left_lane_x), int(right_lane_x)

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_other, y1_other, x2_other, y2_other = box2
    xi1 = max(x1, x1_other)
    yi1 = max(y1, y1_other)
    xi2 = min(x2, x2_other)
    yi2 = min(y2, y2_other)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_other - x1_other) * (y2_other - y1_other)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

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
    print(f"Streaming video: {filename}")
    return Response(process_video(os.path.join(app.config["UPLOAD_FOLDER"], filename)),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        yield b''
        return

    FPS = cap.get(cv2.CAP_PROP_FPS)  # Use actual FPS from video
    FRAME_TIME = 1 / FPS
    prev_frame = None
    frame_count = 0
    prev_tracks = {}

    try:
        with app.app_context():
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("End of video stream reached.")
                    break
                frame_count += 1
                print(f"Processing frame {frame_count}")
                # Skip every 2nd frame to reduce load
                if frame_count % 2 == 0:
                    continue
                frame = cv2.resize(frame, (640, 480))  # Maintain higher resolution
                height, width, _ = frame.shape
                center_x = width // 2

                lane_lines = detect_lanes(frame)
                left_lane_x, right_lane_x = get_ego_lane_bounds(lane_lines, width, height)
                for x1, y1, x2, y2 in lane_lines:
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                gps_data = get_gps_data()
                ego_speed = gps_data["speed"]
                motion_status = detect_motion_changes(prev_frame, frame) if prev_frame is not None else "✅ Normal Motion"
                prev_frame = frame.copy()
                results = model(frame)[0]
                detections = []
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    if class_id in [2, 3, 5, 7]:  # Car, motorcycle, bus, truck
                        detections.append([x1, y1, x2, y2])
                tracked_objects = tracker.update(np.array(detections) if detections else np.empty((0, 4)))
                frontier_vehicle = None
                min_y = height
                # Prioritize closest vehicle in ego lane
                for track in tracked_objects:
                    if len(track) < 5:
                        continue
                    x1, y1, x2, y2, track_id = map(int, track)
                    obj_center_x = (x1 + x2) // 2
                    obj_center_y = y2
                    # Check if vehicle is in ego lane and closer than current min_y
                    if (left_lane_x <= obj_center_x <= right_lane_x and obj_center_y < min_y):
                        min_y = obj_center_y
                        frontier_vehicle = track

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.3  # Reduced font size for smaller labels
                thickness = 1     # Reduced thickness for thinner outline
                padding = 3       # Reduced padding for compactness
                label_spacing = 15  # Reduced spacing between labels

                for track in tracked_objects:
                    if len(track) < 5:
                        continue
                    x1, y1, x2, y2, track_id = map(int, track)
                    color = (255, 0, 0)
                    event_type = "Tracked"
                    ttc = None
                    vehicle_motion = "✅ Normal Motion" if motion_status == "✅ Normal Motion" else motion_status
                    if np.array_equal(track, frontier_vehicle):
                        color = (0, 255, 0)
                        event_type = "Frontier"
                        y_center = (y1 + y2) // 2
                        frontier_speed = estimate_frontier_speed(track_id, y_center, frame_count, FRAME_TIME)
                        distance = height - y2
                        ttc = calculate_ttc(ego_speed, frontier_speed, distance) if frontier_speed and ego_speed else float('inf')
                        if ttc < 2:
                            event_type = "Near Collision"
                        x_center = (x1 + x2) // 2
                        pred_x, pred_y = kalman_tracker.predict_next_position(x_center, y_center)
                        cv2.circle(frame, (pred_x, pred_y), 5, (0, 255, 0), -1)
                        features = pd.DataFrame([[frontier_speed, 0, 0]], columns=["v_Vel", "v_Acc", "Lane_ID"])
                        scaled_features_array = scaler.transform(features)
                        scaled_features = pd.DataFrame(scaled_features_array, columns=["v_Vel", "v_Acc", "Lane_ID"])
                        if anomaly_model.predict(scaled_features)[0] == -1:
                            event_type = f"{event_type} - Anomaly"

                        current_pos = (x_center, y_center)
                        if track_id in prev_tracks:
                            prev_pos = prev_tracks[track_id]
                            dist_moved = ((current_pos[0] - prev_pos[0]) ** 2 + (current_pos[1] - prev_pos[1]) ** 2) ** 0.5
                            speed_px_per_frame = dist_moved / FRAME_TIME
                            if speed_px_per_frame < 0.5:
                                vehicle_motion = "🚨 Sudden Stop Detected!"
                            elif speed_px_per_frame > 5.0:
                                vehicle_motion = "⚠️ Harsh Braking"
                        prev_tracks[track_id] = current_pos

                        is_collision = False
                        if ttc is not None and ttc < 0.3:
                            is_collision = True
                        for other_track in tracked_objects:
                            if not np.array_equal(other_track, track) and len(other_track) >= 5:
                                ox1, oy1, ox2, oy2, other_id = map(int, other_track)
                                iou = calculate_iou([x1, y1, x2, y2], [ox1, oy1, ox2, oy2])
                                if iou > 0.5:
                                    is_collision = True
                                    collided_vehicles.add(other_id)
                                    collision_cooldown[other_id] = frame_count + (5 * FPS)

                        if is_collision and vehicle_motion not in ["Collided", "🚨 Sudden Stop Detected!", "⚠️ Harsh Braking"]:
                            vehicle_motion = "Collided"
                            collided_vehicles.add(track_id)
                            collision_cooldown[track_id] = frame_count + (5 * FPS)

                        # Define all labels and calculate their sizes
                        motion_text = vehicle_motion  # Remove "Motion: " prefix, show only status
                        motion_size, _ = cv2.getTextSize(motion_text, font, font_scale, thickness)
                        motion_width, motion_height = motion_size

                        speed_text = f"Speed: {frontier_speed:.1f} km/h"
                        speed_size, _ = cv2.getTextSize(speed_text, font, font_scale, thickness)
                        speed_width, speed_height = speed_size

                        ttc_text = f"TTC: {ttc if ttc != float('inf') else 'N/A'}s"
                        ttc_size, _ = cv2.getTextSize(ttc_text, font, font_scale, thickness)
                        ttc_width, ttc_height = ttc_size

                        id_text = f"ID: {track_id}"
                        id_size, _ = cv2.getTextSize(id_text, font, font_scale, thickness)
                        id_width, id_height = id_size

                        # Determine the maximum width for consistent background rectangles
                        max_width = max(motion_width, speed_width, ttc_width, id_width)

                        # Define positions for each label, spaced vertically
                        motion_pos = (x1, y1 - 80)
                        speed_pos = (x1, y1 - 80 + label_spacing)
                        ttc_pos = (x1, y1 - 80 + 2 * label_spacing)
                        id_pos = (x1, y1 - 80 + 3 * label_spacing)

                        # Define background positions with padding
                        motion_bg_pos1 = (x1 - padding, motion_pos[1] - motion_height - padding)
                        motion_bg_pos2 = (x1 + max_width + padding, motion_pos[1] + padding)

                        speed_bg_pos1 = (x1 - padding, speed_pos[1] - speed_height - padding)
                        speed_bg_pos2 = (x1 + max_width + padding, speed_pos[1] + padding)

                        ttc_bg_pos1 = (x1 - padding, ttc_pos[1] - ttc_height - padding)
                        ttc_bg_pos2 = (x1 + max_width + padding, ttc_pos[1] + padding)

                        id_bg_pos1 = (x1 - padding, id_pos[1] - id_height - padding)
                        id_bg_pos2 = (x1 + max_width + padding, id_pos[1] + padding)

                        # Use red background for critical events, otherwise black
                        is_critical = vehicle_motion in ["Collided", "⚠️ Harsh Braking", "🚨 Sudden Stop Detected!"]
                        bg_color = (0, 0, 255) if is_critical else (0, 0, 0)  # Red for critical, black otherwise
                        alpha = 0.6  # Transparency for all backgrounds

                        # Draw semi-transparent backgrounds for all labels
                        overlay = frame.copy()
                        cv2.rectangle(overlay, motion_bg_pos1, motion_bg_pos2, bg_color, -1)
                        cv2.rectangle(overlay, speed_bg_pos1, speed_bg_pos2, bg_color, -1)
                        cv2.rectangle(overlay, ttc_bg_pos1, ttc_bg_pos2, bg_color, -1)
                        cv2.rectangle(overlay, id_bg_pos1, id_bg_pos2, bg_color, -1)
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                        # Draw text with black outline and white fill for all labels
                        cv2.putText(frame, motion_text, motion_pos, font, font_scale, (0, 0, 0), thickness + 1)
                        cv2.putText(frame, motion_text, motion_pos, font, font_scale, (255, 255, 255), thickness)

                        cv2.putText(frame, speed_text, speed_pos, font, font_scale, (0, 0, 0), thickness + 1)
                        cv2.putText(frame, speed_text, speed_pos, font, font_scale, (255, 255, 255), thickness)

                        cv2.putText(frame, ttc_text, ttc_pos, font, font_scale, (0, 0, 0), thickness + 1)
                        cv2.putText(frame, ttc_text, ttc_pos, font, font_scale, (255, 255, 255), thickness)

                        cv2.putText(frame, id_text, id_pos, font, font_scale, (0, 0, 0), thickness + 1)
                        cv2.putText(frame, id_text, id_pos, font, font_scale, (255, 255, 255), thickness)

                    if track_id in collided_vehicles and frame_count <= collision_cooldown.get(track_id, 0):
                        color = (0, 0, 255)
                    elif track_id in collided_vehicles:
                        collided_vehicles.remove(track_id)
                        collision_cooldown.pop(track_id, None)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    event = EventLog(vehicle_id=track_id, event_type=event_type,
                                    x1=x1, y1=y1, x2=x2, y2=y2, ttc=ttc,
                                    latitude=gps_data["latitude"], longitude=gps_data["longitude"],
                                    motion_status=vehicle_motion)
                    db.session.add(event)
                    print(f"Added event: {event.vehicle_id}, {event.event_type}, {event.motion_status}, {event.timestamp}")

                # Commit every 30 frames to reduce database load
                if frame_count % 30 == 0:
                    try:
                        db.session.commit()
                        print(f"Committed {frame_count} frames")
                    except Exception as e:
                        print(f"Commit failed: {e}")

                # Encode with higher quality
                _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                yield (b"--frame\r\n"
                      b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

            # Final commit
            try:
                db.session.commit()
                print("Final commit successful")
            except Exception as e:
                print(f"Final commit failed: {e}")

    except Exception as e:
        print(f"Error in process_video: {e}")
    finally:
        cap.release()
        with app.app_context():
            db.session.remove()  # Clean up database session

@app.route("/events", methods=["GET"])
def get_events():
    events = EventLog.query.order_by(EventLog.timestamp.desc()).limit(10).all()
    print(f"Events retrieved: {len(events)}")
    data = [{
        "id": e.id,
        "vehicle_id": e.vehicle_id,
        "event_type": e.event_type,
        "timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "x1": e.x1, "y1": e.y1, "x2": e.x2, "y2": e.y2,
        "ttc": "N/A" if e.ttc is None or e.ttc == float('inf') else e.ttc,
        "latitude": e.latitude,
        "longitude": e.longitude,
        "motion_status": e.motion_status
    } for e in events]
    print(f"Data returned: {data}")
    return jsonify(data)

@app.route("/export_critical_events", methods=["GET"])
def export_critical_events():
    try:
        with app.app_context():
            # Define critical event types based on motion_status
            critical_event_types = [
                "Collided",
                "⚠️ Harsh Braking",
                "🚨 Sudden Stop Detected!"
            ]

            # Query the database for critical events
            critical_events = EventLog.query.filter(EventLog.motion_status.in_(critical_event_types)).all()

            if not critical_events:
                return jsonify({"error": "No critical events found."}), 404

            # Prepare data for Excel
            data = []
            for event in critical_events:
                data.append({
                    "ID": event.id,
                    "Vehicle ID": event.vehicle_id,
                    "Event Type": event.event_type,
                    "Motion Status": event.motion_status,
                    "Timestamp": event.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "X1": event.x1,
                    "Y1": event.y1,
                    "X2": event.x2,
                    "Y2": event.y2,
                    "TTC (s)": "N/A" if event.ttc is None or event.ttc == float('inf') else event.ttc,
                    "Latitude": event.latitude,
                    "Longitude": event.longitude
                })

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Create an in-memory Excel file
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name="Critical Events")
            output.seek(0)

            # Send the file as a downloadable attachment
            return send_file(
                output,
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                as_attachment=True,
                download_name="critical_events.xlsx"
            )

    except Exception as e:
        print(f"Error exporting critical events: {e}")
        return jsonify({"error": "Failed to export critical events."}), 500

# Directory to save the exported Excel file on shutdown
EXPORT_DIR = "exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

def export_critical_events_on_shutdown():
    try:
        with app.app_context():
            critical_event_types = ["Collided", "⚠️ Harsh Braking", "🚨 Sudden Stop Detected!"]
            critical_events = EventLog.query.filter(EventLog.motion_status.in_(critical_event_types)).all()

            if not critical_events:
                print("No critical events found to export on shutdown.")
                return

            data = []
            for event in critical_events:
                data.append({
                    "ID": event.id,
                    "Vehicle ID": event.vehicle_id,
                    "Event Type": event.event_type,
                    "Motion Status": event.motion_status,
                    "Timestamp": event.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "X1": event.x1,
                    "Y1": event.y1,
                    "X2": event.x2,
                    "Y2": event.y2,
                    "TTC (s)": "N/A" if event.ttc is None or event.ttc == float('inf') else event.ttc,
                    "Latitude": event.latitude,
                    "Longitude": event.longitude
                })

            df = pd.DataFrame(data)
            filename = os.path.join(EXPORT_DIR, f"critical_events_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx")
            df.to_excel(filename, index=False, sheet_name="Critical Events")
            print(f"Critical events exported to {filename}")

    except Exception as e:
        print(f"Error exporting critical events on shutdown: {e}")

# Register the shutdown function
atexit.register(export_critical_events_on_shutdown)

if __name__ == "__main__":
    app.run(debug=True)
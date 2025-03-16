# from flask import Flask, request, render_template, jsonify, Response
# import os
# import cv2
# import numpy as np
# from datetime import datetime
# from ultralytics import YOLO
# from flask_sqlalchemy import SQLAlchemy
# from sort import Sort  # SORT Tracker
# from gps_module import get_gps_data  # ✅ Import GPS module
# from vehicle_tracker import VehicleTracker  # ✅ Kalman Filter for prediction
# from motion_detection import detect_motion_changes  # ✅ Motion anomaly detection
#
# app = Flask(__name__)
#
# # ✅ Set Upload Folder
# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
#
# # ✅ Setup SQLite Database2
# app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///events.db"
# app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# db = SQLAlchemy(app)
#
# # ✅ Define Database Model
# class EventLog(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     vehicle_id = db.Column(db.Integer, nullable=False)
#     event_type = db.Column(db.String(50), nullable=False)
#     timestamp = db.Column(db.DateTime, default=datetime.utcnow)
#     x1 = db.Column(db.Integer)
#     y1 = db.Column(db.Integer)
#     x2 = db.Column(db.Integer)
#     y2 = db.Column(db.Integer)
#     ttc = db.Column(db.Float, nullable=True)
#     latitude = db.Column(db.Float, nullable=True)  # ✅ GPS Data
#     longitude = db.Column(db.Float, nullable=True)  # ✅ GPS Data
#
# # ✅ Initialize Database
# with app.app_context():
#     db.create_all()
#
# # ✅ Load YOLOv8 Model
# model = YOLO("yolov8n.pt")
#
# # ✅ Initialize SORT Tracker
# tracker = Sort()
#
# # ✅ Initialize Kalman Filter Tracker
# kalman_tracker = VehicleTracker()
#
# def calculate_ttc(ego_speed, frontier_speed, distance):
#     """Calculates Time-to-Collision (TTC) using speeds in km/h."""
#     if frontier_speed <= ego_speed or distance <= 0:
#         return float('inf')  # No collision risk if ego vehicle is not approaching
#     relative_speed = (frontier_speed - ego_speed) / 3.6  # Convert km/h to m/s
#     return round(distance / relative_speed, 2) if relative_speed > 0 else float('inf')
#
# def estimate_frontier_speed(track_id):
#     """Simulated function to estimate frontier vehicle speed."""
#     return np.random.uniform(20, 80)  # Simulated speed between 20-80 km/h
#
# @app.route("/")
# def index():
#     return render_template("index.html")
#
# @app.route("/upload", methods=["POST"])
# def upload_file():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
#
#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "No selected file"}), 400
#
#     file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
#     file.save(file_path)
#     return jsonify({"filename": file.filename})
#
# @app.route("/video_feed/<filename>")
# def video_feed(filename):
#     return Response(process_video(os.path.join(app.config["UPLOAD_FOLDER"], filename)),
#                     mimetype="multipart/x-mixed-replace; boundary=frame")
#
# def process_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         return
#
#     prev_frame = None
#
#     with app.app_context():
#         while cap.isOpened():
#             success, frame = cap.read()
#             if not success:
#                 cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#                 continue
#
#             height, width, _ = frame.shape
#             center_x = width // 2  # Middle of the frame
#
#             # ✅ Get GPS data
#             gps_data = get_gps_data()
#             ego_speed = gps_data["speed"]  # Speed in km/h
#
#             # ✅ Motion detection (Sudden Braking)
#             motion_status = "✅ Normal Motion"
#             if prev_frame is not None:
#                 motion_status = detect_motion_changes(prev_frame, frame)
#
#             prev_frame = frame.copy()
#
#             # ✅ Run YOLOv8 Detection
#             results = model(frame)[0]
#             detections = []
#             for box in results.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 class_id = int(box.cls[0])
#                 if class_id in [2, 3, 5, 7]:  # ✅ Vehicles
#                     detections.append([x1, y1, x2, y2])
#
#             tracked_objects = tracker.update(np.array(detections) if detections else np.empty((0, 4)))
#
#             # ✅ Frontier Vehicle Identification
#             frontier_vehicle = None
#             min_y = height
#             for track in tracked_objects:
#                 if len(track) < 5:
#                     continue
#                 x1, y1, x2, y2, track_id = map(int, track)
#                 obj_center_x = (x1 + x2) // 2
#                 obj_center_y = y2  # Bottom Y-coordinate
#
#                 if abs(obj_center_x - center_x) < width * 0.2 and obj_center_y < min_y:
#                     min_y = obj_center_y
#                     frontier_vehicle = track
#
#             for track in tracked_objects:
#                 if len(track) < 5:
#                     continue
#                 x1, y1, x2, y2, track_id = map(int, track)
#                 color = (255, 0, 0)
#                 event_type = "Tracked"
#                 ttc = None
#
#                 if np.array_equal(track, frontier_vehicle):
#                     color = (0, 0, 255)
#                     event_type = "Frontier"
#                     frontier_speed = estimate_frontier_speed(track_id)
#                     distance = height - y2  # Approximate distance
#                     ttc = calculate_ttc(ego_speed, frontier_speed, distance)
#                     if ttc < 2:
#                         event_type = "Near Collision"
#
#                     cv2.putText(frame, f"TTC: {ttc}s", (x1, y1 - 40),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
#
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 25),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
#
#                 event = EventLog(vehicle_id=track_id, event_type=event_type,
#                                  x1=x1, y1=y1, x2=x2, y2=y2, ttc=ttc,
#                                  latitude=gps_data["latitude"], longitude=gps_data["longitude"])
#                 db.session.add(event)
#
#             db.session.commit()  # ✅ Commit once per frame for efficiency
#
#             _, buffer = cv2.imencode(".jpg", frame)
#             yield (b"--frame\r\n"
#                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
#     cap.release()
#
# @app.route("/events", methods=["GET"])
# def get_events():
#     events = EventLog.query.order_by(EventLog.timestamp.desc()).limit(10).all()
#     data = [{
#         "id": e.id,
#         "vehicle_id": e.vehicle_id,
#         "event_type": e.event_type,
#         "timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
#         "x1": e.x1, "y1": e.y1, "x2": e.x2, "y2": e.y2,
#         "ttc": e.ttc if e.ttc is not None else "N/A",
#         "latitude": e.latitude,
#         "longitude": e.longitude
#     } for e in events]
#     return jsonify(data)
#
# if __name__ == "__main__":
#     app.run(debug=True)
#
#
# from flask import Flask, request, render_template, jsonify, Response
# import os
# import cv2
# import numpy as np
# from datetime import datetime
# from ultralytics import YOLO
# from flask_sqlalchemy import SQLAlchemy
# from sort import Sort  # SORT Tracker
# from gps_module import get_gps_data  # Assumed functional
# from vehicle_tracker import VehicleTracker  # Kalman Filter for prediction
# from motion_detection import detect_motion_changes  # Motion anomaly detection
# import joblib  # For Isolation Forest
#
# app = Flask(__name__)
#
# # Set Upload Folder
# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
#
# # Setup SQLite Database
# app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///events.db"
# app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# db = SQLAlchemy(app)
#
# # Define Database Model with Motion Status
# class EventLog(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     vehicle_id = db.Column(db.Integer, nullable=False)
#     event_type = db.Column(db.String(50), nullable=False)
#     timestamp = db.Column(db.DateTime, default=datetime.utcnow)
#     x1 = db.Column(db.Integer)
#     y1 = db.Column(db.Integer)
#     x2 = db.Column(db.Integer)
#     y2 = db.Column(db.Integer)
#     ttc = db.Column(db.Float, nullable=True)
#     latitude = db.Column(db.Float, nullable=True)
#     longitude = db.Column(db.Float, nullable=True)
#     motion_status = db.Column(db.String(50), nullable=True)  # Added for motion anomalies
#
# # Initialize Database
# with app.app_context():
#     db.create_all()
#
# # Load YOLOv8 Model
# model = YOLO("yolov8n.pt")
#
# # Initialize SORT Tracker
# tracker = Sort()
#
# # Initialize Kalman Filter Tracker
# kalman_tracker = VehicleTracker()
#
# # Load Isolation Forest Model and Scaler
# anomaly_model = joblib.load("frontier_anomaly_model.pkl")
# scaler = joblib.load("scaler.pkl")
#
# def calculate_ttc(ego_speed, frontier_speed, distance):
#     """Calculates Time-to-Collision (TTC) using speeds in km/h."""
#     if frontier_speed <= ego_speed or distance <= 0:
#         return float('inf')
#     relative_speed = (ego_speed - frontier_speed) / 3.6  # Convert km/h to m/s (corrected logic)
#     return round(distance / relative_speed, 2) if relative_speed > 0 else float('inf')
#
# def estimate_frontier_speed(track_id):
#     """Simulated function to estimate frontier vehicle speed."""
#     return np.random.uniform(20, 80)  # Simulated speed between 20-80 km/h
#
# @app.route("/")
# def index():
#     return render_template("index.html")
#
# @app.route("/upload", methods=["POST"])
# def upload_file():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "No selected file"}), 400
#     file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
#     file.save(file_path)
#     return jsonify({"filename": file.filename})
#
# @app.route("/video_feed/<filename>")
# def video_feed(filename):
#     return Response(process_video(os.path.join(app.config["UPLOAD_FOLDER"], filename)),
#                     mimetype="multipart/x-mixed-replace; boundary=frame")
#
# def process_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         return
#     prev_frame = None
#
#     with app.app_context():
#         while cap.isOpened():
#             success, frame = cap.read()
#             if not success:
#                 cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#                 continue
#
#             height, width, _ = frame.shape
#             center_x = width // 2
#
#             # Get GPS data
#             gps_data = get_gps_data()
#             ego_speed = gps_data["speed"]
#
#             # Motion detection
#             motion_status = detect_motion_changes(prev_frame, frame) if prev_frame is not None else "✅ Normal Motion"
#             prev_frame = frame.copy()
#
#             # Run YOLOv8 Detection
#             results = model(frame)[0]
#             detections = []
#             for box in results.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 class_id = int(box.cls[0])
#                 if class_id in [2, 3, 5, 7]:  # Vehicles
#                     detections.append([x1, y1, x2, y2])
#
#             tracked_objects = tracker.update(np.array(detections) if detections else np.empty((0, 4)))
#
#             # Frontier Vehicle Identification
#             frontier_vehicle = None
#             min_y = height
#             for track in tracked_objects:
#                 if len(track) < 5:
#                     continue
#                 x1, y1, x2, y2, track_id = map(int, track)
#                 obj_center_x = (x1 + x2) // 2
#                 obj_center_y = y2
#                 if abs(obj_center_x - center_x) < width * 0.2 and obj_center_y < min_y:
#                     min_y = obj_center_y
#                     frontier_vehicle = track
#
#             for track in tracked_objects:
#                 if len(track) < 5:
#                     continue
#                 x1, y1, x2, y2, track_id = map(int, track)
#                 color = (255, 0, 0)  # Blue for tracked
#                 event_type = "Tracked"
#                 ttc = None
#
#                 if np.array_equal(track, frontier_vehicle):
#                     color = (0, 0, 255)  # Red for frontier
#                     event_type = "Frontier"
#                     frontier_speed = estimate_frontier_speed(track_id)
#                     distance = height - y2  # Approximate distance
#                     ttc = calculate_ttc(ego_speed, frontier_speed, distance)
#                     if ttc < 2:
#                         event_type = "Near Collision"
#
#                     # Kalman Filter Prediction
#                     x_center = (x1 + x2) // 2
#                     y_center = (y1 + y2) // 2
#                     pred_x, pred_y = kalman_tracker.predict_next_position(x_center, y_center)
#                     cv2.circle(frame, (pred_x, pred_y), 5, (0, 255, 0), -1)  # Green dot for prediction
#
#                     # Isolation Forest Anomaly Detection
#                     features = np.array([[frontier_speed, 0, 0]])  # v_Vel, v_Acc (0), Lane_ID (0) placeholders
#                     scaled_features = scaler.transform(features)
#                     if anomaly_model.predict(scaled_features)[0] == -1:
#                         event_type = f"{event_type} - Anomaly"
#
#                     cv2.putText(frame, f"TTC: {ttc}s", (x1, y1 - 40),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
#
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 25),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
#
#                 # Log event with motion status
#                 event = EventLog(vehicle_id=track_id, event_type=event_type,
#                                  x1=x1, y1=y1, x2=x2, y2=y2, ttc=ttc,
#                                  latitude=gps_data["latitude"], longitude=gps_data["longitude"],
#                                  motion_status=motion_status)
#                 db.session.add(event)
#
#             db.session.commit()
#
#             _, buffer = cv2.imencode(".jpg", frame)
#             yield (b"--frame\r\n"
#                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
#     cap.release()
#
# @app.route("/events", methods=["GET"])
# def get_events():
#     events = EventLog.query.order_by(EventLog.timestamp.desc()).limit(10).all()
#     data = [{
#         "id": e.id,
#         "vehicle_id": e.vehicle_id,
#         "event_type": e.event_type,
#         "timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
#         "x1": e.x1, "y1": e.y1, "x2": e.x2, "y2": e.y2,
#         "ttc": e.ttc if e.ttc is not None else "N/A",
#         "latitude": e.latitude,
#         "longitude": e.longitude,
#         "motion_status": e.motion_status
#     } for e in events]
#     return jsonify(data)
#
# if __name__ == "__main__":
#     app.run(debug=True)

#
# from flask import Flask, request, render_template, jsonify, Response
# import os
# import cv2
# import numpy as np
# import pandas as pd  # Added for DataFrame
# from datetime import datetime
# from ultralytics import YOLO
# from flask_sqlalchemy import SQLAlchemy
# from sort import Sort
# from gps_module import get_gps_data
# from vehicle_tracker import VehicleTracker
# from motion_detection import detect_motion_changes
# import joblib
#
# app = Flask(__name__)
#
# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///events.db"
# app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# db = SQLAlchemy(app)
#
# class EventLog(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     vehicle_id = db.Column(db.Integer, nullable=False)
#     event_type = db.Column(db.String(50), nullable=False)
#     timestamp = db.Column(db.DateTime, default=datetime.utcnow)
#     x1 = db.Column(db.Integer)
#     y1 = db.Column(db.Integer)
#     x2 = db.Column(db.Integer)
#     y2 = db.Column(db.Integer)
#     ttc = db.Column(db.Float, nullable=True)
#     latitude = db.Column(db.Float, nullable=True)
#     longitude = db.Column(db.Float, nullable=True)
#     motion_status = db.Column(db.String(50), nullable=True)
#
# with app.app_context():
#     db.create_all()
#
# model = YOLO("yolov8n.pt")
# tracker = Sort()
# kalman_tracker = VehicleTracker()
# anomaly_model = joblib.load("frontier_anomaly_model.pkl")
# scaler = joblib.load("scaler.pkl")
#
# def calculate_ttc(ego_speed, frontier_speed, distance):
#     if frontier_speed <= ego_speed or distance <= 0:
#         return float('inf')
#     relative_speed = (ego_speed - frontier_speed) / 3.6
#     return round(distance / relative_speed, 2) if relative_speed > 0 else float('inf')
#
# def estimate_frontier_speed(track_id):
#     return np.random.uniform(20, 80)
#
# @app.route("/")
# def index():
#     return render_template("index.html")
#
# @app.route("/upload", methods=["POST"])
# def upload_file():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "No selected file"}), 400
#     file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
#     file.save(file_path)
#     return jsonify({"filename": file.filename})
#
# @app.route("/video_feed/<filename>")
# def video_feed(filename):
#     return Response(process_video(os.path.join(app.config["UPLOAD_FOLDER"], filename)),
#                     mimetype="multipart/x-mixed-replace; boundary=frame")
#
# def process_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         return
#     prev_frame = None
#     with app.app_context():
#         while cap.isOpened():
#             success, frame = cap.read()
#             if not success:
#                 cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#                 continue
#             height, width, _ = frame.shape
#             center_x = width // 2
#             gps_data = get_gps_data()
#             ego_speed = gps_data["speed"]
#             motion_status = detect_motion_changes(prev_frame, frame) if prev_frame is not None else "✅ Normal Motion"
#             prev_frame = frame.copy()
#             results = model(frame)[0]
#             detections = []
#             for box in results.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 class_id = int(box.cls[0])
#                 if class_id in [2, 3, 5, 7]:
#                     detections.append([x1, y1, x2, y2])
#             tracked_objects = tracker.update(np.array(detections) if detections else np.empty((0, 4)))
#             frontier_vehicle = None
#             min_y = height
#             for track in tracked_objects:
#                 if len(track) < 5:
#                     continue
#                 x1, y1, x2, y2, track_id = map(int, track)
#                 obj_center_x = (x1 + x2) // 2
#                 obj_center_y = y2
#                 if abs(obj_center_x - center_x) < width * 0.2 and obj_center_y < min_y:
#                     min_y = obj_center_y
#                     frontier_vehicle = track
#             for track in tracked_objects:
#                 if len(track) < 5:
#                     continue
#                 x1, y1, x2, y2, track_id = map(int, track)
#                 color = (255, 0, 0)
#                 event_type = "Tracked"
#                 ttc = None
#                 if np.array_equal(track, frontier_vehicle):
#                     color = (0, 0, 255)
#                     event_type = "Frontier"
#                     frontier_speed = estimate_frontier_speed(track_id)
#                     distance = height - y2
#                     ttc = calculate_ttc(ego_speed, frontier_speed, distance)
#                     if ttc < 2:
#                         event_type = "Near Collision"
#                     x_center = (x1 + x2) // 2
#                     y_center = (y1 + y2) // 2
#                     pred_x, pred_y = kalman_tracker.predict_next_position(x_center, y_center)
#                     cv2.circle(frame, (pred_x, pred_y), 5, (0, 255, 0), -1)
#                     features = pd.DataFrame([[frontier_speed, 0, 0]], columns=["v_Vel", "v_Acc", "Lane_ID"])
#                     scaled_features = scaler.transform(features)
#                     if anomaly_model.predict(scaled_features)[0] == -1:
#                         event_type = f"{event_type} - Anomaly"
#                     cv2.putText(frame, f"TTC: {ttc}s", (x1, y1 - 40),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 25),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
#                 event = EventLog(vehicle_id=track_id, event_type=event_type,
#                                  x1=x1, y1=y1, x2=x2, y2=y2, ttc=ttc,
#                                  latitude=gps_data["latitude"], longitude=gps_data["longitude"],
#                                  motion_status=motion_status)
#                 db.session.add(event)
#             db.session.commit()
#             _, buffer = cv2.imencode(".jpg", frame)
#             yield (b"--frame\r\n"
#                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
#     cap.release()
#
# @app.route("/events", methods=["GET"])
# def get_events():
#     events = EventLog.query.order_by(EventLog.timestamp.desc()).limit(10).all()
#     data = [{
#         "id": e.id,
#         "vehicle_id": e.vehicle_id,
#         "event_type": e.event_type,
#         "timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
#         "x1": e.x1, "y1": e.y1, "x2": e.x2, "y2": e.y2,
#         "ttc": e.ttc if e.ttc is not None else "N/A",
#         "latitude": e.latitude,
#         "longitude": e.longitude,
#         "motion_status": e.motion_status
#     } for e in events]
#     return jsonify(data)
#
# if __name__ == "__main__":
#     app.run(debug=True)
#
# from flask import Flask, request, render_template, jsonify, Response
# import os
# import cv2
# import numpy as np
# import pandas as pd
# from datetime import datetime
# from ultralytics import YOLO
# from flask_sqlalchemy import SQLAlchemy
# from sort import Sort
# from gps_module import get_gps_data
# from vehicle_tracker import VehicleTracker
# from motion_detection import detect_motion_changes
# import joblib
#
# app = Flask(__name__)
#
# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///events.db"
# app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# db = SQLAlchemy(app)
#
# class EventLog(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     vehicle_id = db.Column(db.Integer, nullable=False)
#     event_type = db.Column(db.String(50), nullable=False)
#     timestamp = db.Column(db.DateTime, default=datetime.utcnow)
#     x1 = db.Column(db.Integer)
#     y1 = db.Column(db.Integer)
#     x2 = db.Column(db.Integer)
#     y2 = db.Column(db.Integer)
#     ttc = db.Column(db.Float, nullable=True)
#     latitude = db.Column(db.Float, nullable=True)
#     longitude = db.Column(db.Float, nullable=True)
#     motion_status = db.Column(db.String(50), nullable=True)
#
# with app.app_context():
#     db.create_all()
#
# model = YOLO("yolov8n.pt")
# tracker = Sort()
# kalman_tracker = VehicleTracker()
# anomaly_model = joblib.load("frontier_anomaly_model.pkl")
# scaler = joblib.load("scaler.pkl")
#
# def calculate_ttc(ego_speed, frontier_speed, distance):
#     if frontier_speed <= ego_speed or distance <= 0:
#         return float('inf')
#     relative_speed = (ego_speed - frontier_speed) / 3.6
#     return round(distance / relative_speed, 2) if relative_speed > 0 else float('inf')
#
# def estimate_frontier_speed(track_id):
#     return np.random.uniform(20, 80)
#
# @app.route("/")
# def index():
#     return render_template("index.html")
#
# @app.route("/upload", methods=["POST"])
# def upload_file():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "No selected file"}), 400
#     file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
#     file.save(file_path)
#     return jsonify({"filename": file.filename})
#
# @app.route("/video_feed/<filename>")
# def video_feed(filename):
#     return Response(process_video(os.path.join(app.config["UPLOAD_FOLDER"], filename)),
#                     mimetype="multipart/x-mixed-replace; boundary=frame")
#
# def process_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         return
#     prev_frame = None
#     with app.app_context():
#         while cap.isOpened():
#             success, frame = cap.read()
#             if not success:
#                 cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#                 continue
#             height, width, _ = frame.shape
#             center_x = width // 2
#             gps_data = get_gps_data()
#             ego_speed = gps_data["speed"]
#             motion_status = detect_motion_changes(prev_frame, frame) if prev_frame is not None else "✅ Normal Motion"
#             prev_frame = frame.copy()
#             results = model(frame)[0]
#             detections = []
#             for box in results.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 class_id = int(box.cls[0])
#                 if class_id in [2, 3, 5, 7]:
#                     detections.append([x1, y1, x2, y2])
#             tracked_objects = tracker.update(np.array(detections) if detections else np.empty((0, 4)))
#             frontier_vehicle = None
#             min_y = height
#             for track in tracked_objects:
#                 if len(track) < 5:
#                     continue
#                 x1, y1, x2, y2, track_id = map(int, track)
#                 obj_center_x = (x1 + x2) // 2
#                 obj_center_y = y2
#                 if abs(obj_center_x - center_x) < width * 0.2 and obj_center_y < min_y:
#                     min_y = obj_center_y
#                     frontier_vehicle = track
#             for track in tracked_objects:
#                 if len(track) < 5:
#                     continue
#                 x1, y1, x2, y2, track_id = map(int, track)
#                 color = (255, 0, 0)
#                 event_type = "Tracked"
#                 ttc = None
#                 if np.array_equal(track, frontier_vehicle):
#                     color = (0, 0, 255)
#                     event_type = "Frontier"
#                     frontier_speed = estimate_frontier_speed(track_id)
#                     distance = height - y2
#                     ttc = calculate_ttc(ego_speed, frontier_speed, distance)
#                     if ttc < 2:
#                         event_type = "Near Collision"
#                     x_center = (x1 + x2) // 2
#                     y_center = (y1 + y2) // 2
#                     pred_x, pred_y = kalman_tracker.predict_next_position(x_center, y_center)
#                     cv2.circle(frame, (pred_x, pred_y), 5, (0, 255, 0), -1)
#                     features = pd.DataFrame([[frontier_speed, 0, 0]], columns=["v_Vel", "v_Acc", "Lane_ID"])
#                     scaled_features_array = scaler.transform(features)
#                     scaled_features = pd.DataFrame(scaled_features_array, columns=["v_Vel", "v_Acc", "Lane_ID"])
#                     if anomaly_model.predict(scaled_features)[0] == -1:
#                         event_type = f"{event_type} - Anomaly"
#                     cv2.putText(frame, f"TTC: {ttc}s", (x1, y1 - 40),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 25),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
#                 event = EventLog(vehicle_id=track_id, event_type=event_type,
#                                  x1=x1, y1=y1, x2=x2, y2=y2, ttc=ttc,
#                                  latitude=gps_data["latitude"], longitude=gps_data["longitude"],
#                                  motion_status=motion_status)
#                 db.session.add(event)
#             db.session.commit()
#             _, buffer = cv2.imencode(".jpg", frame)
#             yield (b"--frame\r\n"
#                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
#     cap.release()
#
# @app.route("/events", methods=["GET"])
# def get_events():
#     events = EventLog.query.order_by(EventLog.timestamp.desc()).limit(10).all()
#     data = [{
#         "id": e.id,
#         "vehicle_id": e.vehicle_id,
#         "event_type": e.event_type,
#         "timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
#         "x1": e.x1, "y1": e.y1, "x2": e.x2, "y2": e.y2,
#         "ttc": e.ttc if e.ttc is not None else "N/A",
#         "latitude": e.latitude,
#         "longitude": e.longitude,
#         "motion_status": e.motion_status
#     } for e in events]
#     return jsonify(data)
#
# if __name__ == "__main__":
#     app.run(debug=True)
#
# from flask import Flask, request, render_template, jsonify, Response
# import os
# import cv2
# import numpy as np
# import pandas as pd
# from datetime import datetime
# from ultralytics import YOLO
# from flask_sqlalchemy import SQLAlchemy
# from sort import Sort
# from gps_module import get_gps_data
# from vehicle_tracker import VehicleTracker
# from motion_detection import detect_motion_changes
# import joblib
#
# app = Flask(__name__)
#
# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///events.db"
# app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# db = SQLAlchemy(app)
#
# # Calibration constants (adjust these based on your video/camera)
# FPS = 30  # Frames per second (assumed; replace with cap.get(cv2.CAP_PROP_FPS) if known)
# PIXELS_PER_METER = 0.1  # Example: 10 pixels = 1 meter (needs calibration)
# FRAME_TIME = 1 / FPS  # Time per frame in seconds
#
# # Store vehicle position history
# vehicle_history = {}  # {track_id: {"last_y": y, "last_frame": frame, "speed": speed_kmh}}
#
#
# class EventLog(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     vehicle_id = db.Column(db.Integer, nullable=False)
#     event_type = db.Column(db.String(50), nullable=False)
#     timestamp = db.Column(db.DateTime, default=datetime.utcnow)
#     x1 = db.Column(db.Integer)
#     y1 = db.Column(db.Integer)
#     x2 = db.Column(db.Integer)
#     y2 = db.Column(db.Integer)
#     ttc = db.Column(db.Float, nullable=True)
#     latitude = db.Column(db.Float, nullable=True)
#     longitude = db.Column(db.Float, nullable=True)
#     motion_status = db.Column(db.String(50), nullable=True)
#
#
# with app.app_context():
#     db.create_all()
#
# model = YOLO("yolov8n.pt")
# tracker = Sort()
# kalman_tracker = VehicleTracker()
# anomaly_model = joblib.load("frontier_anomaly_model.pkl")
# scaler = joblib.load("scaler.pkl")
#
#
# def calculate_ttc(ego_speed, frontier_speed, distance):
#     if frontier_speed <= ego_speed or distance <= 0:
#         return float('inf')
#     relative_speed = (ego_speed - frontier_speed) / 3.6
#     return round(distance / relative_speed, 2) if relative_speed > 0 else float('inf')
#
#
# def estimate_frontier_speed(track_id, y_center, frame_count):
#     """Estimate frontier vehicle speed based on vertical displacement in video."""
#     if track_id not in vehicle_history:
#         # Initialize with no prior speed (default 40 km/h)
#         vehicle_history[track_id] = {"last_y": y_center, "last_frame": frame_count, "speed": 40.0}
#         return 40.0
#
#     # Get previous position and frame
#     last_y = vehicle_history[track_id]["last_y"]
#     last_frame = vehicle_history[track_id]["last_frame"]
#
#     # Calculate time difference
#     time_diff = (frame_count - last_frame) * FRAME_TIME  # Seconds
#
#     if time_diff > 0:
#         # Calculate speed in pixels per second (reverse direction: closer = larger y)
#         displacement = last_y - y_center  # Positive if moving away, negative if closer
#         speed_pixels_per_sec = displacement / time_diff
#
#         # Convert to meters per second, then km/h
#         speed_mps = speed_pixels_per_sec * PIXELS_PER_METER
#         speed_kmh = speed_mps * 3.6
#
#         # Clamp to reasonable range (0-120 km/h) and smooth with previous speed
#         alpha = 0.7  # Smoothing factor
#         new_speed = max(0, min(120, speed_kmh))
#         smoothed_speed = alpha * new_speed + (1 - alpha) * vehicle_history[track_id]["speed"]
#         vehicle_history[track_id]["speed"] = smoothed_speed
#
#     # Update history
#     vehicle_history[track_id]["last_y"] = y_center
#     vehicle_history[track_id]["last_frame"] = frame_count
#
#     return vehicle_history[track_id]["speed"]
#
#
# @app.route("/")
# def index():
#     return render_template("index.html")
#
#
# @app.route("/upload", methods=["POST"])
# def upload_file():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "No selected file"}), 400
#     file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
#     file.save(file_path)
#     return jsonify({"filename": file.filename})
#
#
# @app.route("/video_feed/<filename>")
# def video_feed(filename):
#     return Response(process_video(os.path.join(app.config["UPLOAD_FOLDER"], filename)),
#                     mimetype="multipart/x-mixed-replace; boundary=frame")
#
#
# def process_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         return
#     prev_frame = None
#     frame_count = 0  # Track frame number for speed calculation
#     with app.app_context():
#         while cap.isOpened():
#             success, frame = cap.read()
#             if not success:
#                 cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#                 continue
#             frame_count += 1
#             height, width, _ = frame.shape
#             center_x = width // 2
#             gps_data = get_gps_data()
#             ego_speed = gps_data["speed"]
#             motion_status = detect_motion_changes(prev_frame, frame) if prev_frame is not None else "✅ Normal Motion"
#             prev_frame = frame.copy()
#             results = model(frame)[0]
#             detections = []
#             for box in results.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 class_id = int(box.cls[0])
#                 if class_id in [2, 3, 5, 7]:
#                     detections.append([x1, y1, x2, y2])
#             tracked_objects = tracker.update(np.array(detections) if detections else np.empty((0, 4)))
#             frontier_vehicle = None
#             min_y = height
#             for track in tracked_objects:
#                 if len(track) < 5:
#                     continue
#                 x1, y1, x2, y2, track_id = map(int, track)
#                 obj_center_x = (x1 + x2) // 2
#                 obj_center_y = y2
#                 if abs(obj_center_x - center_x) < width * 0.2 and obj_center_y < min_y:
#                     min_y = obj_center_y
#                     frontier_vehicle = track
#             for track in tracked_objects:
#                 if len(track) < 5:
#                     continue
#                 x1, y1, x2, y2, track_id = map(int, track)
#                 color = (255, 0, 0)
#                 event_type = "Tracked"
#                 ttc = None
#                 if np.array_equal(track, frontier_vehicle):
#                     color = (0, 0, 255)
#                     event_type = "Frontier"
#                     y_center = (y1 + y2) // 2  # Use vertical center for speed
#                     frontier_speed = estimate_frontier_speed(track_id, y_center, frame_count)
#                     distance = height - y2
#                     ttc = calculate_ttc(ego_speed, frontier_speed, distance)
#                     if ttc < 2:
#                         event_type = "Near Collision"
#                     x_center = (x1 + x2) // 2
#                     pred_x, pred_y = kalman_tracker.predict_next_position(x_center, y_center)
#                     cv2.circle(frame, (pred_x, pred_y), 5, (0, 255, 0), -1)
#                     features = pd.DataFrame([[frontier_speed, 0, 0]], columns=["v_Vel", "v_Acc", "Lane_ID"])
#                     scaled_features_array = scaler.transform(features)
#                     scaled_features = pd.DataFrame(scaled_features_array, columns=["v_Vel", "v_Acc", "Lane_ID"])
#                     if anomaly_model.predict(scaled_features)[0] == -1:
#                         event_type = f"{event_type} - Anomaly"
#                     cv2.putText(frame, f"TTC: {ttc}s", (x1, y1 - 40),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
#                     # Optional: Display speed on video
#                     cv2.putText(frame, f"Speed: {frontier_speed:.1f} km/h", (x1, y1 - 60),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 25),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
#                 event = EventLog(vehicle_id=track_id, event_type=event_type,
#                                  x1=x1, y1=y1, x2=x2, y2=y2, ttc=ttc,
#                                  latitude=gps_data["latitude"], longitude=gps_data["longitude"],
#                                  motion_status=motion_status)
#                 db.session.add(event)
#             db.session.commit()
#             _, buffer = cv2.imencode(".jpg", frame)
#             yield (b"--frame\r\n"
#                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
#     cap.release()
#
#
# @app.route("/events", methods=["GET"])
# def get_events():
#     events = EventLog.query.order_by(EventLog.timestamp.desc()).limit(10).all()
#     data = [{
#         "id": e.id,
#         "vehicle_id": e.vehicle_id,
#         "event_type": e.event_type,
#         "timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
#         "x1": e.x1, "y1": e.y1, "x2": e.x2, "y2": e.y2,
#         "ttc": e.ttc if e.ttc is not None else "N/A",
#         "latitude": e.latitude,
#         "longitude": e.longitude,
#         "motion_status": e.motion_status
#     } for e in events]
#     return jsonify(data)
#
#
# if __name__ == "__main__":
#     app.run(debug=True)
#
# from flask import Flask, request, render_template, jsonify, Response
# import os
# import cv2
# import numpy as np
# import pandas as pd
# from datetime import datetime
# from ultralytics import YOLO
# from flask_sqlalchemy import SQLAlchemy
# from sort import Sort
# from gps_module import get_gps_data
# from vehicle_tracker import VehicleTracker
# from motion_detection import detect_motion_changes
# import joblib
# import torch
#
# app = Flask(__name__)
#
# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///events.db"
# app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# db = SQLAlchemy(app)
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")
#
# FPS = 30
# PIXELS_PER_METER = 0.1
# vehicle_history = {}
#
# class EventLog(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     vehicle_id = db.Column(db.Integer, nullable=False)
#     event_type = db.Column(db.String(50), nullable=False)
#     timestamp = db.Column(db.DateTime, default=datetime.utcnow)
#     x1 = db.Column(db.Integer)
#     y1 = db.Column(db.Integer)
#     x2 = db.Column(db.Integer)
#     y2 = db.Column(db.Integer)
#     ttc = db.Column(db.Float, nullable=True)
#     latitude = db.Column(db.Float, nullable=True)
#     longitude = db.Column(db.Float, nullable=True)
#     motion_status = db.Column(db.String(50), nullable=True)
#
# with app.app_context():
#     db.create_all()
#
# model = YOLO("yolov8n.pt").to(device)
# tracker = Sort()
# kalman_tracker = VehicleTracker()
# anomaly_model = joblib.load("frontier_anomaly_model.pkl")
# scaler = joblib.load("scaler.pkl")
#
# def calculate_ttc(ego_speed, frontier_speed, distance):
#     if frontier_speed <= ego_speed or distance <= 0:
#         return float('inf')
#     relative_speed = (ego_speed - frontier_speed) / 3.6
#     return round(distance / relative_speed, 2) if relative_speed > 0 else float('inf')
#
# def estimate_frontier_speed(track_id, y_center, frame_count, frame_time):
#     if track_id not in vehicle_history:
#         vehicle_history[track_id] = {"last_y": y_center, "last_frame": frame_count, "speed": 40.0}
#         return 40.0
#     last_y = vehicle_history[track_id]["last_y"]
#     last_frame = vehicle_history[track_id]["last_frame"]
#     time_diff = (frame_count - last_frame) * frame_time
#     if time_diff > 0:
#         displacement = last_y - y_center
#         speed_pixels_per_sec = displacement / time_diff
#         speed_mps = speed_pixels_per_sec * PIXELS_PER_METER
#         speed_kmh = speed_mps * 3.6
#         alpha = 0.7
#         new_speed = max(0, min(120, speed_kmh))
#         smoothed_speed = alpha * new_speed + (1 - alpha) * vehicle_history[track_id]["speed"]
#         vehicle_history[track_id]["speed"] = smoothed_speed
#     vehicle_history[track_id]["last_y"] = y_center
#     vehicle_history[track_id]["last_frame"] = frame_count
#     return vehicle_history[track_id]["speed"]
#
# @app.route("/")
# def index():
#     return render_template("index.html")
#
# @app.route("/upload", methods=["POST"])
# def upload_file():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "No selected file"}), 400
#     file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
#     file.save(file_path)
#     return jsonify({"filename": file.filename})
#
# @app.route("/video_feed/<filename>")
# def video_feed(filename):
#     print(f"Streaming video: {filename}")  # Debug to confirm request
#     return Response(process_video(os.path.join(app.config["UPLOAD_FOLDER"], filename)),
#                     mimetype="multipart/x-mixed-replace; boundary=frame")
#
# def process_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Failed to open video: {video_path}")
#         return
#     FPS = cap.get(cv2.CAP_PROP_FPS) or 30
#     FRAME_TIME = 1 / FPS
#     prev_frame = None
#     frame_count = 0
#     with app.app_context():
#         while cap.isOpened():
#             success, frame = cap.read()
#             if not success:
#                 cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#                 continue
#             frame_count += 1
#             if frame_count % 2 != 0:  # Skip every other frame
#                 continue
#             frame = cv2.resize(frame, (320, 544))  # Apply optimization
#             height, width, _ = frame.shape
#             center_x = width // 2
#             gps_data = get_gps_data()
#             ego_speed = gps_data["speed"]
#             motion_status = detect_motion_changes(prev_frame, frame) if prev_frame is not None else "✅ Normal Motion"
#             prev_frame = frame.copy()
#             results = model(frame)[0]
#             detections = []
#             for box in results.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 class_id = int(box.cls[0])
#                 if class_id in [2, 3, 5, 7]:
#                     detections.append([x1, y1, x2, y2])
#             tracked_objects = tracker.update(np.array(detections) if detections else np.empty((0, 4)))
#             frontier_vehicle = None
#             min_y = height
#             for track in tracked_objects:
#                 if len(track) < 5:
#                     continue
#                 x1, y1, x2, y2, track_id = map(int, track)
#                 obj_center_x = (x1 + x2) // 2
#                 obj_center_y = y2
#                 if abs(obj_center_x - center_x) < width * 0.2 and obj_center_y < min_y:
#                     min_y = obj_center_y
#                     frontier_vehicle = track
#             for track in tracked_objects:
#                 if len(track) < 5:
#                     continue
#                 x1, y1, x2, y2, track_id = map(int, track)
#                 color = (255, 0, 0)
#                 event_type = "Tracked"
#                 ttc = None
#                 if np.array_equal(track, frontier_vehicle):
#                     color = (0, 0, 255)
#                     event_type = "Frontier"
#                     y_center = (y1 + y2) // 2
#                     frontier_speed = estimate_frontier_speed(track_id, y_center, frame_count, FRAME_TIME)
#                     distance = height - y2
#                     ttc = calculate_ttc(ego_speed, frontier_speed, distance)
#                     if ttc < 2:
#                         event_type = "Near Collision"
#                     x_center = (x1 + x2) // 2
#                     pred_x, pred_y = kalman_tracker.predict_next_position(x_center, y_center)
#                     cv2.circle(frame, (pred_x, pred_y), 5, (0, 255, 0), -1)
#                     features = pd.DataFrame([[frontier_speed, 0, 0]], columns=["v_Vel", "v_Acc", "Lane_ID"])
#                     scaled_features_array = scaler.transform(features)
#                     scaled_features = pd.DataFrame(scaled_features_array, columns=["v_Vel", "v_Acc", "Lane_ID"])
#                     if anomaly_model.predict(scaled_features)[0] == -1:
#                         event_type = f"{event_type} - Anomaly"
#                     cv2.putText(frame, f"TTC: {ttc}s", (x1, y1 - 40),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
#                     cv2.putText(frame, f"Speed: {frontier_speed:.1f} km/h", (x1, y1 - 60),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 25),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
#                 event = EventLog(vehicle_id=track_id, event_type=event_type,
#                                  x1=x1, y1=y1, x2=x2, y2=y2, ttc=ttc,
#                                  latitude=gps_data["latitude"], longitude=gps_data["longitude"],
#                                  motion_status=motion_status)
#                 db.session.add(event)
#             if frame_count % 5 == 0:
#                 db.session.commit()
#             _, buffer = cv2.imencode(".jpg", frame)
#             frame_data = (b"--frame\r\n"
#                           b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
#             yield frame_data
#         db.session.commit()
#         cap.release()
#
# @app.route("/events", methods=["GET"])
# def get_events():
#     events = EventLog.query.order_by(EventLog.timestamp.desc()).limit(10).all()
#     data = [{
#         "id": e.id,
#         "vehicle_id": e.vehicle_id,
#         "event_type": e.event_type,
#         "timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
#         "x1": e.x1, "y1": e.y1, "x2": e.x2, "y2": e.y2,
#         "ttc": e.ttc if e.ttc is not None else "N/A",
#         "latitude": e.latitude,
#         "longitude": e.longitude,
#         "motion_status": e.motion_status
#     } for e in events]
#     return jsonify(data)
#
# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, render_template, jsonify, Response
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


#
# from flask import Flask, request, render_template, jsonify, Response
# import os
# import cv2
# import numpy as np
# import pandas as pd
# from datetime import datetime
# from ultralytics import YOLO
# from flask_sqlalchemy import SQLAlchemy
# from sort import Sort
# from gps_module import get_gps_data
# from vehicle_tracker import VehicleTracker
# from motion_detection import detect_motion_changes
# import joblib
# import torch
#
# app = Flask(__name__)
#
# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///events.db"
# app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# db = SQLAlchemy(app)
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")
#
# PIXELS_PER_METER = 0.1  # Placeholder; calibrate later
# vehicle_history = {}
#
#
# class EventLog(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     vehicle_id = db.Column(db.Integer, nullable=False)
#     event_type = db.Column(db.String(50), nullable=False)
#     timestamp = db.Column(db.DateTime, default=datetime.utcnow)
#     x1 = db.Column(db.Integer)
#     y1 = db.Column(db.Integer)
#     x2 = db.Column(db.Integer)
#     y2 = db.Column(db.Integer)
#     ttc = db.Column(db.Float, nullable=True)
#     latitude = db.Column(db.Float, nullable=True)
#     longitude = db.Column(db.Float, nullable=True)
#     motion_status = db.Column(db.String(50), nullable=True)
#
#
# with app.app_context():
#     db.create_all()
#
# model = YOLO("yolov8n.pt").to(device)
# tracker = Sort()
# kalman_tracker = VehicleTracker()
# anomaly_model = joblib.load("frontier_anomaly_model.pkl")
# scaler = joblib.load("scaler.pkl")
#
#
# def calculate_ttc(ego_speed, frontier_speed, distance):
#     if frontier_speed <= ego_speed or distance <= 0:
#         return float('inf')
#     relative_speed = (ego_speed - frontier_speed) / 3.6
#     return round(distance / relative_speed, 2) if relative_speed > 0 else float('inf')
#
#
# def estimate_frontier_speed(track_id, y_center, frame_count, frame_time):
#     if track_id not in vehicle_history:
#         vehicle_history[track_id] = {"last_y": y_center, "last_frame": frame_count, "speed": 40.0}
#         return 40.0
#     last_y = vehicle_history[track_id]["last_y"]
#     last_frame = vehicle_history[track_id]["last_frame"]
#     time_diff = (frame_count - last_frame) * frame_time
#     if time_diff > 0:
#         displacement = last_y - y_center
#         speed_pixels_per_sec = displacement / time_diff
#         speed_mps = speed_pixels_per_sec * PIXELS_PER_METER
#         speed_kmh = speed_mps * 3.6
#         alpha = 0.7
#         new_speed = max(0, min(120, speed_kmh))
#         smoothed_speed = alpha * new_speed + (1 - alpha) * vehicle_history[track_id]["speed"]
#         vehicle_history[track_id]["speed"] = smoothed_speed
#     vehicle_history[track_id]["last_y"] = y_center
#     vehicle_history[track_id]["last_frame"] = frame_count
#     return vehicle_history[track_id]["speed"]
#
#
# def detect_lanes(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blur, 100, 200)
#     height, width = frame.shape[:2]
#     mask = np.zeros_like(edges)
#     roi_vertices = np.array([
#         [0, height * 0.6], [width, height * 0.6], [width, height], [0, height]
#     ], np.int32)
#     cv2.fillPoly(mask, [roi_vertices], 255)
#     masked_edges = cv2.bitwise_and(edges, mask)
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     lower_white = np.array([0, 0, 200])
#     upper_white = np.array([180, 30, 255])
#     white_mask = cv2.inRange(hsv, lower_white, upper_white)
#     masked_edges = cv2.bitwise_and(masked_edges, masked_edges, mask=white_mask)
#     lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=100, minLineLength=80, maxLineGap=30)
#     lane_lines = []
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             lane_lines.append((x1, y1, x2, y2))
#     return lane_lines
#
#
# def get_ego_lane_bounds(lane_lines, width, height):
#     if not lane_lines:
#         return 0, width
#     left_lane_x = width
#     right_lane_x = 0
#     left_lines = []
#     right_lines = []
#     for x1, y1, x2, y2 in lane_lines:
#         if y1 != y2:
#             slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
#             x_bottom = x1 + (x2 - x1) * (height - y1) / (y2 - y1)
#             if slope > 0.5:
#                 right_lines.append(x_bottom)
#             elif slope < -0.5:
#                 left_lines.append(x_bottom)
#     if left_lines:
#         left_lane_x = max(0, min(left_lines) - 50)
#     if right_lines:
#         right_lane_x = min(width, max(right_lines) + 50)
#     return int(left_lane_x), int(right_lane_x)
#
#
# @app.route("/")
# def index():
#     return render_template("index.html")
#
#
# @app.route("/upload", methods=["POST"])
# def upload_file():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "No selected file"}), 400
#     file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
#     file.save(file_path)
#     return jsonify({"filename": file.filename})
#
#
# @app.route("/video_feed/<filename>")
# def video_feed(filename):
#     print(f"Streaming video: {filename}")
#     return Response(process_video(os.path.join(app.config["UPLOAD_FOLDER"], filename)),
#                     mimetype="multipart/x-mixed-replace; boundary=frame")
#
#
# def process_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Failed to open video: {video_path}")
#         return
#     FPS = cap.get(cv2.CAP_PROP_FPS) or 30
#     FRAME_TIME = 1 / FPS
#     prev_frame = None
#     frame_count = 0
#     prev_tracks = {}  # Store previous positions {track_id: (x, y)}
#     with app.app_context():
#         while cap.isOpened():
#             success, frame = cap.read()
#             if not success:
#                 cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#                 continue
#             frame_count += 1
#             if frame_count % 2 != 0:
#                 continue
#             frame = cv2.resize(frame, (320, 192))
#             height, width, _ = frame.shape
#             center_x = width // 2
#
#             lane_lines = detect_lanes(frame)
#             left_lane_x, right_lane_x = get_ego_lane_bounds(lane_lines, width, height)
#             for x1, y1, x2, y2 in lane_lines:
#                 cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
#             gps_data = get_gps_data()
#             ego_speed = gps_data["speed"]
#             motion_status = detect_motion_changes(prev_frame, frame) if prev_frame is not None else "✅ Normal Motion"
#             prev_frame = frame.copy()
#             results = model(frame)[0]
#             detections = []
#             for box in results.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 class_id = int(box.cls[0])
#                 if class_id in [2, 3, 5, 7]:
#                     detections.append([x1, y1, x2, y2])
#             tracked_objects = tracker.update(np.array(detections) if detections else np.empty((0, 4)))
#             frontier_vehicle = None
#             min_y = height
#             for track in tracked_objects:
#                 if len(track) < 5:
#                     continue
#                 x1, y1, x2, y2, track_id = map(int, track)
#                 obj_center_x = (x1 + x2) // 2
#                 obj_center_y = y2
#                 if (left_lane_x <= x1 and right_lane_x >= x2 and obj_center_y < min_y):
#                     min_y = obj_center_y
#                     frontier_vehicle = track
#
#             font = cv2.FONT_HERSHEY_SIMPLEX  # Define font outside the loop
#             font_scale = 0.35
#             thickness = 1
#
#             for track in tracked_objects:
#                 if len(track) < 5:
#                     continue
#                 x1, y1, x2, y2, track_id = map(int, track)
#                 color = (255, 0, 0)
#                 event_type = "Tracked"
#                 ttc = None
#                 vehicle_motion = "✅ Normal Motion"
#                 if np.array_equal(track, frontier_vehicle):
#                     color = (0, 0, 255)
#                     event_type = "Frontier"
#                     y_center = (y1 + y2) // 2
#                     frontier_speed = estimate_frontier_speed(track_id, y_center, frame_count, FRAME_TIME)
#                     distance = height - y2
#                     ttc = calculate_ttc(ego_speed, frontier_speed, distance)
#                     if ttc < 2:
#                         event_type = "Near Collision"
#                     x_center = (x1 + x2) // 2
#                     pred_x, pred_y = kalman_tracker.predict_next_position(x_center, y_center)
#                     cv2.circle(frame, (pred_x, pred_y), 5, (0, 255, 0), -1)
#                     features = pd.DataFrame([[frontier_speed, 0, 0]], columns=["v_Vel", "v_Acc", "Lane_ID"])
#                     scaled_features_array = scaler.transform(features)
#                     scaled_features = pd.DataFrame(scaled_features_array, columns=["v_Vel", "v_Acc", "Lane_ID"])
#                     if anomaly_model.predict(scaled_features)[0] == -1:
#                         event_type = f"{event_type} - Anomaly"
#
#                     # Detect per-vehicle motion
#                     current_pos = (x_center, y_center)
#                     if track_id in prev_tracks:
#                         prev_pos = prev_tracks[track_id]
#                         dist_moved = ((current_pos[0] - prev_pos[0]) ** 2 + (current_pos[1] - prev_pos[1]) ** 2) ** 0.5
#                         speed_px_per_frame = dist_moved / FRAME_TIME
#                         if speed_px_per_frame < 1.0:  # Threshold for sudden stop (adjust as needed)
#                             vehicle_motion = "🚨 Sudden Stop Detected!"
#                         elif speed_px_per_frame > 10.0:  # Threshold for harsh braking (adjust as needed)
#                             vehicle_motion = "⚠️ Harsh Braking"
#                     prev_tracks[track_id] = current_pos
#
#                     # TTC Label: Yellow background, red font
#                     ttc_text = f"TTC: {ttc if ttc != float('inf') else 'N/A'}s"
#                     ttc_size, _ = cv2.getTextSize(ttc_text, font, font_scale, thickness)
#                     ttc_width, ttc_height = ttc_size
#                     ttc_pos = (x1, y1 - 40)
#                     ttc_bg_pos1 = (ttc_pos[0] - 2, ttc_pos[1] - ttc_height - 2)
#                     ttc_bg_pos2 = (ttc_pos[0] + ttc_width + 2, ttc_pos[1] + 2)
#                     cv2.rectangle(frame, ttc_bg_pos1, ttc_bg_pos2, (0, 255, 255), -1)  # Yellow background
#                     cv2.putText(frame, ttc_text, ttc_pos, font, font_scale, (0, 0, 255), thickness)  # Red font
#
#                     # Speed Label: White background, black font
#                     speed_text = f"Speed: {frontier_speed:.1f} km/h"
#                     speed_size, _ = cv2.getTextSize(speed_text, font, font_scale, thickness)
#                     speed_width, speed_height = speed_size
#                     speed_pos = (x1, y1 - 60)
#                     speed_bg_pos1 = (speed_pos[0] - 2, speed_pos[1] - speed_height - 2)
#                     speed_bg_pos2 = (speed_pos[0] + speed_width + 2, speed_pos[1] + 2)
#                     cv2.rectangle(frame, speed_bg_pos1, speed_bg_pos2, (255, 255, 255), -1)  # White background
#                     cv2.putText(frame, speed_text, speed_pos, font, font_scale, (0, 0, 0), thickness)  # Black font
#
#                     # Motion Label: Orange background, black font
#                     motion_text = f"Motion: {vehicle_motion}"
#                     motion_size, _ = cv2.getTextSize(motion_text, font, font_scale, thickness)
#                     motion_width, motion_height = motion_size
#                     motion_pos = (x1, y1 - 80)
#                     motion_bg_pos1 = (motion_pos[0] - 2, motion_pos[1] - motion_height - 2)
#                     motion_bg_pos2 = (motion_pos[0] + motion_width + 2, motion_pos[1] + 2)
#                     cv2.rectangle(frame, motion_bg_pos1, motion_bg_pos2, (0, 165, 255), -1)  # Orange background
#                     cv2.putText(frame, motion_text, motion_pos, font, font_scale, (0, 0, 0), thickness)  # Black font
#
#                 # ID Label: Green background, black font (for all vehicles)
#                 id_text = f"ID: {track_id}"
#                 id_size, _ = cv2.getTextSize(id_text, font, font_scale, thickness)
#                 id_width, id_height = id_size
#                 id_pos = (x1, y1 - 20)  # Adjusted position to avoid overlap
#                 id_bg_pos1 = (id_pos[0] - 2, id_pos[1] - id_height - 2)
#                 id_bg_pos2 = (id_pos[0] + id_width + 2, id_pos[1] + 2)
#                 cv2.rectangle(frame, id_bg_pos1, id_bg_pos2, (0, 255, 0), -1)  # Green background
#                 cv2.putText(frame, id_text, id_pos, font, font_scale, (0, 0, 0), thickness)  # Black font
#
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#
#                 event = EventLog(vehicle_id=track_id, event_type=event_type,
#                                  x1=x1, y1=y1, x2=x2, y2=y2, ttc=ttc,
#                                  latitude=gps_data["latitude"], longitude=gps_data["longitude"],
#                                  motion_status=vehicle_motion)
#                 db.session.add(event)
#                 print(f"Added event: {event.vehicle_id}, {event.event_type}, {event.timestamp}")
#             try:
#                 if frame_count % 5 == 0:
#                     db.session.commit()
#                     print(f"Committed {frame_count} frames")
#             except Exception as e:
#                 print(f"Commit failed: {e}")
#             _, buffer = cv2.imencode(".jpg", frame)
#             yield (b"--frame\r\n"
#                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\rn")
#         try:
#             db.session.commit()
#         except Exception as e:
#             print(f"Final commit failed: {e}")
#         cap.release()
#
#
# @app.route("/events", methods=["GET"])
# def get_events():
#     events = EventLog.query.order_by(EventLog.timestamp.desc()).limit(10).all()
#     print(f"Events retrieved: {len(events)}")
#     data = [{
#         "id": e.id,
#         "vehicle_id": e.vehicle_id,
#         "event_type": e.event_type,
#         "timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
#         "x1": e.x1, "y1": e.y1, "x2": e.x2, "y2": e.y2,
#         "ttc": "N/A" if e.ttc == float('inf') or e.ttc is None else e.ttc,
#         "latitude": e.latitude,
#         "longitude": e.longitude,
#         "motion_status": e.motion_status
#     } for e in events]
#     print(f"Data returned: {data}")
#     return jsonify(data)
#
#
# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, request, render_template, jsonify, Response
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

from flask import Flask, request, render_template, jsonify, Response
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

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///events.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

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

model = YOLO("yolov8n.pt").to(device)
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
    """Calculate Intersection over Union (IoU) between two boxes [x1, y1, x2, y2]."""
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
        return
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30
    FRAME_TIME = 1 / FPS
    prev_frame = None
    frame_count = 0
    prev_tracks = {}  # Store previous positions {track_id: (x, y)}
    with app.app_context():
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame_count += 1
            if frame_count % 2 != 0:
                continue
            frame = cv2.resize(frame, (320, 192))
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
                if class_id in [2, 3, 5, 7]:
                    detections.append([x1, y1, x2, y2])
            tracked_objects = tracker.update(np.array(detections) if detections else np.empty((0, 4)))
            frontier_vehicle = None
            min_y = height
            for track in tracked_objects:
                if len(track) < 5:
                    continue
                x1, y1, x2, y2, track_id = map(int, track)
                obj_center_x = (x1 + x2) // 2
                obj_center_y = y2
                if (left_lane_x <= x1 and right_lane_x >= x2 and obj_center_y < min_y):
                    min_y = obj_center_y
                    frontier_vehicle = track

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.35
            thickness = 1

            for track in tracked_objects:
                if len(track) < 5:
                    continue
                x1, y1, x2, y2, track_id = map(int, track)
                color = (255, 0, 0)  # Default color: Blue
                event_type = "Tracked"
                ttc = None  # Initialize ttc for all vehicles
                vehicle_motion = "✅ Normal Motion"
                if np.array_equal(track, frontier_vehicle):
                    color = (0, 255, 0)  # Green for frontier vehicle
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

                    # Detect per-vehicle motion
                    current_pos = (x_center, y_center)
                    if track_id in prev_tracks:
                        prev_pos = prev_tracks[track_id]
                        dist_moved = ((current_pos[0] - prev_pos[0]) ** 2 + (current_pos[1] - prev_pos[1]) ** 2) ** 0.5
                        speed_px_per_frame = dist_moved / FRAME_TIME
                        if speed_px_per_frame < 0.5:  # Fine-tuned threshold for sudden stop
                            vehicle_motion = "🚨 Sudden Stop Detected!"
                        elif speed_px_per_frame > 5.0:  # Fine-tuned threshold for harsh braking
                            vehicle_motion = "⚠️ Harsh Braking"
                    prev_tracks[track_id] = current_pos

                    # Improved collision detection
                    is_collision = False
                    if ttc is not None and ttc < 0.3:  # Fine-tuned TTC threshold
                        is_collision = True
                    for other_track in tracked_objects:
                        if not np.array_equal(other_track, track) and len(other_track) >= 5:
                            ox1, oy1, ox2, oy2, other_id = map(int, other_track)
                            iou = calculate_iou([x1, y1, x2, y2], [ox1, oy1, ox2, oy2])
                            if iou > 0.5:  # IoU threshold for collision
                                is_collision = True
                                collided_vehicles.add(other_id)
                                collision_cooldown[other_id] = frame_count + (5 * FPS)

                    if is_collision and vehicle_motion not in ["Collided", "🚨 Sudden Stop Detected!", "⚠️ Harsh Braking"]:
                        vehicle_motion = "Collided"
                        collided_vehicles.add(track_id)
                        collision_cooldown[track_id] = frame_count + (5 * FPS)

                    # TTC Label: Yellow background, red font
                    ttc_text = f"TTC: {ttc if ttc != float('inf') else 'N/A'}s"
                    ttc_size, _ = cv2.getTextSize(ttc_text, font, font_scale, thickness)
                    ttc_width, ttc_height = ttc_size
                    ttc_pos = (x1, y1 - 40)
                    ttc_bg_pos1 = (ttc_pos[0] - 2, ttc_pos[1] - ttc_height - 2)
                    ttc_bg_pos2 = (ttc_pos[0] + ttc_width + 2, ttc_pos[1] + 2)
                    cv2.rectangle(frame, ttc_bg_pos1, ttc_bg_pos2, (0, 255, 255), -1)
                    cv2.putText(frame, ttc_text, ttc_pos, font, font_scale, (0, 0, 255), thickness)

                    # Speed Label: White background, black font
                    speed_text = f"Speed: {frontier_speed:.1f} km/h"
                    speed_size, _ = cv2.getTextSize(speed_text, font, font_scale, thickness)
                    speed_width, speed_height = speed_size
                    speed_pos = (x1, y1 - 60)
                    speed_bg_pos1 = (speed_pos[0] - 2, speed_pos[1] - speed_height - 2)
                    speed_bg_pos2 = (speed_pos[0] + speed_width + 2, speed_pos[1] + 2)
                    cv2.rectangle(frame, speed_bg_pos1, speed_bg_pos2, (255, 255, 255), -1)
                    cv2.putText(frame, speed_text, speed_pos, font, font_scale, (0, 0, 0), thickness)

                    # Motion Label: Orange background, black font
                    motion_text = f"Motion: {vehicle_motion}"
                    motion_size, _ = cv2.getTextSize(motion_text, font, font_scale, thickness)
                    motion_width, motion_height = motion_size
                    motion_pos = (x1, y1 - 80)
                    motion_bg_pos1 = (motion_pos[0] - 2, motion_pos[1] - motion_height - 2)
                    motion_bg_pos2 = (motion_pos[0] + motion_width + 2, motion_pos[1] + 2)
                    cv2.rectangle(frame, motion_bg_pos1, motion_bg_pos2, (0, 165, 255), -1)
                    cv2.putText(frame, motion_text, motion_pos, font, font_scale, (0, 0, 0), thickness)

                # ID Label: Green background, black font (for all vehicles)
                id_text = f"ID: {track_id}"
                id_size, _ = cv2.getTextSize(id_text, font, font_scale, thickness)
                id_width, id_height = id_size
                id_pos = (x1, y1 - 20)
                id_bg_pos1 = (id_pos[0] - 2, id_pos[1] - id_height - 2)
                id_bg_pos2 = (id_pos[0] + id_width + 2, id_pos[1] + 2)
                cv2.rectangle(frame, id_bg_pos1, id_bg_pos2, (0, 255, 0), -1)
                cv2.putText(frame, id_text, id_pos, font, font_scale, (0, 0, 0), thickness)

                # Change color to red if vehicle is collided
                if track_id in collided_vehicles and frame_count <= collision_cooldown.get(track_id, 0):
                    color = (0, 0, 255)  # Red for collided vehicle
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
            try:
                if frame_count % 5 == 0:
                    db.session.commit()
                    print(f"Committed {frame_count} frames")
            except Exception as e:
                print(f"Commit failed: {e}")
            _, buffer = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        try:
            db.session.commit()
        except Exception as e:
            print(f"Final commit failed: {e}")
        cap.release()

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

if __name__ == "__main__":
    app.run(debug=True)
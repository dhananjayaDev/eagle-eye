import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sort import Sort
import logging
import os
from pathlib import Path
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
VIDEO_PATH = Path("uploads/video_0001.mp4")
MODEL_PATH = Path("frontier_classifier.pkl")
FRAME_SKIP = 5
TARGET_SIZE = (640, 480)
MAX_FRAMES_TO_PROCESS = 50  # Limit for testing, adjust as needed

# Load YOLO and tracker
try:
    model = YOLO("yolov8n.pt")
    tracker = Sort()
    logger.info("YOLOv8 and SORT tracker initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize YOLO or SORT: {e}")
    raise

# Load the trained model
try:
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    logger.error(f"Model file {MODEL_PATH} not found. Please ensure it exists in the current directory.")
    raise
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise


# Enhanced lane detection
def detect_lanes(frame):
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        lane_mask = cv2.bitwise_or(white_mask, yellow_mask)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 100, 200)
        height, width = frame.shape[:2]
        roi_vertices = np.array([[0, height * 0.6], [width, height * 0.6], [width, height], [0, height]], np.int32)
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, [roi_vertices], 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        masked_edges = cv2.bitwise_and(masked_edges, lane_mask)

        lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=20)
        lane_lines = [] if lines is None else [line[0] for line in lines]
        logger.debug(f"Detected {len(lane_lines)} lane lines.")
        return lane_lines
    except Exception as e:
        logger.error(f"Error in lane detection: {e}")
        return []


def get_ego_lane_bounds(lane_lines, width, height):
    if not lane_lines:
        logger.warning("No lane lines detected, using full width as ego lane.")
        return 0, width
    left_lane_x, right_lane_x = width, 0
    for x1, y1, x2, y2 in lane_lines:
        if y1 != y2:
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
            x_bottom = x1 + (x2 - x1) * (height - y1) / (y2 - y1)
            if slope > 0.5:
                right_lane_x = min(width, max(right_lane_x, int(x_bottom + 50)))
            elif slope < -0.5:
                left_lane_x = max(0, min(left_lane_x, int(x_bottom - 50)))
    logger.debug(f"Ego lane bounds: ({left_lane_x}, {right_lane_x})")
    return left_lane_x, right_lane_x


# Process video and test the model
def test_model(video_path, max_frames=MAX_FRAMES_TO_PROCESS):
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return

    frame_count = 0
    processed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video reached.")
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        processed_frames += 1
        if processed_frames > max_frames:
            logger.info(f"Reached max frames to process: {max_frames}")
            break

        frame = cv2.resize(frame, TARGET_SIZE)
        height, width = TARGET_SIZE

        # Detect lanes
        lane_lines = detect_lanes(frame)
        left_lane_x, right_lane_x = get_ego_lane_bounds(lane_lines, width, height)

        # YOLO detection and tracking
        try:
            results = model(frame, verbose=False)[0]
            detections = [[int(x) for x in box.xyxy[0]] for box in results.boxes if int(box.cls[0]) in [2, 3, 5, 7]]
            tracks = tracker.update(np.array(detections) if detections else np.empty((0, 4)))
            logger.debug(f"Frame {frame_count}: Detected {len(tracks)} vehicles.")
        except Exception as e:
            logger.error(f"Error in YOLO/tracking: {e}")
            continue

        # Collect features for ML prediction
        test_data = []
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            center_x, center_y = (x1 + x2) // 2, y2
            width_bbox = x2 - x1
            height_bbox = y2 - y1
            distance_from_bottom = height - y2
            in_ego_lane = 1 if left_lane_x <= center_x <= right_lane_x else 0
            relative_x = (center_x - left_lane_x) / (right_lane_x - left_lane_x) if right_lane_x > left_lane_x else 0.5
            test_data.append(
                [x1, y1, x2, y2, center_x, center_y, width_bbox, height_bbox, distance_from_bottom, in_ego_lane,
                 relative_x])

        # Predict frontier vehicle using ML model
        if test_data:
            predictions = clf.predict(test_data)
            frontier_id = tracks[np.argmax(predictions)][4] if any(predictions) else None
        else:
            frontier_id = None

        # Draw on frame for real-time display
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            color = (0, 255, 0) if track_id == frontier_id else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display frame
        cv2.imshow("Frontier Vehicle Tracking (Test Mode)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"Test completed for {processed_frames} frames.")


# Run the test
if __name__ == "__main__":
    test_model(VIDEO_PATH)
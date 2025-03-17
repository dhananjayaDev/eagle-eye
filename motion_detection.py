# import cv2
# import numpy as np
#
# def detect_motion_changes(prev_frame, curr_frame):
#     """Detects motion changes using Optical Flow (Farneback method)."""
#     if prev_frame is None or curr_frame is None:
#         return "⚠️ No Frame Data"
#
#     prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#     curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
#
#     # Compute Optical Flow
#     flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
#                                         0.5, 3, 15, 3, 5, 1.2, 0)
#
#     # Compute magnitude of flow vectors
#     motion_magnitude = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
#
#     # Define thresholds for anomaly detection
#     if motion_magnitude > 10:
#         return "🚨 Sudden Stop Detected!"
#     elif motion_magnitude > 5:
#         return "⚠️ Harsh Braking"
#     else:
#         return "✅ Normal Motion"


import cv2
import numpy as np

def detect_motion_changes(prev_frame, curr_frame):
    """Detects motion changes using Optical Flow (Farneback method)."""
    if prev_frame is None or curr_frame is None:
        return "No Frame Data"
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    motion_magnitude = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
    if motion_magnitude > 10:
        return "Sudden Stop Detected!"
    elif motion_magnitude > 5:
        return "Harsh Braking"
    else:
        return "Normal Motion"
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


def iou(bb_test, bb_gt):
    """Compute IoU between two bounding boxes"""
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) +
              (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o


class KalmanBoxTracker:
    """A single object tracker based on Kalman filtering"""
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)  # 7 state vars (x, y, w, h, vx, vy, vw), 4 measurements
        self.kf.F = np.eye(7)  # State transition matrix
        self.kf.H = np.zeros((4, 7))  # Measurement function
        self.kf.H[:, :4] = np.eye(4)  # Only measure position & size
        self.kf.R[2:, 2:] *= 10  # Adjust measurement uncertainty

        # 🔥 **Fix: Reshape bbox to (4,1) for correct Kalman update**
        self.kf.x[:4] = np.reshape(bbox, (4, 1))

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.time_since_update = 0
        self.history = []

    def update(self, bbox):
        """Update the state with new measurements"""
        self.kf.update(np.reshape(bbox, (4, 1)))  # Ensure correct shape
        self.time_since_update = 0

    def predict(self):
        """Predict the next state"""
        self.kf.predict()
        self.time_since_update += 1
        return self.kf.x[:4].flatten()  # Return only bbox


class Sort:
    """SORT tracker for multiple object tracking"""
    def __init__(self, max_age=5, min_hits=3):
        self.trackers = []
        self.max_age = max_age
        self.min_hits = min_hits

    def update(self, dets):
        """Update tracker with new detections"""
        updated_tracks = []
        for track in self.trackers:
            track.predict()

        matches, unmatched_detections, unmatched_tracks = associate_detections_to_trackers(dets, self.trackers)

        for i, j in matches:
            self.trackers[j].update(dets[i])
            updated_tracks.append([*self.trackers[j].kf.x[:4].flatten(), self.trackers[j].id])  # ✅ Append ID

        for i in unmatched_detections:
            track = KalmanBoxTracker(dets[i])
            self.trackers.append(track)
            updated_tracks.append([*track.kf.x[:4].flatten(), track.id])  # ✅ Append ID

        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        return np.array(updated_tracks, dtype=object)  # ✅ Return array with bbox & track_id




def associate_detections_to_trackers(detections, trackers):
    """Associate new detections with existing trackers using IoU"""
    if len(trackers) == 0:
        return np.empty((0, 2)), np.arange(len(detections)), np.empty((0,))

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk.kf.x[:4].flatten())

    matched_indices = linear_sum_assignment(-iou_matrix)
    matches, unmatched_detections, unmatched_tracks = [], [], []

    for d in range(len(detections)):
        if d not in matched_indices[0]:
            unmatched_detections.append(d)

    for t in range(len(trackers)):
        if t not in matched_indices[1]:
            unmatched_tracks.append(t)

    for m in zip(matched_indices[0], matched_indices[1]):
        if iou_matrix[m[0], m[1]] < 0.3:
            unmatched_detections.append(m[0])
            unmatched_tracks.append(m[1])
        else:
            matches.append(m)

    return np.array(matches), np.array(unmatched_detections), np.array(unmatched_tracks)

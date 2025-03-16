# import numpy as np
# from filterpy.kalman import KalmanFilter
#
# class VehicleTracker:
#     def __init__(self):
#         """Initialize a Kalman Filter for trajectory prediction"""
#         self.kf = KalmanFilter(dim_x=4, dim_z=2)  # 4 state variables (x, y, vx, vy), 2 measurement variables (x, y)
#
#         # State transition matrix (How state changes from one time step to the next)
#         self.kf.F = np.array([[1, 0, 1, 0],
#                               [0, 1, 0, 1],
#                               [0, 0, 1, 0],
#                               [0, 0, 0, 1]])
#
#         # Measurement matrix (How observed data is mapped to state)
#         self.kf.H = np.array([[1, 0, 0, 0],
#                               [0, 1, 0, 0]])
#
#         # Process covariance matrix (Uncertainty in system)
#         self.kf.P *= 10  # Initial covariance
#
#         # Measurement noise (Uncertainty in sensor data)
#         self.kf.R = np.array([[5, 0],
#                               [0, 5]])
#
#         # Process noise (Uncertainty in model)
#         self.kf.Q = np.array([[1, 0, 0, 0],
#                               [0, 1, 0, 0],
#                               [0, 0, 1, 0],
#                               [0, 0, 0, 1]]) * 0.1
#
#     def predict_next_position(self, x, y):
#         """Update filter with current position & predict next position"""
#         self.kf.predict()
#         self.kf.update(np.array([x, y]))
#
#         # Extract predicted position
#         predicted_x, predicted_y = self.kf.x[:2]
#         return int(predicted_x), int(predicted_y)


import numpy as np
from filterpy.kalman import KalmanFilter

class VehicleTracker:
    def __init__(self):
        """Initialize a Kalman Filter for trajectory prediction"""
        self.kf = KalmanFilter(dim_x=4, dim_z=2)  # 4 state vars (x, y, vx, vy), 2 measurements (x, y)
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        self.kf.P *= 10
        self.kf.R = np.array([[5, 0],
                              [0, 5]])
        self.kf.Q = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]]) * 0.1

    def predict_next_position(self, x, y):
        """Update filter with current position & predict next position"""
        self.kf.predict()
        self.kf.update(np.array([x, y]))
        predicted_x, predicted_y = self.kf.x[:2]
        return int(predicted_x), int(predicted_y)
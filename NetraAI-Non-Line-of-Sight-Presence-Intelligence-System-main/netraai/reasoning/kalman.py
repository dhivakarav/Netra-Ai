import numpy as np

class Kalman2D:
    # constant velocity model: state [x,y,vx,vy]
    def __init__(self):
        self.x = None
        self.P = np.eye(4, dtype=np.float32) * 500
        self.Q = np.eye(4, dtype=np.float32) * 0.8
        self.R = np.eye(2, dtype=np.float32) * 12.0

    def predict(self, dt=1.0):
        F = np.array([[1,0,dt,0],
                      [0,1,0,dt],
                      [0,0,1,0],
                      [0,0,0,1]], dtype=np.float32)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        return self.x

    def update(self, meas_xy):
        z = np.array(meas_xy, dtype=np.float32).reshape(2,1)
        if self.x is None:
            self.x = np.array([[z[0,0]],[z[1,0]],[0],[0]], dtype=np.float32)
            return self.x

        H = np.array([[1,0,0,0],
                      [0,1,0,0]], dtype=np.float32)
        y = z - (H @ self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ H) @ self.P
        return self.x

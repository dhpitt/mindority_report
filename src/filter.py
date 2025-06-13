### filter.py
# process detections from hand_rec/detections
# using a bayesian/kalman filter to get smoothed hand positions
# (and eventually) gesture types

from dora import DoraStatus
import pyarrow as pa
import pyarrow.ipc as ipc

from filterpy.kalman import KalmanFilter

class Operator:
    
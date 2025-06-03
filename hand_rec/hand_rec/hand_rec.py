import numpy as np
import torch
import cv2
import sys
import time
from dora import DoraStatus


from torch_mediapipe.blazebase import resize_pad, denormalize_detections
from torch_mediapipe.blazepalm import BlazePalm
from torch_mediapipe.blazehand_landmark import BlazeHandLandmark


CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

class Operator:
    """
    Taking webcam images from dataflow and running hand tracking/
    keypoint detection

    TODO: Add pytorch shmem instead of loading to dev
    """

    def __init__(self):
        self.start_time = time.time()
        self.failure_count = 0

        self.device = "mps" if torch.backends.mps.is_built() else "cpu"

        # load pretrained palm detection model 
        self.palm_detector = BlazePalm().to(self.device)
        self.palm_detector.load_weights("torch_mediapipe/blazepalm.pth")
        self.palm_detector.load_anchors("torch_mediapipe/anchors_palm.npy")
        self.palm_detector.min_score_thresh = .75

        # load pretrained hand keypoint regression
        self.hand_regressor = BlazeHandLandmark().to(self.device)
        self.hand_regressor.load_weights("torch_mediapipe/blazehand_landmark.pth")

    def on_event(
        self,
        dora_event: str,
        send_output,
    ) -> DoraStatus:
        if dora_event["type"] == "INPUT":
            id = dora_event["id"]
            value = dora_event["value"]
            if id == "image":
                image = torch.tensor(value, dtype=torch.uint8).to(self.device)
                image = image.reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 3))
                #print(image.shape)
                #send_output(image.shape)
        return DoraStatus.CONTINUE

    

    
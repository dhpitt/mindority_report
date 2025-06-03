import os
import cv2
import time

from dora import DoraStatus
from utils import LABELS

import pyarrow as pa
import pyarrow.ipc as ipc
import numpy as np

from torch_mediapipe.visualization import (HAND_CONNECTIONS, 
                                           draw_landmarks, 
                                           draw_detections, 
                                           draw_roi)

CI = os.environ.get("CI")

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

FONT = cv2.FONT_HERSHEY_SIMPLEX


class Operator:
    """
    Listen for image and hand detection messages.
    When an image and a detection are both present for a timestamp, plot them. 
    """

    def __init__(self):
        self.detections = {}
        self.images = {}
        self.buffer = ""
        self.submitted = []
        self.lines = []
    
    def plot(self, timestamp):
        """
        When 
        """
        raw_img = self.images[timestamp]
        image = (
            raw_img.to_numpy().reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 3)).copy()
        )

        raw_dets = self.detections[timestamp]
        scalar = raw_dets[0]  # pa.BinaryScalar
        raw_bytes = scalar.as_buffer().to_pybytes()
        buf = pa.py_buffer(raw_bytes)
        reader = ipc.open_stream(buf)
        table = reader.read_all()
        hands = [hand.as_py() for hand in table['hands'][0]]

        for hand in hands:
            landmarks = np.array(hand['landmarks'])
            flag = hand['flag']
            if flag > 0.9:
                draw_landmarks(image, landmarks[:,:2], HAND_CONNECTIONS, size=2)
        '''
        cv2.putText(
            image,
            f"{LABELS[int(label)]}, {confidence:0.2f}",
            (int(max_x), int(max_y)),
            FONT,
            0.5,
            (0, 255, 0),
        )'''

        cv2.putText(
            image, self.buffer, (20, 14 + 21 * 14), FONT, 0.5, (190, 250, 0), 1
        )
        if CI != "true":
            cv2.imshow("frame", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return DoraStatus.STOP
        # clear buffer of the step we plotted
        del self.images[timestamp]
        del self.detections[timestamp]

    def on_event(
        self,
        dora_event,
        send_output,
    ):
        if dora_event["type"] == "INPUT":
            id = dora_event["id"]
            value = dora_event["value"]
            metadata = dora_event["metadata"]
            timestamp = metadata["ts"]
            if id == "image":
                self.images[timestamp] = value
                if self.detections.get(timestamp, False):
                    self.plot(timestamp)

            elif id == "hand_detections":
                self.detections[timestamp] = value
                if self.images.get(timestamp, False):
                    self.plot(timestamp)
                #print(f"event md = {metadata}")
                #self.bboxs = value.to_numpy().reshape((-1, 6))
            elif id == "keyboard_buffer":
                self.buffer = value[0].as_py()
            elif id == "line":
                self.lines += [value.to_pylist()]
            elif "message" in id:
                self.submitted += [
                    {
                        "role": id,
                        "content": value[0].as_py(),
                    }
                ]

        return DoraStatus.CONTINUE

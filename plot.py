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
    Plot image and bounding box
    """

    def __init__(self):
        self.detections = []
        self.buffer = ""
        self.submitted = []
        self.lines = []

    def on_event(
        self,
        dora_event,
        send_output,
    ):
        if dora_event["type"] == "INPUT":
            id = dora_event["id"]
            value = dora_event["value"]
            metadata = dora_event["metadata"]
            if id == "image":

                image = (
                    value.to_numpy().reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 3)).copy()
                )

                for hand in self.detections:
                    landmarks = np.array(hand['landmarks'])
                    flag = hand['flag']
                    if flag > 0.5:
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

                i = 0
                for text in self.submitted[::-1]:
                    color = (
                        (0, 255, 190)
                        if text["role"] == "user_message"
                        else (0, 190, 255)
                    )
                    cv2.putText(
                        image,
                        text["content"],
                        (
                            20,
                            14 + (19 - i) * 14,
                        ),
                        FONT,
                        0.5,
                        color,
                        1,
                    )
                    i += 1

                for line in self.lines:
                    cv2.line(
                        image,
                        (int(line[0]), int(line[1])),
                        (int(line[2]), int(line[3])),
                        (0, 0, 255),
                        2,
                    )

                if CI != "true":
                    cv2.imshow("frame", image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return DoraStatus.STOP
            elif id == "hand_detections":
                scalar = value[0]  # pa.BinaryScalar
                raw_bytes = scalar.as_buffer().to_pybytes()
                buf = pa.py_buffer(raw_bytes)
                reader = ipc.open_stream(buf)
                table = reader.read_all()
                self.detections = [hand.as_py() for hand in table['hands'][0]]
                print(f"event md = {metadata}")
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

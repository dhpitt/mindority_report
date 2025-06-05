import os
import cv2
import time

from dora import DoraStatus
from utils import LABELS

import pyarrow as pa
import pyarrow.ipc as ipc
import numpy as np

import asyncio
import base64
import websockets

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
        self.submitted = []

        # Set up websocket for async comms w/frontend
        self.websocket = None
        self.loop = asyncio.get_event_loop()
        self.ws_task = self.loop.create_task(self._setup_ws())
    
    async def _setup_ws(self):
        try:
            self.websocket = await websockets.connect("ws://localhost:8765")
        except Exception as e:
            print(f"WebSocket error: {e}")
            self.websocket = None
    

    async def _send_frame(self, frame):
        if self.websocket is None or self.websocket.closed:
            await self._setup_ws()
        _, buffer = cv2.imencode(".jpg", frame)
        encoded = base64.b64encode(buffer).decode("utf-8")
        await self.websocket.send(encoded)
    
    def _draw_img(self, timestamp):
        """
        When both the ROIs and image arrive from ts x, 
        grab both from the buffer, delete, and plot them on 
        top of one another. 
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


        '''
        # Legacy plotting via cv2.imshow
        if CI != "true":
        cv2.imshow("frame", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return DoraStatus.STOP
        '''

        # clear buffer of the step we plotted
        del self.images[timestamp]
        del self.detections[timestamp]
        return image

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
            annotated_img = None
            if id == "image":
                self.images[timestamp] = value
                if self.detections.get(timestamp, False):
                    annotated_img = self._draw_img(timestamp)

            elif id == "hand_detections":
                self.detections[timestamp] = value
                if self.images.get(timestamp, False):
                    annotated_img = self._draw_img(timestamp)
            
            if annotated_img is not None:
                self.loop.create_task(self.send_frame(annotated_img))
                

        return DoraStatus.CONTINUE

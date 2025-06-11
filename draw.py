import os
import cv2
import time

from dora import DoraStatus
from utils import LABELS

import pyarrow as pa
import pyarrow.ipc as ipc
import numpy as np

import asyncio
import threading
import base64
import websocket

from torch_mediapipe.visualization import (HAND_CONNECTIONS, 
                                           draw_landmarks, 
                                           draw_detections, 
                                           draw_roi)

CI = os.environ.get("CI")
# for now, this is always true. 
# let's find a better way of switching between opencv and webui

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

FONT = cv2.FONT_HERSHEY_SIMPLEX

class WebSocketStreamer:
    def __init__(self):
        self.ws = websocket.WebSocket()
        self.thread = threading.Thread(target=self._connect_loop, daemon=True)
        self.queue = []
        self.lock = threading.Lock()
        self.thread.start()

    def _connect_loop(self):
        while True:
            #try:
            self.ws.connect("ws://localhost:8765")
            while True:
                if self.queue:
                    with self.lock:
                        frame = self.queue.pop(0)
                    _, buffer = cv2.imencode(".jpg", frame)
                    encoded = base64.b64encode(buffer).decode("utf-8")
                    self.ws.send(encoded)
                    print("Sending image", flush=True)
                else:
                    time.sleep(0.01)
            '''except Exception as e:
                print("WebSocket error:", e)
                time.sleep(2)  # retry'''

    def send(self, frame):
        with self.lock:
            self.queue.append(frame)


from enum import Enum

class CaptureMode(Enum):
    DEFAULT = 0
    DATASET = 1

class CollectMode(Enum):
    ROIS = 0
    LANDMARKS = 1


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
        #self.socket = WebSocketStreamer()
        
        self.mode = CaptureMode.DEFAULT
    
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
            handed = hand['handed']
            bbox = hand['box']
            #print(bbox)
            '''if handed > 0.5:
                hand_color = (255,0,0)
            else:
                hand_color = (0,255,0)'''
            flag = hand['flag']

            if self.mode == CaptureMode.DATASET:
                hand_color = (255,0,0)
            elif self.mode == CaptureMode.DEFAULT:
                hand_color = (0,255,0)
            if flag > 0.9:
                draw_roi(image, roi=bbox)
                draw_landmarks(image, points=landmarks[:,:2], 
                               connections=HAND_CONNECTIONS, size=2,line_color=hand_color)
        '''
        cv2.putText(
            image,
            f"{LABELS[int(label)]}, {confidence:0.2f}",
            (int(max_x), int(max_y)),
            FONT,
            0.5,
            (0, 255, 0),
        )
        '''
        
        # clear buffer of the step we plotted
        del self.images[timestamp]
        del self.detections[timestamp]
        # return mirrored image
        return image[:, ::-1]

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
                #cv2.imwrite("data/latest.jpg", annotated_img)
                
                # Legacy plotting via cv2.imshow
                if CI != "true":
                    cv2.imshow("frame", annotated_img)
                    k = cv2.waitKey(1)
                    if k==27:    # Esc key to stop
                        return DoraStatus.STOP
                    elif k != -1:   # normally -1 returned,so don't print it
                        if k == ord('a'):
                            self.mode = CaptureMode.DATASET
                        elif k == ord('b'):
                            self.mode = CaptureMode.DEFAULT
                else:
                    self.socket.send(annotated_img)

                

        return DoraStatus.CONTINUE

import os
from pathlib import Path
import cv2
import time
from enum import Enum

from dora import DoraStatus
import pyarrow as pa
import pyarrow.ipc as ipc
import numpy as np

import gradio 

from torch_mediapipe.visualization import (HAND_CONNECTIONS, 
                                           draw_landmarks, 
                                           draw_detections, 
                                           draw_roi,
                                           crop_from_corners)

# Enum classes for dataset collection and capture modes

class CaptureMode(Enum):
    DEFAULT = 0
    DATASET = 1

class CollectMode(Enum):
    ROIS = 0
    LANDMARKS = 1

class HandedMode(Enum):
    LEFT = 0
    RIGHT = 1

class Gesture(Enum):
    NONE = 0
    OPEN = 1
    FIST = 2
    POINTER = 3
    FINGER_GUN = 4

HANDED_LABELS = {
    HandedMode.RIGHT: "RIGHT",
    HandedMode.LEFT: "LEFT"
}

CAPTURE_MODE_LABELS = {
    CaptureMode.DEFAULT: "default",
    CaptureMode.DATASET: "recording dataset"
}

DATA_COLLECTION_MODE_LABELS = {
    CollectMode.ROIS: "ROIS",
    CollectMode.LANDMARKS: "HAND LANDMARKS",
    #CollectMode.ALL: "ROIS + LANDMARKS"
}

GESTURE_LABELS = {
    Gesture.NONE: "NONE",
    Gesture.OPEN: "Open",
    Gesture.FIST: "Fist",
    Gesture.POINTER: "Pointer",
    Gesture.FINGER_GUN: "Fingergun"
}

CI = os.environ.get("CI")
# for now, this is always true. 
# let's find a better way of switching between opencv and webui

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
        #self.socket = WebSocketStreamer()
        
        self.capture_mode = CaptureMode.DEFAULT
        self.handed_mode = HandedMode.RIGHT
        self.collection_mode = CollectMode.LANDMARKS
        self.gesture_mode = 0
    
    def _process_img(self, timestamp):
        """
        When both the ROIs and image arrive from ts x, 
        grab both from the buffer, delete, and plot them on 
        top of one another. 

        Optionally, if in processing mode, save image too 
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

            # Color the hands according to the label we're assigning in the dataset
            if self.collection_mode == CollectMode.ROIS:
                if self.handed_mode == HandedMode.RIGHT:
                    hand_color = (0,255,0)
                elif self.handed_mode == HandedMode.LEFT:
                    hand_color = (0,0,255)
            else:
                # otherwise, label hands according to the handedness flag
                if handed:
                    # left
                    hand_color = (0,0,255)
                else:
                    # right
                    hand_color = (0,255,0)

            if flag > 0.9:
                # save data when model is confident
                if self.capture_mode == CaptureMode.DATASET:

                    # in ROI mode, save an unannotated image of the entire ROI
                    if self.collection_mode == CollectMode.ROIS:
                        handedness_root = f"./data/handedness/{HANDED_LABELS[self.handed_mode]}/"
                        n_imgs = len(os.listdir(handedness_root))
                        roi_img = crop_from_corners(image, bbox)
                        cv2.imwrite(f"{handedness_root}/{n_imgs+1}.jpg",
                                    roi_img)
                    

                    # in landmark mode, save the landmark points as a 21x3 tensor
                    elif self.collection_mode == CollectMode.LANDMARKS:
                        gesture_root = Path(
                            f"./data/gestures/{HANDED_LABELS[self.handed_mode]}{Gesture(self.gesture_mode).name}/"
                        )
                        if not gesture_root.exists():
                            gesture_root.mkdir(parents=True)
                        n_files = len([x for x in gesture_root.iterdir()])
                        np.save(gesture_root / f"{n_files+1}.npy", landmarks)
                    
                
                draw_roi(image, roi=bbox, handed_color=hand_color, box_color=(0,0,0))
                draw_landmarks(image, points=landmarks[:,:2], 
                               connections=HAND_CONNECTIONS, size=2,line_color=hand_color)
                #if CollectMode.LANDMARKS:
                
                
        
        '''cv2.putText(
            image,
            f"{handed_labels[]}, {confidence:0.2f}",
            (int(max_x), int(max_y)),
            FONT,
            0.5,
            (0, 255, 0),
        )'''
        
        
        # clear buffer of the step we plotted
        del self.images[timestamp]
        del self.detections[timestamp]
        # return mirrored image
        return np.ascontiguousarray(image[:, ::-1, :])

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
                    annotated_img = self._process_img(timestamp)

            elif id == "hand_detections":
                self.detections[timestamp] = value
                if self.images.get(timestamp, False):
                    annotated_img = self._process_img(timestamp)
            
            # If there's no new image to show at this timestep, show 
            if annotated_img is not None:
                # Legacy plotting via cv2.imshow
                if CI != "true":
                    # show capture mode
                    capture_mode_text = CAPTURE_MODE_LABELS[self.capture_mode]
                    cv2.putText(annotated_img, 
                                text=f"Mode: {capture_mode_text} (d)",
                                org=(50,50),fontFace=FONT,
                                fontScale=0.5,
                                color=(255, 0, 0))
                    
                    # show data collection mode
                    collect_mode_text = DATA_COLLECTION_MODE_LABELS[self.collection_mode]
                    cv2.putText(annotated_img, 
                                text=f"Saving {collect_mode_text} (c)",
                                org=(50,80),fontFace=FONT,
                                fontScale=0.5,
                                color=(255, 0, 0))
                    
                    
                    # show handedness mode if collecting hand ROIs
                    handed_mode_text = HANDED_LABELS[self.handed_mode]
                    if self.collection_mode == CollectMode.ROIS:
                        cv2.putText(annotated_img, 
                                    text=f"Recording {handed_mode_text} hand (h)",
                                    org=(50,110),fontFace=FONT,
                                    fontScale=0.5,
                                    color=(255, 0, 0))
                    elif self.collection_mode == CollectMode.LANDMARKS:
                        gesture_mode_text = Gesture(self.gesture_mode).name
                        cv2.putText(annotated_img, 
                                    text=f"Recording gesture {handed_mode_text}->{gesture_mode_text} (h,g)",
                                    org=(50,110),fontFace=FONT,
                                    fontScale=0.5,
                                    color=(255, 0, 0))
                    
                    cv2.imshow("frame", annotated_img)
                    k = cv2.waitKey(1)
                    if k==27:    # Esc key to stop
                        return DoraStatus.STOP
                    elif k != -1:   # normally -1 returned,so don't print it
                        # change capture mode
                        if k == ord('d'):
                            if self.capture_mode == CaptureMode.DEFAULT:
                                self.capture_mode = CaptureMode.DATASET
                            elif self.capture_mode == CaptureMode.DATASET:
                                self.capture_mode = CaptureMode.DEFAULT
                        # change handedness mode
                        elif k == ord('h'):
                            if self.handed_mode == HandedMode.LEFT:
                                self.handed_mode = HandedMode.RIGHT
                            elif self.handed_mode == HandedMode.RIGHT:
                                self.handed_mode = HandedMode.LEFT

                        elif k == ord('c'):
                            if self.collection_mode == CollectMode.ROIS:
                                self.collection_mode = CollectMode.LANDMARKS
                            elif self.collection_mode == CollectMode.LANDMARKS:
                                self.collection_mode = CollectMode.ROIS
                        
                        elif k == ord('g'):
                            self.gesture_mode = (self.gesture_mode + 1) % len(Gesture)
                else:
                    self.socket.send(annotated_img)

            '''else:
                cv2.imshow("frame", value) '''   

        return DoraStatus.CONTINUE

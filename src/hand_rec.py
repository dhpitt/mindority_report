import time
from dora import DoraStatus
import pyarrow as pa
import pyarrow.ipc as ipc
import numpy as np

import torch
import torchvision.transforms.functional as tvtf
from torchvision.transforms import InterpolationMode 

from torch_mediapipe.blazebase import resize_pad, denormalize_detections
from torch_mediapipe.blazepalm import BlazePalm
from torch_mediapipe.blazehand_landmark import BlazeHandLandmark

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# pyarrow struct to send hand bbox

hand_schema = pa.struct([
    ("box", pa.list_(pa.list_(pa.float32(), 4), 2)),  # shape (2,4)
    ("flag", pa.float32()),                           # confidence
    ("handed", pa.int8()),                            # 0 or 1
    ("landmarks", pa.list_(pa.list_(pa.float32(), 3), 21)),  # shape (21, 3)
])

schema = pa.schema([
    ("hands", pa.list_(hand_schema))  # variable-length list of hands per message
])

def tensor_to_list(t):
    return t.detach().cpu().numpy().tolist()

def make_hand_dict(box, flag, handed, landmarks):
    return {
        "box": tensor_to_list(box),               # shape (2,4)
        "flag": float(flag),                      # single float
        "handed": int(handed),                    # single float (needs sigmoid)
        "landmarks": tensor_to_list(landmarks)  # shape (21,3)
    }


class Operator:
    """
    Taking webcam images from dataflow and running hand tracking/
    keypoint detection

    TODO: Add pytorch shmem instead of loading to device
    """

    def __init__(self):
        self.start_time = time.time()
        self.failure_count = 0

        self.device = "mps" if torch.backends.mps.is_built() else "cpu"

        # load pretrained palm detection model 
        self.palm_detector = BlazePalm().to(self.device)
        self.palm_detector.load_weights("src/torch_mediapipe/blazepalm.pth")
        self.palm_detector.load_anchors("src/torch_mediapipe/anchors_palm.npy")
        self.palm_detector.min_score_thresh = .75

        # load pretrained hand keypoint regression
        self.hand_regressor = BlazeHandLandmark().to(self.device)
        self.hand_regressor.load_weights("src/torch_mediapipe/blazehand_landmark.pth")
        self.hand_regressor.handed.load_state_dict(
            torch.load("src/torch_mediapipe/blazehand_handedness_cls_2.pth")
            )
        #self.hand_regressor.load_weights("torch_mediapipe/blazehand_landmark_trained_handedness.pth")

    def on_event(
        self,
        dora_event: str,
        send_output,
    ) -> DoraStatus:
        event_type = dora_event["type"]
        if event_type == "INPUT":
            id = dora_event["id"]
            value = dora_event["value"]
            if id == "image":
                image = value.to_numpy().reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 3)).copy()
                ## reverse color channels
                # opencv defaults to BGR, whereas mediapipe expects RGB
                image = np.ascontiguousarray(image[:,:,::-1])

                # preprocess image
                #image, scale, pad = resize_pad_tensor(image)
                det_img, _, scale, pad = resize_pad(image)

                ## detect palm and denormalize det
                with torch.no_grad():
                    normalized_palm_detections = self.palm_detector.predict_on_image(det_img)

                    palm_detections = denormalize_detections(normalized_palm_detections, scale, pad)

                ## extract additional keypoints
                # dtypes:
                # xc, yc, scale, theta: floats
                # affine: UNK
                # box: torch.tensor, shape (2,2)
                # flags: torch.tensor of list of confidences for each detected hand
                # handed: torch.tensor, same length as flags, detect handedness (this model sucks)
                # normalized_landmarks: torch.tensor of shape (n_hands, 21, 3) of keypoints
                xc, yc, scale, theta = self.palm_detector.detection2roi(palm_detections.cpu())
                img, affine, box = self.hand_regressor.extract_roi(image, xc, yc, theta, scale)
                with torch.no_grad():
                    flags, handed, normalized_landmarks = self.hand_regressor(img.to(self.device))
                    handed = torch.nn.functional.sigmoid(handed) > 0.5
                # xc, yc, scale, theta: floats

                # denormalize landmarks to plot using affine transform
                landmarks = self.hand_regressor.denormalize_landmarks(normalized_landmarks.cpu(), affine)
            
                # pack each detection (seen in `flags`) into my custom arrow struct
                if len(flags) > 0:
                    hands_data = []
                    for i in range(len(flags)):
                        hands_data.append(
                            make_hand_dict(box[i], flags[i], handed[i], landmarks[i])
                        )
                    # Only send a message if there's at least one confident hand
                    #if len(hands_data) > 0:
                    batch = pa.Table.from_pydict({"hands": [hands_data]}, schema=schema)

                    # serialize batch, since it isn't an arrow array
                    sink = pa.BufferOutputStream()
                    with ipc.new_stream(sink, schema) as writer:
                        writer.write_table(batch)
                    msg_bytes = sink.getvalue().to_pybytes()

                    # place the bytes into a pa array so they're all sent at once
                    msg_array = pa.array([msg_bytes], type=pa.binary())

                    send_output(
                        "detections",
                        msg_array,
                        dora_event["metadata"],
                        )
            
            elif event_type == "STOP":
                print("received stop")
            else:
                print("received unexpected event:", event_type)
        return DoraStatus.CONTINUE


    

    
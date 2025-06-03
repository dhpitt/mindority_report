import time
from dora import DoraStatus
import pyarrow as pa
import pyarrow.ipc as ipc

import torch
import torchvision.transforms.functional as tvtf
from torchvision.transforms import InterpolationMode 

from torch_mediapipe.blazebase import resize_pad, denormalize_detections
from torch_mediapipe.blazepalm import BlazePalm
from torch_mediapipe.blazehand_landmark import BlazeHandLandmark


CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

## helper functions for image processing

def resize_pad_tensor(img: torch.Tensor, output_res=(256,256)):
    c, h_0,w_0, = img.shape
    goal_h, goal_w = output_res

    if h_0>=w_0:
        h1 = goal_h
        w1 = goal_w * w_0 // h_0
        padh = 0
        padw = goal_w - w1
        scale = w_0 / w1
    else:
        h1 = goal_h * h_0 // w_0
        w1 = goal_w
        padh = goal_h - h1
        padw = 0
        scale = h_0 / h1
    
    img = tvtf.resize(img, (h1,w1),
                      interpolation=InterpolationMode.NEAREST)
    padh1 = padh//2
    padh2 = padh//2 + padh%2
    padw1 = padw//2
    padw2 = padw//2 + padw%2
    # pad Left, Top, Right, Bottom acc to tvt.F
    img = tvtf.pad(img, (padw1,padh1,padw2,padh2))

    pad = (int(padh1 * scale), int(padw1 * scale))

    
    return img, scale, pad

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
        "handed": int(handed),                    # single int
        "landmarks": tensor_to_list(landmarks)  # shape (21,3)
    }


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
        event_type = dora_event["type"]
        if event_type == "INPUT":
            id = dora_event["id"]
            value = dora_event["value"]
            if id == "image":
                image = value.to_numpy().reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 3)).copy()

                # preprocess image
                #image, scale, pad = resize_pad_tensor(image)
                det_img, _, scale, pad = resize_pad(image)

                ## detect palm and denormalize det
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
                flags, handed, normalized_landmarks = self.hand_regressor(img.to(self.device))

                # xc, yc, scale, theta: floats

                # denormalize landmarks to plot using affine transform
                landmarks = self.hand_regressor.denormalize_landmarks(normalized_landmarks.cpu(), affine)
            
                # Serialize to Arrow table if len(flags) > 0
                if len(flags) > 0:
                    hands_data = []
                    for i, flag in enumerate(flags):
                        #if flag > 0.5:
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

                    msg_array = pa.array([msg_bytes], type=pa.binary())

                    send_output(
                        "detections",
                        msg_array,
                        dora_event["metadata"],
                        )
                '''send_output(
                "image",
                pa.array(image.ravel()),
                dora_event["metadata"],
                )'''
            elif event_type == "STOP":
                print("received stop")
            else:
                print("received unexpected event:", event_type)
        return DoraStatus.CONTINUE


    

    
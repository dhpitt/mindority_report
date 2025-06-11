import torch
from torch.optim import AdamW
import torchvision.transforms.functional as tvtf
from torchvision.transforms import InterpolationMode 

from torch_mediapipe.blazebase import resize_pad, denormalize_detections
from torch_mediapipe.blazepalm import BlazePalm
from torch_mediapipe.blazehand_landmark import BlazeHandLandmark

device = "mps"

# load pretrained hand keypoint regression
hand_regressor = BlazeHandLandmark().to(device)
hand_regressor.load_weights("torch_mediapipe/blazehand_landmark.pth")

N_EPOCHS = 100
N_
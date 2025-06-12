import torch
from torch.optim import AdamW
import torch.utils.data.dataset
import torchvision.transforms.functional as tvtf
from torchvision.transforms import InterpolationMode 
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import binary_cross_entropy
from torch.nn import BCEWithLogitsLoss
from PIL import Image
from pathlib import Path
import random

from torch_mediapipe.blazebase import resize_pad, denormalize_detections
from torch_mediapipe.blazepalm import BlazePalm
from torch_mediapipe.blazehand_landmark import BlazeHandLandmark

# create dataset class and load data

'''import os
class BinaryClassificationDataset(Dataset):
    def __init__(self, root):
        classes = {
            0: f"{root}/RIGHT",
            1: f"{root}/LEFT",
        }
        self.data_lists = {
            k: [v + f"/{x}" for x in os.listdir(v)] for k,v in classes.items()
        }
        self.lengths = {
            k: len(v) for k,v in self.data_lists.items()
        }
    
    def __len__(self):
        return sum(self.lengths.values())

    def __getitem__(self, index):
        if index < self.lengths[0]:
            data, label = Image.open(self.data_lists[0][index]), 0
        elif index < self.lengths[0] + self.lengths[1]:
            data, label = Image.open(self.data_lists[1][index - self.lengths[0]]), 1
        else:
            raise IndexError
        
        return tvtf.resize(tvtf.to_tensor(data), size=(256,256), interpolation=InterpolationMode.NEAREST), label

    def train_test_split(self, split=0.8):
        return 
train_dataset = BinaryClassificationDataset(root="./data/handedness")
'''

root = Path("./data/handedness")
all_imgs = root.glob("**/*.jpg")
data_list = [(x,0) if "RIGHT" in str(x) else (x,1) for x in all_imgs]

class SimpleBinaryDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_pth, label = self.data_list[index]
        img = Image.open(img_pth)
        return tvtf.resize(tvtf.to_tensor(img), (256,256)), label

random.shuffle(data_list)
train_stop = int(0.8*len(data_list))

train_dataset = SimpleBinaryDataset(data_list[:train_stop])
test_dataset = SimpleBinaryDataset(data_list[train_stop:])

# Prepare device and model
device = "mps"

# load pretrained hand keypoint regression
hand_regressor = BlazeHandLandmark()
hand_regressor.load_weights("torch_mediapipe/blazehand_landmark.pth")

# freeze the hand_regressor's landmark and conf regressor
for p in hand_regressor.parameters():
    p.requires_grad_(False)
#hand_regressor.hand_flag.requires_grad_(False)
#hand_regressor.landmarks.requires_grad_(False)
hand_regressor.handed.requires_grad_(True)

hand_regressor = hand_regressor.to(device)

N_EPOCHS = 20
BATCH_SIZE = 32
LR = 3e-4
DECAY = 1e-6

optimizer = torch.optim.AdamW(hand_regressor.parameters(), lr=LR,weight_decay=DECAY)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
loss_fn = BCEWithLogitsLoss()

for ep in range(N_EPOCHS):
    ep_loss = 0.

    hand_regressor.train()

    for idx, (x,y) in enumerate(train_loader):
        print(f"Progress: {(idx+1)/len(train_loader):.2%}", end='\r')
        x = x.to(device)
        y = y.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        _, y_pred, _ = hand_regressor(x)
        train_loss = loss_fn(y_pred, y)
        ep_loss += (train_loss / y.shape[0])

        train_loss.backward()
        optimizer.step()
    ep_loss /= len(train_loader)
    
    # Eval loop
    hand_regressor.eval()
    total_right = 0.
    total = 0.
    avg_val_loss = 0.
    for idx, (x,y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            _, y_pred, _ = hand_regressor(x)
        val_loss = loss_fn(y_pred, y.to(dtype=torch.float32))
        avg_val_loss += (val_loss / y.shape[0])
        # Compute accuracy
        probs = torch.nn.functional.sigmoid(y_pred)
        preds = (probs > 0.5)
        #print(f"probs:{probs:.2f}")
        #print(f"preds:{preds:.2f}")
        
        total_right += (preds == y).sum()
        total += y.shape[0]
    
    avg_val_loss /= len(test_loader)
    acc = total_right / total
        
        
        
    print(f"\n[Epoch {ep}], avg loss {ep_loss:.2f} | val loss {avg_val_loss:.2f}, acc {acc:.2%}")

torch.save(hand_regressor.handed.state_dict(), 
           "torch_mediapipe/blazehand_handedness_cls_2.pth")






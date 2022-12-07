import os
import torch
import warnings
import cv2
from skimage import io
from PIL import Image
import torchvision
from torch import nn
from torch import optim
from torchvision import transforms
from torchvision import models as M
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import imgaug
import imgaug as ia
from imgaug import augmenters as iaa

import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

torch.manual_seed(1412)
random.seed(1412)
np.random.seed(1412)

#------------------------------------------------------------------------------------------------------------------
## read in data

n_epoch = 50
BATCH_SIZE = 30
LR = 0.001
DROPOUT = 0.5

## Image processing
CHANNELS = 3
IMAGE_SIZE = 256

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
SAVE_MODEL = True



#------------------------------------------------------------------------------------------------------------------

class CustomDataset(Dataset):
    def __init__(self,df,transform=None):
        super().__init__()
        self.path_label = df
        self.transform=transform

    def __len__(self):
        return self.path_label.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patient_id = self.path_label["patient_id"].values[idx]
        image_path = self.path_label["path"].values[idx]
        image = io.imread(image_path)
        label = self.path_label["label"].values[idx]

        if self.transform:
            # image = Image.fromarray(io.imread(image_path))
            image = self.transform(image)

        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))


        X = torch.FloatTensor(image)

        X = torch.reshape(X, (3, IMAGE_SIZE, IMAGE_SIZE))

        y = torch.FloatTensor(label)


        sample = {"patch": X,
                  "label": y,
                  "patient": patient_id}

        return sample

#------------------------------------------------------------------------------------------------------------------

## data augmentation
# color diversity, color normalization

def alltransform(key="train"):

    seq1 = iaa.Sequential([
        iaa.Resize(256),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.CropAndPad(percent=(0.01, 0.02)),
        iaa.MultiplyAndAddToBrightness(mul=(0.7, 1.2), add=(-10, 10)),
        iaa.MultiplyHueAndSaturation(mul_hue=(0.9, 1.1), mul_saturation=(0.8, 1.2)),
        iaa.pillike.EnhanceContrast(factor=(0.75, 1.25)),
        iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(loc=1, scale=(0, 0.05 * 255), per_channel=0.5)),  # probability
        iaa.Add((-20, 5)),
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        iaa.Affine(scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                   translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                   rotate=(-10, 10),
                   shear=(-3, 3))
    ], random_order=True)

    train_sequence = [seq1.augment_image, transforms.ToPILImage()]
    test_val_sequence = [iaa.Resize(256).augment_image, transforms.ToPILImage()]

    train_sequence.extend([transforms.ToTensor()
                              , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_val_sequence.extend([transforms.ToTensor()
                                 , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_transforms = {'train': transforms.Compose(train_sequence), 'test_val': transforms.Compose(test_val_sequence)}

    return data_transforms[key]

#------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------
PATH = os.getcwd()
os.chdir("..")
Excel_PATH = os.getcwd()
for file in os.listdir(Excel_PATH):
    if file[-5:] == '.xlsx':
        FILE_NAME = os.getcwd() + os.path.sep + 'excel.xlsx'
data = pd.read_excel(FILE_NAME)

# split the train and test file (train 70%, test 15%, val 15%)
train, test_val = train_test_split(data, test_size=0.3, random_state=1412
                                   , stratify=data["label"])
test, val = train_test_split(test_val, test_size=0.5, random_state=1412
                             , stratify=test_val["label"])

df_train = CustomDataset(train,transform = alltransform(key="train"))
df_test = CustomDataset(test,transform = alltransform(key="train"))


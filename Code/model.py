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


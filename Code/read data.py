import os
import re
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

import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns

import random
import numpy as np
import pandas as pd
import datetime
from time import time
import gc

from sklearn.model_selection import train_test_split

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

torch.manual_seed(1412)
random.seed(1412)
np.random.seed(1412)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# read in data
ORI_PATH = os.getcwd()
os.chdir("..")
PATH = os.getcwd() + os.path.sep + 'data'
patients = os.listdir(PATH)
print(len(patients))


positive_patches = 0
negative_patches = 0
for patient_id in patients:
    class0_path = os.path.join(PATH, patient_id, str(0))
    class0_patches = os.listdir(class0_path)
    negative_patches += len(class0_patches)

    class1_path = os.path.join(PATH, patient_id, str(1))
    class1_patches = os.listdir(class1_path)
    positive_patches += len(class1_patches)

total_patches = positive_patches + negative_patches

# create a df for image path
data = pd.DataFrame(index=np.arange(0, total_patches)
                    , columns=["patient_id", "path", "label"])

idx = 0
for patient_id in patients:
    for label in [0, 1]:
        class_path = os.path.join(PATH, patient_id, str(label))
        class_patches = os.listdir(class_path)
        for patch in class_patches:
            data.loc[idx, "path"] = os.path.join(class_path, patch)
            data.loc[idx, "label"] = label
            data.loc[idx, "patient_id"] = patient_id
            idx += 1

data["x"] = data["path"].apply(lambda x: int(x.split("_")[-3][1:]))
data["y"] = data["path"].apply(lambda x: int(x.split("_")[-2][1:]))

print(data.shape)
print(data.head())

# data.to_excel("imagepath.xlsx")

# split the train and test file (train 70%, test 15%, val 15%)
train, test_val = train_test_split(data, test_size=0.3, random_state=1412
                                   , stratify=data["label"])
test, val = train_test_split(test_val, test_size=0.5, random_state=1412
                             , stratify=test_val["label"])
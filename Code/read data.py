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


#------------------------------------------------------------------------------------------------------------------

## read in data
ORI_PATH = os.getcwd()
os.chdir("..")
PATH = os.getcwd() + os.path.sep + 'data'
patients = os.listdir(PATH)
#print(len(patients))

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#------------------------------------------------------------------------------------------------------------------

## dataset explore
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


#------------------------------------------------------------------------------------------------------------------

def image_path(patients):
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

    # print(data.shape)
    # print(data.head())

    data.to_excel("excel.xlsx",index = False)
    print('Image path excel is generated!')

    return data

#------------------------------------------------------------------------------------------------------------------

def image_plot(df):
    fig, axs = plt.subplots(5, 10, figsize=(20, 10))

    for n in range(5):
        for m in range(10):
            idx = df[m + 10 * n]
            image = cv2.imread(data.loc[idx, "path"])
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            axs[n, m].imshow(image)
            axs[n, m].grid(False)
            axs[n, m].axis("off")
    plt.show()

#------------------------------------------------------------------------------------------------------------------

def visualize_single_patient(patient_id):
    singlepatient = data[data["patient_id"] == patient_id]
    sns.set()
    plt.scatter(singlepatient["x"],singlepatient["y"],c=singlepatient["label"],cmap="coolwarm",s=10);
    plt.grid(linewidth=1,color="white")

#------------------------------------------------------------------------------------------------------------------
## visulize data

# data = image_path(patients)

Path = os.getcwd()
for file in os.listdir(Path):
    if file[-5:] == '.xlsx':
        FILE_NAME = os.getcwd() + os.path.sep + 'excel.xlsx'
data = pd.read_excel(FILE_NAME)

# vis positive and negative graphs
pos_selection = np.random.choice(data[data.label==1].index.values, size=50, replace=False)
neg_selection = np.random.choice(data[data.label==0].index.values, size=50, replace=False)
#
# image_plot(pos_selection)
# image_plot(neg_selection)
#------------------------------------------------------------------------------------------------------------------

# dive into detail
cancer_perc = data.groupby("patient_id").label.value_counts()/ data.groupby("patient_id").label.size()
cancer_perc = cancer_perc.unstack()

fig,ax = plt.subplots(1,3,figsize=(20,5))
sns.distplot(data.groupby("patient_id").size(), ax=ax[0], color="Orange", kde=False, bins=30)
ax[0].set_xlabel("Number of patches")
ax[0].set_ylabel("Frequency")
ax[0].set_title("How many patches do we have per patient?")
sns.distplot(cancer_perc.loc[:, 1]*100, ax=ax[1], color="Tomato", kde=False, bins=30)
ax[1].set_title("How much percentage of an image is covered by IDC?")
ax[1].set_ylabel("Frequency")
ax[1].set_xlabel("% of patches with IDC")
colors = sns.color_palette('pastel')[0:7]
sns.countplot(data.label, palette="Set2", ax=ax[2])
ax[2].set_xlabel("no(0) versus yes(1)")
ax[2].set_title("How many patches show IDC?")
plt.show()

# label pie plot

colors = sns.color_palette('pastel')[0:5]
plt.figure()
plt.pie(data['label'].value_counts(),labels=data['label'].unique(),colors=colors, autopct='%1.1f%%')
plt.title("How many patches show IDC?")
plt.show()
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
fig, axs = plt.subplots(5,3,figsize=(20, 27))
patient_ids = data["patient_id"].unique()
np.random.shuffle(patient_ids)

for n in range(5):
    for m in range(3):
        patient_id = patient_ids[m+3*n]
        singlepatient = data[data["patient_id"] == patient_id]
        axs[n,m].scatter(singlepatient["x"],singlepatient["y"],c=singlepatient["label"],cmap="coolwarm",s=20)
        axs[n,m].set_title("patient " + str(patient_id) + " " + str(round(cancer_perc.loc[patient_id,"1"],3)))

plt.show()

#------------------------------------------------------------------------------------------------------------------






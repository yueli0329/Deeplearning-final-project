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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils import compute_class_weight
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

torch.manual_seed(1412)
random.seed(1412)
np.random.seed(1412)
THRESHOLD = 0.5
#------------------------------------------------------------------------------------------------------------------
## read in data

n_epoch = 50
BATCH_SIZE = 30
LR = 0.001
DROPOUT = 0.5

## N_NEURONS for MLP
N_NEURONS = 4

## Image processing
CHANNELS = 3
IMAGE_SIZE = 100

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
SAVE_MODEL = True

#.............................................................................
def my_transform(key="train", plot=False):
    train_sequence = [transforms.Resize((50, 50)),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip()]
    val_sequence = [transforms.Resize((50, 50))]
    if plot == False:
        train_sequence.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        val_sequence.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_transforms = {'train': transforms.Compose(train_sequence), 'val': transforms.Compose(val_sequence)}
    return data_transforms[key]
#------------------------------------------------------------------------------------------------------------------
class BreastCancerDataset(Dataset):

    def __init__(self, df, transform=None):
        self.states = df
        self.transform = transform

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        patient_id = self.states.patient_id.values[idx]
        x_coord = self.states.x.values[idx]
        y_coord = self.states.y.values[idx]
        image_path = self.states.path.values[idx]
        image = Image.open(image_path)
        image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        if "target" in self.states.columns.values:
            target = np.int(self.states.target.values[idx])
        else:
            target = None

        return {"image": image,
                "label": target,
                "patient_id": patient_id,
                "x": x_coord,
                "y": y_coord}
#..................................................................................................................

#...................................................................................................................

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

        X = torch.FloatTensor(image)

        X = torch.reshape(X, (3, IMAGE_SIZE, IMAGE_SIZE))

        y = torch.FloatTensor(label)


        sample = {"patch": image,
                  "label": label,
                  "patient": patient_id}

        return sample

#------------------------------------------------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (3, 3))
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(16, 32, (3, 3))
        self.convnorm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d((2, 2))

        self.linear1 = nn.Linear(32*5*5, 400)
        self.linear1_bn = nn.BatchNorm1d(400)
        self.drop = nn.Dropout(DROPOUT)

        self.linear2 = nn.Linear(400, 1)
        self.act = torch.sigmoid()

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        x = self.drop(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
        return self.act(self.linear2(x))

#------------------------------------------------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(num_features, 512)
        self.act1 = torch.relu
        self.convnorm1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)

        self.linear2 = nn.Linear(512, 256)
        self.act2 = torch.relu
        self.convnorm2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)

        self.linear3 = nn.Linear(256, 1)


    def forward(self, x):
        # Just a sequential pass to go through the 2 layers
        x = self.drop1(self.convnorm1(self.act1(self.linear1(x))))
        x = self.drop2(self.convnorm2(self.act2(self.linear2(x))))
        x = self.linear3(x)
        return x

#.................................................................................................................
def save_model(model):
    '''
      Print Model Summary
    '''

    print(model, file=open('summary.txt', "w"))
#------------------------------------------------------------------------------------------------------------------
def model_definition(model, pretrained=False):
    '''
        Define a Keras sequential model
        Compile the model
    '''

    # if pretrained == True:
    #         model = models.resnet50(progress=True, pretrained=pretrained)
    #         model.fc = nn.Linear(2048, 7)
    if model == CNN:
        model = CNN()
        model = model.to(device)

        weights = compute_class_weight(y=train_df.target.values, class_weight="balanced",
                                       classes=train_df.target.unique())
        class_weights = torch.FloatTensor(weights)
        if device.type == "cuda":
            class_weights = class_weights.cuda()
        optimizer = optim.SGD(model.fc.parameters(), lr = LR)
        criterion = nn.CrossEntropyLoss(weight=class_weights)


    elif model == MLP:
        model = MLP(N_NEURONS)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.MSELoss()



    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=0, verbose=True)

    save_model(model)

    return model, optimizer, criterion, scheduler
#...................................................................................................................
def metrics_func(metrics, aggregates, y_true, y_pred):
    '''
    multiple functiosn of metrics to call each function
    f1, cohen, accuracy, mattews correlation
    list of metrics: f1_micro, f1_macro, f1_avg, coh, acc, mat
    list of aggregates : avg, sum
    :return:
    '''

    def f1_score_metric(y_true, y_pred, type):
        '''
            type = micro,macro,weighted,samples
        :param y_true:
        :param y_pred:
        :param average:
        :return: res
        '''
        res = f1_score(y_true, y_pred, average=type)
        return res

    def cohen_kappa_metric(y_true, y_pred):
        res = cohen_kappa_score(y_true, y_pred)
        return res

    def accuracy_metric(y_true, y_pred):
        res = accuracy_score(y_true, y_pred)
        return res

    def matthews_metric(y_true, y_pred):
        res = matthews_corrcoef(y_true, y_pred)
        return res

    def hamming_metric(y_true, y_pred):
        res = hamming_loss(y_true, y_pred)
        return res

    xcont = 1
    xsum = 0
    xavg = 0
    res_dict = {}
    for xm in metrics:
        if xm == 'f1_micro':
            # f1 score average = micro
            xmet = f1_score_metric(y_true, y_pred, 'micro')
        elif xm == 'f1_macro':
            # f1 score average = macro
            xmet = f1_score_metric(y_true, y_pred, 'macro')
        elif xm == 'f1_weighted':
            # f1 score average =
            xmet = f1_score_metric(y_true, y_pred, 'weighted')
        elif xm == 'coh':
             # Cohen kappa
            xmet = cohen_kappa_metric(y_true, y_pred)
        elif xm == 'acc':
            # Accuracy
            xmet =accuracy_metric(y_true, y_pred)
        elif xm == 'mat':
            # Matthews
            xmet =matthews_metric(y_true, y_pred)
        elif xm == 'hlm':
            xmet =hamming_metric(y_true, y_pred)
        else:
            xmet = 0

        res_dict[xm] = xmet

        xsum = xsum + xmet
        xcont = xcont +1

    if 'sum' in aggregates:
        res_dict['sum'] = xsum
    if 'avg' in aggregates and xcont > 0:
        res_dict['avg'] = xsum/xcont
    # Ask for arguments for each metric

    return res_dict
#------------------------------------------------------------------------------------------------------------------
def train_and_test(train_ds, test_ds, list_of_metrics, list_of_agg, save_on, pretrained = False):

    model, optimizer, criterion, scheduler = model_definition(pretrained)

    cont = 0
    train_loss_item = list([])
    test_loss_item = list([])

    pred_labels_per_hist = list([])

    model.phase = 0

    met_test_best = 0
    for epoch in range(n_epoch):
        train_loss, steps_train = 0, 0

        model.train()

        pred_logits, real_labels = np.zeros((1, 1)), np.zeros((1, 1))

        train_hist = list([])
        test_hist = list([])

        with tqdm(total=len(train_ds), desc="Epoch {}".format(epoch)) as pbar:

            for xdata,xtarget in train_ds:

                xdata, xtarget = xdata.to(device), xtarget.to(device)
                optimizer.zero_grad()
                output = model(xdata)

                loss = criterion(output, xtarget)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                cont += 1

                steps_train += 1

                train_loss_item.append([epoch, loss.item()])

                pred_labels_per = output.detach().to(torch.device('cpu')).numpy()

                if len(pred_labels_per_hist) == 0:
                    pred_labels_per_hist = pred_labels_per
                else:
                    pred_labels_per_hist = np.vstack([pred_labels_per_hist, pred_labels_per])

                if len(train_hist) == 0:
                    train_hist = xtarget.cpu().numpy()
                else:
                    train_hist = np.vstack([train_hist, xtarget.cpu().numpy()])

                pbar.update(1)
                pbar.set_postfix_str("Test Loss: {:.5f}".format(train_loss / steps_train))

                pred_logits = np.vstack((pred_logits, output.detach().cpu().numpy()))
                real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))

        pred_labels = pred_logits[1:]
        pred_labels[pred_labels >= THRESHOLD] = 1
        pred_labels[pred_labels < THRESHOLD] = 0

        # Metric Evaluation
        train_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels[1:], pred_labels)

        avg_train_loss = train_loss / steps_train

        ## Finish with Training

        ## Testing the model

        model.eval()

        pred_logits, real_labels = np.zeros((1, 1)), np.zeros((1, 1))

        test_loss, steps_test = 0, 0
        met_test = 0

        with torch.no_grad():

            with tqdm(total=len(test_ds), desc="Epoch {}".format(epoch)) as pbar:

                for xdata,xtarget in test_ds:

                    xdata, xtarget = xdata.to(device), xtarget.to(device)
                    optimizer.zero_grad()
                    output = model(xdata)

                    loss = criterion(output, xtarget)
                    test_loss += loss.item()
                    cont += 1

                    steps_test += 1

                    test_loss_item.append([epoch, loss.item()])

                    pred_labels_per = output.detach().to(torch.device('cpu')).numpy()

                    if len(pred_labels_per_hist) == 0:
                        pred_labels_per_hist = pred_labels_per
                    else:
                        pred_labels_per_hist = np.vstack([pred_labels_per_hist, pred_labels_per])

                    if len(test_hist) == 0:
                        tast_hist = xtarget.cpu().numpy()
                    else:
                        test_hist = np.vstack([test_hist, xtarget.cpu().numpy()])

                    pbar.update(1)
                    pbar.set_postfix_str("Test Loss: {:.5f}".format(test_loss / steps_test))

                    pred_logits = np.vstack((pred_logits, output.detach().cpu().numpy()))
                    real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))

        pred_labels = pred_logits[1:]
        pred_labels[pred_labels >= THRESHOLD] = 1
        pred_labels[pred_labels < THRESHOLD] = 0

        test_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels[1:], pred_labels)

        avg_test_loss = test_loss / steps_test

        xstrres = "Epoch {}: ".format(epoch)
        for met, dat in train_metrics.items():
            xstrres = xstrres +' Train '+met+ ' {:.5f}'.format(dat)


        xstrres = xstrres + " - "
        for met, dat in test_metrics.items():
            xstrres = xstrres + ' Test '+met+ ' {:.5f}'.format(dat)
            if met == save_on:
                met_test = dat

        print(xstrres)  #Print metrics

        if met_test > met_test_best and SAVE_MODEL:

            torch.save(model.state_dict(), "model.pt")
            xdf_dset_results = xdf_dset_test.copy()

            ## The following code creates a string to be saved as 1,2,3,3,
            ## This code will be used to validate the model
            xfinal_pred_labels = []
            for i in range(len(pred_labels)):
                joined_string = ",".join(str(int(e)) for e in pred_labels[i])
                xfinal_pred_labels.append(joined_string)

            xdf_dset_results['results'] = xfinal_pred_labels

            xdf_dset_results.to_excel('results.xlsx', index = False)
            print("The model has been saved!")
            met_test_best = met_test

#------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------
PATH = os.getcwd()
os.chdir("..")
Excel_PATH = os.getcwd()
# read excel file
for file in os.listdir(Excel_PATH):
    if file[-5:] == '.xlsx':
        FILE_NAME = os.getcwd() + os.path.sep + 'excel.xlsx'
data = pd.read_excel(FILE_NAME)

# split the train and test file (train 70%, test 15%, val 15%)
train, test_val = train_test_split(data, test_size=0.3, random_state=1412
                                   , stratify=data["label"])
test, val = train_test_split(test_val, test_size=0.5, random_state=1412
                             , stratify=test_val["label"])

# transform data
train_dataset = BreastCancerDataset(train, transform=my_transform(key="train"))
dev_dataset = BreastCancerDataset(val, transform=my_transform(key="val"))
test_dataset = BreastCancerDataset(test, transform=my_transform(key="val"))

# dataloader
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

dataloaders = {"train": train_dataloader, "dev": dev_dataloader, "test": test_dataloader}

#...........................................................................................................
# 2022/12/04     ლ(◉◞౪◟◉ )ლ   design 好 MLP 跟 CNN model，但還在想辦法改成跟Exam2一樣運作方式  --By Ian



# model = torchvision.models.resnet18(pretrained=False)

# for epoch in range(N_EPOCHS):

#     optimizer.zero_grad()  # It is good practice to do it right before going forward on any model

#     t_pred = model(p)

#     loss = criterion(t, t_pred)
#     loss.backward()
#     optimizer.step()
#     if epoch % PRINT_LOSS_EVERY == 0:
#         print("Epoch {} | Loss {:.5f}".format(epoch, loss.item()))


patients = data.patient_id.unique()

train_ids, sub_test_ids = train_test_split(patients,
                                           test_size=0.3,
                                           random_state=0)
test_ids, dev_ids = train_test_split(sub_test_ids, test_size=0.5, random_state=0)

train_df = data.loc[data.patient_id.isin(train_ids),:].copy()
test_df = data.loc[data.patient_id.isin(test_ids),:].copy()
dev_df = data.loc[data.patient_id.isin(dev_ids),:].copy()
# df_train = CustomDataset(train)
# df_test = CustomDataset(test)
list_of_metrics = ['acc', 'hlm']
list_of_agg = ['avg']


# image_datasets = {"train": train_dataset, "dev": dev_dataset, "test": test_dataset}
# dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "dev", "test"]}


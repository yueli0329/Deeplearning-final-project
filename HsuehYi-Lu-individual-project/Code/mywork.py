import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from PIL import Image
from imgaug import augmenters as iaa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CyclicLR
import torchvision
from torchvision import  transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import os
import cv2
from glob import glob
from skimage.io import imread
from os import listdir

import time
import copy
from tqdm import tqdm_notebook as tqdm

from pip._internal import main
main(['install','ipywidgets'])

#####################################  Setting  #################################################################

NUM_EPOCH = 30
BATCH_SIZE = 32
NUM_CLASSES = 2
run_training = True
find_learning_rate = False
start_lr = 1e-6
#end_lr = 0.1 # initial
end_lr = 0.006
MODEL_PATH =  "../input/breastcancermodel/"
LOSSES_PATH = "../input/breastcancermodel/"
OUTPUT_PATH = ""
model_name = 'Resnet18'  #'VGG16'
imple_CLR = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
np.random.seed(0)


#####################################  Read in dataset  ####################################################

def Read_in_Dataset(folder,base_path):

    total_images = 0
    for n in range(len(folder)):
        patient_id = folder[n]
        for c in [0, 1]:
            patient_path = base_path + "/" + patient_id
            class_path = patient_path + "/" + str(c) + "/"
            subfiles = listdir(class_path)
            total_images += len(subfiles)


    data = pd.DataFrame(index=np.arange(0, total_images), columns=["patient_id", "path", "target"])

    k = 0
    for n in range(len(folder)):
        patient_id = folder[n]
        patient_path = base_path + "/"+patient_id
        for c in [0,1]:
            class_path = patient_path + "/" + str(c) + "/"
            subfiles = listdir(class_path)
            for m in range(len(subfiles)):
                image_path = subfiles[m]
                data.iloc[k]["path"] = class_path + image_path
                data.iloc[k]["target"] = c
                data.iloc[k]["patient_id"] = patient_id
                k += 1

    data.loc[:, "target"] = data.target.astype(np.str)

    return data

def extract_coords(df):
    coord = df.path.str.rsplit("_", n=4, expand=True)
    coord = coord.drop([0, 1, 4], axis=1)
    coord = coord.rename({2: "x", 3: "y"}, axis=1)
    coord.loc[:, "x"] = coord.loc[:,"x"].str.replace("x", "", case=False).astype(np.int)
    coord.loc[:, "y"] = coord.loc[:,"y"].str.replace("y", "", case=False).astype(np.int)
    df.loc[:, "x"] = coord.x.values
    df.loc[:, "y"] = coord.y.values
    return df

def get_cancer_dataframe(patient_id, cancer_id):
    path = base_path +"/" + patient_id + "/" + cancer_id
    files = listdir(path)
    dataframe = pd.DataFrame(files, columns=["filename"])
    path_names = path + "/" + dataframe.filename.values
    dataframe = dataframe.filename.str.rsplit("_", n=4, expand=True)
    dataframe.loc[:, "target"] = np.int(cancer_id)
    dataframe.loc[:, "path"] = path_names
    dataframe = dataframe.drop([0, 1, 4], axis=1)
    dataframe = dataframe.rename({2: "x", 3: "y"}, axis=1)
    dataframe.loc[:, "x"] = dataframe.loc[:,"x"].str.replace("x", "", case=False).astype(np.int)
    dataframe.loc[:, "y"] = dataframe.loc[:,"y"].str.replace("y", "", case=False).astype(np.int)
    return dataframe

def get_patient_dataframe(patient_id):
    df_0 = get_cancer_dataframe(patient_id, "0")
    df_1 = get_cancer_dataframe(patient_id, "1")
    patient_df = df_0.append(df_1)
    return patient_df


def train_test_dev(data):
    patients = data.patient_id.unique()

    train_ids, sub_test_ids = train_test_split(patients,
                                               test_size=0.3,
                                               random_state=0)
    test_ids, dev_ids = train_test_split(sub_test_ids, test_size=0.5, random_state=0)

    train_df = data.loc[data.patient_id.isin(train_ids), :].copy()
    test_df = data.loc[data.patient_id.isin(test_ids), :].copy()
    dev_df = data.loc[data.patient_id.isin(dev_ids), :].copy()

    train_df = extract_coords(train_df)
    test_df = extract_coords(test_df)
    dev_df = extract_coords(dev_df)

    return train_df, test_df, dev_df


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


#####################################  Inital model  ####################################################


def initialize_model(model_name=model_name):
    # if model_name == 'Resnet18':
    #
    #     model = torchvision.models.resnet18(pretrained=True)
    #
    #     num_features = model.fc.in_features
    #
    #     model.fc = nn.Sequential(
    #         nn.Linear(num_features, 512),
    #         nn.ReLU(),
    #         nn.BatchNorm1d(512),
    #         nn.Dropout(0.5),
    #
    #         nn.Linear(512, 256),
    #         nn.ReLU(),
    #         nn.BatchNorm1d(256),
    #         nn.Dropout(0.5),
    #
    #         nn.Linear(256, NUM_CLASSES))
    #     #print(num_features)

    if model_name == 'VGG':

        model = torchvision.models.vgg16(pretrained=True)

        num_features = model.classifier[6].in_features

        model.classifier[6] = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),

            nn.Linear(256, NUM_CLASSES))

    return model


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

#
# def train_loop(model, criterion, optimizer, lr_find=False, scheduler=None, num_epochs=10, lam=0.0, model_name = model_name ):
#     since = time.time()
#     if lr_find:
#         phases = ["train"]
#     else:
#         phases = ["train", "dev", "test"]
#
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0
#
#     loss_dict = {"train": [], "dev": [], "test": []}
#     lam_tensor = torch.tensor(lam, device=device)
#
#     running_loss_dict = {"train": [], "dev": [], "test": []}
#
#     lr_find_loss = []
#     lr_find_lr = []
#     smoothing = 0.2
#
#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         print('-' * 10)
#
#         for phase in phases:
#             if phase == "train":
#                 model.train()
#             else:
#                 model.eval()
#
#             running_loss = 0.0
#             running_corrects = 0
#
#             tk0 = tqdm(dataloaders[phase], total=int(len(dataloaders[phase])))
#
#             counter = 0
#             for bi, d in enumerate(tk0):
#                 inputs = d["image"]
#                 labels = d["label"]
#                 inputs = inputs.to(device, dtype=torch.float)
#                 labels = labels.to(device, dtype=torch.long)
#
#                 # zero the parameter gradients
#                 optimizer.zero_grad()
#
#                 # forward
#                 # track history if only in train
#
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)
#
#                     # backward + optimize only if in training phase
#                     if phase == 'train':
#                         loss.backward()
#
#                         # l2_reg = torch.tensor(0., device=device)
#                         # for param in model.parameters():
#                         # l2_reg = lam_tensor * torch.norm(param)
#
#                         # loss += l2_reg
#
#                         optimizer.step()
#                         # cyclical lr schedule is invoked after each batch
#                         if scheduler is not None:
#                             scheduler.step()
#                             if lr_find:
#                                 lr_step = optimizer.state_dict()["param_groups"][0]["lr"]
#                                 lr_find_lr.append(lr_step)
#                                 if counter == 0:
#                                     lr_find_loss.append(loss.item())
#                                 else:
#                                     smoothed_loss = smoothing * loss.item() + (1 - smoothing) * lr_find_loss[-1]
#                                     lr_find_loss.append(smoothed_loss)
#
#                 # statistics
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)
#
#                 counter += 1
#
#                 tk0.update(1)
#                 tk0.set_postfix({'loss': running_loss / (counter * dataloaders[phase].batch_size),
#                                  'accuracy': running_corrects.double() / (counter * dataloaders[phase].batch_size)})
#                 running_loss_dict[phase].append(running_loss / (counter * dataloaders[phase].batch_size))
#
#             epoch_loss = running_loss / dataset_sizes[phase]
#             loss_dict[phase].append(epoch_loss)
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]
#             print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#                 phase, epoch_loss, epoch_acc))
#
#             # deep copy the model
#             if phase == 'dev' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())
#
#                 OUTPUT_PATH = f"_{model_name}_cuda.pth"
#
#                 torch.save(model.state_dict(), OUTPUT_PATH)
#
#                 print("The model has been saved!")
#
#
#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     print('Best val Acc: {:4f}'.format(best_acc))
#
#     # load best model weights
#     model.load_state_dict(best_model_wts)
#     results = {"model": model,
#                "loss_dict": loss_dict,
#                "running_loss_dict": running_loss_dict,
#                "lr_find": {"lr": lr_find_lr, "loss": lr_find_loss}}
#     return results


def f1_score(preds, targets):
    tp = (preds * targets).sum().to(torch.float32)
    fp = ((1 - targets) * preds).sum().to(torch.float32)
    fn = (targets * (1 - preds)).sum().to(torch.float32)

    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1_score = 2 * precision * recall / (precision + recall + epsilon)
    return f1_score


#####################################  cyclical learning rate  ####################################################
#
# def get_lr_search_scheduler(optimizer, min_lr, max_lr, max_iterations):
#     # max_iterations should be the number of steps within num_epochs_*epoch_iterations
#     # this way the learning rate increases linearily within the period num_epochs*epoch_iterations
#     scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer,
#                                                   base_lr=min_lr,
#                                                   max_lr=max_lr,
#                                                   step_size_up=max_iterations,
#                                                   step_size_down=max_iterations,
#                                                   mode="triangular")
#
#     return scheduler
#
#
# def get_scheduler(optimizer, min_lr, max_lr, stepsize):
#     # suggested_stepsize = 2*num_iterations_within_epoch
#     stepsize_up = np.int(stepsize / 2)
#     scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer,
#                                                   base_lr=min_lr,
#                                                   max_lr=max_lr,
#                                                   step_size_up=stepsize_up,
#                                                   step_size_down=stepsize_up,
#                                                   mode="triangular")
#     return scheduler
#
#
# def find_lr_plot(find_lr_df):
#     fig, ax = plt.subplots(1, 2, figsize=(20, 5))
#     ax[0].plot(find_lr_df.lr.values)
#     ax[1].plot(find_lr_df["smoothed loss"].values)
#     ax[0].set_xlabel("Steps")
#     ax[0].set_ylabel("Learning rate")
#     ax[1].set_xlabel("Steps")
#     ax[1].set_ylabel("Loss");
#     ax[0].set_title("How the learning rate increases during search")
#     ax[1].set_title("How the training loss evolves during search")
#
#     plt.figure(figsize=(20, 5))
#     plt.plot(find_lr_df.lr.values, find_lr_df["smoothed loss"].values, '-', color="tomato");
#     plt.xlabel("Learning rate")
#     plt.xscale("log")
#     plt.ylabel("Smoothed Loss")
#     plt.title("Searching for the optimal learning rate (VGG)")
#     plt.show()
#
#
# def CLR(start_lr,end_lr,train_dataloader,model_name,criterion,model):
#     lr_find_epochs = 1
#
#     if model_name =='Resnet18':
#         optimizer = optim.SGD(model.fc.parameters(), start_lr)
#     if model_name =='VGG16':
#         optimizer = optim.SGD(model.classifier[6].parameters(), start_lr)
#
#     scheduler = get_lr_search_scheduler(optimizer, start_lr, end_lr, lr_find_epochs * len(train_dataloader))
#     results = train_loop(model, criterion, optimizer, lr_find=True, scheduler=scheduler, num_epochs=lr_find_epochs,model_name = model_name)
#     lr_find_lr, lr_find_loss = results["lr_find"]["lr"], results["lr_find"]["loss"]
#
#     find_lr_df = pd.DataFrame(lr_find_loss, columns=["smoothed loss"])
#     find_lr_df.loc[:, "lr"] = lr_find_lr
#     #find_lr_df.to_csv("learning_rate_search.csv", index=False)
#     find_lr_plot(find_lr_df)


#####################################  Training process  ####################################################


def Loss_plot(losses_df,running_losses_df,model_name):

    plt.figure(figsize=(20,5))

    plt.plot(losses_df["train"], '-o', label="train")
    plt.plot(losses_df["dev"], '-o', label="dev")
    plt.plot(losses_df["test"], '-o', label="dev")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted x-entropy")
    plt.title(f"Loss change over epoch - {model_name}")
    plt.legend()
    plt.show()



    fig, ax = plt.subplots(3,1,figsize=(20,15))

    ax[0].plot(running_losses_df["train"], '-o', label="train")
    ax[0].set_xlabel("Step")
    ax[0].set_ylabel("Weighted x-entropy")
    ax[0].set_title("Loss change over steps")
    ax[0].legend();

    ax[1].plot(running_losses_df["dev"], '-o', label="dev", color="orange")
    ax[1].set_xlabel("Step")
    ax[1].set_ylabel("Weighted x-entropy")
    ax[1].set_title("Loss change over steps")
    ax[1].legend();

    ax[2].plot(running_losses_df["test"], '-o', label="test", color="mediumseagreen")
    ax[2].set_xlabel("Step")
    ax[2].set_ylabel("Weighted x-entropy")
    ax[2].set_title("Loss change over steps")
    ax[2].legend()
    2
    plt.show()


def trainer(model,criterion,model_name,start_lr,end_lr,num_epoch,imple_CLR):
    since = time.time()
    if model_name =='Resnet18':
        optimizer = optim.SGD(model.fc.parameters(), start_lr)
        print(f'Training {model_name} model...........')
    if model_name =='VGG16':
        optimizer = optim.SGD(model.classifier[6].parameters(), start_lr)
        print(f'Training {model_name} model...........')

    NUM_EPOCHS = num_epoch

    scheduler = get_scheduler(optimizer, start_lr, end_lr, 2 * NUM_EPOCHS)

    if imple_CLR:
        print(f'Implementing CLR for training process...........')
        results = train_loop(model, criterion, optimizer, scheduler=scheduler, num_epochs=NUM_EPOCHS,model_name = model_name)

    else:
        print(f'Constant learning rate for trainiing process...........')
        results = train_loop(model, criterion,optimizer,num_epochs=NUM_EPOCHS)



    model, loss_dict, running_loss_dict = results["model"], results["loss_dict"], results["running_loss_dict"]


    losses_df = pd.DataFrame(loss_dict["train"], columns=["train"])
    losses_df.loc[:, "dev"] = loss_dict["dev"]
    losses_df.loc[:, "test"] = loss_dict["test"]
    #losses_df.to_csv("losses_breastcancer.csv", index=False)

    running_losses_df = pd.DataFrame(running_loss_dict["train"], columns=["train"])
    running_losses_df.loc[0:len(running_loss_dict["dev"]) - 1, "dev"] = running_loss_dict["dev"]
    running_losses_df.loc[0:len(running_loss_dict["test"]) - 1, "test"] = running_loss_dict["test"]
    #running_losses_df.to_csv("running_losses_breastcancer.csv", index=
    Loss_plot(losses_df,running_losses_df,model_name)

    time_elapsed = time.time() - since
    print('Training process lasts....  {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))



if __name__ == '__main__':

    ORI_PATH = os.getcwd()
    os.chdir("..")
    PATH = os.getcwd() + os.path.sep + 'data'
    folder = os.listdir(PATH)
    base_path=PATH

    data = Read_in_Dataset(folder,base_path)


    # split the train and test file (train 70%, test 15%, val 15%)

    train_df,test_df, dev_df = train_test_dev(data)

    train_dataset = BreastCancerDataset(train_df, transform=alltransform(key="train"))
    dev_dataset = BreastCancerDataset(dev_df, transform=alltransform(key="val"))
    test_dataset = BreastCancerDataset(test_df, transform=alltransform(key="val"))


    image_datasets = {"train": train_dataset, "dev": dev_dataset, "test": test_dataset}
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "dev", "test"]}


    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    dataloaders = {"train": train_dataloader, "dev": dev_dataloader, "test": test_dataloader}


    model = initialize_model(model_name = model_name)
    model.apply(init_weights)
    model = model.to(device)


    weights = compute_class_weight(y=train_df.target.values, class_weight="balanced", classes=train_df.target.unique())
    class_weights = torch.FloatTensor(weights)
    class_weights = class_weights.cuda()
    #print(class_weights)


    criterion = nn.CrossEntropyLoss(weight=class_weights)


    if find_learning_rate:
        CLR(start_lr,end_lr,train_dataloader,model_name,criterion,model)

    if run_training:
        trainer(model,criterion,model_name,start_lr,end_lr,NUM_EPOCH,imple_CLR)


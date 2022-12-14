'''
#@Time      :12/9/22 17:08
#@Author    : Chelsea Li
#@File      :VGG.py
#@Software  :PyCharm
'''
import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary

class VGG16(nn.Module):
    def __init__(self):
        super().__init__()

        #block 1
        self.conv1 = nn.Conv2d(3,64,3,1,1)
        self.conv2 = nn.Conv2d(64,64,3,1,1)
        self.pool1 = nn.MaxPool2d(2)

        #block 2
        self.conv3 = nn.Conv2d(64,128,3,1,1)
        self.conv4 = nn.Conv2d(128,128,3,1,1)
        self.pool2 = nn.MaxPool2d(2)

        #block 3
        self.conv5 = nn.Conv2d(128,256,3,1,1)
        self.conv6 = nn.Conv2d(256,256,3,1,1)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2)

        #block 4
        self.conv8 = nn.Conv2d(256,512,3,1,1)
        self.conv9 = nn.Conv2d(512,512,3,1,1)
        self.conv10 = nn.Conv2d(512,512, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(2)

        #block 5
        self.conv11 = nn.Conv2d(512,512,3,1,1)
        self.conv12 = nn.Conv2d(512,512,3,1,1)
        self.conv13 = nn.Conv2d(512, 512, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(2)

        #FC
        self.linear1 = nn.Linear(512*7*7,4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.linear3 = nn.Linear(4096, 2)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))

        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(F.relu(self.conv7(x)))

        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.pool4(F.relu(self.conv10(x)))

        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.pool5(F.relu(self.conv13(x)))

        x = x.view(-1,512*7*7)

        x = F.relu(self.linear1(F.dropout(x,p=0.5)))
        x = F.relu(self.linear2(F.dropout(x, p=0.5)))
        output = F.sigmoid(self.linear3(x))

        return output

vgg = VGG16()

print(summary(vgg, input_size=(10, 3, 224, 224),device="cpu"))



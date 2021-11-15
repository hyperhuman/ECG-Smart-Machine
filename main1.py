from __future__ import print_function, division
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from ECGDataset import ECGDataset

import gc

import torchvision.transforms as transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

from skimage import io, transform
import pandas as pd


import warnings
warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)
plt.ion()

# Set device
device = torch.device('cuda')
torch.cuda.empty_cache()

dataset = ECGDataset(csv_file='imageLabels.csv', root_dir='data', transforms=transforms.ToTensor())

train_set, test_set = torch.utils.data.random_split(dataset, [319,1])

#torch.cuda.memory_summary(device=None, abbreviated=False)

# sample of a network for our CNN

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 4)

        # linear, fully connected, and dense are the same
        self.fc1 = nn.Linear(in_features=77 * 57 * 32, out_features=2239)
        self.out = nn.Linear(in_features=2239, out_features=5)

        # foward pass

    def forward(self, t):
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = F.relu(self.conv3(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = F.relu(self.fc1(t.reshape(-1, 77 * 57 * 32)))
        t = self.out(t)

        return t


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()



network = Network()


network = network.cuda()
optimizer = optim.Adam(network.parameters(), lr=0.0007)
test_loader = DataLoader(dataset=test_set, batch_size=65, shuffle=True)
for epoch in range(100):
    total_loss = 0
    total_correct = 0
    correct = 0
    train_loader = DataLoader(dataset=train_set, batch_size=65, shuffle=True)
    for batch in train_loader:
        torch.cuda.empty_cache()
        images, labels = batch

        labels = labels.to('cuda')
        preds = network(images.to('cuda'))
        loss = F.cross_entropy(preds, labels)

        optimizer.zero_grad()
        loss.backward()  # back propogation
        optimizer.step()  # Update weights

        total_loss += loss.item()
        total_correct = get_num_correct(preds, labels)
        correct += total_correct

        print("epoch:", epoch, "total correct:", total_correct, "total loss:", total_loss)

    print("CNN got", correct, "321")
# confusion matrix

print(torch.__version__)
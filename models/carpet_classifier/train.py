import logging
from pathlib import Path
from typing import Sequence

import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import random
#import time
from datetime import datetime
import copy

import warnings
from warnings import warn

import torch
from torch import Tensor, nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import torchvision.models as models
# from torchinfo import summary

from data import SampleDataModule
__all__ = ["SampleDataModule"]

logger = logging.getLogger(__name__)


def test_datamodule(class_names, dataloader):
    # dataiter = iter(dataloader)
    # # images, labels = next(dataiter)
    # batch = next(dataiter)
    for i, batch in enumerate(dataloader):
        print("Batch {}:".format(i+1))
        print("\tGround Truth Classes: \t{}".format(' '.join(f'{class_names[batch["class"][j]]:5s}' for j in range(len(batch["image"][:5])))))
        # print(">>>>>>>>>> min: {} \tmax: {}".format(batch["image"].min(), batch["image"].max()))
        image_grid = torchvision.utils.make_grid(batch["image"][:5])
        # image_grid = image_grid / 2 + 0.5 # unnormalize
        inverse_transform = T.Compose([T.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                       T.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),
                                       ])
        plt.imshow(np.transpose(inverse_transform(image_grid).numpy(), (1, 2, 0)))
        plt.show()


def train_model(trainloader, num_classes, epochs=10):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1) #models.vgg16(pretrained=True)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    print(model.classifier)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #exp_lr_scheduler = lr_scheduler.StepLR(optimiser, step_size=7, gamma=0.1)

    for epoch in range(epochs):  # loop over the dataset multiple times
        if epoch == 0: dataset_size = 0 # only dataset calculate size once
        running_loss = 0.0
        running_corrects = 0
        for i, batch in enumerate(trainloader, 0):
            if epoch == 0: dataset_size += len(batch["image"])
            images, labels = batch["image"].to(device), batch["class"].to(device)
            # print(">>>>>>>>>>>>> Original Labels: \t{}".format(labels))
            for idx, label in enumerate(torch.unique(labels).tolist()):
                labels[labels == label] = idx
            # print(">>>>>>>>>>>>> New Labels: \t{}".format(labels))

            optimiser.zero_grad() # zero the parameter gradients

            # forward + backward + optimize
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            # print(">>>>>>>>>>>>> Outputs: \t{}".format(outputs))
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            # update statistics
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.double() / dataset_size
        print('Epoch {}/{}: \tLoss: {:.4f} \tAccuracy: {:.4f}'.format(epoch+1, epochs, epoch_loss, epoch_acc))

    print('Finished Training!')
    # save_path = './cifar_net.pth'
    # torch.save(model.state_dict(), PATH)
    return model


def test_model(model, testloader, all_classes, train_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    for i, batch in enumerate(testloader, 0):
        with torch.no_grad():
            outputs = model(batch["image"].to(device))
        _, preds = torch.max(outputs, 1)
        print("Ground Truth: \t{}".format(' '.join(f'{all_classes[batch["class"][j]]:5s}' for j in range(len(batch["image"])))))
        print("Predicted: \t{}".format(' '.join(f'{train_classes[preds[j]]:5s}' for j in range(len(batch["image"])))))
        # image_grid = torchvision.utils.make_grid(batch["image"])


def evaluate_model():

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


if __name__ == "__main__":
    folder_path = '/home/brionyf/Desktop/Images/training/classification/'
    image_size = 256
    seed = 42
    datamodule = SampleDataModule(folder_path, image_size, seed)
    trainloader = datamodule.train_dataloader()
    testloader = datamodule.test_dataloader()

    # test_datamodule(datamodule.classes, testloader)

    epochs = 20
    train_classes = [c for c in datamodule.classes if c not in datamodule.test_classes]
    model = train_model(trainloader, len(train_classes), epochs)

    test_model(model, testloader, datamodule.classes, train_classes)


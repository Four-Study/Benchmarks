#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import torchvision
import torch.optim as optim

import numpy as np

import os
import argparse
import datetime

tik = datetime.datetime.now()

# hyper-parameters
parser         = argparse.ArgumentParser(description='VGG CIFAR100 Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--n_epoch', default=300, type=int, help='number of epochs')
parser.add_argument('--depth', default=16, type=int, choices=[11, 13, 16, 19], help='depth of net')
parser.add_argument('--norm', default=True, action=argparse.BooleanOptionalAction, help='normalization?')
args           = parser.parse_args()
device         = torch.device("cuda")
ngpu           = torch.cuda.device_count()
nc             = 3
nclass         = 100
lambda_lr      = lambda epoch: 0.5 ** (epoch // 25)
momentum       = 0.9
weight_decay   = 5e-4
loss_criterion = nn.CrossEntropyLoss()


transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
#             transforms.RandomRotation(degrees=(-20, 20)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
A         = torchvision.datasets.CIFAR100('./data', train=True, download=True,transform=transform)
test      = torchvision.datasets.CIFAR100('./data', train=False, download=True,transform=transform)
trainset  = torch.utils.data.DataLoader(A, batch_size=args.batch_size, shuffle=True)
testset   = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=False)


# Neural network structure
if args.norm:
    if args.depth == 11:
        Net  = models.vgg11_bn(pretrained=False, num_classes=nclass)
    if args.depth == 13:
        Net  = models.vgg13_bn(pretrained=False, num_classes=nclass)
    if args.depth == 16:
        Net  = models.vgg16_bn(pretrained=False, num_classes=nclass)
    if args.depth == 19:
        Net  = models.vgg19_bn(pretrained=False, num_classes=nclass)
else:
    if args.depth == 11:
        Net  = models.vgg11(pretrained=False, num_classes=nclass)
    if args.depth == 13:
        Net  = models.vgg13(pretrained=False, num_classes=nclass)
    if args.depth == 16:
        Net  = models.vgg16(pretrained=False, num_classes=nclass)
    if args.depth == 19:
        Net  = models.vgg19(pretrained=False, num_classes=nclass)
        
Net          = Net.to(device)
Net          = nn.DataParallel(Net)
optimizer    = torch.optim.Adam(Net.parameters(), lr=args.lr, betas=(0.5, 0.999))
optimizer    = optim.SGD(Net.parameters(),
                        lr=args.lr,
                        momentum=momentum,
                        weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)


# predict accuracy on a data loader
def accuracy(dataloader):
    correct = 0
    correct5 = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            X, y = data
            X = X.to(device)
            y = y.to(device)
            output = Net(X)
            probs = F.softmax(output, 1)
            sorted_probs, indexes = torch.sort(probs, dim = 1, descending = True)

            for idx, feature in enumerate(indexes):
                ## top-1 accuracy
                if feature[0] == y[idx]:
                    correct += 1
                ## top-5 accuracy
                if y[idx] in feature[:5]:
                    correct5 += 1
                total += 1
    top1 = correct / total
    top5 = correct5 / total
    return (top1, top5)

for epoch in range(args.n_epoch): 
    losses = []
    for data in trainset:  
        X, y = data
        X = X.to(device)
        y = y.to(device) 
        Net.zero_grad()  
        output = Net(X)  
        loss = loss_criterion(output, y)  

        # Backpropergation 
        loss.backward()  
        losses.append(loss.item())
        optimizer.step()  
    lr_scheduler.step()
    losses = np.array(losses)
    print('Epoch', epoch, ':', losses.mean())
    if (epoch % 10) == 0:
        # predict accuracy on test data set
        # record the sorted scores, associate permutation indexes, and ground-truth y
        top1_insample, top5_insample  = accuracy(trainset)
        top1_outsample, top5_outsample = accuracy(testset)
        print('Training Top 1 Accuracy', top1_insample, 'Training Top 5 Accuracy', top5_insample)
        print('Test Top 1 Accuracy', top1_outsample, 'Test Top 5 Accuracy', top5_outsample)


print('Training Top 1 Accuracy', top1_insample, 'Training Top 5 Accuracy', top5_insample)
print('Test Top 1 Accuracy', top1_outsample, 'Test Top 5 Accuracy', top5_outsample)

tok = datetime.datetime.now()
print('execution time:', tok - tik)

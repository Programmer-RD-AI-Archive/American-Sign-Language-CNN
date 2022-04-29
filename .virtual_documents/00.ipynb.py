import os
import cv2
import wandb
import torch
import torchvision
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn import *
from torch.optim import *
from torchvision.transforms import *
from torchvision.models import *
from sklearn.model_selection import *
from tqdm import tqdm


PROJECT_NAME = "American-Sign-Language-CNN"
TEST_SIZE = 0.25
device = 'cuda'
# TODO


def load_data():
    data = []
    X = []
    y = []
    labels = {}
    labels_r = {}
    idx = 0
    for folder_dir in os.listdir('./data/'):
        idx += 1
        labels[folder_dir] = idx
        labels_r[idx] = folder_dir
    for folder_dir in tqdm(os.listdir('./data/')):
        for file_dir in os.listdir(f'./data/{folder_dir}/'):
            img = cv2.imread(f'./data/{folder_dir}/{file_dir}')
            img = cv2.resize(img,(56,56))
            img = img / 255.0
            data.append([
                img,
                np.eye(labels[folder_dir],idx)[-1]
            ])
    np.random.shuffle(data)
    for d_iter in data:
        X.append(d_iter[0])
        y.append(d_iter[1])
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=TEST_SIZE,shuffle=True)
    X_train = torch.from_numpy(np.array(X_train)).view(-1,3,56,56).float().to(device)
    X_test = torch.from_numpy(np.array(X_test)).view(-1,3,56,56).float().to(device)
    y_train = torch.from_numpy(np.array(y_train)).float().to(device)
    y_test = torch.from_numpy(np.array(y_test)).float().to(device)
    return X_train,X_test,y_train,y_test,data,X,y,labels,labels_r,idx


X_train,X_test,y_train,y_test,data,X,y,labels,labels_r,idx = load_data()


len(X_train),len(X_test),len(y_train),len(y_test)


def get_loss(model,X,y,criterion):
    preds = model(X)
    loss = criterion(preds,y)
    return loss.item()


def get_accuracy(model,X,y):
    preds = model(X)
    correct = 0
    total = 0
    for pred,y_iter in zip(preds,y):
        pred = torch.argmax(pred)
        y_iter = torch.argmax(y_iter)
        if pred == y_iter:
            correct += 1
        total += 1
    return round(correct/total,3)


model = resnet18().to(device)
model.fc = Linear(512,idx)
model = model.to(device)
criterion = MSELoss()
optimizer = Adam(model.parameters(),lr=0.001)
batch_size = 32
epochs = 100


wandb.init(project=PROJECT_NAME,name='BaseLine resnet18')
wandb.watch(model)
for _ in tqdm(range(epochs)):
    for idx in range(0,len(X_train),batch_size):
        X_batch = X_train[idx:idx+batch_size].float().view(-1,3,56,56).to(device)
        y_batch = y_train[idx:idx+batch_size].float().to(device)
        preds = model(X_batch)
        loss = criterion(preds,y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    wandb.log({
        'Accuracy Train':get_accuracy(model,X_train,y_train),
        'Loss Train':get_loss(model,X_train,y_train,criterion),
        'Accuracy Test':get_accuracy(model,X_test,y_test),
        'Loss Test':get_loss(model,X_test,y_test,criterion),
    })
    model.train()
wandb.watch(model)
wandb.finish()


torch.save(model,'./save/model.pt')
torch.save(model,'./save/model.pth')
torch.save(criterion,'./save/criterion.pt',)
torch.save(criterion,'./save/criterion.pth')
torch.save(optimizer,'./save/optimizer.pt')
torch.save(optimizer,'./save/optimizer.pth')




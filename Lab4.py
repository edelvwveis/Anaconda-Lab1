# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 08:42:00 2025

@author: edelvwveiss
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('dataset_simple.csv')

scaler = StandardScaler()
X = torch.FloatTensor(scaler.fit_transform(df.iloc[:, 0:2].values))
y = torch.FloatTensor(df.iloc[:, 2].values).reshape(-1, 1)

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class NNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(NNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, out_size),
            nn.Sigmoid()
        )
    
    def forward(self, X):
        return self.layers(X)

inputSize = X_train.shape[1]
hiddenSizes = 10
outputSize = 1  

net = NNet(inputSize, hiddenSizes, outputSize)
lossFn = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

epochs = 500
for i in range(epochs):
    optimizer.zero_grad()
    pred = net(X_train)
    loss = lossFn(pred, y_train)
    loss.backward()
    optimizer.step()
    if i % 50 == 0:
        print(f'Эпоха {i+1}, Ошибка: {loss.item():.4f}')

# Обучающая выбрка
with torch.no_grad():
    pred_train = net(X_train)
    pred_labels_train = (pred_train > 0.5).float()
    accuracy_train = (pred_labels_train == y_train).float().mean()
    err_train = sum(abs(y_train - pred_labels_train)) / 2
    print(f"\nТочность на обучающей выборке: {accuracy_train.item() * 100:.2f}%")
    print(f"Ошибок на обучающей выборке: {err_train.item()}")

# Тестовая выборка
with torch.no_grad():
    pred_test = net(X_test)
    pred_labels_test = (pred_test > 0.5).float()
    accuracy_test = (pred_labels_test == y_test).float().mean()
    err_test = sum(abs(y_test - pred_labels_test)) / 2
    print(f"\nТочность на тестовой выборке: {accuracy_test.item() * 100:.2f}%")
    print(f"Ошибок на тестовой выборке: {err_test.item()}")
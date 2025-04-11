# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 15:53:30 2025

@author: edelvwveiss
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Загрузка и подготовка данных
df = pd.read_csv('data.csv')

# Признаки и метки
X = torch.tensor(df.iloc[:, :4].values, dtype=torch.float32)
y = torch.tensor(np.where(df.iloc[:, 4] == "Iris-setosa", 1, -1), dtype=torch.float32).view(-1, 1)

# Линейная модель
model = nn.Linear(4, 1)

print('Начальные веса: ', model.weight.data)
print('Начальный сдвиг (bias): ', model.bias.data)

# Функция потерь и оптимизатор
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Обучение
iteration = 1
while True:
    pred = model(X)
    loss = criterion(pred, y)
    print(f'Ошибка на {iteration} итерации: {loss.item():.4f}')
    
    optimizer.zero_grad()  # Обнуляем градиенты перед backward
    loss.backward()
    optimizer.step()
    
    iteration += 1
    if loss.item() < 1:
        break

# Определяем порог разделения
min_val, max_val = pred.min().item(), pred.max().item()
threshold = (min_val + max_val) / 2

# Интерфейс для ввода данных пользователем
print("\nВведите четыре параметра цветка, чтобы определить его вид.")
while True:
    try:
        inputs = [float(input(f"Введите признак {i+1}: ")) for i in range(4)]
        z = torch.tensor(inputs, dtype=torch.float32)
        with torch.no_grad():
            output = model(z)
        print("\nВид цветка: ", "Iris-setosa" if output.item() >= threshold else "Iris-versicolor")
    except ValueError:
        print("Ошибка ввода. Пожалуйста, введите числовые значения.")
    print()

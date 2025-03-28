import torch
import random

x = torch.randint(1, 10, (1,), dtype=torch.int32)
print('Исходный тензор:', x.item())

x = x.to(torch.float64)
print('Тензор после преобразования к float64:', x.item())
n = 3 if (int(input('Введите номер по списку в ЭИОС: ')) % 2 == 0) else 2

x = x ** n
print(f'Тензор после возведения в степень {n}:', x.item())

random_multiplier = random.randint(1, 10)
x *= random_multiplier
print(f'Тензор после умножения на случайное значение {random_multiplier}:', x.item())

exp_value = torch.exp(x)
print('Тензор после взятия экспоненты:', exp_value.item())

x_grad = torch.tensor(x.item(), dtype=torch.float64, requires_grad=True)
z = torch.exp(x_grad)
z.backward()
print('Производная:', x_grad.grad.item())
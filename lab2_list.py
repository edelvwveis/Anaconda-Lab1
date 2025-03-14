import random

# Создаем список из случайных чисел
list_of_numbers = [random.randint(1, 100) for _ in range(10)]  # Список из 10 случайных чисел от 1 до 100

# 2. Создаем цикл для суммирования четных чисел
sum_of_numbers = 0
for num in list_of_numbers:
    if num % 2 == 0:  # Проверка на четность
        sum_of_numbers += num

# 3. Выводим сумму
print(f"Список чисел:{list_of_numbers}")
print(f"Сумма четных чисел в списке: {sum_of_numbers}")

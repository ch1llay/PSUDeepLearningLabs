import torch
import random
import pandas as pd
import torch.nn as nn


def task_two():
    # выбираем устройство: GPU если доступен, иначе CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print(f'Устройство: {device}')

    df = pd.read_csv('data.csv', header=None)

    X = df.iloc[1:, :4].values.astype(float)
    y = df.iloc[1:, 4].values

    unique_labels = list(set(y))

    label_to_num = {label: i for i, label in enumerate(unique_labels)}
    y_numeric = [label_to_num[label] for label in y]

    # переносим данные на устройство
    X = torch.FloatTensor(X).to(device)
    y = torch.FloatTensor(y_numeric).to(device)

    model = nn.Sequential(
        nn.Linear(4, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
        nn.Sigmoid()
    ).to(device)  # переносим модель на то же устройство

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    size_epoch = 100

    for epoch in range(size_epoch):
        pred = model(X)
        pred = pred.squeeze()
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1}/{size_epoch}], Loss: {loss.item():.4f}')

    with torch.no_grad():
        outputs = model(X).squeeze()
        predicted = (outputs > 0.5).float()
        accuracy = (predicted == y).float().mean()
        print(f'\nТочность: {accuracy * 100:.2f}%')

def task_one():
    x = torch.tensor(random.randint(1, 3), dtype=torch.int32)
    print(f"1. Исходный тензор: {x}, тип: {x.dtype}")

    x = x.to(torch.float32)
    print(f"2. После преобразования: {x}, тип: {x.dtype}")

    n = 2
    x.requires_grad_(True)
    step1 = x ** n
    print(f"3.1. После возведения в степень {n}: {step1}")

    multiplier = random.uniform(1, 3)
    step2 = step1 * multiplier
    print(f"3.2. После умножения на {multiplier:.2f}: {step2}")

    step3 = torch.exp(step2)
    print(f"3.3. После взятия экспоненты: {step3}")

    step3.backward()
    print(f"4. Производная d(результат)/dx: {x.grad}")

if __name__ == '__main__':
    task_one()
    task_two()
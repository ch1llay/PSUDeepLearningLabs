import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class NNet_regression(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size)
        )

    def forward(self, X):
        return self.layers(X)


def train_model(epohs, X, y, net, lossFn, optimizer):
    losses = []

    for i in range(0, epohs):
        pred = net.forward(X)  # прямой проход - делаем предсказания
        loss = lossFn(pred.squeeze(), y)  # считаем ошибу
        optimizer.zero_grad()  # обнуляем градиенты
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if i % 10 == 0:
            print('Ошибка на ' + str(i + 1) + ' итерации: ', loss.item())

    return losses


def predict_income(age, net, X_mean, X_std, y_mean, y_std):
    with torch.no_grad():
        # Нормализуем входной возраст
        age_normalized = (torch.Tensor([[age]]) - X_mean) / X_std
        # Предсказываем на нормализованных данных
        pred_normalized = net.forward(age_normalized)
        # Денормализуем предсказание
        income_pred = pred_normalized * y_std + y_mean
        return income_pred.item()


def evaluate_model(net, X, y_raw, y_mean, y_std):
    with torch.no_grad():
        pred_normalized = net.forward(X)

    # Денормализация предсказаний
    predictions = pred_normalized * y_std + y_mean

    print('\nПервые 10 предсказаний:')
    for i in range(10):
        print(
            f'Возраст: {df.iloc[i, 0]}, Фактический доход: {df.iloc[i, 1]}, Предсказанный доход: {predictions[i].item():.2f}')

    mae = torch.mean(torch.abs(torch.Tensor(y_raw) - predictions.squeeze()))
    print(f'\nСредняя абсолютная ошибка (MAE): {mae.item():.2f}')
    print(f'Относительная ошибка: {(mae.item() / y_mean * 100):.1f}%')

    return predictions


def plot_results(df, predictions):
    plt.figure(figsize=(12, 8))

    sorted_indices = np.argsort(df['age'].values)
    ages_sorted = df['age'].values[sorted_indices]
    incomes_sorted = df['income'].values[sorted_indices]
    predictions_sorted = predictions.detach().numpy()[sorted_indices]

    plt.scatter(ages_sorted, incomes_sorted, alpha=0.7, label='Фактические данные', color='blue')
    plt.plot(ages_sorted, predictions_sorted, label='Предсказания модели', color='red', linewidth=2)

    plt.xlabel('Возраст')
    plt.ylabel('Доход')
    plt.title('Предсказание дохода по возрасту (с нормализацией)')
    plt.legend()
    plt.grid(True)
    plt.savefig('Предсказание дохода по возрасту.jpg')
    plt.show()


    plt.figure(figsize=(10, 6))
    plt.scatter(df['income'].values, predictions.detach().numpy(), alpha=0.7)
    min_val = min(df['income'].min(), predictions.min().item())
    max_val = max(df['income'].max(), predictions.max().item())

    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Идеальная предсказательная линия')
    plt.xlabel('Фактический доход')
    plt.ylabel('Предсказанный доход')
    plt.title('Сравнение фактических и предсказанных значений')
    plt.legend()
    plt.grid(True)
    plt.savefig('Сравнение фактических и предсказанных значений.jpg')
    plt.show()


def plot_losses(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, color='red', linewidth=2)
    plt.xlabel('Эпоха')
    plt.ylabel('Ошибка (MSE)')
    plt.title('Изменение ошибки в процессе обучения')
    plt.grid(True)
    plt.savefig('Изменение ошибки в процессе обучения.jpg')
    plt.show()


def make_predictions(net, X_mean, X_std, y_mean, y_std, test_ages):
    print('\nПредсказания для новых возрастов:')
    for age in test_ages:
        predicted_income = predict_income(age, net, X_mean, X_std, y_mean, y_std)
        print(f'Возраст {age}: предсказанный доход {predicted_income:.2f}')


if __name__ == '__main__':
    input_size = 1
    hidden_size = 32
    output_size = 1
    learning_rate = 0.01
    epochs = 200

    df = pd.read_csv('dataset_simple.csv')

    X_raw = df[['age']].values
    y_raw = df['income'].values

    X_mean, X_std = X_raw.mean(), X_raw.std()
    y_mean, y_std = y_raw.mean(), y_raw.std()

    # нормализация данных, чтобы значения находились в одном диапазоне
    X = torch.Tensor((X_raw - X_mean) / X_std)
    y = torch.Tensor((y_raw - y_mean) / y_std)

    net = NNet_regression(input_size, hidden_size, output_size)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    losses = train_model(epochs, X, y, net, loss_fn, optimizer)

    plot_losses(losses)

    predictions = evaluate_model(net, X, y_raw, y_mean, y_std)

    plot_results(df, predictions)

    # предсказания для новых данных
    test_ages = [20, 25, 30, 35, 40, 45, 50, 55, 60]
    make_predictions(net, X_mean, X_std, y_mean, y_std, test_ages)

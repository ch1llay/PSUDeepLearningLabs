import pandas as pd # библиотека pandas нужна для работы с данными
import matplotlib.pyplot as plt # matplotlib для построения графиков
import numpy as np # numpy для работы с векторами и матрицами
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  нужен для projection='3d'

def neuron(w,x):
    if((w[1]*x[0]+w[2]*x[1]+w[3]*x[2]+w[0])>=0):
        predict = 1
    else: 
        predict = -1
    return predict

def train_perceptron(X, y):
    w = np.random.random(4)
    eta = 0.01  # скорость обучения
    w_iter = [] # пустой список, в него будем добавлять веса, чтобы потом построить график
    for xi, target, j in zip(X, y, range(X.shape[0])):
        predict = neuron(w,xi)
        w[1:] += (eta * (target - predict)) * xi # target - predict - это и есть ошибка
        w[0] += eta * (target - predict)
        # каждую 10ю итерацию будем сохранять набор весов в специальном списке
        if(j%10==0):
            w_iter.append(w.tolist())
    return w, w_iter

def print_error(X, y, w):
    sum_err = 0
    for xi, target in zip(X, y):
        predict = neuron(w,xi)
        sum_err += (target - predict)/2

    print("Всего ошибок: ", sum_err)

    correct = 0
    total = len(X)
    for xi, target in zip(X, y):
        predict = neuron(w, xi)
        if predict == target:
            correct += 1

    accuracy = correct / total * 100
    print(f"\nТочность классификации: {accuracy:.2f}%")
    pass


def main() -> None:
    df = pd.read_csv('data.csv')
    y = df.iloc[:, 4].values
    y = np.where(y == "Iris-setosa", 1, -1)
    X = df.iloc[:, [0, 1, 2]].values

    w, w_iter = train_perceptron(X, y)
    print_error(X, y, w)
    w_final = w  # сохраняем обученные веса (ниже w переиспользуется в цикле)

    # --- 2D: проекция на признаки 1 и 2 + анимация обучения ---
    plt.figure()
    xl = np.linspace(min(X[:, 0]), max(X[:, 0]))

    # сначала рисуем сами точки данных по признакам
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', marker='o')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], color='blue', marker='x')

    for i, w in zip(range(len(w_iter)), w_iter):
        yl = -(xl * w[1] + w[0]) / w[2]  # уравнение линии
        plt.plot(xl, yl)  # строим разделяющую границу
        plt.text(xl[-1], yl[-1], i, dict(size=10, color='gray'))
        plt.pause(1)

    plt.text(xl[-1] - 0.3, yl[-1], 'END', dict(size=14, color='red'))
    plt.xlabel('признак 1')
    plt.ylabel('признак 2')

    # --- 3D: все три признака + разделяющая плоскость ---
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2],
               color='red', marker='o', label='Iris-setosa')
    ax.scatter(X[y == -1, 0], X[y == -1, 1], X[y == -1, 2],
               color='blue', marker='x', label='остальные')

    # граница: w1*x0 + w2*x1 + w3*x2 + w0 = 0
    #   =>  x2 = -(w1*x0 + w2*x1 + w0) / w3
    x0r = np.linspace(min(X[:, 0]), max(X[:, 0]), 20)
    x1r = np.linspace(min(X[:, 1]), max(X[:, 1]), 20)
    xx0, xx1 = np.meshgrid(x0r, x1r)
    xx2 = -(w_final[1] * xx0 + w_final[2] * xx1 + w_final[0]) / w_final[3]
    ax.plot_surface(xx0, xx1, xx2, alpha=0.3, color='green')

    ax.set_xlabel('признак 1')
    ax.set_ylabel('признак 2')
    ax.set_zlabel('признак 3')
    ax.legend()

    plt.show()

if __name__ == '__main__':
    main()
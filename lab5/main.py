import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import random
import copy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time

DATA_PATH = "./custom_dataset"
BATCH_SIZE = 100
EPOCHS = 20
SEED = 42
VAL_RATIO = 0.15            # доля train, уходящая в валидацию
IMAGES_PER_CLASS_VIZ = 3    # сколько примеров каждого класса рисуем


def set_seed(seed=SEED):
    # Без фиксации seed запуски невоспроизводимы (разный split, разная
    # инициализация головы) — для отчёта это обязательно.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_split(targets, val_ratio, seed):
    """Индексы train/val с сохранением баланса классов (на каждый класс
    своя доля в val). Простой random split может перекосить редкий класс."""
    by_class = {}
    for idx, t in enumerate(targets):
        by_class.setdefault(int(t), []).append(idx)

    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for cls, idxs in sorted(by_class.items()):
        idxs = idxs[:]
        rng.shuffle(idxs)
        k = max(1, int(round(len(idxs) * val_ratio)))
        val_idx += idxs[:k]
        train_idx += idxs[k:]
    return train_idx, val_idx


def load_data():
    # AlexNet обучался на ImageNet -> те же mean/std и вход 224x224
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    # train: аугментация = регуляризация. Модель каждый раз видит слегка
    # другую картинку (случайный кроп + отражение) и меньше переобучается.
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm,
    ])
    # val/test: детерминированное преобразование без случайности —
    # чтобы метрика честно измеряла качество, а не «повезло с кропом».
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        norm,
    ])

    train_root = os.path.join(DATA_PATH, 'train')
    test_root = os.path.join(DATA_PATH, 'test')

    # Один и тот же train-каталог читаем дважды: с аугментацией (для
    # обучения) и без неё (для валидации тех же файлов по индексам).
    train_full_aug = torchvision.datasets.ImageFolder(train_root, transform=train_tf)
    train_full_eval = torchvision.datasets.ImageFolder(train_root, transform=eval_tf)
    test_dataset = torchvision.datasets.ImageFolder(test_root, transform=eval_tf)

    class_names = train_full_aug.classes
    num_classes = len(class_names)

    # Валидацию вырезаем ИЗ train. test трогаем только один раз в самом
    # конце — иначе мониторинг по test = утечка и нечестная оценка.
    train_idx, val_idx = stratified_split(train_full_aug.targets, VAL_RATIO, SEED)
    train_dataset = torch.utils.data.Subset(train_full_aug, train_idx)
    val_dataset = torch.utils.data.Subset(train_full_eval, val_idx)

    print(f"Найдены классы: {class_names}")
    print(f"Количество классов: {num_classes}")
    print(f"Размер выборок train/val/test: "
          f"{len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")

    # num_workers>0  - загрузка/transform идут в отдельных процессах
    #                  параллельно обучению, GPU не ждёт CPU;
    # pin_memory     - быстрее копирование RAM -> видеопамять;
    # persistent_workers - не пересоздавать процессы каждую эпоху.
    pin = torch.cuda.is_available()
    loader_kwargs = dict(num_workers=4, pin_memory=pin, persistent_workers=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, class_names, num_classes


def create_model(num_classes):
    # Определим устройство для работы
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")

    # Загружаем предобученную AlexNet
    net = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)

    # Замораживаем веса feature extractor
    for param in net.parameters():
        param.requires_grad = False

    # AlexNet classifier состоит из 7 модулей, индекс 6 - Linear(4096, 1000).
    # Берём все слои кроме последнего и дописываем свой выходной слой.
    new_classifier = net.classifier[:-1]
    new_classifier.add_module('6', nn.Linear(4096, num_classes))
    net.classifier = new_classifier

    net = net.to(device)
    return net, device


def evaluate_model(model, loader, device):
    model.eval()
    use_amp = (device.type == 'cuda')
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / max(total, 1)


def train_model(model, train_loader, val_loader, device, epochs):
    lossFn = nn.CrossEntropyLoss()
    # Оптимизируем ТОЛЬКО размороженные параметры (наш новый слой).
    # momentum=0.9 заметно ускоряет сходимость линейной головы.
    trainable = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(trainable, lr=0.01, momentum=0.9)

    # AMP (mixed precision): ускоряет AlexNet на 224x224 и экономит
    # видеопамять; на CPU автоматически отключается.
    use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    train_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val = -1.0
    best_state = None
    best_epoch = -1

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = lossFn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            batch_count += 1

            if i % 10 == 0:
                print(f'Эпоха {epoch + 1}/{epochs}, Батч {i}, Ошибка: {loss.item():.4f}')

        avg_loss = epoch_loss / batch_count
        train_losses.append(avg_loss)

        # train accuracy меряем eval-проходом (без аугментации в самой
        # метрике он всё равно близок) — нужен, чтобы видеть разрыв
        # train/val, т.е. переобучение.
        train_acc = evaluate_model(model, train_loader, device)
        val_acc = evaluate_model(model, val_loader, device)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Лучшую модель выбираем по валидации, а не по последней эпохе.
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

        print(f'Эпоха {epoch + 1}/{epochs} завершена. '
              f'Ошибка: {avg_loss:.4f}, train: {train_acc:.2f}%, val: {val_acc:.2f}%')
        print('-' * 50)

    training_time = time.time() - start_time
    print(f'Обучение завершено за {training_time:.2f} секунд')

    # Возвращаем веса лучшей по валидации эпохи (а не последней).
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f'Восстановлены веса лучшей эпохи ({best_epoch}), val={best_val:.2f}%')

    return train_losses, train_accuracies, val_accuracies


def visualize_predictions(model, test_loader, class_names, device):
    images_per_class = IMAGES_PER_CLASS_VIZ
    num_classes = len(class_names)
    test_inputs_by_class = {i: [] for i in range(num_classes)}
    test_classes_by_class = {i: [] for i in range(num_classes)}

    # Просматриваем тестовый датасет и набираем по N примеров на класс.
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            for i in range(len(labels)):
                class_idx = labels[i].item()
                if len(test_inputs_by_class[class_idx]) < images_per_class:
                    test_inputs_by_class[class_idx].append(images[i])
                    test_classes_by_class[class_idx].append(labels[i])

            if all(len(test_inputs_by_class[i]) >= images_per_class
                   for i in range(num_classes)):
                break

    test_inputs = []
    test_classes = []
    for class_idx in range(num_classes):
        for j in range(len(test_inputs_by_class[class_idx])):
            test_inputs.append(test_inputs_by_class[class_idx][j])
            test_classes.append(test_classes_by_class[class_idx][j])

    test_inputs = torch.stack(test_inputs)
    test_classes = torch.tensor(test_classes)

    use_amp = (device.type == 'cuda')
    with torch.no_grad():
        test_inputs_gpu = test_inputs.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            predictions = model(test_inputs_gpu)
        _, predicted_classes = torch.max(predictions, 1)
        probabilities = torch.nn.functional.softmax(predictions.float(), dim=1)
    # на CPU для безопасной индексации списка class_names и отрисовки
    predicted_classes = predicted_classes.cpu()
    probabilities = probabilities.cpu()

    num_images = len(test_inputs)
    ncols = images_per_class
    nrows = (num_images + ncols - 1) // ncols

    # constrained_layout сам резервирует место под 3-строчные подписи и
    # suptitle, поэтому текст больше не налезает на картинку соседнего ряда.
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(2.6 * ncols, 3.1 * nrows),
                             constrained_layout=True)

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for idx in range(num_images):
        img = test_inputs[idx].numpy().transpose((1, 2, 0))
        img = std * img + mean
        img = np.clip(img, 0, 1)

        true_label = class_names[test_classes[idx]]
        pred_label = class_names[predicted_classes[idx]]
        confidence = probabilities[idx][predicted_classes[idx]].item()

        axes[idx].imshow(img)
        color = 'green' if true_label == pred_label else 'red'
        axes[idx].set_title(f'Истина: {true_label}\nПредсказание: {pred_label}\n'
                            f'Уверенность: {confidence:.2f}',
                            color=color, fontsize=8, pad=4)
        axes[idx].axis('off')

    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f'Примеры предсказаний (по {images_per_class} изображения от каждого класса)',
                 fontsize=13)
    fig.savefig('predictions.png', dpi=110, bbox_inches='tight')
    plt.show()


def plot_training_results(train_losses, train_accuracies, val_accuracies,
                          test_accuracy, epochs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    xs = range(1, epochs + 1)

    ax1.plot(xs, train_losses, 'b-', marker='o')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Ошибка')
    ax1.set_title('Функция потерь во время обучения')
    ax1.grid(True)

    # Две кривые на одном графике: расхождение train/val = переобучение.
    ax2.plot(xs, train_accuracies, 'g-', marker='o', label='train')
    ax2.plot(xs, val_accuracies, 'r-', marker='o', label='val')
    # test — одно число (меряется 1 раз в конце), рисуем горизонталью.
    ax2.axhline(test_accuracy, color='purple', linestyle='--',
                label=f'test (финал): {test_accuracy:.1f}%')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Точность (%)')
    ax2.set_title('Точность: train / val / test')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()


def save_model(model, path='alexnet_custom_classes.pth'):
    torch.save(model.state_dict(), path)
    print(f"\nМодель сохранена в файл: {path}")


if __name__ == '__main__':
    # нужно для корректного spawn процессов-загрузчиков на Windows
    torch.multiprocessing.freeze_support()
    set_seed(SEED)
    # cuDNN подберёт быстрейшие алгоритмы свёрток под фиксированный
    # размер входа 224x224 (вход не меняется -> даёт прирост).
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader, test_loader, class_names, num_classes = load_data()
    model, device = create_model(num_classes)

    initial_accuracy = evaluate_model(model, val_loader, device)
    print(f'Точность модели до обучения (val): {initial_accuracy:.2f}%')

    print(f"\nНачало обучения на {EPOCHS} эпох")
    train_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, device, EPOCHS)

    # test трогаем РОВНО один раз — это и есть честная оценка обобщения.
    test_accuracy = evaluate_model(model, test_loader, device)
    print(f'\nИтоговая точность на тесте (не использовался в обучении): {test_accuracy:.2f}%')
    print(f'Лучшая точность на валидации: {max(val_accuracies):.2f}%')

    plot_training_results(train_losses, train_accuracies, val_accuracies,
                          test_accuracy, EPOCHS)

    visualize_predictions(model, test_loader, class_names, device)
    save_model(model)

"""Распознавание музыкального инструмента на своих картинках.

Идея: модель уже обучена (main.py сохранил веса в .pth), поэтому здесь
её НЕ обучаем заново — только загружаем и делаем предсказание.

Как пользоваться:
    1. Запусти один раз main.py (он создаст alexnet_custom_classes.pth).
    2. Скачай из интернета свои картинки инструментов и положи их
       в папку lab5/my_images/.
    3. Запусти:  python predict.py

Скрипт выведет предсказанный класс и уверенность для каждой картинки
и сохранит общий коллаж в my_predictions.png.
"""

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

DATA_PATH = "./custom_dataset"
MODEL_PATH = "alexnet_custom_classes.pth"
IMAGES_DIR = "./my_images"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_class_names():
    """ImageFolder в main.py сортирует классы по алфавиту, и сеть училась
    именно в этом порядке. Берём имена классов из того же каталога, чтобы
    индекс выхода сети точно соответствовал нужному инструменту."""
    train_dir = os.path.join(DATA_PATH, "train")
    if not os.path.isdir(train_dir):
        raise SystemExit(f"Нет каталога {train_dir} — сначала запусти split_dataset.py")
    classes = sorted(d.name for d in os.scandir(train_dir) if d.is_dir())
    if not classes:
        raise SystemExit(f"В {train_dir} нет подпапок-классов")
    return classes


def build_model(num_classes, device):
    """Собираем ту же архитектуру, что в обучении (AlexNet с заменённой
    головой), и грузим в неё СВОИ дообученные веса (ImageNet не нужен)."""
    net = torchvision.models.alexnet(weights=None)
    new_classifier = net.classifier[:-1]
    new_classifier.add_module('6', nn.Linear(4096, num_classes))
    net.classifier = new_classifier

    if not os.path.isfile(MODEL_PATH):
        raise SystemExit(f"Нет файла модели {MODEL_PATH} — сначала запусти main.py")
    state = torch.load(MODEL_PATH, map_location=device)
    net.load_state_dict(state)
    return net.to(device).eval()


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

    class_names = get_class_names()
    model = build_model(len(class_names), device)

    # Преобразование ровно как для val/test в main.py (та же нормировка
    # ImageNet) — иначе сеть получит «не те» числа и ошибётся.
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    os.makedirs(IMAGES_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(IMAGES_DIR)
                   if os.path.splitext(f)[1].lower() in IMG_EXTS)
    if not files:
        raise SystemExit(
            f"Положи свои картинки в папку:\n  {os.path.abspath(IMAGES_DIR)}\n"
            f"Форматы: {', '.join(sorted(IMG_EXTS))}. Потом запусти снова.")

    n = len(files)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 4.2),
                             constrained_layout=True)
    if n == 1:
        axes = [axes]

    print(f"\nНайдено картинок: {n}")
    print("-" * 60)
    for ax, fname in zip(axes, files):
        # .convert('RGB') — скачанные картинки бывают RGBA / серые / CMYK,
        # а сеть ждёт ровно 3 канала.
        img = Image.open(os.path.join(IMAGES_DIR, fname)).convert('RGB')
        x = tf(img).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]

        conf, idx = probs.max(0)
        label = class_names[idx.item()]

        # Топ-3 — чтобы видеть, в чём сеть «сомневалась».
        top3 = torch.topk(probs, k=min(3, len(class_names)))
        top3_str = ", ".join(f"{class_names[i]} {p * 100:.1f}%"
                             for p, i in zip(top3.values, top3.indices))
        print(f"{fname:<28} -> {label:<12} ({conf.item() * 100:.1f}%)  "
              f"| топ-3: {top3_str}")

        ax.imshow(img)
        ax.set_title(f"{fname}\n{label}  ({conf.item() * 100:.0f}%)",
                     fontsize=10)
        ax.axis('off')

    print("-" * 60)
    fig.suptitle("Распознавание инструментов на своих картинках из интернета",
                 fontsize=13)
    fig.savefig("my_predictions.png", dpi=110, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()

"""Стратифицированное разбиение датасета music_instruments на train/test.

Берёт каждый класс из music_instruments/ и раскладывает его файлы
по custom_dataset/train/<class>/ и custom_dataset/test/<class>/
в заданном соотношении (по умолчанию 80/20), отдельно для каждого класса.

Запуск (из папки lab5):
    python split_dataset.py
или с параметрами:
    python split_dataset.py --ratio 0.8 --seed 42
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

# Windows-консоль по умолчанию не в UTF-8 → русский текст превращается
# в кракозябры. Переключаем поток вывода, если это возможно.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Разбиение датасета на train/test")
    parser.add_argument("--src", default="music_instruments",
                        help="папка с исходными классами (относительно скрипта)")
    parser.add_argument("--dst", default="custom_dataset",
                        help="папка для результата (относительно скрипта)")
    parser.add_argument("--ratio", type=float, default=0.8,
                        help="доля train (0..1), остальное в test")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed для воспроизводимого перемешивания")
    return parser.parse_args()


def main():
    args = parse_args()

    src_dir = (SCRIPT_DIR / args.src).resolve()
    dst_dir = (SCRIPT_DIR / args.dst).resolve()

    if not src_dir.is_dir():
        raise SystemExit(f"Не найдена папка с датасетом: {src_dir}")
    if not 0.0 < args.ratio < 1.0:
        raise SystemExit(f"--ratio должен быть в (0, 1), а не {args.ratio}")

    # Защита от грязного выхода: убираем старый результат целиком,
    # чтобы не смешать файлы прошлых запусков.
    if dst_dir.exists():
        print(f"[!] Папка {dst_dir} уже существует — удаляю её перед раскладкой.")
        shutil.rmtree(dst_dir)

    # Только подпапки = классы; одиночный .csv в корне сюда не попадёт.
    class_dirs = sorted(p for p in src_dir.iterdir() if p.is_dir())
    if not class_dirs:
        raise SystemExit(f"В {src_dir} нет подпапок-классов")

    rng = random.Random(args.seed)

    totals = {"train": 0, "test": 0}
    print(f"Источник : {src_dir}")
    print(f"Результат: {dst_dir}")
    print(f"Соотношение train/test: {args.ratio:.0%}/{1 - args.ratio:.0%}, seed={args.seed}")
    print("-" * 52)
    print(f"{'класс':<14}{'train':>8}{'test':>8}{'всего':>9}")
    print("-" * 52)

    for class_dir in class_dirs:
        files = sorted(f for f in class_dir.iterdir()
                       if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
        if not files:
            print(f"{class_dir.name:<14}{'— пусто, пропускаю —':>25}")
            continue

        rng.shuffle(files)

        n_train = round(len(files) * args.ratio)
        # Гарантируем хотя бы 1 файл в test и хотя бы 1 в train.
        n_train = max(1, min(n_train, len(files) - 1))

        split = {
            "train": files[:n_train],
            "test": files[n_train:],
        }

        for subset, subset_files in split.items():
            out_dir = dst_dir / subset / class_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in subset_files:
                shutil.copy2(f, out_dir / f.name)
            totals[subset] += len(subset_files)

        print(f"{class_dir.name:<14}{len(split['train']):>8}"
              f"{len(split['test']):>8}{len(files):>9}")

    print("-" * 52)
    grand_total = totals["train"] + totals["test"]
    print(f"{'ИТОГО':<14}{totals['train']:>8}{totals['test']:>8}{grand_total:>9}")

    # Проверка: набор классов в train и test должен совпадать —
    # именно это ожидает torchvision.datasets.ImageFolder в main.py.
    train_classes = sorted(p.name for p in (dst_dir / "train").iterdir() if p.is_dir())
    test_classes = sorted(p.name for p in (dst_dir / "test").iterdir() if p.is_dir())
    if train_classes == test_classes:
        print(f"\n[OK] Классы train и test совпадают ({len(train_classes)} шт.):")
        print("     " + ", ".join(train_classes))
        print(f"\nГотово. Теперь можно запускать main.py (он ждёт ./{args.dst}).")
    else:
        print("\n[!] ВНИМАНИЕ: наборы классов train/test не совпадают:")
        print(f"    только в train: {sorted(set(train_classes) - set(test_classes))}")
        print(f"    только в test : {sorted(set(test_classes) - set(train_classes))}")


if __name__ == "__main__":
    main()

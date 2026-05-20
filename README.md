# PSUDeepLearningLabs

Лабораторные работы по курсу **«Глубокое обучение»**, 3 семестр магистратуры ПГУ.

Все работы выполнены на Python 3.13 / PyTorch (CUDA 12.6) в среде `dnnlab`. Каждая папка — самостоятельный мини-проект с кодом, входными данными и (где есть) графиками результатов.

## Содержание

| № | Папка | Тема | Что изучено |
|---|---|---|---|
| 1 | [lab1](lab1) | Установка и настройка среды | conda-окружение `dnnlab`, проверка работоспособности Python / NumPy / matplotlib |
| 2 | [lab2](lab2) | Основы Python + однонейронный перцептрон | список ↔ функции, NumPy, ручная реализация перцептрона Розенблатта на датасете Iris (1 vs остальные) |
| 3 | [lab3](lab3) | Основы PyTorch | тензоры, autograd, обучение простой `nn.Sequential` модели на Iris |
| 4 | [lab4](lab4) | Полносвязная сеть для регрессии | предсказание дохода по возрасту, MSE, нормализация входов/выходов, графики потерь и качества |
| 5 | [lab5](lab5) | Классификация изображений (CNN, transfer learning) | дообучение **AlexNet** на собственном датасете музыкальных инструментов; stratified split, аугментация, AMP, отдельный скрипт предсказания на «своих» картинках |
| 6 | [lab6](lab6) | No-code платформы машинного обучения | обзор и сравнение готовых ML-сервисов (отчёт в `lab6.docx`) |
| 7 | [lab7](lab7) | Детекция объектов (YOLOv8) | дообучение **YOLOv8** на собственной разметке котов, инференс на тестовых изображениях |
| 8 | [lab8](lab8) | Локальная LLM | запуск **Qwen2.5-7B-Instruct-1M** через `transformers`, суммаризация и Q&A по англоязычной статье, два варианта промптов |

## Что в каждой лабе

### [lab1 — Установка среды](lab1)
- [environment.yml](lab1/environment.yml) — полный экспорт conda-окружения `dnnlab` (Python 3.13, NumPy, matplotlib, PyTorch и др.).
- [test.py](lab1/test.py) — проверочный скрипт: импортирует пакеты и печатает имя активного окружения.
- [result.txt](lab1/result.txt) — лог удачного запуска.

### [lab2 — Python и нейрон](lab2)
- [Lab2_python_basics.py](lab2/Lab2_python_basics.py) — генерация списка случайных чисел и сумма чётных.
- [lab2_neuron.py](lab2/lab2_neuron.py) — собственная реализация одного нейрона с пороговой функцией активации, обучение правилом перцептрона, визуализация разделяющей прямой (2D) и плоскости (3D) на признаках Iris.
- [data.csv](lab2/data.csv) — датасет Iris.

### [lab3 — Основы PyTorch](lab3)
- [main.py](lab3/main.py):
  - `task_one` — создание тензора, `requires_grad`, цепочка операций (степень → умножение → exp) и автоматическое дифференцирование через `.backward()`.
  - `task_two` — простая `Sequential`-сеть (Linear → ReLU → Linear → Sigmoid) для бинарной классификации Iris с оптимизатором Adam.

### [lab4 — Регрессия на PyTorch](lab4)
- [main.py](lab4/main.py) — модель `NNet_regression` (две скрытых ReLU-полносвязки), обучение через `MSELoss` + SGD, нормализация признаков и таргета, денормализация предсказаний.
- Сохранённые графики:
  - [Изменение ошибки в процессе обучения.jpg](lab4/Изменение%20ошибки%20в%20процессе%20обучения.jpg)
  - [Предсказание дохода по возрасту.jpg](lab4/Предсказание%20дохода%20по%20возрасту.jpg)
  - [Сравнение фактических и предсказанных значений.jpg](lab4/Сравнение%20фактических%20и%20предсказанных%20значений.jpg)

### [lab5 — Image classification (AlexNet)](lab5)
Transfer learning: ImageNet-предобученный AlexNet, замороженный feature extractor, дообученная классификационная голова под собственные классы музыкальных инструментов.

- [split_dataset.py](lab5/split_dataset.py) — стратифицированное разбиение `music_instruments/` на `custom_dataset/train` и `custom_dataset/test` (80/20, фиксированный seed).
- [main.py](lab5/main.py) — обучение: аугментация (RandomResizedCrop + Flip), внутренняя валидация (15 % от train, стратифицированно), SGD + momentum, mixed precision (`torch.autocast` + `GradScaler`), выбор лучшей модели по val-метрике, тест ровно один раз в конце.
- [predict.py](lab5/predict.py) — инференс уже обученной модели на пользовательских картинках из [my_images/](lab5/my_images), вывод топ-3 классов и коллажа.
- [alexnet_custom_classes.pth](lab5/alexnet_custom_classes.pth) — веса дообученной модели.
- Графики: [training_results.png](lab5/training_results.png), [predictions.png](lab5/predictions.png), [my_predictions.png](lab5/my_predictions.png).

### [lab6 — No-code платформы](lab6)
- [lab6.docx](lab6/lab6.docx) — отчёт: обзор no-code/low-code ML-инструментов, сравнение возможностей.

### [lab7 — Object detection (YOLOv8)](lab7)
- [main.py](lab7/main.py) — обучение `YOLOv8n` на собственной разметке (закомментировано) и инференс лучшими весами (`masks/train4/weights/best.pt`) на тестовых картинках котов из [cats/](lab7/cats), запись аннотированных результатов в [result/](lab7/result).
- [masked.yaml](lab7/masked.yaml) — конфиг датасета (`nc: 1`, класс `cat`).
- [images/](lab7/images), [labels/](lab7/labels), [masks/](lab7/masks) — изображения, разметка YOLO-формата и веса обученной модели.

### [lab8 — Локальная LLM](lab8)
- [main.py](lab8/main.py) — загрузка `Qwen/Qwen2.5-7B-Instruct-1M` через `transformers` (`torch.bfloat16`, `device_map="auto"`), чат-шаблон, генерация ответа на два разных промпта по одной и той же статье, запись отчёта в `summary_report.txt`. Запускался на Kaggle (GPU T4 ×2).
- [ENG_article.txt](lab8/ENG_article.txt) — англоязычная статья (история нейронных сетей).
- [prompt1.txt](lab8/prompt1.txt) — простой пользовательский промпт с тремя вопросами по тексту.
- [prompt2.txt](lab8/prompt2.txt) — расширенный промпт с ролью «точный аналитик», требованием цитировать текст и явно отмечать отсутствие информации.
- [summary_report.txt](lab8/summary_report.txt) — итоговый отчёт со сравнением ответов модели на оба промпта.

## Запуск

Все работы используют общее окружение `dnnlab` из [lab1/environment.yml](lab1/environment.yml):

```bash
conda env create -f lab1/environment.yml
conda activate dnnlab
```

Дополнительно по лабам:
- **lab5** — установить `torchvision` (входит в `dnnlab`); датасет `music_instruments/` ожидается рядом со скриптом.
- **lab7** — `pip install ultralytics opencv-python`; веса `yolov8s.pt` подтянутся автоматически.
- **lab8** — `pip install transformers accelerate`; модель ~15 ГБ, удобнее запускать в Kaggle (GPU T4 ×2) или Colab, а не локально.

Каждую лабу запускать из её папки:

```bash
cd lab3
python main.py
```

## Стек

- **Язык/среда:** Python 3.13, conda env `dnnlab`, Spyder / VS Code.
- **Базовые библиотеки:** NumPy, pandas, matplotlib.
- **DL-фреймворки:** PyTorch (CUDA 12.6), torchvision, Ultralytics YOLOv8, HuggingFace `transformers`.
- **Аппаратное:** локально — NVIDIA GPU с CUDA 12.6; lab8 запускалась на Kaggle (Tesla T4).

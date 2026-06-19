# Vzglyad-Detect Training Module

Русский | [English](#english)

---

<a id="русский"></a>
## Русский

Этот модуль содержит переписанный с нуля код для обучения моделей архитектуры `vzglyad-detect`, **не использует** код Ultralytics под лицензией GPL-3 / AGPL-3. 

Весь код написан с использованием стандартных библиотек (`torch`, `opencv-python`, `numpy`) и может свободно использоваться в коммерческих проектах без необходимости открывать исходный код вашего приложения. <br><br>

Для запуска поместите всё по данной структуре (или скачайте из релизов):
```text
Vzglyad/
├── inference (скачайте https://github.com/AlexeyRF/VZGLYAD-Detect)/
│   ├── example.py
│   ├── example_on_screen.py
│   ├── README.md
│   ├── LICENSE
│   └── vzglyad_vision/
│       ├── __init__.py
│       ├── blocks.py
│       ├── inference.py 
│       ├── model.py
│       └── utils.py
└── train/ (эти коды)
    ├── dataset.py
    ├── loss.py
    ├── metrics.py
    ├── README.md
    ├── train.py
    └── universal_model.yaml
```
### Особенности
- **Своя архитектура** (`universal_model.yaml`)
- **Встроенные аугментации**
- **Target Assignment без GPL**: Написана кастомная функция потерь (`loss.py`), которая использует пространственное сопоставление (Spatial Target Assignment), GIoU и BCE Loss.

### Структура
- `dataset.py` - загрузчик данных. Реализует аугментации и `letterbox` паддинг.
- `loss.py` - реализация функции потерь. 
- `train.py` - основной скрипт цикла обучения. Загружает архитектуру из `yaml` словаря или из `.pth` модели.
- `universal_model.yaml` - конфиг архитектуры для старта обучения с нуля.

### Как использовать

#### 1. Подготовка данных
Данные должны быть в стандартном формате:
```text
dataset/
├── images/
│   ├── image1.jpg
│   └── image2.jpg
└── labels/
    ├── image1.txt
    └── image2.txt
```
*Note: Если у вас нет готовой структуры с train val, то скрипт сам разделит их 80/20. соотвтетственно*
#### 2. Запуск обучения

**Вариант А: Обучение с нуля из словаря YAML (универсальная модель)**
```bash
python train.py --data-dir path/to/dataset --cfg universal_model.yaml --scale n --epochs 100 --batch-size 16
```

**Вариант Б: Дообучение базовой модели (.pth)**
```bash
python train.py --data-dir path/to/dataset --model-path ../inference/your_model.pth --epochs 100
```

**Доступные аргументы:**
- `--data-dir` (обязательный): путь к папке с датасетом.
- `--cfg`: путь к YAML конфигу для старта с нуля.
- `--model-path`: путь к базовой модели `.pth` для дообучения.
- `--scale`: масштаб модели (например, `n`, `s`, `m`, `l`, `x`), если используется `--cfg`.
- `--no-augment`: отключить встроенные аугментации.
- `--epochs`: количество эпох (по умолчанию 100).
- `--batch-size`: размер батча (по умолчанию 16).
- `--img-size`: размер входного изображения (по умолчанию 640).
- `--lr`: learning rate (по умолчанию 0.001).

Обученная модель сохраняется в папку `runs/train/`.

---

<a id="english"></a>
<a id="english"></a>
## English

This module contains a written-from-scratch training code for the `vzglyad-detect` architecture, **not using** Ultralytics code under the GPL-3 / AGPL-3 license.

All code is written using standard libraries (`torch`, `opencv-python`, `numpy`) and can be freely used in commercial projects without the requirement to open-source your application.<br><br>
To run, place everything in this structure (or download from the releases):
```text
Vzglyad/
├── inference (download https://github.com/AlexeyRF/VZGLYAD-Detect)/
│   ├── example.py
│   ├── example_on_screen.py
│   ├── README.md
│   ├── LICENSE
│   └── vzglyad_vision/
│       ├── __init__.py
│       ├── blocks.py
│       ├── inference.py 
│       ├── model.py
│       └── utils.py
└── train/ (this repo)
    ├── dataset.py
    ├── loss.py
    ├── metrics.py
    ├── README.md
    ├── train.py
    └── universal_model.yaml
```
### Features
- **Custom Architecture Layout** (`universal_model.yaml`)
- **Built-in Augmentations & Auto-split**: Automatically applies advanced data augmentations on the fly. Automatically splits the dataset 80/20 into train/val if explicit split folders are missing.
- **GPL-free Target Assignment**: A custom loss function (`loss.py`) is implemented using Spatial Target Assignment, GIoU, and BCE Loss.
- **Metrics & Plots**: Automatically generates validation metrics (Precision, Recall, mAP@0.5) and dataset/training plots.

### Structure
- `dataset.py` - Dataloader and Dataset implementation. Handles on-the-fly augmentations, auto-split, and `letterbox` padding.
- `loss.py` - Loss function implementation. 
- `train.py` - Main training loop script. Dynamically loads architecture from a `yaml` dictionary or a base `.pth` model.
- `universal_model.yaml` - A configuration template for training custom detection models from scratch.

### How to use

#### 1. Data Preparation
Data must be in the standard format:
```text
dataset/
├── images/
│   ├── image1.jpg
│   └── image2.jpg
└── labels/
    ├── image1.txt
    └── image2.txt
```
*Note: If you don't have separate `images/train` and `images/val` folders, the script will automatically split the dataset 80/20.*

#### 2. Running Training

**Option A: Train from scratch using a YAML dictionary (universal model)**
```bash
python train.py --data-dir path/to/dataset --cfg universal_model.yaml --scale n --epochs 100 --batch-size 16
```

**Option B: Fine-tune from a base `.pth` model**
```bash
python train.py --data-dir path/to/dataset --model-path ../inference/your_model.pth --epochs 100
```

**Available Arguments:**
- `--data-dir` (required): Path to the dataset directory.
- `--cfg`: Path to a YAML configuration file to train from scratch.
- `--model-path`: Path to a base `.pth` model to fine-tune.
- `--scale`: Model scale parameter (e.g., `n`, `s`, `m`, `l`, `x`) when using `--cfg`.
- `--no-augment`: Disable built-in data augmentations.
- `--epochs`: Number of epochs (default: 100).
- `--batch-size`: Batch size (default: 16).
- `--img-size`: Input image size (default: 640).
- `--lr`: Learning rate (default: 0.001).

The trained model, plots, and dataset statistics are saved in the `runs/train/` directory.

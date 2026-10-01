"""Скачивает и раскладывает реальные наборы данных для работы.

banknote.csv, predict_examples.csv - Banknote Authentication (UCI):
    признаки снимков настоящих и поддельных банкнот (часть 1);
images/digit_N/ - рукописные цифры Optical Recognition of Handwritten
    Digits (UCI, входит в scikit-learn) в виде PNG (части 2 и 3);
tables/ - Iris (UCI), по одному CSV на каждый вид ириса (часть 2).

Данные уже лежат в репозитории, скрипт нужен только чтобы
получить их заново. Требуется scikit-learn.

Запуск из корня проекта:  python get_datasets.py --output data/input
"""

import argparse
import os
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.datasets import load_digits, load_iris

BANKNOTE_URL = os.getenv(
    "BANKNOTE_URL",
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "banknote_authentication.csv",
)
BANKNOTE_HEADER = "variance,skewness,curtosis,entropy,class"


def save_banknote(folder, url, holdout):
    """Скачивает банкноты, часть строк откладывает для предсказания.

    :param folder: папка для сохранения
    :param url: адрес CSV-файла
    :param holdout: сколько строк отложить в predict_examples.csv
    """
    with urllib.request.urlopen(url, timeout=30) as response:
        lines = response.read().decode("utf-8").split()
    rows = [line.strip() for line in lines if line.strip()]
    random = np.random.default_rng(1)
    picked = set(random.choice(len(rows), holdout, replace=False).tolist())
    train_rows = [row for number, row in enumerate(rows)
                  if number not in picked]
    test_rows = [rows[number] for number in sorted(picked)]
    (folder / "banknote.csv").write_text(
        BANKNOTE_HEADER + "\n" + "\n".join(train_rows) + "\n",
        encoding="utf-8")
    # в файле для предсказания нет столбца с ответом
    header = BANKNOTE_HEADER.rsplit(",", 1)[0]
    features = [row.rsplit(",", 1)[0] for row in test_rows]
    answers = [row.rsplit(",", 1)[1] for row in test_rows]
    (folder / "predict_examples.csv").write_text(
        header + "\n" + "\n".join(features) + "\n", encoding="utf-8")
    print(f"banknote.csv: {len(train_rows)} строк, отложено {holdout} "
          f"(правильные классы: {', '.join(answers)})")


def save_digits(folder, per_class):
    """Сохраняет рукописные цифры как PNG в папки digit_0 ... digit_9.

    :param per_class: сколько картинок каждой цифры сохранить
    """
    digits = load_digits()
    counters = {}
    for image, label in zip(digits.images, digits.target):
        if counters.get(label, 0) >= per_class:
            continue
        counters[label] = counters.get(label, 0) + 1
        sub_folder = folder / f"digit_{label}"
        sub_folder.mkdir(parents=True, exist_ok=True)
        # яркость в наборе от 0 до 16, переводим в 0..255
        pixels = (image / 16.0 * 255).astype(np.uint8)
        Image.fromarray(pixels, mode="L").save(
            sub_folder / f"{label}_{counters[label]:03d}.png")
    print(f"images: {sum(counters.values())} картинок цифр")


def save_iris(folder):
    """Сохраняет ирисы в три CSV-файла (по виду цветка)."""
    iris = load_iris()
    folder.mkdir(parents=True, exist_ok=True)
    header = "sepal_length,sepal_width,petal_length,petal_width"
    for code, name in enumerate(iris.target_names):
        part = iris.data[iris.target == code]
        np.savetxt(folder / f"iris_{name}.csv", part, delimiter=",",
                   header=header, comments="", fmt="%.1f")
    print(f"tables: {len(iris.target_names)} файла с ирисами")


def main():
    """Разбирает аргументы и сохраняет все наборы."""
    parser = argparse.ArgumentParser(description="Загрузка наборов данных")
    parser.add_argument("--output", default=os.getenv("DATA_PATH",
                                                      "data/input"))
    parser.add_argument("--per-class", type=int, default=30,
                        help="сколько картинок каждой цифры сохранить")
    parser.add_argument("--holdout", type=int, default=6,
                        help="сколько банкнот отложить для предсказания")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_banknote(output, BANKNOTE_URL, args.holdout)
    save_digits(output / "images", args.per_class)
    save_iris(output / "tables")


if __name__ == "__main__":
    main()

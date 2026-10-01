"""Функции аугментации картинок.

Все функции работают с массивами NumPy формы (высота, ширина, 3)
со значениями от 0 до 1. Функция augment_image вызывается
в отдельных процессах, поэтому она объявлена на уровне модуля.
"""

import os
import zlib

import numpy as np


def rotate(image, angle_deg):
    """Поворачивает картинку на произвольный угол вокруг центра.

    Для каждой точки новой картинки берется ближайшая точка
    исходной. Углы, куда ничего не попало, остаются черными.

    :param image: массив (h, w, 3)
    :param angle_deg: угол поворота в градусах
    :return: повернутый массив той же формы
    """
    height, width = image.shape[:2]
    angle = np.deg2rad(angle_deg)
    center_y = (height - 1) / 2
    center_x = (width - 1) / 2
    rows, cols = np.mgrid[0:height, 0:width]
    # откуда в исходной картинке берется каждая точка
    source_x = (np.cos(angle) * (cols - center_x)
                + np.sin(angle) * (rows - center_y) + center_x)
    source_y = (-np.sin(angle) * (cols - center_x)
                + np.cos(angle) * (rows - center_y) + center_y)
    source_x = np.round(source_x).astype(int)
    source_y = np.round(source_y).astype(int)
    inside = ((source_x >= 0) & (source_x < width)
              & (source_y >= 0) & (source_y < height))
    result = np.zeros_like(image)
    result[inside] = image[source_y[inside], source_x[inside]]
    return result


def flip(image, horizontal=True):
    """Отражает картинку по горизонтали или по вертикали."""
    return image[:, ::-1] if horizontal else image[::-1, :]


def add_noise(image, strength, random):
    """Добавляет случайный (гауссов) шум.

    :param strength: стандартное отклонение шума
    :param random: генератор случайных чисел
    """
    noise = random.normal(0.0, strength, image.shape)
    return np.clip(image + noise, 0.0, 1.0)


def change_brightness(image, factor):
    """Делает картинку светлее (factor > 1) или темнее (factor < 1)."""
    return np.clip(image * factor, 0.0, 1.0)


def blur(image, passes=2):
    """Размывает картинку средним по соседним пикселям (3x3).

    :param passes: сколько раз повторить размытие
    """
    result = image
    for _ in range(passes):
        padded = np.pad(result, ((1, 1), (1, 1), (0, 0)), mode="edge")
        height, width = result.shape[:2]
        total = np.zeros_like(result)
        # складываем картинку, сдвинутую на 1 пиксель во все стороны
        for shift_y in range(3):
            for shift_x in range(3):
                total += padded[shift_y:shift_y + height,
                                shift_x:shift_x + width]
        result = total / 9.0
    return result


def file_seed(name, base_seed):
    """Считает зерно для генератора по имени файла.

    Так одна и та же картинка получает одинаковые изменения
    и в последовательной, и в параллельной версии.
    """
    return (zlib.crc32(name.encode("utf-8")) + base_seed) % (2 ** 32)


def augment_image(name, image, copies, base_seed):
    """Создает несколько измененных копий одной картинки.

    Каждая копия: поворот, отражение, яркость, шум, иногда размытие.

    :param name: имя файла (нужно для зерна и отчета)
    :param image: исходный массив (h, w, 3)
    :param copies: сколько копий сделать
    :param base_seed: общее зерно
    :return: кортеж (имя, список массивов: оригинал + копии,
             номер процесса, который делал работу)
    """
    random = np.random.default_rng(file_seed(name, base_seed))
    results = [image.astype(np.float32)]
    for _ in range(copies):
        augmented = rotate(image, random.uniform(-30, 30))
        if random.random() < 0.5:
            augmented = flip(augmented, horizontal=True)
        if random.random() < 0.2:
            augmented = flip(augmented, horizontal=False)
        augmented = change_brightness(augmented, random.uniform(0.7, 1.3))
        augmented = add_noise(augmented, 0.03, random)
        if random.random() < 0.3:
            augmented = blur(augmented)
        results.append(augmented.astype(np.float32))
    return name, results, os.getpid()

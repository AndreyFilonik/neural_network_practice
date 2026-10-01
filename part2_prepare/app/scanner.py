"""Обход каталога и преобразование файлов в массивы NumPy."""

import logging
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
TABLE_EXTENSIONS = {".csv"}


class PrepareError(Exception):
    """Общая ошибка подготовки данных."""


class NoFilesFoundError(PrepareError):
    """В каталоге не найдено файлов с нужными расширениями."""


class MixedDataError(PrepareError):
    """Картинки и таблицы нельзя сложить в один массив."""


def normalize_extensions(raw_extensions):
    """Приводит расширения к виду '.png' (нижний регистр, с точкой).

    Можно передать список ['png', 'jpg'] или строку 'png,jpg'.

    :param raw_extensions: список строк
    :return: множество расширений
    """
    result = set()
    for item in raw_extensions:
        for part in item.split(","):
            part = part.strip().lower()
            if part:
                result.add(part if part.startswith(".") else f".{part}")
    return result


def find_files(root, extensions):
    """Рекурсивно ищет файлы с нужными расширениями.

    :param root: папка, с которой начинается поиск
    :param extensions: множество расширений
    :return: отсортированный список путей
    """
    root = Path(root)
    if not root.is_dir():
        raise PrepareError(f"Папка не найдена: {root}")
    files = [path for path in root.rglob("*")
             if path.is_file() and path.suffix.lower() in extensions]
    logger.debug("В папке %s найдено файлов: %d", root, len(files))
    return sorted(files)


def load_image(path, image_size):
    """Читает картинку, приводит к RGB и одному размеру, делит на 255.

    :param path: путь к файлу
    :param image_size: сторона квадрата, к которому приводится картинка
    :return: массив float32 формы (size, size, 3) со значениями 0..1
    """
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size))
        return np.asarray(image, dtype=np.float32) / 255.0


def load_table(path):
    """Читает CSV-файл (первая строка - заголовок).

    :param path: путь к файлу
    :return: двумерный массив float32
    """
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2,
                      dtype=np.float32)


def min_max_normalize(table):
    """Приводит каждый столбец таблицы к диапазону от 0 до 1.

    :param table: двумерный массив
    :return: нормализованный массив
    """
    column_min = table.min(axis=0)
    column_range = table.max(axis=0) - column_min
    column_range[column_range == 0] = 1.0
    return (table - column_min) / column_range


def detect_data_kind(files):
    """Определяет, что лежит в списке: картинки или таблицы.

    :return: строка "image" или "table"
    """
    suffixes = {path.suffix.lower() for path in files}
    has_images = bool(suffixes & IMAGE_EXTENSIONS)
    has_tables = bool(suffixes & TABLE_EXTENSIONS)
    if has_images and has_tables:
        raise MixedDataError(
            "Найдены и картинки, и CSV. Запустите prepare отдельно "
            "для каждого типа (--ext png jpg или --ext csv)"
        )
    if not has_images and not has_tables:
        raise PrepareError("Поддерживаются только картинки и CSV-файлы")
    return "image" if has_images else "table"


def backup_existing(output_file):
    """Если файл результата уже есть, переименовывает его в .bak.

    :param output_file: путь к файлу результата
    """
    if output_file.exists():
        backup = output_file.with_name(output_file.name + ".bak")
        shutil.move(str(output_file), str(backup))
        logger.info("Старый файл перенесен в %s", backup)


def prepare_dataset(root, extensions, output_file, image_size):
    """Главная функция команды prepare.

    :param root: папка с исходными файлами
    :param extensions: множество расширений
    :param output_file: путь к итоговому .npy
    :param image_size: размер, к которому приводятся картинки
    :return: словарь со статистикой
    """
    start = time.perf_counter()
    files = find_files(root, extensions)
    if not files:
        raise NoFilesFoundError(
            f"В папке {root} нет файлов с расширениями "
            f"{', '.join(sorted(extensions))}"
        )
    kind = detect_data_kind(files)
    arrays = []
    skipped = []
    for path in files:
        try:
            if kind == "image":
                arrays.append(load_image(path, image_size))
            else:
                arrays.append(load_table(path))
            logger.debug("Прочитан файл %s", path)
        except (OSError, ValueError, UnidentifiedImageError) as error:
            logger.warning("Файл %s пропущен: %s", path, error)
            skipped.append(str(path))
    if not arrays:
        raise PrepareError("Ни один файл не удалось прочитать")

    if kind == "image":
        dataset = np.stack(arrays)
    else:
        columns = {array.shape[1] for array in arrays}
        if len(columns) != 1:
            raise PrepareError(
                f"У CSV-файлов разное число столбцов: {sorted(columns)}"
            )
        dataset = min_max_normalize(np.vstack(arrays))

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    free_space = shutil.disk_usage(output_file.parent).free
    if dataset.nbytes > free_space:
        raise PrepareError("Недостаточно места на диске для сохранения")
    backup_existing(output_file)
    np.save(output_file, dataset)

    total_size = sum(path.stat().st_size for path in files)
    return {
        "kind": kind,
        "files_found": len(files),
        "files_loaded": len(arrays),
        "files_skipped": len(skipped),
        "source_size_kb": round(total_size / 1024, 1),
        "array_shape": dataset.shape,
        "array_size_kb": round(dataset.nbytes / 1024, 1),
        "value_range": (float(dataset.min()), float(dataset.max())),
        "output_file": str(output_file),
        "elapsed_sec": round(time.perf_counter() - start, 3),
    }

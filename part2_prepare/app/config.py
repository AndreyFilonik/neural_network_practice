"""Настройки программы.

Порядок: аргумент командной строки -> переменная окружения
(или файл .env) -> значение по умолчанию.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import find_dotenv, load_dotenv
except ImportError:  # библиотека не установлена - работаем без .env
    load_dotenv = None

DEFAULTS = {
    "DATA_PATH": "../data/input",
    "OUTPUT_DIR": "../data/output",
    "LOG_DIR": "../logs",
    "LOG_LEVEL": "INFO",
    "IMAGE_SIZE": "64",
}


@dataclass
class Settings:
    """Набор настроек программы."""

    data_path: Path
    output_dir: Path
    log_dir: Path
    log_level: str
    image_size: int


def read_env(name):
    """Возвращает переменную окружения или значение по умолчанию.

    :param name: имя переменной
    :return: строка со значением
    """
    return os.getenv(name, DEFAULTS[name])


def load_settings():
    """Загружает .env (если есть) и собирает настройки.

    Файл .env ищется в текущей папке и выше по дереву,
    поэтому можно держать один .env в корне проекта.

    :return: объект Settings
    """
    if load_dotenv is not None:
        load_dotenv(find_dotenv(usecwd=True))
    try:
        image_size = int(read_env("IMAGE_SIZE"))
    except ValueError:
        image_size = int(DEFAULTS["IMAGE_SIZE"])
        logging.warning("IMAGE_SIZE должен быть числом, беру %d",
                        image_size)
    return Settings(
        data_path=Path(read_env("DATA_PATH")),
        output_dir=Path(read_env("OUTPUT_DIR")),
        log_dir=Path(read_env("LOG_DIR")),
        log_level=read_env("LOG_LEVEL").upper(),
        image_size=image_size,
    )

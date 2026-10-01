"""Настройка логирования: вывод в консоль и в файл."""

import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(level="INFO", log_dir="logs", file_name="prepare.log"):
    """Настраивает корневой логгер.

    :param level: уровень логирования (DEBUG, INFO, WARNING, ERROR)
    :param log_dir: папка для файла лога
    :param file_name: имя файла лога
    :return: путь к файлу лога
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / file_name
    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format=LOG_FORMAT,
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )
    # сторонние библиотеки пишут слишком много отладочных сообщений
    logging.getLogger("PIL").setLevel(logging.WARNING)
    return log_file

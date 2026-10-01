"""Утилита командной строки: команды prepare и doctor.

Запуск из папки part2_prepare:
    python -m app.cli prepare --path ../data/input/images --ext png jpg
    python -m app.cli doctor
"""

import argparse
import importlib.util
import logging
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from app.config import load_settings
from app.logging_config import setup_logging
from app.scanner import PrepareError, normalize_extensions, prepare_dataset

logger = logging.getLogger("cli")

# библиотеки для проверки: (имя для import, имя пакета, обязательная ли)
LIBRARIES = [
    ("numpy", "numpy", True),
    ("PIL", "Pillow", True),
    ("dotenv", "python-dotenv", True),
    ("matplotlib", "matplotlib", False),
    ("torch", "torch", False),
    ("tensorflow", "tensorflow", False),
]


def build_parser(settings):
    """Создает парсер аргументов с двумя подкомандами.

    :param settings: настройки по умолчанию (из .env)
    :return: объект ArgumentParser
    """
    # общие параметры логирования для обеих команд
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--log-level", default=settings.log_level,
                        help="уровень логирования (DEBUG, INFO, ...)")
    common.add_argument("--log-dir", default=str(settings.log_dir),
                        help="папка для файла лога")

    parser = argparse.ArgumentParser(
        prog="prepare-tool",
        description="Подготовка данных для нейронной сети",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare", parents=[common],
        help="собрать файлы из папки в один .npy")
    prepare.add_argument("--path", default=str(settings.data_path),
                         help="папка с исходными файлами")
    prepare.add_argument("--ext", nargs="+", default=["png", "jpg"],
                         help="расширения файлов, например: png jpg")
    prepare.add_argument("--output",
                         default=str(settings.output_dir / "dataset.npy"),
                         help="путь к итоговому файлу .npy")
    prepare.add_argument("--size", type=int, default=settings.image_size,
                         help="размер картинок после приведения (пиксели)")

    commands.add_parser("doctor", parents=[common],
                        help="проверить окружение")
    return parser


def run_prepare(args):
    """Выполняет команду prepare и печатает статистику.

    :param args: разобранные аргументы
    :return: код завершения (0 - успешно)
    """
    extensions = normalize_extensions(args.ext)
    logger.info("Папка: %s, расширения: %s", args.path,
                ", ".join(sorted(extensions)))
    if args.size <= 0:
        logger.error("Размер картинки должен быть больше нуля")
        return 1
    try:
        stats = prepare_dataset(args.path, extensions, args.output,
                                args.size)
    except PrepareError as error:
        logger.error("%s", error)
        return 1
    print("\n----- Статистика -----")
    print(f"Тип данных:          {stats['kind']}")
    print(f"Найдено файлов:      {stats['files_found']}")
    print(f"Прочитано:           {stats['files_loaded']}")
    print(f"Пропущено:           {stats['files_skipped']}")
    print(f"Размер исходников:   {stats['source_size_kb']} КБ")
    print(f"Форма массива:       {stats['array_shape']}")
    print(f"Размер массива:      {stats['array_size_kb']} КБ")
    low, high = stats["value_range"]
    print(f"Диапазон значений:   {low:.3f} ... {high:.3f}")
    print(f"Файл результата:     {stats['output_file']}")
    print(f"Время обработки:     {stats['elapsed_sec']} с")
    return 0


def run_command(command):
    """Запускает внешнюю команду и возвращает ее вывод.

    :param command: список (программа и аргументы)
    :return: кортеж (успешно ли, текст вывода)
    """
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=15, check=False)
    except FileNotFoundError:
        return False, "команда не найдена"
    except subprocess.TimeoutExpired:
        return False, "команда не ответила за 15 секунд"
    output = (result.stdout or result.stderr).strip()
    return result.returncode == 0, output


def check_library(import_name, package_name):
    """Проверяет, установлена ли библиотека, и узнает ее версию.

    :return: строка с версией или None, если библиотеки нет
    """
    if importlib.util.find_spec(import_name) is None:
        return None
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "версия неизвестна"


def run_doctor(settings):
    """Выполняет команду doctor: проверка окружения.

    :param settings: текущие настройки
    :return: код завершения (0 - все обязательное есть)
    """
    problems = 0
    print("----- Python -----")
    print(f"Версия:        {platform.python_version()} "
          f"({platform.python_implementation()})")
    print(f"Интерпретатор: {sys.executable}")
    print(f"Система:       {platform.system()} {platform.release()}")
    _, pip_info = run_command([sys.executable, "-m", "pip", "--version"])
    print(f"pip:           {pip_info}")

    print("\n----- Библиотеки -----")
    for import_name, package_name, required in LIBRARIES:
        version = check_library(import_name, package_name)
        if version:
            print(f"[OK]   {package_name:<14} {version}")
        elif required:
            print(f"[НЕТ]  {package_name:<14} нужно установить")
            problems += 1
        else:
            print(f"[--]   {package_name:<14} не установлен "
                  f"(не обязателен)")

    print("\n----- GPU / CUDA -----")
    if shutil.which("nvidia-smi") is None:
        print("nvidia-smi не найден - CUDA недоступна, "
              "работа будет на процессоре")
    else:
        success, gpu_info = run_command([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader"])
        state = "найдена видеокарта" if success else "ошибка nvidia-smi"
        print(f"{state}: {gpu_info}")

    print("\n----- Папки и настройки -----")
    print(f"Рабочая папка: {Path.cwd()}")
    print(f"DATA_PATH:     {settings.data_path} "
          f"({'есть' if settings.data_path.exists() else 'не найдена'})")
    print(f"OUTPUT_DIR:    {settings.output_dir}")
    print(f"LOG_LEVEL:     {settings.log_level}")
    print(f"Процессоров:   {os.cpu_count()}")

    print("\nИтог:", "все в порядке" if problems == 0
          else f"не хватает обязательных библиотек: {problems}")
    return 0 if problems == 0 else 1


def main(argv=None):
    """Точка входа: разбирает аргументы и запускает нужную команду.

    :param argv: список аргументов (None - взять из командной строки)
    :return: код завершения
    """
    settings = load_settings()
    args = build_parser(settings).parse_args(argv)
    log_file = setup_logging(args.log_level, args.log_dir)
    logger.debug("Лог пишется в файл %s", log_file)
    if args.command == "prepare":
        return run_prepare(args)
    return run_doctor(settings)


if __name__ == "__main__":
    sys.exit(main())

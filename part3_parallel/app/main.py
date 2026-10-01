"""Точка входа третьей части.

Запуск из папки part3_parallel:
    python -m app.main --path ../data/input/images --threads 4 --processes 4
"""

import argparse
import logging
import os
import sys
from pathlib import Path

try:
    from dotenv import find_dotenv, load_dotenv
except ImportError:  # без python-dotenv работаем только с переменными ОС
    load_dotenv = None

from app.performance import compare

LOG_FORMAT = ("%(asctime)s [%(levelname)s] [%(processName)s/"
              "%(threadName)s] %(message)s")


def env_int(name, default):
    """Читает целое число из переменной окружения.

    :param name: имя переменной
    :param default: значение, если переменной нет или она не число
    """
    try:
        return int(os.getenv(name, default))
    except ValueError:
        logging.warning("%s должен быть числом, беру %d", name, default)
        return default


def build_parser():
    """Создает парсер аргументов. Значения по умолчанию берутся из ENV."""
    cpu_count = os.cpu_count() or 2
    parser = argparse.ArgumentParser(
        description="Многопоточное чтение и многопроцессорная аугментация")
    parser.add_argument("--path",
                        default=os.getenv("DATA_PATH", "../data/input"),
                        help="папка с картинками")
    parser.add_argument("--ext", nargs="+", default=["png", "jpg", "jpeg"],
                        help="расширения файлов")
    parser.add_argument("--output-dir",
                        default=os.getenv("OUTPUT_DIR", "../data/output"),
                        help="папка для отчета и результата")
    parser.add_argument("--threads", type=int,
                        default=env_int("NUM_THREADS", 4),
                        help="количество потоков-читателей")
    parser.add_argument("--processes", type=int,
                        default=env_int("NUM_PROCESSES", cpu_count),
                        help="количество процессов")
    parser.add_argument("--copies", type=int,
                        default=env_int("AUG_COPIES", 4),
                        help="сколько копий делать из каждой картинки")
    parser.add_argument("--size", type=int,
                        default=env_int("IMAGE_SIZE", 64),
                        help="размер картинок после чтения")
    parser.add_argument("--seed", type=int, default=42,
                        help="зерно случайных чисел")
    parser.add_argument("--save-npy", action="store_true",
                        help="сохранить расширенный набор в augmented.npy")
    parser.add_argument("--log-level",
                        default=os.getenv("LOG_LEVEL", "INFO"),
                        help="уровень логирования")
    parser.add_argument("--log-dir", default=os.getenv("LOG_DIR", "../logs"),
                        help="папка для файла лога")
    return parser


def setup_logging(level, log_dir):
    """Настраивает вывод лога в консоль и в файл parallel.log."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_path / "parallel.log",
                                      encoding="utf-8")],
        force=True,
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)


def main(argv=None):
    """Разбирает аргументы, запускает сравнение и печатает итог.

    :return: код завершения
    """
    if load_dotenv is not None:
        load_dotenv(find_dotenv(usecwd=True))
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, args.log_dir)
    logger = logging.getLogger("main")

    for name in ("threads", "processes", "size"):
        if getattr(args, name) < 1:
            logger.error("Параметр --%s должен быть больше нуля", name)
            return 1
    if args.copies < 0:
        logger.error("Параметр --copies не может быть отрицательным")
        return 1
    if not Path(args.path).is_dir():
        logger.error("Папка не найдена: %s", args.path)
        return 1

    extensions = {ext if ext.startswith(".") else f".{ext}"
                  for ext in (item.lower() for item in args.ext)}
    settings = {
        "data_path": args.path,
        "extensions": extensions,
        "output_dir": args.output_dir,
        "threads": args.threads,
        "processes": args.processes,
        "copies": args.copies,
        "image_size": args.size,
        "seed": args.seed,
        "save_npy": args.save_npy,
    }
    try:
        report = compare(settings)
    except KeyboardInterrupt:
        logger.warning("Программа остановлена пользователем")
        return 1
    if report["files"] == 0:
        logger.error("В папке нет подходящих картинок")
        return 1

    print("\n----- Результат -----")
    print(f"Файлов: {report['files']}, всего картинок после "
          f"аугментации: {report['images_total']}")
    print(f"Последовательно: {report['sequential_time_sec']} с")
    print(f"Параллельно:     {report['parallel_time_sec']} с "
          f"({report['reader_threads']} потоков, "
          f"{report['processes']} процессов)")
    print(f"Ускорение:       {report['speedup']}x")
    print(f"Результаты совпадают: "
          f"{'да' if report['results_match'] else 'нет'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

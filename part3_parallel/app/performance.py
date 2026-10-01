"""Сравнение последовательной и параллельной обработки."""

import json
import logging
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from app.producer_consumer import AugmentationPipeline, run_sequential

logger = logging.getLogger(__name__)


def results_are_equal(first, second):
    """Проверяет, что две версии дали одинаковые картинки.

    :param first: словарь {имя: список массивов}
    :param second: такой же словарь
    :return: True, если совпадают имена и все массивы
    """
    if first.keys() != second.keys():
        return False
    for name, images in first.items():
        other = second[name]
        if len(images) != len(other):
            return False
        for first_image, second_image in zip(images, other):
            if not np.allclose(first_image, second_image):
                return False
    return True


def save_dataset(results, file_path):
    """Сохраняет все картинки (оригиналы и копии) в один .npy.

    :param results: словарь {имя: список массивов}
    :param file_path: путь к файлу
    :return: форма сохраненного массива
    """
    arrays = [image for name in sorted(results) for image in results[name]]
    dataset = np.stack(arrays)
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, dataset)
    return dataset.shape


def compare(settings):
    """Запускает обе версии, сравнивает время и пишет отчет в JSON.

    :param settings: словарь настроек (см. main.py)
    :return: словарь с отчетом
    """
    extensions = settings["extensions"]
    logger.info("=== Последовательная версия ===")
    start = time.perf_counter()
    sequential = run_sequential(settings["data_path"], extensions,
                                settings["image_size"], settings["copies"],
                                settings["seed"])
    sequential_time = time.perf_counter() - start

    logger.info("=== Параллельная версия ===")
    pipeline = AugmentationPipeline(
        settings["data_path"], extensions, settings["image_size"],
        settings["threads"], settings["processes"], settings["copies"],
        settings["seed"])
    start = time.perf_counter()
    parallel = pipeline.run()
    parallel_time = time.perf_counter() - start

    total_images = sum(len(images) for images in parallel.values())
    report = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "files": len(parallel),
        "copies_per_file": settings["copies"],
        "images_total": total_images,
        "image_size": settings["image_size"],
        "reader_threads": settings["threads"],
        "processes": settings["processes"],
        "worker_processes_used": len(pipeline.worker_pids),
        "sequential_time_sec": round(sequential_time, 3),
        "parallel_time_sec": round(parallel_time, 3),
        "speedup": round(sequential_time / parallel_time, 2),
        "results_match": results_are_equal(sequential, parallel),
        "errors": pipeline.errors,
    }
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "performance_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=4),
                           encoding="utf-8")
    logger.info("Отчет сохранен в %s", report_path)

    if settings["save_npy"] and parallel:
        shape = save_dataset(parallel, output_dir / "augmented.npy")
        logger.info("Расширенный набор сохранен, форма массива %s", shape)
    return report

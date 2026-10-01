"""Паттерн производитель-потребитель для чтения и аугментации картинок.

Поток Producer кладет пути к файлам в очередь, потоки Reader читают
картинки и кладут массивы во вторую очередь, главный поток отдает
массивы в пул процессов, а результат приходит в callback.
"""

import logging
import queue
import threading
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.augment import augment_image

logger = logging.getLogger(__name__)

# специальный объект: сигнал "работа закончена"
STOP = None


def find_images(root, extensions):
    """Рекурсивно ищет картинки с нужными расширениями.

    :param root: папка с картинками
    :param extensions: множество расширений вида {'.png', '.jpg'}
    :return: генератор путей
    """
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def read_image(path, image_size):
    """Читает картинку в массив (size, size, 3) со значениями от 0 до 1.

    :param path: путь к файлу
    :param image_size: сторона квадрата
    :return: массив float32
    """
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size))
        return np.asarray(image, dtype=np.float32) / 255.0


class AugmentationPipeline:
    """Многопоточное чтение и многопроцессорная аугментация картинок."""

    def __init__(self, root, extensions, image_size=64, n_readers=4,
                 n_processes=4, copies=4, seed=42, queue_size=32):
        """Сохраняет настройки и создает очереди.

        :param root: папка с картинками
        :param extensions: множество расширений
        :param image_size: размер картинок после чтения
        :param n_readers: количество потоков-читателей
        :param n_processes: количество процессов для аугментации
        :param copies: сколько копий делать из одной картинки
        :param seed: зерно случайных чисел
        :param queue_size: максимальная длина очередей
        """
        self.root = Path(root)
        self.extensions = extensions
        self.image_size = image_size
        self.n_readers = n_readers
        self.n_processes = n_processes
        self.copies = copies
        self.seed = seed
        # ограничение длины очереди не дает забить всю память
        self.path_queue = queue.Queue(maxsize=queue_size)
        self.array_queue = queue.Queue(maxsize=queue_size)
        self.lock = threading.Lock()
        self.files_found = 0
        self.files_read = 0
        self.files_augmented = 0
        self.errors = []
        self.results = {}
        self.worker_pids = set()

    def _producer(self):
        """Поток-производитель: кладет пути к файлам в очередь."""
        logger.info("Начинаю обход папки %s", self.root)
        for path in find_images(self.root, self.extensions):
            self.path_queue.put(path)
            with self.lock:
                self.files_found += 1
        # каждому читателю отправляем сигнал остановки
        for _ in range(self.n_readers):
            self.path_queue.put(STOP)
        logger.info("Обход закончен, найдено файлов: %d", self.files_found)

    def _reader(self):
        """Поток-потребитель: читает файл и кладет массив в очередь."""
        logger.info("Поток запущен")
        read_by_me = 0
        while True:
            path = self.path_queue.get()
            if path is STOP:
                self.array_queue.put(STOP)
                logger.info("Поток завершен, прочитано файлов: %d",
                            read_by_me)
                break
            try:
                array = read_image(path, self.image_size)
            except (OSError, UnidentifiedImageError) as error:
                logger.warning("Не удалось прочитать %s: %s", path, error)
                with self.lock:
                    self.errors.append(f"{path}: {error}")
                continue
            # ключ - путь относительно корня, чтобы не путать одинаковые
            # имена из разных подпапок
            key = path.relative_to(self.root).as_posix()
            self.array_queue.put((key, array))
            read_by_me += 1
            with self.lock:
                self.files_read += 1
                count = self.files_read
            logger.debug("Прочитан %s (всего %d)", key, count)

    def _on_result(self, result):
        """Callback пула: сохраняет результат аугментации."""
        name, images, worker_pid = result
        with self.lock:
            self.results[name] = images
            self.worker_pids.add(worker_pid)
            self.files_augmented += 1
            count = self.files_augmented
        logger.debug("%s обработан в процессе %d", name, worker_pid)
        if count % 20 == 0:
            logger.info("Аугментировано файлов: %d (последний - в процессе "
                        "PID %d)", count, worker_pid)

    def _on_error(self, error):
        """Callback пула на случай ошибки в процессе."""
        logger.error("Ошибка в процессе аугментации: %s", error)
        with self.lock:
            self.errors.append(str(error))

    def run(self):
        """Запускает потоки и пул процессов, ждет завершения.

        :return: словарь {имя файла: список массивов}
        """
        start = time.perf_counter()
        threads = [threading.Thread(target=self._producer, name="Producer",
                                    daemon=True)]
        for number in range(1, self.n_readers + 1):
            threads.append(threading.Thread(target=self._reader,
                                            name=f"Reader-{number}",
                                            daemon=True))
        for thread in threads:
            thread.start()
        logger.info("Запущено потоков-читателей: %d, процессов: %d",
                    self.n_readers, self.n_processes)

        pool = Pool(processes=self.n_processes)
        try:
            finished_readers = 0
            while finished_readers < self.n_readers:
                item = self.array_queue.get()
                if item is STOP:
                    finished_readers += 1
                    continue
                name, array = item
                pool.apply_async(augment_image,
                                 (name, array, self.copies, self.seed),
                                 callback=self._on_result,
                                 error_callback=self._on_error)
            # новых задач не будет, ждем окончания всех процессов
            pool.close()
            pool.join()
        except BaseException:
            # при любой аварии (в том числе Ctrl+C) не оставляем процессы
            logger.warning("Обработка прервана, завершаю процессы")
            pool.terminate()
            pool.join()
            raise
        finally:
            for thread in threads:
                thread.join(timeout=5)
        alive = [thread.name for thread in threads if thread.is_alive()]
        if alive:
            logger.warning("Не завершились потоки: %s", ", ".join(alive))
        logger.info("Параллельная обработка завершена за %.2f с, "
                    "работало процессов: %d", time.perf_counter() - start,
                    len(self.worker_pids))
        return self.results


def run_sequential(root, extensions, image_size, copies, seed):
    """Та же обработка, но в одном потоке без процессов (для сравнения).

    :return: словарь {имя файла: список массивов}
    """
    start = time.perf_counter()
    results = {}
    root = Path(root)
    for path in find_images(root, extensions):
        try:
            array = read_image(path, image_size)
        except (OSError, UnidentifiedImageError) as error:
            logger.warning("Не удалось прочитать %s: %s", path, error)
            continue
        key = path.relative_to(root).as_posix()
        name, images, _ = augment_image(key, array, copies, seed)
        results[name] = images
    logger.info("Последовательная обработка завершена за %.2f с",
                time.perf_counter() - start)
    return results

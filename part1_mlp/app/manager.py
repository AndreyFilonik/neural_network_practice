"""Класс для загрузки, нормализации и разбиения данных."""

import csv
from pathlib import Path

import numpy as np

from app.exceptions import (
    DataFileError,
    InvalidInputError,
    MismatchedDataError,
)


class DatasetManager:
    """Работа с набором данных из CSV-файла.

    В файле первая строка - заголовки, последний столбец - ответ.
    """

    METHODS = ("standard", "minmax")

    def __init__(self):
        """Создает пустой менеджер данных."""
        self.feature_names = []
        self.target_name = ""
        self.features = None
        self.targets = None
        self.method = None
        self._shift = None
        self._scale = None
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

    @property
    def is_loaded(self):
        """True, если данные уже загружены."""
        return self.features is not None

    @property
    def n_features(self):
        """Количество признаков (столбцов без ответа)."""
        return len(self.feature_names)

    def load_csv(self, file_path, delimiter=","):
        """Читает CSV-файл.

        :param file_path: путь к файлу
        :param delimiter: разделитель столбцов
        :return: количество загруженных строк
        :raises DataFileError: если файла нет или в нем ошибки
        """
        path = Path(file_path)
        if not path.is_file():
            raise DataFileError(f"Файл не найден: {path}")
        try:
            with path.open(encoding="utf-8") as file:
                rows = list(csv.reader(file, delimiter=delimiter))
        except (OSError, UnicodeDecodeError, csv.Error) as error:
            raise DataFileError(f"Не удалось прочитать файл: {error}") \
                from error
        if len(rows) < 2:
            raise DataFileError("В файле нет данных, только заголовок")
        header = rows[0]
        if len(header) < 2:
            raise DataFileError("Нужно минимум два столбца: признак и ответ")
        values = []
        for line_number, row in enumerate(rows[1:], start=2):
            if not row:
                continue
            if len(row) != len(header):
                raise DataFileError(
                    f"Строка {line_number}: ожидалось {len(header)} "
                    f"значений, найдено {len(row)}"
                )
            try:
                values.append([float(item) for item in row])
            except ValueError as error:
                raise DataFileError(
                    f"Строка {line_number}: нечисловое значение ({error})"
                ) from error
        table = np.array(values, dtype=float)
        self.feature_names = header[:-1]
        self.target_name = header[-1]
        self.features = table[:, :-1]
        self.targets = table[:, -1:]
        self.method = None
        return table.shape[0]

    def normalize(self, method="standard"):
        """Нормализует признаки.

        standard - (x - среднее) / стандартное отклонение;
        minmax - (x - min) / (max - min).

        :param method: способ нормализации
        """
        if method not in self.METHODS:
            raise InvalidInputError(
                f"Неизвестный способ нормализации: {method}")
        if method == "standard":
            shift = self.features.mean(axis=0)
            scale = self.features.std(axis=0)
        else:
            shift = self.features.min(axis=0)
            scale = self.features.max(axis=0) - shift
        # если столбец постоянный, делим на 1, чтобы не было деления на 0
        scale[scale == 0] = 1.0
        self._shift = shift
        self._scale = scale
        self.method = method
        self.features = (self.features - shift) / scale

    def transform(self, features):
        """Нормализует новые примеры теми же параметрами, что и выборку.

        :param features: массив новых примеров
        :return: нормализованный массив
        """
        features = np.asarray(features, dtype=float)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.shape[1] != self.n_features:
            raise MismatchedDataError(
                f"Нужно {self.n_features} признаков, "
                f"получено {features.shape[1]}"
            )
        if self.method is None:
            return features
        return (features - self._shift) / self._scale

    def split(self, test_size=0.2, seed=None):
        """Делит данные на обучающую и тестовую выборки.

        :param test_size: доля тестовой выборки (от 0 до 1)
        :param seed: число для перемешивания (можно None)
        :return: кортеж (train_x, test_x, train_y, test_y)
        """
        if not 0 < test_size < 1:
            raise InvalidInputError(
                "Доля тестовой выборки должна быть от 0 до 1")
        n_samples = self.features.shape[0]
        n_test = max(1, int(round(n_samples * test_size)))
        if n_test >= n_samples:
            raise MismatchedDataError(
                "Слишком мало данных для такого разбиения")
        order = np.random.default_rng(seed).permutation(n_samples)
        test_index = order[:n_test]
        train_index = order[n_test:]
        self.train_x = self.features[train_index]
        self.train_y = self.targets[train_index]
        self.test_x = self.features[test_index]
        self.test_y = self.targets[test_index]
        return self.train_x, self.test_x, self.train_y, self.test_y

    def load_features_file(self, file_path, delimiter=","):
        """Читает CSV только с признаками (для предсказания).

        Если в файле есть лишний последний столбец с ответом,
        он отбрасывается.

        :param file_path: путь к файлу
        :return: массив признаков
        """
        path = Path(file_path)
        if not path.is_file():
            raise DataFileError(f"Файл не найден: {path}")
        try:
            table = np.loadtxt(path, delimiter=delimiter, skiprows=1,
                               ndmin=2)
        except (OSError, ValueError) as error:
            raise DataFileError(f"Ошибка в файле: {error}") from error
        if table.shape[1] == self.n_features + 1:
            table = table[:, :-1]
        return table

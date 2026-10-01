"""Консольное меню для работы с нейронной сетью.

Запуск из папки part1_mlp:  python -m app.main
"""

import os

from app.exceptions import (
    DataNotLoadedError,
    MismatchedDataError,
    NetworkError,
    NetworkNotCreatedError,
    InvalidInputError,
)
from app.manager import DatasetManager
from app.models import ACTIVATIONS, NeuralNetwork
from app.utils import (
    ask_float,
    ask_int,
    ask_text,
    parse_example,
    parse_layers,
    plot_history,
)

MENU = """
========== Нейронная сеть ==========
1. Создать сеть
2. Загрузить данные из CSV
3. Обучить сеть
4. Сделать предсказание
5. Сохранить / загрузить веса
6. Показать график ошибки
7. Выход
===================================="""

# значения по умолчанию можно поменять через переменные окружения
DEFAULT_WEIGHTS = os.getenv("WEIGHTS_FILE", "weights.json")
DEFAULT_PLOT = os.getenv("PLOT_FILE", "loss_plot.png")


class ConsoleApp:
    """Консольное приложение: хранит сеть и данные, обрабатывает меню."""

    def __init__(self):
        """Создает приложение без сети и без данных."""
        self.network = None
        self.dataset = DatasetManager()
        self.actions = {
            "1": self.create_network,
            "2": self.load_data,
            "3": self.train_network,
            "4": self.make_prediction,
            "5": self.weights_menu,
            "6": self.show_plot,
        }

    def _require_network(self):
        """Проверяет, что сеть уже создана."""
        if self.network is None:
            raise NetworkNotCreatedError(
                "Сначала создайте сеть (пункт 1) или загрузите веса (пункт 5)"
            )

    def _require_data(self):
        """Проверяет, что данные уже загружены."""
        if not self.dataset.is_loaded:
            raise DataNotLoadedError("Сначала загрузите данные (пункт 2)")

    def create_network(self):
        """Пункт 1: создание сети."""
        if self.dataset.is_loaded:
            print(f"Подсказка: в данных {self.dataset.n_features} "
                  f"признаков, значит входной слой = "
                  f"{self.dataset.n_features}")
        layers = parse_layers(ask_text(
            "Размеры слоев через пробел (вход скрытые... выход)", "4 8 1"))
        activation = ask_text(
            f"Функция активации ({'/'.join(ACTIVATIONS)})", "sigmoid")
        self.network = NeuralNetwork(layers, activation)
        print(f"Готово. {self.network}")

    def load_data(self):
        """Пункт 2: загрузка, нормализация и разбиение данных."""
        default_path = os.getenv("DATA_FILE", "../data/input/banknote.csv")
        path = ask_text("Путь к CSV-файлу", default_path)
        rows = self.dataset.load_csv(path)
        method = ask_text("Нормализация (standard/minmax)", "standard")
        if method not in DatasetManager.METHODS:
            raise InvalidInputError("Нужно выбрать standard или minmax")
        self.dataset.normalize(method)
        test_size = ask_float("Доля тестовой выборки", "0.2",
                              minimum=0.0, maximum=1.0)
        train_x, test_x, _, _ = self.dataset.split(test_size)
        print(f"Загружено строк: {rows}")
        print(f"Признаки: {', '.join(self.dataset.feature_names)}")
        print(f"Целевая переменная: {self.dataset.target_name}")
        print(f"Обучающая выборка: {len(train_x)}, "
              f"тестовая: {len(test_x)}")

    def train_network(self):
        """Пункт 3: обучение сети."""
        self._require_network()
        self._require_data()
        if self.network.layer_sizes[0] != self.dataset.n_features:
            raise MismatchedDataError(
                f"Во входном слое {self.network.layer_sizes[0]} нейронов, "
                f"а в данных {self.dataset.n_features} признаков"
            )
        learning_rate = ask_float("Скорость обучения", "0.5")
        epochs = ask_int("Число эпох", "200")
        batch_size = ask_int("Размер батча (1 - по одному примеру)", "16")
        print("Обучение...")
        self.network.train(self.dataset.train_x, self.dataset.train_y,
                           epochs, learning_rate, batch_size,
                           print_every=max(1, epochs // 10))
        train_loss, train_acc = self.network.evaluate(
            self.dataset.train_x, self.dataset.train_y)
        test_loss, test_acc = self.network.evaluate(
            self.dataset.test_x, self.dataset.test_y)
        print(f"Обучающая выборка: MSE = {train_loss:.4f}, "
              f"точность = {train_acc:.1f}%")
        print(f"Тестовая выборка:  MSE = {test_loss:.4f}, "
              f"точность = {test_acc:.1f}%")

    def make_prediction(self):
        """Пункт 4: предсказание для введенных примеров или из файла."""
        self._require_network()
        self._require_data()
        source = ask_text("1 - ввести вручную, 2 - из файла", "1")
        if source == "1":
            print(f"Введите {self.dataset.n_features} признаков через ';' "
                  f"({'; '.join(self.dataset.feature_names)}).")
            print("Пустая строка - закончить ввод.")
            examples = []
            while True:
                line = input("> ").strip()
                if not line:
                    break
                examples.append(parse_example(line))
            if not examples:
                print("Примеры не введены")
                return
            if len({len(example) for example in examples}) != 1:
                raise MismatchedDataError(
                    "У примеров разное количество признаков")
        elif source == "2":
            path = ask_text("Путь к CSV с примерами")
            examples = self.dataset.load_features_file(path)
        else:
            raise InvalidInputError("Нужно выбрать 1 или 2")
        features = self.dataset.transform(examples)
        outputs = self.network.predict(features)
        for number, value in enumerate(outputs[:, 0], start=1):
            label = 1 if value >= 0.5 else 0
            print(f"Пример {number}: выход сети = {value:.3f}, "
                  f"класс = {label}")

    def weights_menu(self):
        """Пункт 5: сохранение или загрузка весов."""
        choice = ask_text("1 - сохранить веса, 2 - загрузить веса", "1")
        if choice == "1":
            self._require_network()
            path = ask_text("Файл для сохранения", DEFAULT_WEIGHTS)
            self.network.save(path)
            print(f"Веса сохранены в {path}")
        elif choice == "2":
            path = ask_text("Файл с весами", DEFAULT_WEIGHTS)
            self.network = NeuralNetwork.load(path)
            print(f"Веса загружены. {self.network}")
        else:
            raise InvalidInputError("Нужно выбрать 1 или 2")

    def show_plot(self):
        """Пункт 6: график ошибки."""
        self._require_network()
        if not self.network.loss_history:
            print("Сеть еще не обучалась, графика нет")
            return
        try:
            path = plot_history(self.network.loss_history, DEFAULT_PLOT)
        except ImportError:
            print("matplotlib не установлен, вывожу ошибку текстом:")
            for epoch, loss in enumerate(self.network.loss_history, 1):
                print(f"{epoch}: {loss:.5f}")
            return
        print(f"График сохранен в {path}")

    def run(self):
        """Главный цикл меню."""
        while True:
            print(MENU)
            choice = input("Выберите пункт: ").strip()
            if choice == "7":
                print("Выход из программы")
                break
            action = self.actions.get(choice)
            if action is None:
                print("Такого пункта нет, введите число от 1 до 7")
                continue
            try:
                action()
            except NetworkError as error:
                print(f"Ошибка: {error}")


def main():
    """Точка входа в программу."""
    try:
        ConsoleApp().run()
    except (KeyboardInterrupt, EOFError):
        print("\nПрограмма остановлена")


if __name__ == "__main__":
    main()

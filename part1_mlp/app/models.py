"""Классы функций активации и полносвязной нейронной сети."""

import json
from pathlib import Path

import numpy as np

from app.exceptions import (
    InvalidLayerSizeError,
    MismatchedDataError,
    UnknownActivationError,
    WeightsFileError,
)


class Activation:
    """Базовый класс функции активации.

    Наследники должны реализовать методы apply и derivative.
    """

    name = "base"

    def apply(self, values):
        """Вычисляет значение функции.

        :param values: массив взвешенных сумм нейронов
        :return: массив выходов нейронов
        """
        raise NotImplementedError

    def derivative(self, outputs):
        """Вычисляет производную через уже посчитанные выходы нейронов.

        :param outputs: массив выходов нейронов (результат apply)
        :return: массив значений производной
        """
        raise NotImplementedError


class Sigmoid(Activation):
    """Сигмоида: f(z) = 1 / (1 + e^(-z))."""

    name = "sigmoid"

    def apply(self, values):
        """Возвращает значение сигмоиды."""
        # ограничиваем значения, чтобы exp не переполнялась
        values = np.clip(values, -500, 500)
        return 1.0 / (1.0 + np.exp(-values))

    def derivative(self, outputs):
        """Производная сигмоиды: f * (1 - f)."""
        return outputs * (1.0 - outputs)


class ReLU(Activation):
    """ReLU: f(z) = max(0, z)."""

    name = "relu"

    def apply(self, values):
        """Возвращает значение ReLU."""
        return np.maximum(0.0, values)

    def derivative(self, outputs):
        """Производная ReLU: 1, если выход больше нуля, иначе 0."""
        return (outputs > 0).astype(float)


# словарь доступных функций активации
ACTIVATIONS = {
    Sigmoid.name: Sigmoid,
    ReLU.name: ReLU,
}


def get_activation(name):
    """Создает объект функции активации по названию.

    :param name: название функции ("sigmoid" или "relu")
    :return: объект класса Activation
    :raises UnknownActivationError: если такой функции нет
    """
    key = str(name).strip().lower()
    if key not in ACTIVATIONS:
        raise UnknownActivationError(
            f"Функция активации '{name}' не поддерживается. "
            f"Доступно: {', '.join(ACTIVATIONS)}"
        )
    return ACTIVATIONS[key]()


class NeuralNetwork:
    """Полносвязная нейронная сеть (многослойный перцептрон).

    На скрытых слоях используется выбранная функция активации,
    на выходном слое всегда сигмоида (ответ от 0 до 1).
    Функция ошибки - среднеквадратичная (MSE).
    """

    def __init__(self, layer_sizes, activation="sigmoid", seed=None):
        """Создает сеть и заполняет веса случайными числами.

        :param layer_sizes: список размеров слоев, например [4, 8, 1]
        :param activation: функция активации скрытых слоев
        :param seed: число для генератора случайных чисел (можно None)
        :raises InvalidLayerSizeError: если размеры слоев неверные
        """
        self.layer_sizes = self._check_layers(layer_sizes)
        self.hidden_activation = get_activation(activation)
        self.output_activation = Sigmoid()
        self.random = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        self.loss_history = []
        self._layer_outputs = []
        self._init_weights()

    @staticmethod
    def _check_layers(layer_sizes):
        """Проверяет список размеров слоев и возвращает его копию."""
        if len(layer_sizes) < 3:
            raise InvalidLayerSizeError(
                "Нужно минимум три слоя: входной, скрытый и выходной"
            )
        checked = []
        for size in layer_sizes:
            if not isinstance(size, (int, np.integer)) or size <= 0:
                raise InvalidLayerSizeError(
                    f"Размер слоя должен быть целым числом больше 0: {size}"
                )
            checked.append(int(size))
        return checked

    def _init_weights(self):
        """Заполняет веса случайными числами, смещения - нулями."""
        self.weights = []
        self.biases = []
        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            # разброс зависит от числа входов, чтобы сеть лучше училась
            scale = np.sqrt(2.0 / (n_in + n_out))
            self.weights.append(self.random.normal(0.0, scale, (n_in, n_out)))
            self.biases.append(np.zeros((1, n_out)))

    def set_activation(self, name):
        """Меняет функцию активации скрытых слоев.

        :param name: название новой функции
        """
        self.hidden_activation = get_activation(name)

    def _check_input(self, features):
        """Приводит вход к двумерному массиву и проверяет число признаков."""
        features = np.asarray(features, dtype=float)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.shape[1] != self.layer_sizes[0]:
            raise MismatchedDataError(
                f"Сеть ожидает {self.layer_sizes[0]} признаков, "
                f"а получено {features.shape[1]}"
            )
        return features

    def forward(self, features):
        """Прямое распространение сигнала.

        Для каждого слоя: z = a * W + b, затем a = f(z).

        :param features: массив примеров размером (n, число входов)
        :return: выход сети размером (n, число выходов)
        """
        layer_output = self._check_input(features)
        self._layer_outputs = [layer_output]
        last_index = len(self.weights) - 1
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            weighted_sum = layer_output @ weight + bias
            if index == last_index:
                layer_output = self.output_activation.apply(weighted_sum)
            else:
                layer_output = self.hidden_activation.apply(weighted_sum)
            self._layer_outputs.append(layer_output)
        return layer_output

    def backward(self, targets, learning_rate):
        """Обратное распространение ошибки и шаг градиентного спуска.

        Перед вызовом нужно выполнить forward для этого же батча.

        :param targets: правильные ответы размером (n, число выходов)
        :param learning_rate: скорость обучения
        """
        outputs = self._layer_outputs[-1]
        batch_size = targets.shape[0]
        # ошибка выходного слоя: (выход - ответ) * производная сигмоиды
        output_derivative = self.output_activation.derivative(outputs)
        delta = (outputs - targets) * output_derivative
        for index in range(len(self.weights) - 1, -1, -1):
            previous_output = self._layer_outputs[index]
            grad_weight = previous_output.T @ delta / batch_size
            grad_bias = np.sum(delta, axis=0, keepdims=True) / batch_size
            if index > 0:
                # ошибку переносим на предыдущий слой до обновления весов
                delta = (delta @ self.weights[index].T) * \
                    self.hidden_activation.derivative(previous_output)
            self.weights[index] -= learning_rate * grad_weight
            self.biases[index] -= learning_rate * grad_bias

    @staticmethod
    def mse(outputs, targets):
        """Среднеквадратичная ошибка.

        :return: число - средний квадрат разницы
        """
        return float(np.mean((outputs - targets) ** 2))

    def _prepare_targets(self, targets, n_samples):
        """Приводит ответы к форме (n, число выходов)."""
        targets = np.asarray(targets, dtype=float)
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        if targets.shape[0] != n_samples:
            raise MismatchedDataError(
                "Количество примеров и ответов не совпадает"
            )
        if targets.shape[1] != self.layer_sizes[-1]:
            raise MismatchedDataError(
                f"Выходной слой имеет {self.layer_sizes[-1]} нейронов, "
                f"а в ответах {targets.shape[1]} столбцов"
            )
        return targets

    def train(self, features, targets, epochs, learning_rate,
              batch_size=1, print_every=10):
        """Обучает сеть мини-батчами.

        :param features: обучающие примеры
        :param targets: правильные ответы
        :param epochs: количество эпох
        :param learning_rate: скорость обучения
        :param batch_size: размер батча (1 - обучение по одному примеру)
        :param print_every: как часто печатать ошибку (в эпохах)
        :return: список ошибок по эпохам
        """
        features = self._check_input(features)
        targets = self._prepare_targets(targets, features.shape[0])
        n_samples = features.shape[0]
        batch_size = max(1, min(batch_size, n_samples))
        for epoch in range(1, epochs + 1):
            # перемешиваем примеры перед каждой эпохой
            order = self.random.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                batch_index = order[start:start + batch_size]
                self.forward(features[batch_index])
                self.backward(targets[batch_index], learning_rate)
            epoch_loss = self.mse(self.forward(features), targets)
            self.loss_history.append(epoch_loss)
            if print_every and (epoch % print_every == 0 or epoch == 1):
                print(f"Эпоха {epoch:4d}/{epochs}  ошибка (MSE): "
                      f"{epoch_loss:.5f}")
        return self.loss_history

    def predict(self, features):
        """Возвращает ответ сети для новых примеров.

        :param features: один пример или массив примеров
        :return: массив выходов сети
        """
        return self.forward(features)

    def evaluate(self, features, targets):
        """Считает ошибку и точность на выборке.

        Точность считается для задачи из двух классов (порог 0.5).

        :return: кортеж (mse, точность в процентах)
        """
        outputs = self.predict(features)
        targets = self._prepare_targets(targets, outputs.shape[0])
        accuracy = np.mean((outputs >= 0.5) == (targets >= 0.5)) * 100
        return self.mse(outputs, targets), float(accuracy)

    def save(self, file_path):
        """Сохраняет размеры слоев, активацию и веса в JSON-файл.

        :param file_path: путь к файлу
        :raises WeightsFileError: если записать файл не получилось
        """
        path = Path(file_path)
        content = {
            "layer_sizes": self.layer_sizes,
            "activation": self.hidden_activation.name,
            "weights": [weight.tolist() for weight in self.weights],
            "biases": [bias.tolist() for bias in self.biases],
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(content), encoding="utf-8")
        except OSError as error:
            raise WeightsFileError(
                f"Не удалось сохранить веса: {error}"
            ) from error

    @classmethod
    def load(cls, file_path):
        """Создает сеть из JSON-файла с весами.

        :param file_path: путь к файлу
        :return: объект NeuralNetwork
        :raises WeightsFileError: если файла нет или он поврежден
        """
        path = Path(file_path)
        if not path.is_file():
            raise WeightsFileError(f"Файл не найден: {path}")
        try:
            content = json.loads(path.read_text(encoding="utf-8"))
            network = cls(content["layer_sizes"], content["activation"])
            weights = [np.array(item, dtype=float)
                       for item in content["weights"]]
            biases = [np.array(item, dtype=float)
                      for item in content["biases"]]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise WeightsFileError(
                f"Файл с весами поврежден: {error}"
            ) from error
        for weight, expected in zip(weights, network.weights):
            if weight.shape != expected.shape:
                raise WeightsFileError(
                    "Размеры весов в файле не совпадают с размерами слоев"
                )
        network.weights = weights
        network.biases = biases
        return network

    def __str__(self):
        """Краткое описание сети для вывода в консоль."""
        layers = " -> ".join(str(size) for size in self.layer_sizes)
        return f"Сеть: {layers}, активация: {self.hidden_activation.name}"

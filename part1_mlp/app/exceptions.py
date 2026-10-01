"""Собственные исключения для первой части работы."""


class NetworkError(Exception):
    """Базовое исключение программы. От него наследуются остальные."""


class InvalidLayerSizeError(NetworkError):
    """Размеры слоев заданы неправильно (меньше двух слоев или число <= 0)."""


class UnknownActivationError(NetworkError):
    """Указана функция активации, которой нет в программе."""


class MismatchedDataError(NetworkError):
    """Размерность данных не совпадает с размерами сети."""


class DataFileError(NetworkError):
    """Файл с данными не найден или имеет неверный формат."""


class WeightsFileError(NetworkError):
    """Ошибка при сохранении или загрузке файла с весами."""


class NetworkNotCreatedError(NetworkError):
    """Попытка работать с сетью до ее создания."""


class DataNotLoadedError(NetworkError):
    """Попытка обучить сеть до загрузки данных."""


class InvalidInputError(NetworkError):
    """Пользователь ввел некорректное значение в меню."""

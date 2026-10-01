"""Вспомогательные функции для ввода данных и графика ошибки."""

from pathlib import Path

from app.exceptions import DataFileError, InvalidInputError


def ask_text(prompt, default=None):
    """Спрашивает строку у пользователя.

    :param prompt: текст вопроса
    :param default: значение, если пользователь нажал Enter
    :return: введенная строка
    """
    suffix = f" [{default}]" if default is not None else ""
    answer = input(f"{prompt}{suffix}: ").strip()
    if not answer:
        if default is None:
            raise InvalidInputError("Значение не может быть пустым")
        return str(default)
    return answer


def ask_int(prompt, default=None, minimum=1):
    """Спрашивает целое число.

    :param minimum: минимально допустимое значение
    :return: целое число
    """
    answer = ask_text(prompt, default)
    try:
        number = int(answer)
    except ValueError as error:
        raise InvalidInputError(f"'{answer}' - не целое число") from error
    if number < minimum:
        raise InvalidInputError(f"Число должно быть не меньше {minimum}")
    return number


def ask_float(prompt, default=None, minimum=0.0, maximum=None):
    """Спрашивает дробное число (можно писать через точку или запятую).

    :return: число float
    """
    answer = ask_text(prompt, default).replace(",", ".")
    try:
        number = float(answer)
    except ValueError as error:
        raise InvalidInputError(f"'{answer}' - не число") from error
    if number <= minimum or (maximum is not None and number >= maximum):
        limit = f" и меньше {maximum}" if maximum is not None else ""
        raise InvalidInputError(f"Число должно быть больше {minimum}{limit}")
    return number


def parse_layers(text):
    """Переводит строку вида '4 8 1' или '4,8,1' в список чисел.

    :param text: строка с размерами слоев
    :return: список целых чисел
    """
    parts = text.replace(",", " ").split()
    try:
        return [int(part) for part in parts]
    except ValueError as error:
        raise InvalidInputError(
            "Размеры слоев нужно вводить целыми числами"
        ) from error


def parse_example(text):
    """Переводит строку признаков через ';' или пробел в список чисел.

    :param text: строка вида '1.5; 2; 3'
    :return: список чисел
    """
    parts = text.replace(";", " ").split()
    try:
        return [float(part.replace(",", ".")) for part in parts]
    except ValueError as error:
        raise InvalidInputError("Признаки должны быть числами") from error


def plot_history(loss_history, file_path):
    """Строит график ошибки по эпохам и сохраняет его в файл.

    :param loss_history: список ошибок
    :param file_path: куда сохранить картинку
    :return: путь к сохраненному файлу
    """
    import matplotlib.pyplot as plt

    path = Path(file_path)
    epochs = range(1, len(loss_history) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss_history, color="tab:blue")
    plt.title("Изменение ошибки при обучении")
    plt.xlabel("Эпоха")
    plt.ylabel("Ошибка (MSE)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    try:
        plt.savefig(path)
    except OSError as error:
        plt.close()
        raise DataFileError(f"Не удалось сохранить график: {error}") \
            from error
    plt.show()
    plt.close()
    return path

# Практическое задание по дисциплине «Основы нейронных сетей»

Работа состоит из трех частей, каждая лежит в своей папке и запускается отдельно.

| Папка | Что внутри |
|---|---|
| `part1_mlp` | нейронная сеть на NumPy (класс `NeuralNetwork`), загрузка данных (`DatasetManager`), консольное меню |
| `part2_prepare` | CLI-утилита: команды `prepare` и `doctor` |
| `part3_parallel` | многопоточное чтение и многопроцессорная аугментация картинок, сравнение времени |
| `data/input` | тестовые данные |
| `data/output` | результаты работы (в git не попадают) |
| `logs` | логи (в git не попадают) |

## Установка

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux:   source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env      # Linux: cp .env.example .env
```

## Данные

Используются реальные открытые наборы данных, они уже лежат в `data/input`:

| Файл / папка | Набор данных | Где используется |
|---|---|---|
| `banknote.csv`, `predict_examples.csv` | Banknote Authentication (UCI Machine Learning Repository) - признаки снимков 1372 настоящих и поддельных банкнот | часть 1 |
| `images/digit_0 ... digit_9` | Optical Recognition of Handwritten Digits (UCI, входит в scikit-learn) - 300 рукописных цифр 8x8 | части 2 и 3 |
| `tables/iris_*.csv` | Iris (UCI) - 150 цветков ириса, по одному файлу на вид | часть 2 |

В `predict_examples.csv` лежат 6 банкнот, которые убраны из `banknote.csv`,
то есть сеть их при обучении не видит. Правильные классы: 0, 0, 0, 0, 1, 1.

Скачать данные заново (нужен scikit-learn):

```bash
pip install scikit-learn
python get_datasets.py --output data/input
```

## Запуск

```bash
cd part1_mlp      && python -m app.main
cd part2_prepare  && python -m app.cli doctor
cd part3_parallel && python -m app.main
```

Подробнее - в README каждой части.

## Проверка стиля

```bash
flake8 .
```

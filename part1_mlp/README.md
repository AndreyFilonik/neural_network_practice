# Часть 1. Многослойный перцептрон

Полносвязная нейронная сеть, написанная на NumPy без готовых фреймворков.

## Файлы

- `app/models.py` - классы функций активации (`Sigmoid`, `ReLU`) и класс `NeuralNetwork`
- `app/manager.py` - класс `DatasetManager` (чтение CSV, нормализация, разбиение train/test)
- `app/exceptions.py` - собственные исключения (`InvalidLayerSizeError`, `MismatchedDataError` и др.)
- `app/utils.py` - ввод чисел с проверкой, график ошибки
- `app/main.py` - консольное меню

## Запуск

```bash
cd part1_mlp
python -m app.main
```

Пример работы: пункт 2 (загрузить `../data/input/banknote.csv`), пункт 1 (сеть `4 8 1`),
пункт 3 (обучение), пункт 4 (предсказание). Файл для проверки предсказания
из файла - `../data/input/predict_examples.csv`.

## Данные

`banknote.csv` - набор Banknote Authentication (UCI). Признаки получены
по снимкам банкнот: variance, skewness, curtosis, entropy.
Последний столбец `class` - 0 (подлинная) или 1 (поддельная).

# Часть 2. Подготовка данных (CLI)

## Файлы

- `app/cli.py` - разбор аргументов (argparse), команды `prepare` и `doctor`
- `app/config.py` - настройки из CLI, переменных окружения и `.env`
- `app/scanner.py` - обход папки, чтение картинок и CSV, нормализация, сохранение в `.npy`
- `app/logging_config.py` - логирование в консоль и в `logs/prepare.log`

## Запуск

```bash
cd part2_prepare
python -m app.cli prepare --path ../data/input/images --ext png jpg --output ../data/output/images.npy
python -m app.cli prepare --path ../data/input/tables --ext csv --output ../data/output/tables.npy
python -m app.cli prepare --path ../data/input/images --ext png --size 32 --log-level DEBUG
python -m app.cli doctor
```

## Настройки

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `DATA_PATH` | `../data/input` | папка с данными |
| `OUTPUT_DIR` | `../data/output` | папка для результата |
| `LOG_DIR` | `../logs` | папка для логов |
| `LOG_LEVEL` | `INFO` | уровень логирования |
| `IMAGE_SIZE` | `64` | размер картинок после приведения |

Аргументы командной строки важнее переменных окружения.

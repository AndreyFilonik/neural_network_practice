# Часть 3. Многопоточность и многопроцессорность

## Файлы

- `app/producer_consumer.py` - поток-производитель, потоки-читатели, пул процессов
- `app/augment.py` - аугментация: поворот, отражение, яркость, шум, размытие
- `app/performance.py` - сравнение последовательной и параллельной версии, `performance_report.json`
- `app/main.py` - точка входа, аргументы командной строки

## Запуск

```bash
cd part3_parallel
python -m app.main --path ../data/input/images --threads 4 --processes 4
python -m app.main --path ../data/input/images --processes 2 --copies 6 --save-npy
```

Отчет о времени сохраняется в `../data/output/performance_report.json`,
лог - в `../logs/parallel.log`.

## Настройки

`DATA_PATH`, `OUTPUT_DIR`, `LOG_DIR`, `LOG_LEVEL`, `IMAGE_SIZE`,
`NUM_THREADS`, `NUM_PROCESSES`, `AUG_COPIES` - через `.env` или переменные
окружения. Аргументы командной строки важнее.

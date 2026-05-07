# whisper-parser-py

`whisper-parser-py` — скрипт для транскрипции аудио с помощью `faster-whisper`.

## Требования

- Python 3.9+
- `faster-whisper`
- `pyyaml`

Установка:

```bash
pip install faster-whisper pyyaml
```

## Примечание по GPU

Для режима NVIDIA GPU библиотека `ctranslate2` обычно требует CUDA 12 и cuDNN 9.
Если транскрипция на GPU не запускается, сначала проверьте совместимость CUDA/cuDNN.

## Конфигурация

Откройте и настройте `config.yaml`:

- `model_size`: например `large-v3` или `distil-large-v3`
- `device`: `cpu`, `cuda` или `auto`
- `compute_type`: например `int8`, `float16`, `int8_float16`, `float32`
- `processing_mode`: `single` (один файл) или `all` (все найденные файлы)
- `audio_extensions`: список поддерживаемых расширений (по умолчанию `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.aac`, `.mp4`, `.wma`)
- `use_batched_inference`: включить `BatchedInferencePipeline`
- `batch_size`: размер батча для batched inference
- `vad_filter`, `vad_parameters`: параметры VAD-фильтрации
- `word_timestamps`: включить пометки времени по словам
- `language`: фиксированный язык или `null` для автоопределения
- `condition_on_previous_text`: дополнительный параметр декодирования (актуален для distil-моделей)

## Использование

1. Поместите аудиофайлы в папку, указанную в `audio_folder_name`.
2. Запустите скрипт:

```bash
python whisper.py
```

Логи транскрипции сохраняются в папку `transcription_folder_name` с расширением `.log`.

## Важно

`model.transcribe(...)` возвращает генератор сегментов. Фактическая транскрипция начинается при итерации по `segments`.

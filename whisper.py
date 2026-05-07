import os
import threading
import time
from pathlib import Path

from faster_whisper import BatchedInferencePipeline, WhisperModel

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "LOGGER: Не найден пакет PyYAML. Установите его командой: pip install pyyaml"
    ) from exc


VALID_DEVICES = {"cpu", "cuda", "auto"}
VALID_COMPUTE_TYPES = {
    "default",
    "auto",
    "int8",
    "int8_float16",
    "int8_float32",
    "float16",
    "float32",
}
DEFAULT_AUDIO_EXTENSIONS = [
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
    ".aac",
    ".mp4",
    ".wma",
]


def load_config() -> dict:
    config_path = Path(__file__).with_name("config.yaml")
    if not config_path.exists():
        raise SystemExit(f"LOGGER: Файл конфигурации не найден: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise SystemExit("LOGGER: config.yaml должен содержать YAML-объект (map).")
    return loaded


def validate_config(config: dict) -> None:
    beam_size = int(config.get("beam_size", 5))
    heartbeat_interval_s = int(config.get("heartbeat_interval_s", 5))
    batch_size = int(config.get("batch_size", 16))
    device = str(config.get("device", "cpu")).strip().lower()
    compute_type = str(config.get("compute_type", "int8")).strip().lower()
    processing_mode = str(config.get("processing_mode", "single")).strip().lower()
    audio_extensions = config.get("audio_extensions", DEFAULT_AUDIO_EXTENSIONS)

    if beam_size < 1:
        raise SystemExit("LOGGER: `beam_size` должен быть >= 1.")
    if heartbeat_interval_s < 1:
        raise SystemExit("LOGGER: `heartbeat_interval_s` должен быть >= 1.")
    if batch_size < 1:
        raise SystemExit("LOGGER: `batch_size` должен быть >= 1.")
    if device not in VALID_DEVICES:
        raise SystemExit(
            f"LOGGER: `device` должен быть одним из {sorted(VALID_DEVICES)}, сейчас: {device}"
        )
    if compute_type not in VALID_COMPUTE_TYPES:
        raise SystemExit(
            "LOGGER: `compute_type` не поддерживается. "
            f"Ожидается одно из {sorted(VALID_COMPUTE_TYPES)}, сейчас: {compute_type}"
        )
    if processing_mode not in {"single", "all"}:
        raise SystemExit(
            "LOGGER: `processing_mode` должен быть 'single' или 'all'."
        )
    if not isinstance(audio_extensions, list) or not audio_extensions:
        raise SystemExit(
            "LOGGER: `audio_extensions` должен быть непустым списком расширений."
        )

def resolve_base_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(
        "LOGGER: Ни один путь из `base_path_candidates` не найден.\n"
        "LOGGER: Укажите корректные пути в `config.yaml`."
    )


def normalize_audio_extensions(raw_extensions: list) -> set[str]:
    normalized: set[str] = set()
    for ext in raw_extensions:
        ext_str = str(ext).strip().lower()
        if not ext_str:
            continue
        if not ext_str.startswith("."):
            ext_str = "." + ext_str
        normalized.add(ext_str)
    if not normalized:
        raise SystemExit("LOGGER: `audio_extensions` не содержит корректных расширений.")
    return normalized


def pick_audio_files(
    audio_dir: Path, processing_mode: str, allowed_extensions: set[str]
) -> list[Path]:
    audio_files = sorted(
        p
        for p in audio_dir.iterdir()
        if p.is_file() and p.suffix.lower() in allowed_extensions
    )

    if not audio_files:
        allowed = ", ".join(sorted(allowed_extensions))
        raise SystemExit(
            f"LOGGER: В папке нет аудиофайлов ({allowed}): {audio_dir.resolve()}\n"
            "LOGGER: Добавьте файл поддерживаемого формата и запустите скрипт снова."
        )

    if processing_mode == "single":
        if len(audio_files) > 1:
            files_str = ", ".join(p.name for p in audio_files)
            raise SystemExit(
                "LOGGER: В режиме `single` найдено несколько аудиофайлов "
                f"({len(audio_files)}): {files_str}\n"
                "LOGGER: Оставьте один файл или включите `processing_mode: all`."
            )
        return [audio_files[0]]

    return audio_files


def heartbeat(stop_event: threading.Event, start_time: float, interval_s: int) -> None:
    while not stop_event.wait(interval_s):
        elapsed_s = int(time.time() - start_time)
        mm = elapsed_s // 60
        ss = elapsed_s % 60
        print(f"LOGGER: прогресс: прошло времени {mm:02d}:{ss:02d}", flush=True)


def build_transcribe_kwargs(config: dict) -> dict:
    kwargs = {
        "beam_size": int(config.get("beam_size", 5)),
        "word_timestamps": bool(config.get("word_timestamps", False)),
        "condition_on_previous_text": bool(config.get("condition_on_previous_text", True)),
    }

    language = config.get("language")
    if language is not None:
        language = str(language).strip()
        if language:
            kwargs["language"] = language

    return kwargs


def transcribe_single_audio(
    model: WhisperModel,
    audio_path: Path,
    transcription_dir: Path,
    config: dict,
) -> None:
    use_batched_inference = bool(config.get("use_batched_inference", False))
    transcribe_kwargs = build_transcribe_kwargs(config)
    batch_size = int(config.get("batch_size", 16))

    print(f"LOGGER: Файл для транскрипции: {audio_path.name}", flush=True)
    run_mode = "batched" if use_batched_inference else "single-pass"
    print(
        "LOGGER: Запуск транскрипции "
        f"(device={config['device']}, compute_type={config['compute_type']}, "
        f"model={config['model_size']}, mode={run_mode}, beam_size={config['beam_size']})",
        flush=True,
    )

    if use_batched_inference:
        pipeline = BatchedInferencePipeline(model=model)
        segments, info = pipeline.transcribe(
            str(audio_path), batch_size=batch_size, **transcribe_kwargs
        )
    else:
        segments, info = model.transcribe(str(audio_path), **transcribe_kwargs)

    header_line = "LOGGER: Язык '%s' с вероятностью %f" % (
        info.language,
        info.language_probability,
    )
    print(header_line)

    log_path = transcription_dir / audio_path.with_suffix(".log").name
    log_path.write_text(header_line, encoding="utf-8")

    with log_path.open("a", encoding="utf-8") as f:
        for segment in segments:
            line = "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)
            print(line, flush=True)
            f.write("\n" + line)
            f.flush()

    print(f"LOGGER: Транскрипция сохранена: {log_path}")


def main() -> None:
    config = load_config()
    validate_config(config)

    hf_token = str(config.get("hf_token", "")).strip()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

    config["model_size"] = str(config.get("model_size", "large-v3")).strip()
    config["device"] = str(config.get("device", "cpu")).strip().lower()
    config["compute_type"] = str(config.get("compute_type", "int8")).strip().lower()
    config["beam_size"] = int(config.get("beam_size", 5))
    config["batch_size"] = int(config.get("batch_size", 16))
    config["heartbeat_interval_s"] = int(config.get("heartbeat_interval_s", 5))
    config["processing_mode"] = str(config.get("processing_mode", "single")).strip().lower()
    config["audio_extensions"] = normalize_audio_extensions(
        config.get("audio_extensions", DEFAULT_AUDIO_EXTENSIONS)
    )

    base_path_candidates = [
        Path(str(raw).replace("~", str(Path.home()), 1))
        for raw in config.get(
            "base_path_candidates",
            ["~/OneDrive/Рабочий стол", "~/OneDrive/Desktop", "~/Desktop"],
        )
    ]
    base_path = resolve_base_path(base_path_candidates)

    transcription_dir = base_path / str(
        config.get("transcription_folder_name", "transcription")
    )
    audio_dir = base_path / str(config.get("audio_folder_name", "audio"))
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcription_dir.mkdir(parents=True, exist_ok=True)

    audio_files = pick_audio_files(
        audio_dir, config["processing_mode"], config["audio_extensions"]
    )
    model = WhisperModel(
        config["model_size"],
        device=config["device"],
        compute_type=config["compute_type"],
    )

    start_time = time.time()
    stop_event = threading.Event()
    threading.Thread(
        target=heartbeat,
        args=(stop_event, start_time, config["heartbeat_interval_s"]),
        daemon=True,
    ).start()

    try:
        for index, audio_path in enumerate(audio_files, start=1):
            if len(audio_files) > 1:
                print(
                    f"LOGGER: [{index}/{len(audio_files)}] Обработка {audio_path.name}",
                    flush=True,
                )
            transcribe_single_audio(model, audio_path, transcription_dir, config)
    except Exception as exc:
        print(f"LOGGER: Ошибка транскрипции: {exc}", flush=True)
        print("LOGGER: Проверьте модель, аудиофайл и настройки CUDA/cuDNN.", flush=True)
        raise SystemExit(1) from exc
    finally:
        stop_event.set()

    duration_s = time.time() - start_time
    print(f"LOGGER: Время транскрипции: {duration_s:.2f} сек")


if __name__ == "__main__":
    main()

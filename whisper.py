import threading
import time
import os
from pathlib import Path

from faster_whisper import WhisperModel
try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "LOGGER: Не найден пакет PyYAML. Установите его командой: pip install pyyaml"
    ) from exc


def load_config() -> dict:
    config_path = Path(__file__).with_name("config.yaml")
    if not config_path.exists():
        raise SystemExit(f"LOGGER: Файл конфигурации не найден: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise SystemExit("LOGGER: config.yaml должен содержать YAML-объект (map).")
    return loaded


config = load_config()

HF_TOKEN = str(config.get("hf_token", "")).strip()
MODEL_SIZE = str(config.get("model_size", "large-v3"))
BASE_PATH_CANDIDATES = [
    Path(str(raw).replace("~", str(Path.home()), 1))
    for raw in config.get(
        "base_path_candidates",
        ["~/OneDrive/Рабочий стол", "~/OneDrive/Desktop", "~/Desktop"],
    )
]
AUDIO_FOLDER_NAME = str(config.get("audio_folder_name", "audio"))
TRANSCRIPTION_FOLDER_NAME = str(config.get("transcription_folder_name", "transcription"))
BEAM_SIZE = int(config.get("beam_size", 5))
DEVICE = str(config.get("device", "cpu"))
COMPUTE_TYPE = str(config.get("compute_type", "int8"))
HEARTBEAT_INTERVAL_S = int(config.get("heartbeat_interval_s", 5))

if HF_TOKEN:
    # Set both common names so downstream libraries can pick token reliably.
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

# Resolve base path using candidate paths from config.
base_path = None
for cand in BASE_PATH_CANDIDATES:
    if cand.exists():
        base_path = cand
        break

if base_path is None:
    raise SystemExit(
        "LOGGER: Ни один путь из `base_path_candidates` не найден.\n"
        "LOGGER: Укажите корректные пути в `config.yaml`."
    )

transcription_dir = base_path / TRANSCRIPTION_FOLDER_NAME
audio_dir = base_path / AUDIO_FOLDER_NAME

audio_dir.mkdir(parents=True, exist_ok=True)
transcription_dir.mkdir(parents=True, exist_ok=True)

mp3_files = sorted(
    p for p in audio_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp3"
)

if len(mp3_files) == 0:
    raise SystemExit(
        f"LOGGER: В папке нет mp3 файлов: {audio_dir.resolve()}\n"
        "LOGGER: Положите в папку ровно один .mp3 файл."
    )

if len(mp3_files) > 1:
    files_str = ", ".join(p.name for p in mp3_files)
    raise SystemExit(
        f"LOGGER: В папке найдено несколько mp3 файлов ({len(mp3_files)}): {files_str}\n"
        "LOGGER: Оставьте только один .mp3 файл и запустите скрипт снова."
    )

audio_path = mp3_files[0]

def transcribe_with(device: str, compute_type: str):
    """Run transcribe and return (segments, info)."""
    model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type)
    return model.transcribe(str(audio_path), beam_size=BEAM_SIZE)

start_time = time.time()
stop_event = threading.Event()


def heartbeat() -> None:
    # Periodic progress so the terminal doesn't look "stuck" during model download/initialization.
    interval_s = HEARTBEAT_INTERVAL_S
    while not stop_event.wait(interval_s):
        elapsed_s = int(time.time() - start_time)
        mm = elapsed_s // 60
        ss = elapsed_s % 60
        print(f"LOGGER: прогресс: прошло времени {mm:02d}:{ss:02d}", flush=True)


threading.Thread(target=heartbeat, daemon=True).start()

try:
    print(f"LOGGER: Файл для транскрипции: {audio_path.name}", flush=True)
    print("LOGGER: Запуск транскрипции (CPU)...", flush=True)
    segments, info = transcribe_with(DEVICE, COMPUTE_TYPE)
finally:
    stop_event.set()

header_line = (
    "LOGGER: Язык '%s' с вероятностью %f"
    % (info.language, info.language_probability)
)
print(header_line)

log_path = transcription_dir / audio_path.with_suffix(".log").name

log_path.write_text(header_line, encoding="utf-8")

# Update log incrementally so you can see progress even before completion.
with log_path.open("a", encoding="utf-8") as f:
    for segment in segments:
        line = "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)
        print(line, flush=True)
        f.write("\n" + line)
        f.flush()

duration_s = time.time() - start_time
duration_line = f"LOGGER: Время транскрипции: {duration_s:.2f} сек"
with log_path.open("a", encoding="utf-8") as f:
    f.write("\n" + duration_line + "\n")

print(f"LOGGER: {duration_line}")
print(f"LOGGER: Транскрипция сохранена: {log_path}")

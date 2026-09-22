## Файлы

| Файл | Назначение |
|---|---|
| `Dockerfile` | Образ для NVIDIA GPU (CUDA) |
| `Dockerfile.cpu` | Образ без GPU — для AMD и любых машин без CUDA |
| `docker-compose.yml` / `docker-compose.cpu.yml` | Запуск через compose |
| `process_video.py` | Основной скрипт |
| `patch_omniparser.py` | Чинит два бага апстрима, которые всплывают на кадрах без текста |

## Сборка

GPU:
```bash
docker build -t omniparser-video:latest .
```

CPU (AMD):
```bash
docker build -f Dockerfile.cpu -t omniparser-video:cpu .
```

## Запуск

```bash
mkdir -p input output cache/easyocr cache/paddleocr
cp /путь/к/video.mp4 input/video.mp4

docker run --rm \
  -v "$(pwd)/input:/data/input" \
  -v "$(pwd)/output:/data/output" \
  -v "$(pwd)/cache/easyocr:/root/.EasyOCR" \
  -v "$(pwd)/cache/paddleocr:/root/.paddleocr" \
  omniparser-video:cpu \
  /data/input/video.mp4 -o /data/output/result.json \
  --fps 0.33 --imgsz 480 --no-captions
```

Для GPU-образа: добавить `--gpus all`, убрать `--no-captions`.

Результат — `output/result.json`.

## Основные флаги

| Флаг | Описание | По умолчанию |
|---|---|---|
| `--fps` | Сколько кадров в секунду видео анализировать (`0` = каждый кадр) | `0.33` |
| `--max-frames` | Ограничить число обрабатываемых кадров (для теста) | без ограничения |
| `--no-captions` | Не описывать иконки через Florence-2 — сильно быстрее на CPU | выкл |
| `--imgsz` | Разрешение для YOLO (меньше = быстрее, менее точно) | `640` |
| `--device` | `auto` / `cpu` / `cuda` | `auto` |
| `--save-annotated` | Сохранять размеченные PNG кадров | выкл |

## Формат JSON

```jsonc
{
  "video_file": "video.mp4",
  "source_fps": 29.97,
  "frames_processed": 4,
  "frames": [
    {
      "frame_index": 0,
      "timestamp_sec": 0.0,
      "timestamp": "0:00:00",
      "elements": [
        { "type": "text", "bbox": [0.0, 0.0, 0.09, 0.12], "content": "Settings", "interactivity": false },
        { "type": "icon", "bbox": [0.85, 0.02, 0.91, 0.07], "content": null, "interactivity": true }
      ]
    }
  ]
}
```

`bbox` — `[x1, y1, x2, y2]`, нормализовано (0–1) относительно размеров кадра.

## Замечания

- На CPU лучше начинать с `--fps 0.2–0.5`, `--imgsz 320–480`, `--no-captions`.

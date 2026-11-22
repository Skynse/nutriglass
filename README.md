# nutriglass

Measures how long a meal takes from video: YOLO finds the dish on every frame, and SAM 3
measures how much of it is covered by food every couple of seconds. The meal runs from when
food is served until the dish stays (nearly) empty.

## Setup

```
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv
uv sync
```

## Run the monitor

```
uv run python src/eater/monitor.py data/videos/<video> [--show]      # a video file
uv run python src/eater/monitor.py 0 --show                          # a webcam
```

Outputs go to `runs/monitor/<name>/`: `timeline.csv`, `summary.json` (meal start/end/duration),
`fill_curve.png` and `annotated.mp4`. See `--help` for thresholds and sampling rates.

## Evaluate

`eval_plate_state.ipynb` (kernel "Python 3 (nutriglass)") compares YOLO alone with YOLO + SAM 3
on the eval images in `data/eval/` and on monitor runs in `runs/monitor/`.

## Layout

```
src/eater/monitor.py        eating-duration monitor (YOLO + SAM 3)
src/eater/models/sam3/      SAM 3 weights
data/eval/                  eval images (empty.png, full.png)
data/videos/                test videos
training/datasets/          food, plates_glasses, unified (food + plates, 4 classes), bowl_frames
training/runs/detect/       YOLO training runs; current model: unified-1-runpod/weights/best.pt
training/weights/           pretrained YOLO base weights
training/visualize.py       live YOLO preview on a video
training/label_video_frames.py  auto-label video frames for fine-tuning
experiments/qwen_vl/        earlier Qwen3-VL food-description test
docs/                       notes and ideas
```

## Train

```
cd training
yolo train model=weights/yolo26s.pt data=datasets/unified/data.yaml imgsz=640 epochs=60
```

"""Auto-label video frames of a single bowl/plate with a known state.

The detector already finds the bowl reliably; on unseen bowls it's the class
that's wrong. So we keep its best box and overwrite the class with the state we
know from the video's timeline. Each frame is written twice: the full frame,
and a zoomed crop around the bowl (close-ups like the eval images).

The last --val-frac of the frames (in time order) go to valid/, so near-duplicate
neighbouring frames don't leak between splits.

Usage:
  python3 label_video_frames.py WEIGHTS FRAMES_DIR CLASS OUT_DIR --prefix salmon_empty
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("weights")
parser.add_argument("frames", type=Path)
parser.add_argument("cls", choices=["food", "plate_empty"])
parser.add_argument("out", type=Path)
parser.add_argument("--prefix", required=True)
parser.add_argument("--conf", type=float, default=0.25)
parser.add_argument("--val-frac", type=float, default=0.2)
parser.add_argument("--crop-scale", type=float, default=1.6, help="crop size relative to the box")
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

CLASS_IDS = {"food": 0, "plate_empty": 1}  # matches unified/data.yaml
BOWL_CLASSES = {0, 1}

model = YOLO(args.weights)
frames = sorted(args.frames.glob("*.jpg"))
n_val = int(len(frames) * args.val_frac)
kept = skipped = 0

for i, path in enumerate(frames):
    split = "valid" if i >= len(frames) - n_val else "train"
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    r = model.predict(img, conf=args.conf, device=args.device, verbose=False)[0]
    boxes = [(conf, box) for cls, conf, box in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())
             if int(cls) in BOWL_CLASSES]
    if not boxes:
        skipped += 1
        continue
    _, (x0, y0, x1, y1) = max(boxes)

    # Crop around the bowl, clamped to the frame.
    cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * args.crop_scale / 2
    cx0, cy0 = int(max(0, cx - half)), int(max(0, cy - half))
    cx1, cy1 = int(min(w, cx + half)), int(min(h, cy + half))

    for suffix, image, (ox, oy, iw, ih) in (
        ("", img, (0, 0, w, h)),
        ("_crop", img[cy0:cy1, cx0:cx1], (cx0, cy0, cx1 - cx0, cy1 - cy0)),
    ):
        bx0, by0 = max(0, x0 - ox), max(0, y0 - oy)
        bx1, by1 = min(iw, x1 - ox), min(ih, y1 - oy)
        line = f"{CLASS_IDS[args.cls]} {(bx0 + bx1) / 2 / iw:.6f} {(by0 + by1) / 2 / ih:.6f} {(bx1 - bx0) / iw:.6f} {(by1 - by0) / ih:.6f}\n"
        name = f"{args.prefix}_{path.stem}{suffix}"
        for sub in ("images", "labels"):
            (args.out / split / sub).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.out / split / "images" / f"{name}.jpg"), image)
        (args.out / split / "labels" / f"{name}.txt").write_text(line)
    kept += 1

print(f"{args.prefix}: labeled {kept} frames ({kept * 2} images), skipped {skipped} with no bowl box")

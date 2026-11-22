"""Play a video with live YOLO food detections, kept in sync with real time.

Usage: python3 visualize.py WEIGHTS VIDEO [--conf 0.25]
Keys: q/Esc quit, space pause, +/- change confidence threshold.
"""
import argparse
import time

import cv2
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("weights")
parser.add_argument("video")
parser.add_argument("--conf", type=float, default=0.25)
args = parser.parse_args()

model = YOLO(args.weights)
cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
window = "nutriglass food detector"
cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window, 1280, 720)

conf = args.conf
start = time.perf_counter()
frame_idx = 0
paused = False
infer_ms = 0.0

while True:
    if not paused:
        # Skip frames we've fallen behind on so playback stays at real-time speed.
        target = int((time.perf_counter() - start) * fps)
        while frame_idx < target - 1:
            if not cap.grab():
                break
            frame_idx += 1
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        t0 = time.perf_counter()
        result = model.predict(frame, conf=conf, verbose=False)[0]
        infer_ms = 0.9 * infer_ms + 0.1 * (time.perf_counter() - t0) * 1000

        shown = result.plot()
        scores = result.boxes.conf.tolist()
        hud = (
            f"t={frame_idx / fps:6.1f}s  boxes={len(scores)}  "
            f"max conf={max(scores, default=0):.2f}  thresh={conf:.2f}  infer={infer_ms:.0f}ms"
        )
        cv2.rectangle(shown, (0, 0), (shown.shape[1], 44), (0, 0, 0), -1)
        cv2.putText(shown, hud, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imshow(window, shown)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
        break
    if key == ord(" "):
        paused = not paused
        # Shift the clock so resuming doesn't jump ahead by the paused duration.
        pause_started = time.perf_counter() if paused else pause_started
        if not paused:
            start += time.perf_counter() - pause_started
    elif key in (ord("+"), ord("=")):
        conf = min(conf + 0.05, 0.95)
    elif key == ord("-"):
        conf = max(conf - 0.05, 0.05)

cap.release()
cv2.destroyAllWindows()

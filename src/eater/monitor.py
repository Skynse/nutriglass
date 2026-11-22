"""Eating-duration monitor: YOLO finds the dish, SAM 3 measures how full it is.

YOLO runs on every analysed frame (fast, ~10 ms) and gives the live dish box.
Every --sam-every seconds SAM 3 segments the food, the dish and any hands; the
fill level is the share of the *visible* dish covered by food, so a hand over
the bowl doesn't read as "less food". Food outside the dish (a lemon on the
board) is ignored.

The meal starts when the smoothed fill level first reaches --start-fill (food
served) and ends once it stays at or below --end-fill for --end-hold seconds.

Outputs in --out: timeline.csv (one row per SAM sample), summary.json,
fill_curve.png and, unless --no-video, an annotated video.

Usage:
  uv run python src/eater/monitor.py VIDEO_OR_CAMERA_INDEX [--show]
"""
import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YOLO = ROOT / "training/runs/detect/unified-1-runpod/weights/best.pt"
DEFAULT_SAM = ROOT / "src/eater/models/sam3"
DISH_CLASSES = {"food", "plate_empty"}  # YOLO classes that mark a dish


class FillMeter:
    """SAM 3 fill level: food area / visible dish area."""

    def __init__(self, model_dir, dish_prompt, threshold=0.4):
        self.dtype = torch.bfloat16
        self.model = Sam3Model.from_pretrained(model_dir, dtype=self.dtype).to("cuda").eval()
        self.processor = Sam3Processor.from_pretrained(model_dir)
        self.prompts = {"food": "food", "dish": dish_prompt, "hand": "hand"}
        self.text = {k: self.processor(text=p, return_tensors="pt").to("cuda") for k, p in self.prompts.items()}
        self.threshold = threshold

    @staticmethod
    def _filled_hull(mask):
        """Largest dish region with its interior filled (food sits on top of the dish mask)."""
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = np.zeros(mask.shape, np.uint8)
        if contours:
            cv2.drawContours(out, [cv2.convexHull(max(contours, key=cv2.contourArea))], -1, 1, -1)
        return out.astype(bool)

    @torch.inference_mode()
    def measure(self, frame_bgr):
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=img, return_tensors="pt").to("cuda")
        vision = self.model.get_vision_features(pixel_values=inputs["pixel_values"].to(self.dtype))
        sizes = inputs["original_sizes"].tolist()
        masks = {}
        for key, text in self.text.items():
            out = self.model(vision_embeds=vision, **text)
            r = self.processor.post_process_instance_segmentation(
                out, threshold=self.threshold, mask_threshold=0.5, target_sizes=sizes)[0]
            masks[key] = r["masks"].any(0).cpu().numpy() if len(r["masks"]) else np.zeros(frame_bgr.shape[:2], bool)

        dish = self._filled_hull(masks["dish"])
        visible = dish & ~masks["hand"]
        food = masks["food"] & visible
        visible_frac = visible.sum() / dish.sum() if dish.any() else 0.0
        # No dish in view, or most of it hidden behind hands: no reliable reading.
        fill = float(food.sum() / visible.sum()) if visible_frac >= 0.4 else None
        return {"fill": fill, "visible_frac": float(visible_frac), "food": food, "dish": dish, "hand": masks["hand"]}


class MealTracker:
    def __init__(self, start_fill, end_fill, end_hold, window=5):
        self.start_fill, self.end_fill, self.end_hold = start_fill, end_fill, end_hold
        self.recent = deque(maxlen=window)
        self.state = "WAITING"
        self.meal_start = self.meal_end = None
        self._low_since = None

    def update(self, t, fill):
        if fill is not None:
            self.recent.append(fill)
        smooth = float(np.median(self.recent)) if self.recent else None
        if smooth is None:
            return None
        if self.state == "WAITING" and smooth >= self.start_fill:
            self.state, self.meal_start = "EATING", t
        elif self.state == "EATING":
            if smooth <= self.end_fill:
                self._low_since = self._low_since if self._low_since is not None else t
                if t - self._low_since >= self.end_hold:
                    self.state, self.meal_end = "FINISHED", self._low_since
            else:
                self._low_since = None
        return smooth

    def elapsed(self, t):
        if self.meal_start is None:
            return None
        return (self.meal_end if self.meal_end is not None else t) - self.meal_start


def draw(frame, t, dets, sam, smooth, tracker, history, width=1280):
    vis = frame.copy()
    if sam is not None:
        vis[sam["hand"]] = (0.6 * vis[sam["hand"]] + 0.4 * np.array([255, 120, 0])).astype(np.uint8)
        vis[sam["food"]] = (0.55 * vis[sam["food"]] + 0.45 * np.array([0, 220, 0])).astype(np.uint8)
        contours, _ = cv2.findContours(sam["dish"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 200, 255), 3)
    for name, conf, (x0, y0, x1, y1) in dets:
        cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 255), 2)
        cv2.putText(vis, f"yolo {name} {conf:.2f}", (int(x0), int(y0) - 8), 0, 0.8, (255, 255, 255), 2)

    vis = cv2.resize(vis, (width, int(width * vis.shape[0] / vis.shape[1])))
    h, w = vis.shape[:2]
    elapsed = tracker.elapsed(t)
    hud = (f"t={t:6.1f}s  fill={'--' if smooth is None else f'{smooth:4.0%}'}  {tracker.state}"
           + ("" if elapsed is None else f"  meal {int(elapsed // 60)}:{int(elapsed % 60):02d}"))
    cv2.rectangle(vis, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(vis, hud, (12, 28), 0, 0.85, (255, 255, 255), 2)

    # Fill-level sparkline, bottom right.
    gw, gh, pad = 360, 110, 12
    x0, y0 = w - gw - pad, h - gh - pad
    cv2.rectangle(vis, (x0, y0), (x0 + gw, y0 + gh), (0, 0, 0), -1)
    for level, col in ((tracker.start_fill, (0, 200, 0)), (tracker.end_fill, (0, 0, 220))):
        y = int(y0 + gh - level * gh)
        cv2.line(vis, (x0, y), (x0 + gw, y), col, 1)
    pts = [(ts, f) for ts, f in history if f is not None]
    if len(pts) > 1:
        t_max = max(pts[-1][0], 1e-6)
        poly = np.array([(x0 + int(ts / t_max * (gw - 1)), int(y0 + gh - f * gh)) for ts, f in pts], np.int32)
        cv2.polylines(vis, [poly], False, (255, 255, 255), 2)
    return vis


def save_plot(rows, tracker, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = [r["t"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t, [np.nan if r["fill_raw"] == "" else r["fill_raw"] for r in rows], ".", color="0.6", label="raw sample")
    ax.plot(t, [np.nan if r["fill_smooth"] == "" else r["fill_smooth"] for r in rows], "-", color="C0", label="smoothed")
    ax.axhline(tracker.start_fill, color="C2", lw=0.8, ls="--", label="start threshold")
    ax.axhline(tracker.end_fill, color="C3", lw=0.8, ls="--", label="end threshold")
    if tracker.meal_start is not None:
        ax.axvspan(tracker.meal_start, tracker.meal_end if tracker.meal_end is not None else t[-1], color="C1", alpha=0.12,
                   label="meal")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("food / visible dish")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="video file, or a camera index such as 0")
    p.add_argument("--yolo", default=str(DEFAULT_YOLO))
    p.add_argument("--sam", default=str(DEFAULT_SAM))
    p.add_argument("--out", type=Path, help="output dir (default runs/monitor/<source name>)")
    p.add_argument("--fps", type=float, default=10, help="frames analysed per second of video")
    p.add_argument("--sam-every", type=float, default=2.0, help="seconds between SAM 3 fill measurements")
    p.add_argument("--dish-prompt", default="bowl", help='SAM 3 prompt for the dish, e.g. "plate"')
    p.add_argument("--start-fill", type=float, default=0.30)
    p.add_argument("--end-fill", type=float, default=0.15)
    p.add_argument("--end-hold", type=float, default=10.0, help="seconds the fill must stay low to end the meal")
    p.add_argument("--show", action="store_true", help="live preview window (q to quit)")
    p.add_argument("--no-video", action="store_true", help="skip writing the annotated video")
    args = p.parse_args()

    camera = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if camera else args.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = 1 if camera else max(1, round(src_fps / args.fps))
    out_dir = args.out or ROOT / "runs/monitor" / ("camera" if camera else Path(args.source).stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading models... ", end="", flush=True)
    yolo = YOLO(args.yolo)
    meter = FillMeter(args.sam, args.dish_prompt)
    tracker = MealTracker(args.start_fill, args.end_fill, args.end_hold)
    print("done")

    writer = None
    rows, history = [], []
    sam, smooth, last_sam_t = None, None, -1e9
    frame_idx, wall0 = 0, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = time.perf_counter() - wall0 if camera else frame_idx / src_fps
            for _ in range(step - 1):  # skip to the next analysed frame
                cap.grab()
            frame_idx += step

            r = yolo.predict(frame, conf=0.25, device=0, verbose=False)[0]
            dets = [(yolo.names[int(c)], float(s), b) for c, s, b in
                    zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())
                    if yolo.names[int(c)] in DISH_CLASSES]
            top = max(dets, key=lambda d: d[1]) if dets else None

            if t - last_sam_t >= args.sam_every:
                last_sam_t = t
                sam = meter.measure(frame)
                smooth = tracker.update(t, sam["fill"])
                history.append((t, smooth))
                rows.append({"t": round(t, 2),
                             "fill_raw": "" if sam["fill"] is None else round(sam["fill"], 4),
                             "fill_smooth": "" if smooth is None else round(smooth, 4),
                             "dish_visible": round(sam["visible_frac"], 3),
                             "state": tracker.state,
                             "yolo_class": top[0] if top else "", "yolo_conf": round(top[1], 3) if top else ""})

            if args.show or not args.no_video:
                vis = draw(frame, t, dets, sam, smooth, tracker, history)
                if not args.no_video:
                    if writer is None:
                        writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 src_fps / step if not camera else args.fps, vis.shape[1::-1])
                    writer.write(vis)
                if args.show:
                    cv2.imshow("nutriglass monitor", vis)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
            if len(rows) % 30 == 1 and rows[-1]["t"] == round(t, 2):
                print(f"  t={t:6.1f}s  fill={rows[-1]['fill_smooth']}  {tracker.state}  "
                      f"({frame_idx / max(time.perf_counter() - wall0, 1e-6):.0f} video frames/s)", flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if not rows:
        raise SystemExit("no frames analysed")
    with open(out_dir / "timeline.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    save_plot(rows, tracker, out_dir / "fill_curve.png")
    summary = {
        "source": args.source,
        "meal_start_s": tracker.meal_start,
        "meal_end_s": tracker.meal_end,
        "duration_s": None if tracker.meal_start is None or tracker.meal_end is None
        else round(tracker.meal_end - tracker.meal_start, 1),
        "final_state": tracker.state,
        "samples": len(rows),
        "processing_s": round(time.perf_counter() - wall0, 1),
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("meal_start_s", "meal_end_s", "duration_s", "final_state", "processing_s")}))
    print(f"outputs in {out_dir}")


if __name__ == "__main__":
    main()

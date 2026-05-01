# SafeDriveVision — Starter

Real-time driver monitoring that detects:

- **Phone use** (YOLOv8, COCO `cell phone` class)
- **Drowsiness** via face landmarks (MediaPipe Face Mesh):
  - Eye Aspect Ratio (EAR) → eye closure
  - Mouth Aspect Ratio (MAR) → yawning
  - Head pose pitch → nodding / inattention

This is a simplified architecture compared to the dissertation (which used FaceBoxes + Sim3DR + Basel Face Model). MediaPipe gives you 468 face landmarks plus head pose out of the box, with one pip install.

## Setup

Recommended Python: 3.9–3.11 (MediaPipe's wheels are most reliable here).

```bash
cd safedrivevision
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

The first run will auto-download `yolov8n.pt` (~6 MB).

## Run

```bash
python main.py            # default webcam
python main.py 1          # webcam index 1
python main.py clip.mp4   # video file
```

Press **`q`** in the video window to quit.

## Optional audio alerts

Create a `sounds/` folder next to `main.py` and drop in:

- `phone.mp3` — phone detected
- `drowsy.mp3` — sustained eye closure
- `yawn.mp3` — yawn detected
- `attention.mp3` — head down / looking away

If files are missing, alerts still print to the console.

## Tuning

All thresholds are at the top of `main.py`:

| Constant | Default | What it does |
|---|---|---|
| `EAR_THRESHOLD` | 0.21 | EAR below this counts as "eyes closed" |
| `EAR_CONSEC_FRAMES` | 30 | sustained closed-eye frames before drowsy alert (~1s @ 30fps) |
| `MAR_THRESHOLD` | 0.60 | MAR above this counts as a yawn |
| `HEAD_PITCH_THRESHOLD` | 20° | head pitch deviation that triggers attention alert |
| `PHONE_CONFIDENCE` | 0.40 | YOLO minimum confidence for phone class |
| `ALERT_COOLDOWN` | 4.0 s | suppresses repeat alerts of the same type |

The dissertation calibrates EAR per-driver on first startup (records baseline EAR, then sets the threshold relative to it). That's a good next step — easy to add: capture the first ~5 seconds of video, average the driver's EAR, then set `EAR_THRESHOLD = baseline * 0.75`.

## What's deliberately simplified vs the paper

| Paper | This starter |
|---|---|
| YOLOv5 + custom training | YOLOv8 pretrained on COCO (phone class already there) |
| FaceBoxes for face detection | MediaPipe Face Mesh handles detection + landmarks together |
| Sim3DR + BFM for head pose | `cv2.solvePnP` with 6 landmarks (much simpler, ~equally accurate for pitch) |
| ONNX Runtime | Not used — Ultralytics handles inference. Add later for speed if needed. |
| Per-driver EAR calibration | Fixed threshold (add calibration as next step) |
| 4 separate MP3 alert types | Same — drop your own MP3s into `sounds/` |

## Suggested next steps

1. **Per-driver EAR calibration** on startup (5-second baseline)
2. **NAR (Nose Aspect Ratio)** — the paper computes this; trivial with MediaPipe landmarks
3. **Train a custom phone-in-hand detector** if COCO false-positives bother you (the paper does this)
4. **Edge deployment** — convert YOLO to ONNX/TFLite, port to Raspberry Pi or Jetson Nano
5. **Mobile build** — TensorFlow Lite versions of both models, wrap with React Native or Flutter

## Troubleshooting

- **`mediapipe` won't install** — use Python 3.9, 3.10, or 3.11. 3.12 wheels arrived later and can be patchy on some platforms.
- **No webcam found** — try `python main.py 1` or higher indexes.
- **Slow / low FPS** — close other apps; consider `yolov8n` (already the smallest). For more speed, export to ONNX: `yolo export model=yolov8n.pt format=onnx`.
- **EAR always high (no drowsy alerts firing in test)** — that's correct unless you actually close your eyes for ~1 s. Lower `EAR_CONSEC_FRAMES` to test more aggressively.

"""
SafeDriveVision - Starter
==========================
Real-time driver monitoring:
  1) Phone usage detection (YOLOv8 via Ultralytics)
  2) Drowsiness detection from face landmarks (MediaPipe Face Mesh)
       - Eye Aspect Ratio (EAR)        -> eye closure
       - Mouth Aspect Ratio (MAR)      -> yawning
       - Head pose (pitch via solvePnP)-> head nodding / inattention

Quit with the 'q' key.

Run:
    python main.py            # uses default webcam (device 0)
    python main.py 1          # use webcam device 1
    python main.py video.mp4  # use a video file
"""

import os
import sys
import time
import threading

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("Missing dependency. Run:  pip install ultralytics")

try:
    import mediapipe as mp
except ImportError:
    sys.exit("Missing dependency. Run:  pip install mediapipe")


# =====================================================================
# CONFIG
# =====================================================================
EAR_THRESHOLD        = 0.21   # eyes considered closed below this
EAR_CONSEC_FRAMES    = 20     # ~0.67 s @ 30 fps of sustained closure -> drowsy
MAR_THRESHOLD        = 0.60   # mouth open wide -> yawn
HEAD_PITCH_THRESHOLD = 20.0   # degrees of forward nodding -> attention alert
PHONE_CONFIDENCE     = 0.40   # YOLO threshold for phone class

ALERT_COOLDOWN = 4.0          # seconds between repeated alerts of same type
OVERLAY_DURATION = 1.5        # seconds the visual flash stays on screen

PHONE_CLASS_ID = 67           # COCO class id for "cell phone"


# =====================================================================
# MEDIAPIPE FACE MESH LANDMARK INDICES
# =====================================================================
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH     = [78, 308, 13, 14]
HEAD_POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]

MODEL_POINTS_3D = np.array([
    (   0.0,    0.0,    0.0),
    (   0.0, -330.0,  -65.0),
    (-225.0,  170.0, -135.0),
    ( 225.0,  170.0, -135.0),
    (-150.0, -150.0, -125.0),
    ( 150.0, -150.0, -125.0),
], dtype=np.float64)


# =====================================================================
# GEOMETRY HELPERS
# =====================================================================
def _dist(p1, p2):
    return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))


def eye_aspect_ratio(landmarks, eye_idx, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_idx]
    v1 = _dist(pts[1], pts[5])
    v2 = _dist(pts[2], pts[4])
    horiz = _dist(pts[0], pts[3])
    return (v1 + v2) / (2.0 * horiz) if horiz > 0 else 0.0


def mouth_aspect_ratio(landmarks, mouth_idx, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in mouth_idx]
    horiz = _dist(pts[0], pts[1])
    vert  = _dist(pts[2], pts[3])
    return vert / horiz if horiz > 0 else 0.0


def head_pose_pitch(landmarks, w, h):
    image_points = np.array([
        (landmarks[i].x * w, landmarks[i].y * h)
        for i in HEAD_POSE_LANDMARKS
    ], dtype=np.float64)

    focal = float(w)
    cam_matrix = np.array([
        [focal,  0,     w / 2.0],
        [0,      focal, h / 2.0],
        [0,      0,     1      ],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    ok, rvec, _ = cv2.solvePnP(
        MODEL_POINTS_3D, image_points, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    sy    = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
    yaw   = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
    roll  = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
    return pitch, yaw, roll


# =====================================================================
# ALERT MANAGER  — synthesized audio (no external files needed)
# =====================================================================
_ALERT_COLORS = {
    'phone':     (0,   0,   255),   # red
    'drowsy':    (0,   165, 255),   # orange
    'yawn':      (0,   220, 220),   # yellow
    'attention': (255, 100,   0),   # blue-ish
}
_ALERT_LABELS = {
    'phone':     'PUT DOWN YOUR PHONE!',
    'drowsy':    'WAKE UP!',
    'yawn':      'FEELING SLEEPY?',
    'attention': 'EYES ON THE ROAD!',
}


class AlertManager:
    def __init__(self):
        self._last_played = {}
        self._sounds      = {}
        self._pygame      = None
        self._active      = {}   # alert_type -> expiry timestamp

        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self._pygame = pygame
            self._sounds = self._build_sounds()
            print("[audio] Synthesized alert tones ready.")
        except Exception as e:
            print(f"[audio] disabled ({e}). Console-only alerts.")

    # ------------------------------------------------------------------
    def _build_sounds(self):
        SR = 44100

        def tone(freq, dur):
            n = int(dur * SR)
            if freq == 0:
                return np.zeros(n, dtype=np.float32)
            t = np.linspace(0, dur, n, endpoint=False, dtype=np.float32)
            return np.sin(2 * np.pi * freq * t, dtype=np.float32)

        def make(segments, volume=0.88):
            wave = np.concatenate([tone(f, d) for f, d in segments])
            wave = (np.clip(wave, -1, 1) * volume * 32767).astype(np.int16)
            arr  = np.ascontiguousarray(np.column_stack([wave, wave]))
            return self._pygame.sndarray.make_sound(arr)

        return {
            # rapid urgent beeping
            'phone':     make([(880,.12),(0,.06),(880,.12),(0,.06),(880,.12),(0,.06),(880,.15)]),
            # loud alternating alarm
            'drowsy':    make([(440,.22),(580,.22),(440,.22),(580,.22),(440,.22),(580,.22)], volume=1.0),
            # triple medium beep
            'yawn':      make([(660,.18),(0,.08),(660,.18),(0,.08),(660,.22)]),
            # two-tone warning
            'attention': make([(523,.16),(0,.06),(523,.16),(0,.06),(698,.28)]),
        }

    # ------------------------------------------------------------------
    def trigger(self, alert_type, message):
        now = time.time()
        if now - self._last_played.get(alert_type, 0) < ALERT_COOLDOWN:
            return
        self._last_played[alert_type] = now
        self._active[alert_type]      = now + OVERLAY_DURATION
        print(f"[ALERT/{alert_type}] {message}")
        if self._pygame and alert_type in self._sounds:
            threading.Thread(
                target=lambda: self._sounds[alert_type].play(),
                daemon=True,
            ).start()

    # ------------------------------------------------------------------
    def draw_overlay(self, frame):
        now = time.time()
        fired = {t: exp for t, exp in self._active.items() if now < exp}
        self._active = fired
        if not fired:
            return frame

        # Use the most recently triggered alert for color/label
        alert_type = max(fired, key=fired.get)
        color  = _ALERT_COLORS.get(alert_type, (255, 255, 255))
        label  = _ALERT_LABELS.get(alert_type, 'ALERT!')

        # Pulsing border — alpha oscillates at 4 Hz
        pulse = 0.35 + 0.25 * abs(np.sin(now * 4 * np.pi))
        overlay = frame.copy()
        thickness = 30
        h, w = frame.shape[:2]
        cv2.rectangle(overlay, (0, 0), (w, h), color, thickness * 2)
        cv2.addWeighted(overlay, pulse, frame, 1 - pulse, 0, frame)

        # Big centered alert text
        font  = cv2.FONT_HERSHEY_DUPLEX
        scale = min(w, h) / 400.0 * 1.6
        thick = max(2, int(scale * 2))
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        cx = (w - tw) // 2
        cy = h // 2 + th // 2
        cv2.putText(frame, label, (cx, cy), font, scale, (0, 0, 0),   thick + 4, cv2.LINE_AA)
        cv2.putText(frame, label, (cx, cy), font, scale, color,        thick,     cv2.LINE_AA)

        return frame


# =====================================================================
# MAIN LOOP
# =====================================================================
def main(source=0):
    print("Loading YOLOv8 (downloads ~6 MB on first run)...")
    yolo = YOLO("yolov8n.pt")

    print("Loading MediaPipe Face Mesh...")
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    alerts = AlertManager()
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Could not open video source: {source}")

    eye_closed_frames = 0
    fps_count, fps_ts, fps_display = 0, time.time(), 0.0

    print("Running. Press 'q' in the video window to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ---------- 1) Phone detection ----------
        phone_detected = False
        results = yolo(frame, verbose=False, conf=PHONE_CONFIDENCE,
                       classes=[PHONE_CLASS_ID])
        for r in results:
            for box in r.boxes:
                phone_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"PHONE {conf:.2f}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if phone_detected:
            alerts.trigger("phone", "Mobile phone in frame")

        # ---------- 2) Face / drowsiness ----------
        ear = mar = pitch = 0.0
        face_found = False
        face_results = face_mesh.process(rgb)
        if face_results.multi_face_landmarks:
            face_found = True
            lm  = face_results.multi_face_landmarks[0].landmark
            ear = (eye_aspect_ratio(lm, LEFT_EYE, w, h) +
                   eye_aspect_ratio(lm, RIGHT_EYE, w, h)) / 2.0
            mar = mouth_aspect_ratio(lm, MOUTH, w, h)
            pitch, _, _ = head_pose_pitch(lm, w, h)

            for idx in LEFT_EYE + RIGHT_EYE + MOUTH:
                x, y = int(lm[idx].x * w), int(lm[idx].y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            if ear < EAR_THRESHOLD:
                eye_closed_frames += 1
                if eye_closed_frames >= EAR_CONSEC_FRAMES:
                    alerts.trigger("drowsy",
                                   f"Sustained eye closure (EAR={ear:.2f})")
            else:
                eye_closed_frames = 0

            if mar > MAR_THRESHOLD:
                alerts.trigger("yawn", f"Yawn detected (MAR={mar:.2f})")

            if abs(pitch) > HEAD_PITCH_THRESHOLD:
                alerts.trigger("attention",
                               f"Head deviation (pitch={pitch:+.1f} deg)")

        # ---------- 3) Alert visual overlay ----------
        frame = alerts.draw_overlay(frame)

        # ---------- 4) FPS counter ----------
        fps_count += 1
        if time.time() - fps_ts >= 1.0:
            fps_display = fps_count / (time.time() - fps_ts)
            fps_count   = 0
            fps_ts      = time.time()

        # ---------- 5) HUD overlay ----------
        hud = [
            f"FPS:   {fps_display:5.1f}",
            f"EAR:   {ear:5.3f}   (thr {EAR_THRESHOLD})",
            f"MAR:   {mar:5.3f}   (thr {MAR_THRESHOLD})",
            f"Pitch: {pitch:+6.1f} deg",
            f"Phone: {'YES' if phone_detected else 'no'}",
            f"Eyes-closed frames: {eye_closed_frames}/{EAR_CONSEC_FRAMES}",
        ]
        if not face_found:
            hud.append("!! NO FACE DETECTED !!")

        for i, line in enumerate(hud):
            y     = 24 + i * 22
            color = (0, 0, 255) if line.startswith("!!") else (255, 255, 255)
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color,     1, cv2.LINE_AA)

        cv2.imshow("SafeDriveVision  (press 'q' to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------
if __name__ == "__main__":
    src = 0
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        src = int(arg) if arg.isdigit() else arg
    main(source=src)

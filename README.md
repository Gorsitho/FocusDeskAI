# FocusDesk AI

A desktop application that estimates a user's observable working state from their webcam.

It reports **only observable visual signals** (presence, head orientation, posture, a visible phone).
It is not a psychological or medical assessment.

## What it shows

| Signal | Source |
| --- | --- |
| Person present | MediaPipe Pose / Face, YOLO `person` |
| Face present | MediaPipe Face Landmarker |
| Head yaw / pitch / roll | Facial transformation matrix from MediaPipe |
| Posture (Upright / Leaning / Slouching) | Shoulder and nose landmarks from MediaPipe Pose |
| Cell phone | Ultralytics YOLO (COCO class `cell phone`) |
| Looking at the phone | Head direction compared with the phone's position in the frame |
| State | Rule-based: `FOCUSED`, `DISTRACTED`, `IDLE`, `AWAY`, plus a manual `BREAK` |

## Pipeline

```
Webcam → OpenCV → MediaPipe (face, pose) → head pose (yaw/pitch/roll)
Webcam → YOLO (cell phone, person)
          ↓
   Feature extraction (pose, gaze, activity)
          ↓
   Rule-based focus state
          ↓
   PySide6 dashboard
```

Capture and inference run in a `QThread`, so the interface never blocks on the camera.

## Workspace setup

At startup a setup window asks for the number of monitors (1–3) and where the camera sits
(top center, top of the left/right monitor, left side, right side, below center).
From this the app computes the range of head yaw/pitch that means "looking at a monitor":
for example, with the camera on the left side, working means looking to the camera's right.
Both values can be changed later under **⚙ Settings** and are saved to
`%APPDATA%\FocusDeskAI\user_settings.json`.

## State rules

Rules are evaluated in priority order. Nothing is decided from a single frame: every rule
needs a behaviour to last, head angles are smoothed, short interruptions (< 1 s) pause a
behaviour timer instead of resetting it, and a new state is shown only after it has
persisted for 1 s.

1. **AWAY** – no person detected for 3 s.
2. **DISTRACTED (phone)** – a phone is visible **and** the head is turned away from the monitors
   towards it, for the phone duration (default 3 s). A phone on the desk while you look at a
   monitor stays `FOCUSED`.
3. **DISTRACTED** – head turned outside the monitor zone (left/right/up/down, strong head roll),
   or the face is hidden while the body is visible, for the distraction duration (default 5 s).
4. **IDLE** – present but almost motionless for the idle duration (default 60 s).
5. **FOCUSED** – otherwise.

**BREAK** is set with the ☕ Break button and pauses distraction detection until you resume.

Durations, detection sensitivity (width of the monitor zone and phone-gaze tolerance), and the
distraction sound (on/off, volume) are all editable in **⚙ Settings**.

## Controls

| Button | Action |
| --- | --- |
| ☕ Break / ▶ Resume | Enter or leave the `BREAK` state |
| ↺ New session | Reset the per-state session timers |
| ⏱ Show/Hide timers | Toggle the time spent in each state |
| ⚙ Settings | Workspace, durations, sensitivity, sound |

While `DISTRACTED`, a soft chime repeats every 3 s and stops as soon as the state changes.

## Setup

Requires Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

## Run

From the project root:

```bash
python -m app.main
```

On first launch the MediaPipe models (`face_landmarker.task`, `pose_landmarker_lite.task`)
and the YOLO weights (`yolo11n.pt`) are downloaded into `models/`.

If a component cannot be loaded (no camera, model download failure, PyTorch unavailable),
the app keeps running and reports it in the status line. The camera is retried automatically.

## Tests

```bash
python -m pytest
```

## Project layout

```
app/
  config/    settings and thresholds
  vision/    camera, face, pose, head pose, object detection
  features/  posture, gaze and activity features, rule-based state
  ui/        PySide6 main window, camera preview, dashboard
  ml/        (planned) model training and inference
  data/      (planned) data collection and labelling
  database/  (planned) persistence
```

## License

MIT

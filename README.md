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
| Looking at the phone | Head direction compared with the phone's bounding box in the frame |
| Monitor being looked at | Head pose vs. the configured 3D monitor layout |
| State | Rule-based: `FOCUSED`, `DISTRACTED`, `AWAY`, plus a manual `BREAK` |

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

## Workspace setup (3D)

At startup a setup window shows a **top view of your desk**: you, up to three monitors, and
the webcam. You can

- choose 1, 2 or 3 monitors (arranged by default side by side on a semicircle facing you),
- drag yourself or any monitor (every monitor stays within **0.3–1 m** of you),
- rotate a monitor with its round handle, the mouse wheel or the angle slider
  (double-click a monitor to turn it towards you),
- set the distance to the nearest monitor with a slider,
- pick the monitor and edge (top, bottom, left, right) the webcam is mounted on.

Each monitor rectangle is projected to the head yaw/pitch needed to look at it from your
seat, relative to the camera. The shaded wedges in the plan show these "allowed" directions.
Looking at **any** configured monitor is focus; only looking clearly outside all of them
(plus a sensitivity-dependent tolerance) for the configured time counts as distraction.

Head angles are expressed relative to the line towards the camera (MediaPipe measures them
relative to the lens axis), and a slow, outlier-protected **automatic pitch calibration**
removes the user's vertical head-pose bias (shown under *Head → Pitch calibration*).
If a monitor lies so far to the side that the face leaves the camera's view, the app keeps
crediting that monitor while the face is out of view after a large turn towards it.

Everything is editable later under **⚙ Settings** and saved to
`%APPDATA%\FocusDeskAI\user_settings.json` (older settings files are migrated).

## State rules

States: `FOCUSED`, `DISTRACTED`, `AWAY`, and the manual `BREAK`.

Rules are evaluated in priority order. Nothing is decided from a single frame: every rule
needs a behaviour to last, head angles are smoothed, short interruptions (< 1 s) pause a
behaviour timer instead of resetting it, and a new state is shown only after it has
persisted for 1 s.

1. **AWAY** – nobody detected for 3 s, or no movement at all for the
   *No movement before away* time (default 30 s; catches an empty chair or a coat that the
   detectors mistake for a person).
2. **DISTRACTED (phone)** – a phone is visible **and** the head points towards the phone's
   bounding box, for the phone duration (default 1 s). A phone on the desk while you look at
   a monitor stays `FOCUSED`; a phone held up close to your face in your line of sight counts
   even when a monitor is behind it.
3. **DISTRACTED** – looking outside every monitor (left/right/up/down/between monitors,
   strong head roll), or the face is hidden while the body is visible, for the distraction
   duration (default 1 s).
4. **FOCUSED** – otherwise.

**BREAK** is set with the ☕ Break button and pauses distraction detection until you resume.

## Settings

| Section | Options |
| --- | --- |
| Workspace | monitors, positions, angles, distance, camera mount |
| Detection | looking away before distracted, looking at phone before distracted, no movement before away, sensitivity |
| Sound | on/off, volume, test |
| Display | show detection landmarks (off by default; detection runs either way) |
| Language | English, Español, Deutsch – applies to the whole application and is remembered |

Typed values accept both `7.5` and `7,5`; out-of-range values are clamped, never dropped.

## Controls

| Button | Action |
| --- | --- |
| ▶ START / ■ STOP SESSION | Start a new session (timers from 00:00) / end it and save its log |
| ☕ Break / ▶ Resume | Enter or leave the `BREAK` state |
| ⏱ Show/Hide timers | Toggle the time spent in each state |
| ⚙ Settings | Workspace, detection, sound, display, language |

While `DISTRACTED`, a soft chime repeats every 3 s and stops as soon as the state changes.

## Sessions

Detection runs as soon as the app starts, but time is only counted during a session.
**START** begins a new session with all timers at 00:00; **STOP SESSION** ends it and
writes exactly one file (closing the app also ends and saves a running session):

```
%LOCALAPPDATA%\FocusDeskAI\sessions\session_0000.log, session_0001.log, ...
```

Each file holds the session number, exact start and end time (local, with UTC offset),
total duration and the duration and share of `FOCUSED`, `DISTRACTED`, `AWAY` and `BREAK`
(plus `NO DATA` for time without a camera image). Durations are measured with a
monotonic clock, so clock changes cannot distort them; the wall clock is only used for the
timestamps. Numbers continue after the highest existing file; files are never overwritten.

## Logs

Technical events (startup/shutdown, model loading, camera, worker thread, YOLO, phone and
state changes, warnings and unexpected errors with tracebacks) are written to

```
%LOCALAPPDATA%\FocusDeskAI\FocusDeskAI.log         (rotating, 3 backups)
%LOCALAPPDATA%\FocusDeskAI\FocusDeskAI_native.log  (native crashes, MediaPipe C++ output)
```

This works without a console, including PyInstaller builds with `console=False`, where
library output (Ultralytics, download progress) is redirected into the log. Session logs
are separate from the technical log.

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

## Build

```bash
python -m PyInstaller FocusDeskAI.spec --noconfirm        # windowed (console=False)
python -m PyInstaller FocusDeskAI_Debug.spec --noconfirm  # same, with a console
```

Both specs bundle the three model files from `models/`, so download them first by running
the app once from source. If a model is missing in a build, it is downloaded to
`%LOCALAPPDATA%\FocusDeskAI\models` instead of the (possibly read-only) install folder.

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

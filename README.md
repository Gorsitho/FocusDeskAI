# FocusDesk AI

A desktop application that estimates a user's observable working state from their webcam.

It reports **only observable visual signals** (presence, head and eye direction, eye opening, posture,
a visible phone).
It is not a psychological or medical assessment.

## What it shows

| Signal | Source |
| --- | --- |
| Present | **The face** (MediaPipe Face Landmarker); a body alone does not count |
| Person visible | MediaPipe Pose / Face, YOLO `person` (shown, but not presence) |
| Head yaw / pitch / roll | Facial transformation matrix from MediaPipe |
| Eye direction | Iris position between the eye corners (MediaPipe iris landmarks) |
| Eyes open / partly closed / closed | Eye aspect ratio vs. the user's learned open eye, MediaPipe blink score |
| Posture (Upright / Leaning / Slouching) | Shoulder and nose landmarks from MediaPipe Pose |
| Cell phone | Ultralytics YOLO (`cell phone`, and `remote` when held), full frame plus enlarged hand crops |
| Phone in hand / at the ear | Phone box vs. wrist and finger landmarks (MediaPipe Pose) and the face |
| Looking at the phone | Gaze (head + eyes) compared with the phone's bounding box in the frame |
| Monitor being looked at | Gaze (head + eyes) vs. the configured 3D monitor layout |
| State | Rule-based: `FOCUSED`, `DISTRACTED`, `AWAY`, plus a manual `BREAK` |

## Pipeline

```
Webcam → OpenCV → MediaPipe (face + iris, pose) → head pose, eye state, eye direction
Webcam → YOLO (cell phone, remote, person; extra pass on the hands)
          ↓
   Feature extraction (pose, gaze = head + eyes, phone evidence, activity)
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

## Eyes

The **eye direction** (iris position between the eye corners) is added to the head
direction, so the gaze, not just the head, decides whether you look at a monitor. A glance
away with the eyes alone counts, and a slightly turned head with the eyes on the screen
stays focused. While the eyes are measured, the tolerance around the monitors on that axis
is halved, because the remaining uncertainty is smaller. A slow calibration learns the
neutral iris position while the head points at a monitor.

The **eye state** is open, partly closed or closed. It compares the eyelid gap (corrected
for the head angle) with *your* normal open eye, which is learned, and uses MediaPipe's
blink score as a second opinion. Looking down lowers the eyelids naturally, so the limits
are relaxed the further you look down (lower screen, keyboard, notebook). Partly closed eyes
never count as a distraction on their own; closures shorter than a blink are ignored.

## Phone detection

Only objects that are very likely a phone count; background objects are filtered out in
several steps:

- **Zoomed verification:** every phone box below 0.75 confidence is cut out with some
  context, enlarged and checked by YOLO again. Background objects rarely pass this.
- **Plausibility:** relative to your face, a phone must be large enough to be near you
  (not far behind you), smaller than a screen, not above your head, within reach to the
  side, and not pen-shaped. A phone in your hand passes these checks anywhere; without a
  visible face only a held phone counts.
- **Confirmation:** one very clear box (0.75 or more) is enough. Otherwise several YOLO
  runs must agree **and** at least one box must be strong (0.55 or more) or the phone must
  be in your hand. Repeated weak boxes of a background object are never enough.
- A second YOLO pass looks at enlarged crops around your hands, where small phones are.
- Phones in the hand are often labelled `remote` by YOLO; a remote counts when it is held
  or at your face, never when it lies on the desk.
- A phone in your hand is kept for 4 s when YOLO loses it (fingers hide it), and YOLO runs
  more often while a phone is around.
- A phone counts as **in use** when your gaze (head and eyes) points at it, when it is held
  up in front of your face, when it is held at your ear (a call), or when it is in your hand
  while your face is briefly not visible.

Everything is editable later under **⚙ Settings** and saved to
`%APPDATA%\FocusDeskAI\user_settings.json` (older settings files are migrated).

## State rules

States: `FOCUSED`, `DISTRACTED`, `AWAY`, and the manual `BREAK`.

Rules are evaluated in priority order. Nothing is decided from a single frame: every rule
needs a behaviour to last, head angles are smoothed, short interruptions (< 1 s) pause a
behaviour timer instead of resetting it, and a new state is shown only after it has
persisted for 1 s.

1. **AWAY** – **no face** for 3 s, even if a body or another person is still visible
   (reason *Face not visible*; *Nobody in front of the camera* when nothing is visible).
   Shorter face dropouts (head bowed over a phone or a notebook, a quick turn) are bridged
   by the body as before. Also AWAY: no movement at all for the
   *No movement before away* time (default 30 s; catches an empty chair or a coat that the
   detectors mistake for a person).
2. **DISTRACTED (phone)** – a phone is in use (see *Phone detection*), for the phone
   duration (default 1 s). A phone on the desk while you look at a monitor stays `FOCUSED`;
   a phone held up close to your face in your line of sight counts even when a monitor is
   behind it, and so does a call.
3. **DISTRACTED (eyes closed)** – eyes closed (not blinking) for 2 s.
4. **DISTRACTED** – the gaze is outside every monitor (left/right/up/down/between monitors,
   strong head roll), or the face is hidden while the body is visible, for the distraction
   duration (default 1 s).
5. **FOCUSED** – otherwise.

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

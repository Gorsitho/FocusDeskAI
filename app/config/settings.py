"""
This file holds the central configuration of FocusDesk AI.
It defines the fixed numbers (limits, sizes, times) that the detection and the
state rules use, so they can be changed in one place without changing the pipeline code.

Main parts:

* Workspace, MonitorPlacement, CameraEdge: the 3D desk layout (user, monitors, webcam).
* StudyMethod: Computer, Tablet / Notebook or Mixed.
* CameraSettings, ModelSettings, DetectionSettings, FeatureSettings, StateSettings:
  the settings for each step of the pipeline, collected in Settings.

The values that the user can change in the app live in app/config/user_settings.py.
apply_user_settings() copies them into these settings.
Almost every other module reads its settings from this file.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from app.config import paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# In a PyInstaller build PROJECT_ROOT is the bundle's _internal folder.
MODELS_DIR = PROJECT_ROOT / "models"


def model_path(name: str) -> Path:
    """Bundled model if present; otherwise where it may be downloaded to.

    A frozen build's install folder can be read-only, so missing models are
    downloaded to the per-user data folder instead.
    """
    bundled = MODELS_DIR / name
    if bundled.exists() or not paths.is_frozen():
        return bundled
    return paths.user_models_dir() / name

# Workspace limits, in metres.
MAX_MONITOR_DISTANCE = 1.0
MIN_MONITOR_DISTANCE = 0.3
MAX_MONITORS = 3
DEFAULT_MONITOR_DISTANCE = 0.65
DEFAULT_MONITOR_WIDTH = 0.53  # a 24" 16:9 monitor
DEFAULT_MONITOR_HEIGHT = 0.30


class CameraEdge(str, Enum):
    """Edge of the chosen monitor the webcam is mounted on."""

    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"


class StudyMethod(str, Enum):
    """What the user studies with, i.e. which head directions count as focused."""

    COMPUTER = "computer"  # looking at the configured monitors
    TABLET = "tablet"  # looking down at a tablet or notebook on the desk
    MIXED = "mixed"  # either of the above


@dataclass(frozen=True)
class MonitorPlacement:
    """A monitor in the top-down workspace plan.

    Coordinates are metres: x to the user's right, z forward (away from the user).
    `angle` is the rotation around the vertical axis in degrees, clockwise seen
    from above; 0 means the screen faces straight back along -z.
    """

    x: float
    z: float
    angle: float = 0.0
    width: float = DEFAULT_MONITOR_WIDTH
    height: float = DEFAULT_MONITOR_HEIGHT


@dataclass(frozen=True)
class Workspace:
    """Approximate 3D desk layout: where the user sits and where the monitors are."""

    person_x: float = 0.0
    person_z: float = 0.0
    monitors: tuple[MonitorPlacement, ...] = (MonitorPlacement(0.0, DEFAULT_MONITOR_DISTANCE),)
    camera_monitor: int = 0
    camera_edge: CameraEdge = CameraEdge.TOP

    @property
    def monitor_count(self) -> int:
        return len(self.monitors)


def arc_layout(count: int, distance: float = DEFAULT_MONITOR_DISTANCE,
               person: tuple[float, float] = (0.0, 0.0)) -> tuple[MonitorPlacement, ...]:
    """Monitors side by side on a semicircle around the person, each facing them."""
    count = max(1, min(MAX_MONITORS, count))
    distance = max(MIN_MONITOR_DISTANCE, min(MAX_MONITOR_DISTANCE, distance))
    # Angular step so that neighbouring screens just touch (plus a small bezel gap).
    step = math.degrees(2 * math.atan((DEFAULT_MONITOR_WIDTH / 2 + 0.015) / distance))
    monitors = []
    for i in range(count):
        angle = (i - (count - 1) / 2.0) * step
        rad = math.radians(angle)
        monitors.append(MonitorPlacement(
            x=person[0] + distance * math.sin(rad), z=person[1] + distance * math.cos(rad), angle=angle,
        ))
    return tuple(monitors)


@dataclass(frozen=True)
class CameraSettings:
    """Which webcam to use, its image size and frame rate."""
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    reconnect_interval_s: float = 2.0


@dataclass(frozen=True)
class ModelSettings:
    """Where the AI model files are, and where to download them from."""
    face_landmarker_path: Path = field(default_factory=lambda: model_path("face_landmarker.task"))
    face_landmarker_url: str = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    )
    pose_landmarker_path: Path = field(default_factory=lambda: model_path("pose_landmarker_lite.task"))
    pose_landmarker_url: str = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    )
    # Ultralytics downloads the weights automatically on first use.
    yolo_weights_path: Path = field(default_factory=lambda: model_path("yolo11n.pt"))


@dataclass(frozen=True)
class DetectionSettings:
    """Confidence limits for the detectors and how often YOLO runs."""
    min_face_confidence: float = 0.5
    min_pose_confidence: float = 0.5
    # Person boxes.
    yolo_confidence: float = 0.4
    # Weakest phone box YOLO reports. Weak boxes are often background objects, so they
    # must also pass the zoomed verification below and the checks in PhoneTracker.
    phone_confidence: float = 0.35
    # COCO "remote": phones held in the hand are often labelled as one. Only used when held.
    remote_confidence: float = 0.45
    # Verification: every phone box below phone_verify_skip is cut out (with some context,
    # enlarged to phone_verify_imgsz pixels) and YOLO must find a phone there again with at
    # least phone_verify_confidence. Background objects rarely survive this. 0 turns it off.
    phone_verify_imgsz: int = 320
    phone_verify_confidence: float = 0.4
    phone_verify_skip: float = 0.75
    # Side of the verification crop relative to the box's longer side (at least 128 px).
    phone_verify_context: float = 2.5
    # YOLO is the most expensive stage; running it every N frames keeps the
    # pipeline responsive while still catching a phone within a fraction of a second.
    yolo_every_n_frames: int = 5
    # While a phone is (or was just) visible YOLO runs more often, to follow it closely.
    yolo_every_n_frames_active: int = 2
    # Second YOLO pass on crops around the hands (input size in pixels), which finds
    # small phones in the hand that the full-frame pass misses. 0 turns it off.
    hand_roi_imgsz: int = 320
    # Side of a hand crop relative to the shoulder width (at least hand_roi_min_px).
    hand_roi_scale: float = 1.1
    hand_roi_min_px: int = 160
    # Minimum landmark visibility for a pose keypoint to be trusted.
    min_landmark_visibility: float = 0.5
    # Hands are often half out of view; wrists/fingers need less visibility to be used.
    min_hand_visibility: float = 0.3


@dataclass(frozen=True)
class FeatureSettings:
    """Numbers used to turn detections into features (head direction, phone, posture, timers)."""
    workspace: Workspace = field(default_factory=Workspace)
    study_method: StudyMethod = StudyMethod.COMPUTER
    # Vertical position of the monitor centres relative to the user's eyes (metres).
    monitor_center_height: float = -0.12
    # Fraction of a gaze shift performed by turning the head; the eyes do the rest.
    head_yaw_ratio: float = 0.7
    head_pitch_ratio: float = 0.5
    # Tolerance (degrees) added around every monitor; derived from the sensitivity setting.
    attention_margin: float = 16.0
    # Extra tolerance while already looking at a monitor, so small head movements at its
    # edge do not count as looking away (hysteresis).
    monitor_exit_margin: float = 6.0
    # Beyond this head yaw the face landmarker usually loses the face.
    face_tracking_limit: float = 45.0
    # Vertical field of view MediaPipe's face geometry assumes; used to express the
    # head pose relative to the line towards the camera rather than its optical axis.
    camera_vertical_fov: float = 63.0
    # Automatic pitch calibration: learns the user's head-pitch bias while they look
    # at the monitors. Samples further than `calibration_max_error` from the expected
    # pitch are ignored, and the learned offset is capped.
    calibration_window_s: float = 120.0
    calibration_min_samples: int = 20
    calibration_sample_every_s: float = 0.5
    calibration_max_error: float = 20.0
    calibration_max_offset: float = 15.0
    # Tablet / notebook and mixed study: any head pitch below this (degrees, relative to
    # the line towards the camera) counts as looking down at the desk, in any direction.
    desk_pitch_max: float = 0.0
    # Head roll beyond this (e.g. resting the head on a hand) counts as off-screen.
    max_head_roll: float = 35.0
    # Extra tolerance on top of the monitor zones for a turned torso (pose only).
    torso_margin: float = 35.0
    # Max angle between head direction and phone direction to count as looking at it.
    phone_gaze_max_angle: float = 45.0
    # The head must be turned at least this far (degrees) for its direction to be trusted.
    phone_gaze_min_turn: float = 8.0
    # "Held up near the face": phone box at least this size and its centre within this
    # distance of the face centre, both relative to the face height in the image.
    phone_near_min_size: float = 0.6
    phone_near_max_distance: float = 1.8
    # Keep "looking at the phone" this long when the face is briefly lost.
    phone_gaze_hold_s: float = 1.0
    # A phone stays "visible" this long after YOLO last saw it (YOLO misses frames).
    phone_hold_s: float = 2.5
    # ... and this long while it is in the user's hand (the fingers often hide it).
    phone_hold_in_hand_s: float = 4.0
    # Phone evidence: every YOLO run adds the box confidence; runs without the phone
    # multiply it by phone_miss_decay. A phone is confirmed by one very clear box
    # (phone_instant_confidence), or by evidence of at least phone_confirm_evidence when at
    # least one box was strong (phone_strong_confidence) or the phone is in the user's hand
    # or at the face. Repeated weak boxes of a background object are never enough.
    phone_confirm_evidence: float = 1.0
    phone_instant_confidence: float = 0.75
    phone_strong_confidence: float = 0.55
    phone_miss_decay: float = 0.5
    phone_max_evidence: float = 2.0
    # Plausibility relative to the user's face (sizes in face heights, distance in face
    # widths). A phone near the user appears at least phone_min_face_size large; smaller
    # boxes are far behind the user. Boxes larger than phone_max_face_size (a screen, a
    # picture frame), above the head, or further than phone_max_reach to the side are
    # rejected unless a hand holds them. Boxes longer than phone_max_aspect : 1 are not phones.
    phone_min_face_size: float = 0.3
    phone_max_face_size: float = 2.5
    phone_max_reach: float = 3.5
    phone_max_aspect: float = 4.0
    # Held in the hand: a wrist or finger point within this distance of the phone box,
    # relative to the box's longer side.
    phone_hand_max_distance: float = 0.6
    # Held at the ear (a call): the box centre at face height, at most this many face
    # widths beside the face, and at least phone_ear_min_size face heights large.
    phone_ear_max_offset: float = 0.9
    phone_ear_min_size: float = 0.35
    # --- eyes ---------------------------------------------------------------------
    # Converts the iris offset inside the eye (fraction of the eye width) into an
    # eye rotation: sin(angle) = offset * gain. ~2.5 for a 30 mm eye and 12 mm eyeball.
    eye_gaze_gain: float = 2.5
    # How strongly the measured eye rotation changes the head-only direction (0 = ignore
    # the eyes, 1 = full). The vertical iris position is less precise than the horizontal.
    eye_yaw_weight: float = 1.0
    eye_pitch_weight: float = 0.7
    # The attention margin is meant for the uncertainty of a head-only direction (how much
    # the eyes add is unknown). With the eyes measured on an axis, that axis uses this
    # fraction of the margin, so a look away with the eyes is noticed.
    eye_margin_scale: float = 0.5
    # Time constant of the eye-gaze smoothing.
    eye_smoothing_s: float = 0.15
    # The two eyes must agree within this many degrees, or the eye gaze is not used.
    eye_max_disagreement: float = 20.0
    # Learned neutral iris position (like the pitch calibration): samples while the
    # head points at a monitor; capped at eye_calibration_max_offset degrees.
    eye_calibration_window_s: float = 120.0
    eye_calibration_min_samples: int = 20
    eye_calibration_sample_every_s: float = 0.25
    eye_calibration_max_error: float = 15.0
    eye_calibration_max_offset: float = 10.0
    # Eye openness = eye aspect ratio / the user's normal open value. The normal value is
    # learned (a high percentile over eye_baseline_window_s); this is the starting guess.
    eye_open_ear: float = 0.28
    eye_baseline_window_s: float = 60.0
    eye_baseline_min_samples: int = 30
    eye_baseline_percentile: float = 85.0
    # Openness below these ratios is "closed" / "partially closed".
    eye_closed_ratio: float = 0.5
    eye_partial_ratio: float = 0.75
    # Blink blendshape (0..1, when MediaPipe provides it): closed needs at least
    # eye_blink_min; at eye_blink_closed or more the eye counts as closed with a
    # partly-open eye aspect ratio too.
    eye_blink_min: float = 0.35
    eye_blink_closed: float = 0.75
    # Looking down lowers the upper eyelids. Between eye_down_start and eye_down_full
    # degrees of downward gaze the closed/partial thresholds shrink by up to eye_down_relax,
    # so reading the lower screen, a keyboard or a notebook is not "eyes closing".
    eye_down_start: float = 5.0
    eye_down_full: float = 35.0
    eye_down_relax: float = 0.45
    # Eye closures shorter than this are blinks and do not interrupt anything.
    eye_blink_grace_s: float = 0.3
    # Time constant of the exponential smoothing applied to yaw/pitch/roll.
    head_smoothing_s: float = 0.25
    # A behaviour timer survives interruptions shorter than this.
    behavior_grace_s: float = 1.0
    # Posture thresholds.
    max_shoulder_tilt: float = 10.0
    min_head_height_ratio: float = 0.35
    max_head_offset_ratio: float = 0.35
    # Seconds of history used for temporal smoothing.
    history_window_s: float = 3.0


@dataclass(frozen=True)
class StateSettings:
    """Times used by the state rules: when the state becomes DISTRACTED or AWAY."""
    # No *face* for this long -> AWAY, even if a body is still visible. A body without a
    # face only bridges shorter face dropouts (head bowed over a phone or a notebook).
    away_after_s: float = 3.0
    # Someone "visible" but without any movement for this long -> AWAY (an empty
    # chair, a coat or a photo that the detectors mistake for a person).
    still_away_after_s: float = 30.0
    # Continuous time looking away from the monitors before DISTRACTED.
    look_away_after_s: float = 1.0
    # Continuous time looking at a visible phone before DISTRACTED.
    phone_after_s: float = 1.0
    # Continuous time with the eyes closed (not blinking) before DISTRACTED.
    eyes_closed_after_s: float = 2.0
    # Mean normalised landmark displacement per second below which nothing moves.
    still_motion_threshold: float = 0.01
    # A candidate state must persist this long before it is displayed (prevents flicker).
    min_state_duration_s: float = 1.0


@dataclass(frozen=True)
class Settings:
    """All pipeline settings together. The shared default instance is `settings` below."""
    camera: CameraSettings = field(default_factory=CameraSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    features: FeatureSettings = field(default_factory=FeatureSettings)
    state: StateSettings = field(default_factory=StateSettings)


settings = Settings()

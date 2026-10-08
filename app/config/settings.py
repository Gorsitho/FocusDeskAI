"""Central configuration for FocusDesk AI.

All thresholds used by the rule-based state logic live here so they can be
tuned without touching the pipeline code. Values the user can change from the
UI live in `app.config.user_settings` and are folded into these dataclasses
with `apply_user_settings`.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


class CameraPosition(str, Enum):
    """Where the webcam sits relative to the monitors, as seen by the user."""

    TOP_CENTER = "top_center"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    LEFT = "left"
    RIGHT = "right"
    BOTTOM_CENTER = "bottom_center"

    @property
    def label(self) -> str:
        return _CAMERA_POSITION_LABELS[self]

    def available_for(self, monitor_count: int) -> bool:
        # With a single monitor "above the left/right monitor" is the same as top center.
        return monitor_count > 1 or self not in (CameraPosition.TOP_LEFT, CameraPosition.TOP_RIGHT)


_CAMERA_POSITION_LABELS = {
    CameraPosition.TOP_CENTER: "Top center",
    CameraPosition.TOP_LEFT: "Top of left monitor",
    CameraPosition.TOP_RIGHT: "Top of right monitor",
    CameraPosition.LEFT: "Left side",
    CameraPosition.RIGHT: "Right side",
    CameraPosition.BOTTOM_CENTER: "Below center",
}


@dataclass(frozen=True)
class CameraSettings:
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    reconnect_interval_s: float = 2.0


@dataclass(frozen=True)
class ModelSettings:
    face_landmarker_path: Path = MODELS_DIR / "face_landmarker.task"
    face_landmarker_url: str = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    )
    pose_landmarker_path: Path = MODELS_DIR / "pose_landmarker_lite.task"
    pose_landmarker_url: str = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    )
    # Ultralytics downloads the weights automatically on first use.
    yolo_weights_path: Path = MODELS_DIR / "yolo11n.pt"


@dataclass(frozen=True)
class DetectionSettings:
    min_face_confidence: float = 0.5
    min_pose_confidence: float = 0.5
    yolo_confidence: float = 0.4
    # YOLO is the most expensive stage; running it every N frames keeps the
    # pipeline responsive while still catching a phone within a fraction of a second.
    yolo_every_n_frames: int = 5
    # Minimum landmark visibility for a pose keypoint to be trusted.
    min_landmark_visibility: float = 0.5


@dataclass(frozen=True)
class FeatureSettings:
    # Workspace layout; together these define the head angles that count as "on screen".
    monitor_count: int = 1
    camera_position: CameraPosition = CameraPosition.TOP_CENTER
    # Head rotation (degrees) needed to sweep across one monitor. The eyes do part
    # of the work, so this is smaller than the monitor's true visual angle.
    monitor_yaw_span: float = 30.0
    monitor_pitch_span: float = 15.0
    # Gap between the outer monitor edge and a camera mounted at the side.
    side_camera_offset: float = 5.0
    # Tolerance added around the monitors; derived from the sensitivity setting.
    attention_margin: float = 14.0
    # Head roll beyond this (e.g. resting the head on a hand) counts as off-screen.
    max_head_roll: float = 35.0
    # Extra tolerance on top of the screen zone for a turned torso (pose only).
    torso_margin: float = 35.0
    # Max angle between head direction and phone direction to count as looking at it.
    phone_gaze_max_angle: float = 45.0
    # A phone stays "visible" this long after YOLO last saw it (YOLO misses frames).
    phone_hold_s: float = 1.0
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
    away_after_s: float = 3.0
    # Continuous time looking away from the monitors before DISTRACTED.
    look_away_after_s: float = 5.0
    # Continuous time looking at a visible phone before DISTRACTED.
    phone_after_s: float = 3.0
    still_idle_after_s: float = 60.0
    # Mean normalised landmark displacement per second below which the user is "still".
    still_motion_threshold: float = 0.01
    # A candidate state must persist this long before it is displayed (prevents flicker).
    min_state_duration_s: float = 1.0


@dataclass(frozen=True)
class Settings:
    camera: CameraSettings = field(default_factory=CameraSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    features: FeatureSettings = field(default_factory=FeatureSettings)
    state: StateSettings = field(default_factory=StateSettings)


settings = Settings()

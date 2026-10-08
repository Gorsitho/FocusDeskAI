"""Screen-attention features derived from head orientation and the monitor layout."""

import math
from dataclasses import dataclass
from enum import Enum

from app.config.settings import CameraPosition, FeatureSettings
from app.vision.head_pose import HeadPose


class Attention(str, Enum):
    """Where the user's attention appears to be in the current frame."""

    ON_SCREEN = "On screen"
    LEFT = "Looking left"
    RIGHT = "Looking right"
    DOWN = "Looking down"
    UP = "Looking up"
    HEAD_TILTED = "Head tilted"
    PHONE = "Looking at phone"
    BODY_TURNED = "Body turned away"
    NO_FACE = "Face not visible"
    ABSENT = "Nobody visible"


@dataclass(frozen=True)
class ScreenZone:
    """Head angles (degrees, HeadPose convention) for which the user faces a monitor."""

    yaw_min: float
    yaw_max: float
    pitch_min: float
    pitch_max: float
    # Same box without the tolerance margin: exactly the monitors.
    core_yaw: tuple[float, float]
    core_pitch: tuple[float, float]

    @property
    def yaw_center(self) -> float:
        return (self.core_yaw[0] + self.core_yaw[1]) / 2.0

    @property
    def pitch_center(self) -> float:
        return (self.core_pitch[0] + self.core_pitch[1]) / 2.0

    def contains(self, yaw: float, pitch: float) -> bool:
        return self.yaw_min <= yaw <= self.yaw_max and self.pitch_min <= pitch <= self.pitch_max

    def in_core(self, yaw: float, pitch: float, slack: float = 0.0) -> bool:
        return (self.core_yaw[0] - slack <= yaw <= self.core_yaw[1] + slack
                and self.core_pitch[0] - slack <= pitch <= self.core_pitch[1] + slack)


def compute_screen_zone(cfg: FeatureSettings) -> ScreenZone:
    """Map the monitor layout and camera position to a box of head angles.

    Monitors sit side by side. Angles are first expressed as "degrees to the
    user's right of the monitor block centre", then shifted by the camera
    position: the head pose is measured relative to the camera, so a camera on
    the left of the monitors sees the user looking to its right when they work.
    """
    n, span, margin = cfg.monitor_count, cfg.monitor_yaw_span, cfg.attention_margin
    left, right = -n * span / 2.0, n * span / 2.0

    position = cfg.camera_position
    if position is CameraPosition.TOP_LEFT and n > 1:
        camera_x = left + span / 2.0
    elif position is CameraPosition.TOP_RIGHT and n > 1:
        camera_x = right - span / 2.0
    elif position is CameraPosition.LEFT:
        camera_x = left - cfg.side_camera_offset
    elif position is CameraPosition.RIGHT:
        camera_x = right + cfg.side_camera_offset
    else:
        camera_x = 0.0

    # HeadPose yaw is positive towards the user's left, i.e. the opposite of "rightwards".
    core_yaw = (camera_x - right, camera_x - left)

    v = cfg.monitor_pitch_span
    if position is CameraPosition.BOTTOM_CENTER:
        core_pitch = (0.0, v)  # monitors are above the camera
    elif position in (CameraPosition.LEFT, CameraPosition.RIGHT):
        core_pitch = (-v / 2.0, v / 2.0)
    else:
        core_pitch = (-v, 0.0)  # camera on top: the user looks slightly down at the screens

    return ScreenZone(
        yaw_min=core_yaw[0] - margin,
        yaw_max=core_yaw[1] + margin,
        pitch_min=core_pitch[0] - margin,
        pitch_max=core_pitch[1] + margin,
        core_yaw=core_yaw,
        core_pitch=core_pitch,
    )


@dataclass
class GazeFeatures:
    # None when the head pose is unknown (no face).
    facing_screen: bool | None = None
    looking_sideways: bool = False
    looking_down: bool = False
    looking_up: bool = False
    head_tilted: bool = False
    # Off-screen direction from the user's point of view, or ON_SCREEN / NO_FACE.
    direction: Attention = Attention.NO_FACE


def extract_gaze_features(
    head_pose: HeadPose | None, cfg: FeatureSettings, zone: ScreenZone | None = None
) -> GazeFeatures:
    if head_pose is None:
        return GazeFeatures()
    zone = zone or compute_screen_zone(cfg)

    turned_left = head_pose.yaw > zone.yaw_max
    turned_right = head_pose.yaw < zone.yaw_min
    down = head_pose.pitch < zone.pitch_min
    up = head_pose.pitch > zone.pitch_max
    tilted = abs(head_pose.roll) > cfg.max_head_roll

    if down:
        direction = Attention.DOWN
    elif turned_left:
        direction = Attention.LEFT
    elif turned_right:
        direction = Attention.RIGHT
    elif up:
        direction = Attention.UP
    elif tilted:
        direction = Attention.HEAD_TILTED
    else:
        direction = Attention.ON_SCREEN

    return GazeFeatures(
        facing_screen=direction is Attention.ON_SCREEN,
        looking_sideways=turned_left or turned_right,
        looking_down=down,
        looking_up=up,
        head_tilted=tilted,
        direction=direction,
    )


class HeadPoseSmoother:
    """Exponential smoothing of yaw/pitch/roll that adapts to the frame rate."""

    def __init__(self, time_constant_s: float, reset_after_s: float = 0.5):
        self._tau = time_constant_s
        self._reset_after = reset_after_s
        self._state: HeadPose | None = None
        self._last_time: float | None = None

    def update(self, timestamp: float, pose: HeadPose | None) -> HeadPose | None:
        if pose is None:
            if self._last_time is not None and timestamp - self._last_time > self._reset_after:
                self._state = None
            return None
        if self._state is None or self._tau <= 0:
            self._state = pose
        else:
            dt = max(timestamp - self._last_time, 0.0)
            alpha = 1.0 - math.exp(-dt / self._tau)
            prev = self._state
            self._state = HeadPose(
                yaw=prev.yaw + alpha * (pose.yaw - prev.yaw),
                pitch=prev.pitch + alpha * (pose.pitch - prev.pitch),
                roll=prev.roll + alpha * (pose.roll - prev.roll),
            )
        self._last_time = timestamp
        return self._state

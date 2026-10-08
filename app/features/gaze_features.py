"""
This file decides where the user is looking: at a monitor, at the desk, or somewhere else.
It uses the head direction (yaw, pitch, roll) and the 3D desk layout.

How it works:

* The workspace (Workspace in app/config/settings.py) places the user's eyes
  and every monitor in metres.
* compute_screen_zones() turns each monitor into the range of head angles that
  are needed to look at it (a "monitor zone"). Angles are measured from the
  camera, like MediaPipe's head pose: looking straight into the camera is yaw = pitch = 0.
* extract_gaze_features() compares the current head angles with these zones.
  It also follows the study method: for Tablet / Notebook and Mixed, looking
  down counts as looking at the desk.
* HeadPoseSmoother makes the head angles calmer (less jitter), and
  PitchCalibrator learns the small up/down error of each user and camera.

It is used by app/features/feature_pipeline.py and app/features/phone_features.py.
"""

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from app.config.settings import CameraEdge, FeatureSettings, MonitorPlacement, StudyMethod, Workspace
from app.vision.head_pose import HeadPose

_CAMERA_GAP = 0.02  # webcam distance from the monitor edge, metres


class Attention(str, Enum):
    """Where the user's attention appears to be in the current frame."""

    ON_SCREEN = "on_screen"
    DESK = "desk"  # down at a tablet / notebook (only when the study method allows it)
    LEFT = "left"
    RIGHT = "right"
    DOWN = "down"
    UP = "up"
    BETWEEN = "between"  # inside the workspace span but not on any monitor
    HEAD_TILTED = "head_tilted"
    PHONE = "phone"
    BODY_TURNED = "body_turned"
    NO_FACE = "no_face"
    ABSENT = "absent"


@dataclass(frozen=True)
class MonitorZone:
    """Head angles (degrees, HeadPose convention) that look at one monitor."""

    index: int
    core_yaw: tuple[float, float]
    core_pitch: tuple[float, float]

    @property
    def center(self) -> tuple[float, float]:
        return (sum(self.core_yaw) / 2.0, sum(self.core_pitch) / 2.0)

    def contains(self, yaw: float, pitch: float, margin: float) -> bool:
        return (self.core_yaw[0] - margin <= yaw <= self.core_yaw[1] + margin
                and self.core_pitch[0] - margin <= pitch <= self.core_pitch[1] + margin)


@dataclass(frozen=True)
class ScreenZones:
    """The zones of all monitors, plus the tolerance (margin) around each of them."""
    monitors: tuple[MonitorZone, ...]
    margin: float

    def monitor_at(self, yaw: float, pitch: float, margin: float | None = None) -> int | None:
        """Index of the monitor being looked at, preferring the closest centre on overlaps."""
        margin = self.margin if margin is None else margin
        hits = [z for z in self.monitors if z.contains(yaw, pitch, margin)]
        if not hits:
            return None
        return min(hits, key=lambda z: math.hypot(yaw - z.center[0], pitch - z.center[1])).index

    @property
    def yaw_min(self) -> float:
        return min(z.core_yaw[0] for z in self.monitors) - self.margin

    @property
    def yaw_max(self) -> float:
        return max(z.core_yaw[1] for z in self.monitors) + self.margin

    @property
    def pitch_min(self) -> float:
        return min(z.core_pitch[0] for z in self.monitors) - self.margin

    @property
    def pitch_max(self) -> float:
        return max(z.core_pitch[1] for z in self.monitors) + self.margin


def _monitor_axes(monitor: MonitorPlacement) -> np.ndarray:
    """Unit vector along the screen towards the user's right (when facing the screen)."""
    rad = math.radians(monitor.angle)
    return np.array([math.cos(rad), 0.0, -math.sin(rad)])


def monitor_center(monitor: MonitorPlacement, cfg: FeatureSettings) -> np.ndarray:
    return np.array([monitor.x, cfg.monitor_center_height, monitor.z])


def camera_point(workspace: Workspace, cfg: FeatureSettings) -> np.ndarray:
    """3D position of the webcam, from its monitor and the edge it is mounted on."""
    monitor = workspace.monitors[workspace.camera_monitor]
    center, right = monitor_center(monitor, cfg), _monitor_axes(monitor)
    up = np.array([0.0, 1.0, 0.0])
    edge = workspace.camera_edge
    if edge is CameraEdge.BOTTOM:
        return center - up * (monitor.height / 2 + _CAMERA_GAP)
    if edge is CameraEdge.LEFT:
        return center - right * (monitor.width / 2 + _CAMERA_GAP)
    if edge is CameraEdge.RIGHT:
        return center + right * (monitor.width / 2 + _CAMERA_GAP)
    return center + up * (monitor.height / 2 + _CAMERA_GAP)


def _direction_angles(vector: np.ndarray) -> tuple[float, float]:
    """Azimuth (positive to the user's right) and elevation (positive up), degrees."""
    azimuth = math.degrees(math.atan2(vector[0], vector[2]))
    elevation = math.degrees(math.atan2(vector[1], math.hypot(vector[0], vector[2])))
    return azimuth, elevation


def _wrap(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def head_angles_towards(point: np.ndarray, workspace: Workspace, cfg: FeatureSettings) -> tuple[float, float]:
    """Head yaw/pitch (HeadPose convention, relative to the camera) to look at a 3D point."""
    eyes = np.array([workspace.person_x, 0.0, workspace.person_z])
    cam_az, cam_el = _direction_angles(camera_point(workspace, cfg) - eyes)
    az, el = _direction_angles(point - eyes)
    # The camera is assumed to be aimed at the user, so its direction is the zero point.
    # HeadPose yaw is positive towards the user's left, i.e. negative azimuth.
    gaze_yaw, gaze_pitch = -_wrap(az - cam_az), el - cam_el
    return gaze_yaw * cfg.head_yaw_ratio, gaze_pitch * cfg.head_pitch_ratio


def compute_screen_zones(cfg: FeatureSettings) -> ScreenZones:
    """Calculate the head angles that are needed to look at each monitor (one zone per monitor)."""
    workspace = cfg.workspace
    zones = []
    for index, monitor in enumerate(workspace.monitors):
        center, right = monitor_center(monitor, cfg), _monitor_axes(monitor)
        up = np.array([0.0, 1.0, 0.0])
        yaws, pitches = [], []
        for sx in (-0.5, 0.0, 0.5):
            for sy in (-0.5, 0.0, 0.5):
                point = center + right * (sx * monitor.width) + up * (sy * monitor.height)
                yaw, pitch = head_angles_towards(point, workspace, cfg)
                yaws.append(yaw)
                pitches.append(pitch)
        zones.append(MonitorZone(index, (min(yaws), max(yaws)), (min(pitches), max(pitches))))
    return ScreenZones(tuple(zones), cfg.attention_margin)


@dataclass
class GazeFeatures:
    # None when the head pose is unknown (no face).
    """Where the user is looking in one frame (made by extract_gaze_features)."""
    facing_screen: bool | None = None
    # Index of the monitor being looked at.
    monitor: int | None = None
    looking_sideways: bool = False
    looking_down: bool = False
    looking_up: bool = False
    head_tilted: bool = False
    # Looking down at the desk, which counts as focused for tablet / mixed study.
    on_desk: bool = False
    direction: Attention = Attention.NO_FACE


def extract_gaze_features(
    head_pose: HeadPose | None, cfg: FeatureSettings, zones: ScreenZones | None = None,
    previous_monitor: int | None = None,
) -> GazeFeatures:
    """`previous_monitor`, the monitor looked at in the last frame, is kept while the head
    stays within `monitor_exit_margin` beyond its zone."""
    if head_pose is None:
        return GazeFeatures()
    zones = zones or compute_screen_zones(cfg)

    method = cfg.study_method
    tilted = abs(head_pose.roll) > cfg.max_head_roll
    if method is StudyMethod.TABLET:
        # No monitors: any downward direction is the desk; everything else is off it.
        monitor = None
        down = head_pose.pitch < cfg.desk_pitch_max
        up = not down
        turned_left = not down and head_pose.yaw > zones.yaw_max
        turned_right = not down and head_pose.yaw < zones.yaw_min
        on_desk = down
    else:
        monitor = None if tilted else zones.monitor_at(head_pose.yaw, head_pose.pitch)
        if monitor is None and not tilted and previous_monitor is not None:
            zone = zones.monitors[previous_monitor]
            if zone.contains(head_pose.yaw, head_pose.pitch, zones.margin + cfg.monitor_exit_margin):
                monitor = previous_monitor
        turned_left = head_pose.yaw > zones.yaw_max
        turned_right = head_pose.yaw < zones.yaw_min
        down = head_pose.pitch < zones.pitch_min
        up = head_pose.pitch > zones.pitch_max
        # Mixed: any downward direction off the monitors is the desk.
        on_desk = method is StudyMethod.MIXED and monitor is None and head_pose.pitch < cfg.desk_pitch_max
    on_desk = on_desk and not tilted

    if monitor is not None:
        direction = Attention.ON_SCREEN
    elif tilted:
        direction = Attention.HEAD_TILTED
    elif on_desk:
        direction = Attention.DESK
    elif down:
        direction = Attention.DOWN
    elif turned_left:
        direction = Attention.LEFT
    elif turned_right:
        direction = Attention.RIGHT
    elif up:
        direction = Attention.UP
    else:
        direction = Attention.BETWEEN

    return GazeFeatures(
        facing_screen=monitor is not None,
        monitor=monitor,
        looking_sideways=turned_left or turned_right,
        looking_down=down,
        looking_up=up,
        head_tilted=tilted,
        on_desk=on_desk,
        direction=direction,
    )


def relative_to_camera_line(
    pose: HeadPose, anchor: tuple[float, float], frame_size: tuple[int, int], vertical_fov: float
) -> HeadPose:
    """Express a head pose relative to the line from the face to the camera.

    MediaPipe reports rotation relative to the camera's optical axis, so a face
    that is off-centre in the image and looks straight into the lens has a
    non-zero yaw/pitch. Subtracting the face's angular position fixes that.
    """
    width, height = frame_size
    focal = (height / 2.0) / math.tan(math.radians(vertical_fov / 2.0))
    # Face on the image right (raw frame = the user's left of the axis) must turn to
    # its right (negative yaw) to look into the lens; face below centre must look up.
    offset_x = math.degrees(math.atan((anchor[0] - width / 2.0) / focal))
    offset_y = math.degrees(math.atan((anchor[1] - height / 2.0) / focal))
    return HeadPose(yaw=pose.yaw + offset_x, pitch=pose.pitch - offset_y, roll=pose.roll)


class PitchCalibrator:
    """Learns the user's head-pitch bias from the time spent looking at monitors.

    Head-pose estimates carry a per-person, per-camera vertical bias of several
    degrees. While the head points (horizontally) at a monitor and its pitch is
    plausibly on that monitor, the difference to the monitor's expected pitch is
    sampled; the running median is the bias. Large deviations (looking at a
    phone, the ceiling) are ignored, so distraction cannot "train" itself away.
    """

    def __init__(self, cfg: FeatureSettings):
        self._cfg = cfg
        self._samples: list[tuple[float, float]] = []  # (timestamp, pitch error)
        self._last_sample = -math.inf
        self.offset = 0.0

    def update(self, timestamp: float, pose: HeadPose, zones: ScreenZones) -> float:
        cfg = self._cfg
        if timestamp - self._last_sample >= cfg.calibration_sample_every_s:
            for zone in zones.monitors:
                if zone.core_yaw[0] - zones.margin <= pose.yaw <= zone.core_yaw[1] + zones.margin:
                    error = pose.pitch - zone.center[1]
                    if abs(error) <= cfg.calibration_max_error:
                        self._samples.append((timestamp, error))
                        self._last_sample = timestamp
                    break
        self._samples = [s for s in self._samples if timestamp - s[0] <= cfg.calibration_window_s]
        if len(self._samples) >= cfg.calibration_min_samples:
            median = float(np.median([e for _, e in self._samples]))
            limit = cfg.calibration_max_offset
            self.offset = max(-limit, min(limit, median))
        return self.offset


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

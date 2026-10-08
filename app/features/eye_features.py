"""
This file reads the user's eyes from the MediaPipe face points:

* Eye state: open, partially closed or closed. It is based on the eye aspect
  ratio (eyelid gap / eye width), corrected for the head angle and compared with
  the user's own normal open eye, which is learned. MediaPipe's blink score is
  used as a second opinion.
* Eye gaze: where the eyes look relative to the head, from the iris position
  between the eye corners. app/features/gaze_features.py adds it to the head
  direction, so a glance with the eyes alone (at a phone, out of the window) counts.

Looking down lowers the upper eyelids, which looks like "closing" to the camera.
So the closed/partial limits are relaxed the further down the user looks, and a
partially closed eye never counts as a distraction by itself: only eyes that are
really closed for longer than a blink do (see app/features/feature_pipeline.py).

It is used by app/features/feature_pipeline.py.
"""

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from app.config.settings import FeatureSettings
from app.vision.head_pose import HeadPose


@dataclass(frozen=True)
class _EyePoints:
    """MediaPipe face mesh indices of one eye ("left" = the user's left)."""
    outer: int
    inner: int
    # Upper and lower eyelid points, as pairs facing each other.
    lids: tuple[tuple[int, int], tuple[int, int]]
    iris: int
    blink: str
    look_down: str


RIGHT_EYE = _EyePoints(33, 133, ((160, 144), (158, 153)), 468, "eyeBlinkRight", "eyeLookDownRight")
LEFT_EYE = _EyePoints(263, 362, ((385, 380), (387, 373)), 473, "eyeBlinkLeft", "eyeLookDownLeft")
_LANDMARKS_WITH_IRIS = 478
# Real eyes: the distance between the eye centres is about twice an eye's width.
_EYE_SPACING_RANGE = (1.2, 3.5)
_MIN_EYE_WIDTH_PX = 6.0
# A wide-open eye has an aspect ratio of about 0.3-0.45; far more is a tracking error.
_MAX_ASPECT_RATIO = 0.7
# The neutral iris position is only learned while the head points this close (degrees)
# to a monitor's centre, where the eye rotation needed to look at it is well known.
_NEUTRAL_NEAR_CENTER = 8.0


class EyeState(str, Enum):
    OPEN = "open"
    PARTIAL = "partial"
    CLOSED = "closed"
    UNKNOWN = "unknown"


@dataclass
class EyeMeasurement:
    """Raw values of both eyes in one frame (made by measure_eyes)."""
    # Eye aspect ratio (mean of both eyes), not yet corrected for the head angle.
    aspect_ratio: float
    # Eye rotation inside the head in degrees, HeadPose convention (yaw > 0 towards the
    # user's left, pitch > 0 up). None when the iris position is not trustworthy.
    yaw: float | None = None
    pitch: float | None = None
    # Mean MediaPipe blink / look-down scores of both eyes (0..1), when available.
    blink: float | None = None
    look_down: float | None = None
    # Iris centres in pixels, for drawing.
    irises: tuple[tuple[float, float], ...] = ()


@dataclass
class EyeFeatures:
    """Eye state and eye gaze for one frame (made by EyeAnalyzer)."""
    state: EyeState = EyeState.UNKNOWN
    # 1.0 = the user's normal open eye.
    openness: float | None = None
    # Calibrated eye rotation inside the head (degrees); None when not usable.
    yaw: float | None = None
    pitch: float | None = None
    # How far the eyes and head look down (degrees); lowers the eyelids naturally.
    downward: float = 0.0
    irises: tuple[tuple[float, float], ...] = ()


def measure_eyes(
    landmarks: np.ndarray | None, frame_size: tuple[int, int], cfg: FeatureSettings,
    blendshapes: dict[str, float] | None = None,
) -> EyeMeasurement | None:
    """Measure eye opening and iris position. None when the eyes cannot be measured."""
    if landmarks is None or len(landmarks) < _LANDMARKS_WITH_IRIS:
        return None
    width, height = frame_size
    pts = landmarks[:, :2].astype(np.float64) * np.array([width, height])

    mids, widths, ratios = [], [], []
    for eye in (RIGHT_EYE, LEFT_EYE):
        outer, inner = pts[eye.outer], pts[eye.inner]
        eye_width = float(np.linalg.norm(outer - inner))
        if eye_width < _MIN_EYE_WIDTH_PX:
            return None
        gap = np.mean([np.linalg.norm(pts[a] - pts[b]) for a, b in eye.lids])
        mids.append((outer + inner) / 2.0)
        widths.append(eye_width)
        ratios.append(float(gap) / eye_width)

    # The line from the user's right eye to the left one: the head's horizontal axis in
    # the image (raw frame: +x when upright), which also removes head roll.
    across = mids[1] - mids[0]
    spacing = float(np.linalg.norm(across))
    if not _EYE_SPACING_RANGE[0] <= spacing / float(np.mean(widths)) <= _EYE_SPACING_RANGE[1]:
        return None
    u = across / spacing
    v = np.array([-u[1], u[0]])  # image "down" for an upright head

    if float(np.mean(ratios)) > _MAX_ASPECT_RATIO:
        return None

    blink = _mean_score(blendshapes, RIGHT_EYE.blink, LEFT_EYE.blink)
    look_down = _mean_score(blendshapes, RIGHT_EYE.look_down, LEFT_EYE.look_down)
    measurement = EyeMeasurement(aspect_ratio=float(np.mean(ratios)), blink=blink, look_down=look_down)

    # Each iris is matched to the nearer eye, which is robust to index-order mix-ups.
    irises = [pts[RIGHT_EYE.iris], pts[LEFT_EYE.iris]]
    if np.linalg.norm(irises[0] - mids[1]) < np.linalg.norm(irises[0] - mids[0]):
        irises.reverse()
    measurement.irises = tuple((float(p[0]), float(p[1])) for p in irises)

    yaws, pitches = [], []
    for mid, eye_width, iris in zip(mids, widths, irises):
        offset = iris - mid
        horizontal, vertical = float(offset @ u) / eye_width, float(offset @ v) / eye_width
        if abs(horizontal) > 0.45 or abs(vertical) > 0.45:
            return measurement  # iris outside its eye: a tracking error
        yaws.append(_asin_deg(horizontal * cfg.eye_gaze_gain))
        pitches.append(-_asin_deg(vertical * cfg.eye_gaze_gain))
    if abs(yaws[0] - yaws[1]) > cfg.eye_max_disagreement or abs(pitches[0] - pitches[1]) > cfg.eye_max_disagreement:
        return measurement
    # The eye that appears wider faces the camera more and is measured more precisely.
    weights = np.array(widths) / float(np.sum(widths))
    measurement.yaw = float(weights @ yaws)
    measurement.pitch = float(weights @ pitches)
    return measurement


class EyeAnalyzer:
    """Turns eye measurements into a smoothed, calibrated eye state and eye gaze.

    It learns two things per user and camera:
    * the normal open-eye aspect ratio (a high percentile of recent values), and
    * the neutral iris position (like PitchCalibrator in gaze_features.py): while
      the head points near a monitor's centre, the eyes are expected to do the rest
      of the turn towards that centre.
    """

    def __init__(self, cfg: FeatureSettings):
        self._cfg = cfg
        self._open_samples: deque[tuple[float, float]] = deque()
        self._last_open_sample = -math.inf
        self.open_ear = cfg.eye_open_ear
        self._gaze_samples: deque[tuple[float, float, float]] = deque()  # (t, yaw error, pitch error)
        self._last_gaze_sample = -math.inf
        self.neutral = (0.0, 0.0)
        self._smoothed: tuple[float, float] | None = None
        self._last_time: float | None = None
        # Last usable eye direction (time, yaw, pitch), kept through blinks.
        self._last_gaze: tuple[float, float | None, float | None] | None = None

    def reconfigure(self, cfg: FeatureSettings) -> None:
        self._cfg = cfg

    def update(
        self, timestamp: float, measurement: EyeMeasurement | None, head: HeadPose | None,
        monitor_center: tuple[float, float] | None = None, reference_pitch: float = 0.0,
    ) -> EyeFeatures:
        """`head` is the (calibrated) head pose relative to the line towards the camera.

        `monitor_center` is the zone centre (yaw, pitch) of the monitor the head points
        at, if any.

        `reference_pitch` is the head pitch for looking at the monitors; looking further
        down than that lowers the eyelids.
        """
        cfg = self._cfg
        if measurement is None or head is None:
            self._smoothed = self._last_gaze = None
            return EyeFeatures()

        # The camera sees the eye at an angle: the eye width shrinks with the head yaw
        # and the eyelid gap with the head pitch.
        correction = math.cos(math.radians(min(abs(head.yaw), 60.0))) / math.cos(
            math.radians(min(abs(head.pitch), 60.0)))
        ear = measurement.aspect_ratio * min(max(correction, 0.6), 1.6)

        downward = self._downward(measurement, head, reference_pitch)
        state = self._classify(ear, measurement.blink, downward)
        # Learn the normal open eye during ordinary screen work (the 85th percentile
        # ignores the occasional half-closed sample).
        if (state is not EyeState.CLOSED and (monitor_center is not None or downward < cfg.eye_down_start)
                and (measurement.blink is None or measurement.blink < cfg.eye_blink_min)):
            self._learn_open(timestamp, ear)

        yaw = pitch = None
        if state is not EyeState.CLOSED and measurement.yaw is not None:
            if monitor_center is not None and state is EyeState.OPEN:
                self._learn_neutral(timestamp, measurement, head, monitor_center)
            raw = (measurement.yaw - self.neutral[0], measurement.pitch - self.neutral[1])
            yaw, pitch = self._smooth(timestamp, raw)
            # The lowered eyelid hides the top of the iris, so its height is only used
            # when the eye is fully open.
            if state is not EyeState.OPEN:
                pitch = None
        else:
            self._smoothed = None
        if yaw is not None:
            self._last_gaze = (timestamp, yaw, pitch)
        elif self._last_gaze is not None and timestamp - self._last_gaze[0] <= cfg.eye_blink_grace_s:
            # A blink hides the iris for a moment: keep the direction from just before.
            _, yaw, pitch = self._last_gaze

        return EyeFeatures(
            state=state, openness=ear / self.open_ear, yaw=yaw, pitch=pitch,
            downward=downward, irises=measurement.irises,
        )

    def _downward(self, measurement: EyeMeasurement, head: HeadPose, reference_pitch: float) -> float:
        """Degrees the gaze is below the monitors, from the best evidence (the largest)."""
        cfg = self._cfg
        reference_gaze = reference_pitch / cfg.head_pitch_ratio
        # Without an eye measurement, assume the eyes follow the head (head-ratio model).
        estimates = [reference_gaze - head.pitch / cfg.head_pitch_ratio]
        if measurement.pitch is not None:
            estimates.append(reference_gaze - (head.pitch + measurement.pitch - self.neutral[1]))
        if measurement.look_down is not None:
            # MediaPipe's look-down score is already ~0.3 while reading a screen.
            estimates.append(max(measurement.look_down - 0.3, 0.0) / 0.7 * cfg.eye_down_full)
        return max(0.0, max(estimates))

    def _classify(self, ear: float, blink: float | None, downward: float) -> EyeState:
        cfg = self._cfg
        span = max(cfg.eye_down_full - cfg.eye_down_start, 1e-6)
        lowered = min(max((downward - cfg.eye_down_start) / span, 0.0), 1.0)
        relax = cfg.eye_down_relax * lowered
        openness = ear / self.open_ear
        closed_limit = cfg.eye_closed_ratio * (1.0 - relax)
        partial_limit = cfg.eye_partial_ratio * (1.0 - relax)
        if openness < closed_limit and (blink is None or blink >= cfg.eye_blink_min):
            return EyeState.CLOSED
        # A very strong blink score also closes a half-open eye, but not while looking
        # far down, where MediaPipe's blink score rises on its own.
        if blink is not None and blink >= cfg.eye_blink_closed and openness < partial_limit and lowered < 0.5:
            return EyeState.CLOSED
        if openness < partial_limit:
            return EyeState.PARTIAL
        return EyeState.OPEN

    def _learn_open(self, timestamp: float, ear: float) -> None:
        cfg = self._cfg
        if timestamp - self._last_open_sample >= 0.1:
            self._open_samples.append((timestamp, ear))
            self._last_open_sample = timestamp
        while self._open_samples and timestamp - self._open_samples[0][0] > cfg.eye_baseline_window_s:
            self._open_samples.popleft()
        if len(self._open_samples) >= cfg.eye_baseline_min_samples:
            value = float(np.percentile([e for _, e in self._open_samples], cfg.eye_baseline_percentile))
            self.open_ear = min(max(value, 0.15), 0.45)

    def _learn_neutral(
        self, timestamp: float, measurement: EyeMeasurement, head: HeadPose, monitor_center: tuple[float, float],
    ) -> None:
        cfg = self._cfg
        center_yaw, center_pitch = monitor_center
        near_center = (abs(head.yaw - center_yaw) <= _NEUTRAL_NEAR_CENTER
                       and abs(head.pitch - center_pitch) <= _NEUTRAL_NEAR_CENTER)
        if near_center and timestamp - self._last_gaze_sample >= cfg.eye_calibration_sample_every_s:
            # The real gaze towards the centre is centre / ratio (zones are in head units);
            # the eyes are expected to do what the head does not.
            expected_yaw = center_yaw / cfg.head_yaw_ratio - head.yaw
            expected_pitch = center_pitch / cfg.head_pitch_ratio - head.pitch
            errors = (measurement.yaw - expected_yaw, measurement.pitch - expected_pitch)
            if max(abs(e) for e in errors) <= cfg.eye_calibration_max_error:
                self._gaze_samples.append((timestamp, *errors))
                self._last_gaze_sample = timestamp
        while self._gaze_samples and timestamp - self._gaze_samples[0][0] > cfg.eye_calibration_window_s:
            self._gaze_samples.popleft()
        if len(self._gaze_samples) >= cfg.eye_calibration_min_samples:
            limit = cfg.eye_calibration_max_offset
            samples = np.array([(y, p) for _, y, p in self._gaze_samples])
            yaw, pitch = np.median(samples, axis=0)
            self.neutral = (float(np.clip(yaw, -limit, limit)), float(np.clip(pitch, -limit, limit)))

    def _smooth(self, timestamp: float, value: tuple[float, float]) -> tuple[float, float]:
        tau = self._cfg.eye_smoothing_s
        if self._smoothed is None or self._last_time is None or tau <= 0:
            self._smoothed = value
        else:
            alpha = 1.0 - math.exp(-max(timestamp - self._last_time, 0.0) / tau)
            self._smoothed = tuple(s + alpha * (v - s) for s, v in zip(self._smoothed, value))
        self._last_time = timestamp
        return self._smoothed


def _mean_score(blendshapes: dict[str, float] | None, *names: str) -> float | None:
    if not blendshapes:
        return None
    values = [blendshapes[n] for n in names if n in blendshapes]
    return float(np.mean(values)) if values else None


def _asin_deg(value: float) -> float:
    return math.degrees(math.asin(max(-1.0, min(1.0, value))))

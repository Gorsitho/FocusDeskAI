"""Phone features: is a phone visible, and is the user actually looking at it?

A phone lying on the desk is not a distraction by itself. It only matters when
the head is turned away from the monitors *towards* the phone.
"""

import math
from dataclasses import dataclass

import numpy as np

from app.config.settings import FeatureSettings
from app.features.gaze_features import ScreenZone
from app.features.pose_features import PoseFeatures, Posture
from app.vision.head_pose import HeadPose
from app.vision.object_detection import ObjectResult

PHONE_LABEL = "cell phone"
# MediaPipe face mesh index of the nose tip.
NOSE_TIP = 1


@dataclass
class PhoneFeatures:
    visible: bool = False
    # Centre of the phone box in pixels of the raw (non-mirrored) frame.
    center: tuple[float, float] | None = None
    looking_at_phone: bool = False


class PhoneTracker:
    """Keeps the last phone detection alive briefly, since YOLO misses frames."""

    def __init__(self, hold_s: float):
        self._hold = hold_s
        self._center: tuple[float, float] | None = None
        self._last_seen: float | None = None

    def update(self, timestamp: float, objects: ObjectResult) -> tuple[float, float] | None:
        phones = [d for d in objects.detections if d.label == PHONE_LABEL]
        if phones:
            best = max(phones, key=lambda d: d.confidence)
            x1, y1, x2, y2 = best.box
            self._center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            self._last_seen = timestamp
        elif self._last_seen is None or timestamp - self._last_seen > self._hold:
            self._center = None
        return self._center


def face_anchor(face_landmarks: np.ndarray | None, frame_size: tuple[int, int]) -> tuple[float, float] | None:
    """Nose tip in pixels, used as the origin of the head direction."""
    if face_landmarks is None or len(face_landmarks) <= NOSE_TIP:
        return None
    width, height = frame_size
    x, y = face_landmarks[NOSE_TIP, :2]
    return float(x * width), float(y * height)


def is_looking_at_phone(
    phone_center: tuple[float, float] | None,
    head_pose: HeadPose | None,
    anchor: tuple[float, float] | None,
    zone: ScreenZone,
    pose: PoseFeatures,
    cfg: FeatureSettings,
) -> bool:
    if phone_center is None:
        return False

    if head_pose is None or anchor is None:
        # Without a face, a dropped head over a visible phone (typically a phone in
        # the lap) is the only reliable cue.
        return pose.posture is Posture.SLOUCHING

    # Looking anywhere at the monitors is never phone use, even with a phone in view.
    if zone.in_core(head_pose.yaw, head_pose.pitch, slack=0.3 * cfg.attention_margin):
        return False

    # Head direction relative to the monitor centre, projected onto the raw image:
    # yaw > 0 (user's left) points to +x, pitch > 0 (up) points to -y.
    head_x = head_pose.yaw - zone.yaw_center
    head_y = -(head_pose.pitch - zone.pitch_center)
    phone_x = phone_center[0] - anchor[0]
    phone_y = phone_center[1] - anchor[1]
    if math.hypot(phone_x, phone_y) < 1e-6:
        return True  # phone held right in front of the face
    return _angle_between((head_x, head_y), (phone_x, phone_y)) <= cfg.phone_gaze_max_angle


def _angle_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    norm = math.hypot(*a) * math.hypot(*b)
    if norm < 1e-9:
        return 180.0
    cos = (a[0] * b[0] + a[1] * b[1]) / norm
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))

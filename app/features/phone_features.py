"""
This file decides if a phone is visible and if the user is really looking at it.

A phone lying on the desk is not a distraction by itself. It only counts when
the head is turned away from every monitor and points *towards* the phone.
A phone held up close to the face also counts, even in front of a monitor.

Main parts:

* PhoneTracker: keeps the last phone position for a short time, because YOLO
  sometimes misses the phone in single frames.
* is_looking_at_phone(): compares the direction of the nose with the position
  of the phone in the image.

It uses the phone boxes from app/vision/object_detection.py and the monitor
zones from app/features/gaze_features.py. It is used by app/features/feature_pipeline.py.
"""

import math
from dataclasses import dataclass

import numpy as np

from app.config.settings import FeatureSettings
from app.features.gaze_features import ScreenZones
from app.features.pose_features import PoseFeatures, Posture
from app.vision.head_pose import HeadPose
from app.vision.object_detection import ObjectResult

PHONE_LABEL = "cell phone"
# MediaPipe face mesh index of the nose tip.
NOSE_TIP = 1


@dataclass
class PhoneFeatures:
    """Phone information for one frame."""
    visible: bool = False
    # Centre of the phone box in pixels of the raw (non-mirrored) frame.
    center: tuple[float, float] | None = None
    looking_at_phone: bool = False
    # Angle between the head direction and the direction of the phone (degrees).
    gaze_angle: float | None = None
    # Held up close to the face (large and next to it), as opposed to lying on the desk.
    near_face: bool = False


class PhoneTracker:
    """Keeps the last phone detection alive briefly, since YOLO misses frames."""

    def __init__(self, hold_s: float):
        self._hold = hold_s
        self._center: tuple[float, float] | None = None
        self.box: tuple[int, int, int, int] | None = None
        self._last_seen: float | None = None

    def update(self, timestamp: float, objects: ObjectResult) -> tuple[float, float] | None:
        phones = [d for d in objects.detections if d.label == PHONE_LABEL]
        if phones:
            best = max(phones, key=lambda d: d.confidence)
            x1, y1, x2, y2 = best.box
            self._center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            self.box = best.box
            self._last_seen = timestamp
        elif self._last_seen is None or timestamp - self._last_seen > self._hold:
            self._center = self.box = None
        return self._center


def phone_near_face(
    box: tuple[int, int, int, int] | None, face_landmarks: np.ndarray | None,
    frame_size: tuple[int, int], cfg: FeatureSettings,
) -> bool:
    """Is the phone held up close to the face rather than lying further away?

    A phone in the hand near the face looks large (comparable to the face) and sits
    right next to it in the image; one on the desk is small and far below.
    """
    if box is None or face_landmarks is None:
        return False
    width, height = frame_size
    face_h = float(face_landmarks[:, 1].max() - face_landmarks[:, 1].min()) * height
    if face_h < 1.0:
        return False
    face_center = (float(face_landmarks[:, 0].mean()) * width, float(face_landmarks[:, 1].mean()) * height)
    x1, y1, x2, y2 = box
    phone_size = max(x2 - x1, y2 - y1)
    distance = math.hypot((x1 + x2) / 2.0 - face_center[0], (y1 + y2) / 2.0 - face_center[1])
    return (phone_size >= cfg.phone_near_min_size * face_h
            and distance <= cfg.phone_near_max_distance * face_h)


def face_anchor(face_landmarks: np.ndarray | None, frame_size: tuple[int, int]) -> tuple[float, float] | None:
    """Nose tip in pixels, used as the origin of the head direction."""
    if face_landmarks is None or len(face_landmarks) <= NOSE_TIP:
        return None
    width, height = frame_size
    x, y = face_landmarks[NOSE_TIP, :2]
    return float(x * width), float(y * height)


def nose_forward(head_pose: HeadPose) -> tuple[float, float]:
    """Where the nose points, projected onto the raw (non-mirrored) image.

    This is the face's forward axis from the head pose: yaw > 0 (towards the
    user's left) points to +x and pitch > 0 (up) to -y. The length is the sine of
    the angle between the nose and the camera axis, so a face looking straight at
    the camera has (almost) no direction.
    """
    yaw, pitch = math.radians(head_pose.yaw), math.radians(head_pose.pitch)
    return math.sin(yaw) * math.cos(pitch), -math.sin(pitch)


def phone_gaze_angle(
    phone_center: tuple[float, float], head_pose: HeadPose, anchor: tuple[float, float], cfg: FeatureSettings
) -> float | None:
    """Angle between the nose's forward direction and the nose-to-phone direction.

    `anchor` is the nose tip landmark. Returns None when the head is too close to
    "looking into the camera" for its direction to mean anything.
    """
    forward = nose_forward(head_pose)
    if math.hypot(*forward) < math.sin(math.radians(cfg.phone_gaze_min_turn)):
        return None
    phone_x, phone_y = phone_center[0] - anchor[0], phone_center[1] - anchor[1]
    if math.hypot(phone_x, phone_y) < 1e-6:
        return 0.0  # phone held right in front of the face
    return _angle_between(forward, (phone_x, phone_y))


def nose_ray_hits_box(
    anchor: tuple[float, float], head_pose: HeadPose, box: tuple[int, int, int, int], cfg: FeatureSettings,
    padding: float = 0.15,
) -> bool:
    """Does the ray from the nose tip along the nose direction cross the phone box?

    Complements the angle test for large, close phones: the ray can pass through
    the box even when the box centre is at a wider angle.
    """
    dx, dy = nose_forward(head_pose)
    if math.hypot(dx, dy) < math.sin(math.radians(cfg.phone_gaze_min_turn)):
        return False
    x1, y1, x2, y2 = box
    pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
    x1, x2, y1, y2 = x1 - pad_x, x2 + pad_x, y1 - pad_y, y2 + pad_y
    # Slab test for a ray starting at the nose (t >= 0).
    t_min, t_max = 0.0, math.inf
    for origin, direction, low, high in ((anchor[0], dx, x1, x2), (anchor[1], dy, y1, y2)):
        if abs(direction) < 1e-9:
            if not low <= origin <= high:
                return False
            continue
        t1, t2 = (low - origin) / direction, (high - origin) / direction
        t_min, t_max = max(t_min, min(t1, t2)), min(t_max, max(t1, t2))
        if t_min > t_max:
            return False
    return True


def is_looking_at_phone(
    phone_center: tuple[float, float] | None,
    head_pose: HeadPose | None,
    anchor: tuple[float, float] | None,
    zones: ScreenZones,
    pose: PoseFeatures,
    cfg: FeatureSettings,
    direction: HeadPose | None = None,
    near_face: bool = False,
    box: tuple[int, int, int, int] | None = None,
) -> bool:
    """A visible phone counts only when the nose/head points towards it.

    `head_pose` decides whether a monitor is being looked at; `direction` (defaults
    to it) is the head direction compared with the phone's position in the image,
    starting at the nose tip `anchor`.
    """
    if phone_center is None:
        return False

    if head_pose is None or anchor is None:
        # Without a face, a dropped head over a visible phone (typically a phone in
        # the lap) is the only reliable cue.
        return pose.posture is Posture.SLOUCHING

    # Looking at a monitor is not phone use, even with a phone in view: a phone on the
    # desk or a stand never counts while the head points at a monitor. The exception is
    # a phone held up close to the face in the line of sight, which hides the monitor.
    on_monitor = zones.monitor_at(head_pose.yaw, head_pose.pitch, margin=0.3 * zones.margin) is not None
    if on_monitor and not near_face:
        return False

    nose_direction = direction or head_pose
    angle = phone_gaze_angle(phone_center, nose_direction, anchor, cfg)
    if angle is not None and angle <= cfg.phone_gaze_max_angle:
        return True
    return box is not None and nose_ray_hits_box(anchor, nose_direction, box, cfg)


def _angle_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    norm = math.hypot(*a) * math.hypot(*b)
    if norm < 1e-9:
        return 180.0
    cos = (a[0] * b[0] + a[1] * b[1]) / norm
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))

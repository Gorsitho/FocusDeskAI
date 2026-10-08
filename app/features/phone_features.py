"""
This file decides if a phone is visible and if the user is really using it.

A phone lying on the desk is not a distraction by itself. It only counts when
the gaze (head and eyes) leaves every monitor and points *towards* the phone.
A phone held up close to the face also counts, even in front of a monitor, and
so does a phone held at the ear (a call).

Main parts:

* plausible_phone(): rejects boxes that cannot be the user's phone (far behind
  the user, above the head, far to the side, the size of a screen).
* PhoneTracker: only trusts a phone after a very clear box, or after repeated
  boxes of which at least one was strong or which the user holds; repeated weak
  boxes of a background object are never enough. It keeps a confirmed phone for a
  while when YOLO misses it (longer while it is in the hand, which hides it).
* phone_in_hand() / phone_at_ear() / phone_near_face(): where the phone is
  relative to the hands (pose points) and the face.
* is_looking_at_phone(): compares the gaze direction with the position of the
  phone in the image.

It uses the phone boxes from app/vision/object_detection.py, the hand points from
app/vision/pose_detection.py and the monitor zones from app/features/gaze_features.py.
It is used by app/features/feature_pipeline.py.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from app.config.settings import FeatureSettings
from app.features.gaze_features import ScreenZones
from app.features.pose_features import PoseFeatures, Posture
from app.vision.head_pose import HeadPose
from app.vision.object_detection import PHONE_LABEL, Detection, box_iou
from app.vision.pose_detection import PoseLandmark

_HAND_POINTS = (PoseLandmark.LEFT_WRIST, PoseLandmark.RIGHT_WRIST, PoseLandmark.LEFT_INDEX, PoseLandmark.RIGHT_INDEX)
# Extra angle tolerance for a phone in the hand: it is far more likely to be in use.
_IN_HAND_EXTRA_ANGLE = 10.0
# MediaPipe face mesh index of the nose tip.
NOSE_TIP = 1


@dataclass
class PhoneFeatures:
    """Phone information for one frame."""
    # A confirmed phone is (or was very recently) visible.
    visible: bool = False
    # Centre of the phone box in pixels of the raw (non-mirrored) frame.
    center: tuple[float, float] | None = None
    box: tuple[int, int, int, int] | None = None
    # YOLO confidence of the last box.
    confidence: float = 0.0
    # The phone is being used: looked at, held up in front of the face or at the ear.
    looking_at_phone: bool = False
    # Angle between the gaze direction and the direction of the phone (degrees).
    gaze_angle: float | None = None
    # Held up close to the face (large and next to it), as opposed to lying on the desk.
    near_face: bool = False
    # A hand (wrist or finger) is at the phone.
    in_hand: bool = False
    # Held at the ear, as in a call.
    at_ear: bool = False


class PhoneTracker:
    """Follows one phone over time.

    Fresh YOLO results add evidence (the box confidence) and runs without the phone
    reduce it. The phone only becomes visible after one very clear box
    (`phone_instant_confidence`), or once the evidence reaches `phone_confirm_evidence`
    and at least one box was strong (`phone_strong_confidence`) or the phone is held
    (`supported`). A confirmed phone stays visible for `phone_hold_s` after YOLO last
    saw it, or `phone_hold_in_hand_s` while it is in the hand.
    """

    def __init__(self, cfg: FeatureSettings):
        self._cfg = cfg
        self.box: tuple[int, int, int, int] | None = None
        self.confidence = 0.0
        self.evidence = 0.0
        self.confirmed = False
        # A strong (or held) box of the current object has been seen.
        self._strong = False
        # Set by the caller: the phone is in the user's hand (affects the hold time).
        self.in_hand = False
        self._last_seen: float | None = None

    def reconfigure(self, cfg: FeatureSettings) -> None:
        self._cfg = cfg

    @property
    def center(self) -> tuple[float, float] | None:
        if self.box is None or not self.confirmed:
            return None
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def update(
        self, timestamp: float, candidates: list[Detection], fresh: bool = True,
        supported: Callable[[Detection], bool] | None = None,
    ) -> tuple[float, float] | None:
        """Returns the centre of the confirmed phone, if any.

        `fresh` is False when the detections repeat an earlier YOLO run (YOLO does not
        run on every frame): they keep the phone alive but add no evidence.
        `supported(detection)` tells if the user holds the box (in the hand, at the face).
        """
        cfg = self._cfg
        best = self._pick(candidates)
        if best is not None:
            if self.box is not None and not self.confirmed and box_iou(best.box, self.box) == 0.0:
                self.evidence, self._strong = 0.0, False  # a different object somewhere else: start again
            if fresh:
                self.evidence = min(cfg.phone_max_evidence, self.evidence + best.confidence)
                self._strong = (self._strong or best.confidence >= cfg.phone_strong_confidence
                                or (supported is not None and supported(best)))
            self.box, self.confidence, self._last_seen = best.box, best.confidence, timestamp
            self.confirmed = (self.confirmed or best.confidence >= cfg.phone_instant_confidence
                              or (self._strong and self.evidence >= cfg.phone_confirm_evidence))
        else:
            if fresh:
                self.evidence *= cfg.phone_miss_decay
            hold = cfg.phone_hold_in_hand_s if self.in_hand else cfg.phone_hold_s
            if self._last_seen is None or timestamp - self._last_seen > hold:
                self.box, self.confidence, self.evidence = None, 0.0, 0.0
                self.confirmed = self.in_hand = self._strong = False
        return self.center

    def _pick(self, candidates: list[Detection]) -> Detection | None:
        """Prefer the phone already followed; otherwise the most confident (real phones first)."""
        if not candidates:
            return None
        if self.box is not None:
            same = [d for d in candidates if box_iou(d.box, self.box) > 0.1]
            if same:
                return max(same, key=lambda d: d.confidence)
        return max(candidates, key=lambda d: d.confidence * (1.0 if d.label == PHONE_LABEL else 0.8))


def hand_points(
    pose_landmarks: np.ndarray | None, frame_size: tuple[int, int], min_visibility: float
) -> list[tuple[float, float]]:
    """Visible wrist and index-finger points in pixels."""
    if pose_landmarks is None:
        return []
    width, height = frame_size
    return [(float(pose_landmarks[i, 0] * width), float(pose_landmarks[i, 1] * height))
            for i in _HAND_POINTS if pose_landmarks[i, 3] >= min_visibility]


def phone_in_hand(box: tuple[int, int, int, int] | None, hands: list[tuple[float, float]],
                  cfg: FeatureSettings) -> bool:
    """Is a hand point on or right next to the phone box?"""
    if box is None or not hands:
        return False
    x1, y1, x2, y2 = box
    reach = cfg.phone_hand_max_distance * max(x2 - x1, y2 - y1)
    for x, y in hands:
        dx, dy = max(x1 - x, 0.0, x - x2), max(y1 - y, 0.0, y - y2)
        if math.hypot(dx, dy) <= reach:
            return True
    return False


def plausible_phone(
    det: Detection, face_landmarks: np.ndarray | None, frame_size: tuple[int, int], held: bool,
    cfg: FeatureSettings,
) -> bool:
    """Can this box be the user's phone, judged by its shape, size and position?

    `held`: a hand holds it or it is at the face, which makes it plausible wherever it is.
    Otherwise the face is the reference: a phone on the user's desk or in the lap
    appears at least `phone_min_face_size` face heights large, never as large as a
    screen, not above the head and within reach to the side. Without a face there is
    no reference, so only held phones count.
    """
    x1, y1, x2, y2 = det.box
    long_side, short_side = max(x2 - x1, y2 - y1), max(min(x2 - x1, y2 - y1), 1)
    if long_side / short_side > cfg.phone_max_aspect:
        return False  # a pen, a cable, an edge
    if held:
        return True
    if face_landmarks is None:
        return False
    width, height = frame_size
    fx1, fx2 = float(face_landmarks[:, 0].min()) * width, float(face_landmarks[:, 0].max()) * width
    fy1, fy2 = float(face_landmarks[:, 1].min()) * height, float(face_landmarks[:, 1].max()) * height
    face_w, face_h = fx2 - fx1, fy2 - fy1
    if face_w < 1.0 or face_h < 1.0:
        return True  # no usable face size: the other checks decide
    if not cfg.phone_min_face_size * face_h <= long_side <= cfg.phone_max_face_size * face_h:
        return False
    if y2 < fy1:
        return False  # entirely above the head: a shelf or a wall behind the user
    return abs((x1 + x2) / 2.0 - (fx1 + fx2) / 2.0) <= cfg.phone_max_reach * face_w


def phone_at_ear(
    box: tuple[int, int, int, int] | None, face_landmarks: np.ndarray | None,
    frame_size: tuple[int, int], cfg: FeatureSettings,
) -> bool:
    """Is the phone held at the side of the head, as in a call?"""
    if box is None or face_landmarks is None:
        return False
    width, height = frame_size
    fx1, fx2 = float(face_landmarks[:, 0].min()) * width, float(face_landmarks[:, 0].max()) * width
    fy1, fy2 = float(face_landmarks[:, 1].min()) * height, float(face_landmarks[:, 1].max()) * height
    face_w, face_h = fx2 - fx1, fy2 - fy1
    if face_w < 1.0 or face_h < 1.0:
        return False
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    # Distance of the box centre outside the face's left/right edge (negative = inside).
    beside = max(fx1 - cx, cx - fx2)
    return (fy1 <= cy <= fy2
            and -0.25 * face_w <= beside <= cfg.phone_ear_max_offset * face_w
            and max(x2 - x1, y2 - y1) >= cfg.phone_ear_min_size * face_h)


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
    in_hand: bool = False,
) -> bool:
    """A visible phone counts only when the gaze points towards it.

    `head_pose` (the gaze: head and eyes combined) decides whether a monitor is
    being looked at; `direction` (defaults to it) is the gaze direction compared
    with the phone's position in the image, starting at the nose tip `anchor`.
    """
    if phone_center is None:
        return False

    if head_pose is None or anchor is None:
        # Without a face, a phone in the hand or a dropped head over a visible phone
        # (typically a phone in the lap) are the only reliable cues.
        return in_hand or pose.posture is Posture.SLOUCHING

    # Looking at a monitor is not phone use, even with a phone in view: a phone on the
    # desk or a stand never counts while the head points at a monitor. The exception is
    # a phone held up close to the face in the line of sight, which hides the monitor.
    on_monitor = zones.monitor_at(head_pose.yaw, head_pose.pitch, margin=0.3 * zones.margin) is not None
    if on_monitor and not near_face:
        return False

    nose_direction = direction or head_pose
    angle = phone_gaze_angle(phone_center, nose_direction, anchor, cfg)
    max_angle = cfg.phone_gaze_max_angle + (_IN_HAND_EXTRA_ANGLE if in_hand else 0.0)
    if angle is not None and angle <= max_angle:
        return True
    return box is not None and nose_ray_hits_box(anchor, nose_direction, box, cfg)


def _angle_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    norm = math.hypot(*a) * math.hypot(*b)
    if norm < 1e-9:
        return 180.0
    cos = (a[0] * b[0] + a[1] * b[1]) / norm
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))

"""Posture features from upper-body pose landmarks."""

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from app.config.settings import DetectionSettings, FeatureSettings
from app.vision.pose_detection import PoseLandmark, PoseResult


class Posture(str, Enum):
    UPRIGHT = "Upright"
    SLOUCHING = "Slouching"
    LEANING = "Leaning"
    UNKNOWN = "Unknown"


@dataclass
class PoseFeatures:
    posture: Posture = Posture.UNKNOWN
    # Angle of the shoulder line relative to horizontal, in degrees.
    shoulder_tilt: float | None = None
    # Vertical nose-to-shoulder distance divided by shoulder width; drops when slouching.
    head_height_ratio: float | None = None
    # Horizontal nose offset from the shoulder midpoint divided by shoulder width.
    head_offset_ratio: float | None = None
    # Rotation of the shoulder line around the vertical axis, in degrees, using the
    # HeadPose sign convention (positive = body turned towards the user's left).
    torso_yaw: float | None = None


def extract_pose_features(
    pose: PoseResult,
    frame_size: tuple[int, int],
    feature_cfg: FeatureSettings,
    detection_cfg: DetectionSettings,
) -> PoseFeatures:
    if not pose.detected or pose.landmarks is None:
        return PoseFeatures()

    lm = pose.landmarks
    needed = [PoseLandmark.NOSE, PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER]
    if np.any(lm[needed, 3] < detection_cfg.min_landmark_visibility):
        return PoseFeatures()

    # Work in pixels so that the non-square frame does not distort angles and ratios.
    width, height = frame_size
    scale = np.array([width, height], dtype=np.float32)
    nose = lm[PoseLandmark.NOSE, :2] * scale
    left = lm[PoseLandmark.LEFT_SHOULDER, :2] * scale
    right = lm[PoseLandmark.RIGHT_SHOULDER, :2] * scale

    shoulder_width = float(np.linalg.norm(left - right))
    if shoulder_width < 1.0:
        return PoseFeatures()

    mid = (left + right) / 2.0
    dx, dy = abs(left[0] - right[0]), abs(left[1] - right[1])
    shoulder_tilt = math.degrees(math.atan2(dy, dx))
    head_height_ratio = float(mid[1] - nose[1]) / shoulder_width
    head_offset_ratio = float(nose[0] - mid[0]) / shoulder_width
    # MediaPipe pose z is roughly in the same scale as x. In the raw (non-mirrored)
    # frame the user's left shoulder is on the image right, and turning left pushes
    # it away from the camera (larger z).
    dz = float(lm[PoseLandmark.LEFT_SHOULDER, 2] - lm[PoseLandmark.RIGHT_SHOULDER, 2]) * width
    torso_yaw = math.degrees(math.atan2(dz, float(left[0] - right[0])))

    return PoseFeatures(
        posture=classify_posture(shoulder_tilt, head_height_ratio, head_offset_ratio, feature_cfg),
        shoulder_tilt=shoulder_tilt,
        head_height_ratio=head_height_ratio,
        head_offset_ratio=head_offset_ratio,
        torso_yaw=torso_yaw,
    )


def classify_posture(
    shoulder_tilt: float,
    head_height_ratio: float,
    head_offset_ratio: float,
    cfg: FeatureSettings,
) -> Posture:
    if head_height_ratio < cfg.min_head_height_ratio:
        return Posture.SLOUCHING
    if shoulder_tilt > cfg.max_shoulder_tilt or abs(head_offset_ratio) > cfg.max_head_offset_ratio:
        return Posture.LEANING
    return Posture.UPRIGHT

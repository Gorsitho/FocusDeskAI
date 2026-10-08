"""
This file finds the user's body points (nose, ears, shoulders, hands, hips) in a camera
frame with MediaPipe Pose Landmarker.

The background worker in app/ui/main_window.py uses it. The points are used by
app/features/pose_features.py, app/features/phone_features.py (a phone in the hand)
and, to measure movement, by app/features/activity_features.py.
"""

import logging
from dataclasses import dataclass
from enum import IntEnum

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from app.config.settings import DetectionSettings, ModelSettings
from app.vision import ensure_model

logger = logging.getLogger(__name__)


class PoseLandmark(IntEnum):
    """Subset of the 33 MediaPipe pose landmarks used by the feature layer."""

    NOSE = 0
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_HIP = 23
    RIGHT_HIP = 24


@dataclass
class PoseResult:
    """Result of pose detection for one frame."""
    detected: bool
    # (33, 4) normalised x, y, z and visibility.
    landmarks: np.ndarray | None = None


class PoseDetector:
    """Runs MediaPipe Pose Landmarker on video frames."""
    def __init__(self, models: ModelSettings, detection: DetectionSettings):
        model_path = ensure_model(models.pose_landmarker_path, models.pose_landmarker_url)
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=detection.min_pose_confidence,
            min_pose_presence_confidence=detection.min_pose_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        logger.info("Pose landmarker ready (%s)", model_path)

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> PoseResult:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.pose_landmarks:
            return PoseResult(detected=False)

        landmarks = np.array(
            [(p.x, p.y, p.z, p.visibility if p.visibility is not None else 0.0) for p in result.pose_landmarks[0]],
            dtype=np.float32,
        )
        return PoseResult(detected=True, landmarks=landmarks)

    def close(self) -> None:
        self._landmarker.close()

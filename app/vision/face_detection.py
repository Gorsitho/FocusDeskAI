"""
This file finds the user's face in a camera frame with MediaPipe Face Landmarker.
It returns 478 face points and a 3D transform of the head.

app/vision/head_pose.py turns the transform into head angles. The background
worker in app/ui/main_window.py uses this detector.
"""

import logging
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from app.config.settings import DetectionSettings, ModelSettings
from app.vision import ensure_model

logger = logging.getLogger(__name__)


@dataclass
class FaceResult:
    """Result of face detection for one frame."""
    detected: bool
    # (478, 3) normalised x, y in [0, 1] and relative depth z.
    landmarks: np.ndarray | None = None
    # 4x4 transform from MediaPipe's canonical face model to camera space.
    transform: np.ndarray | None = None


class FaceDetector:
    """Runs MediaPipe Face Landmarker on video frames."""
    def __init__(self, models: ModelSettings, detection: DetectionSettings):
        model_path = ensure_model(models.face_landmarker_path, models.face_landmarker_url)
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=detection.min_face_confidence,
            min_face_presence_confidence=detection.min_face_confidence,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        logger.info("Face landmarker ready (%s)", model_path)

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> FaceResult:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            return FaceResult(detected=False)

        landmarks = np.array([(p.x, p.y, p.z) for p in result.face_landmarks[0]], dtype=np.float32)
        transform = None
        if result.facial_transformation_matrixes:
            transform = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64)
        return FaceResult(detected=True, landmarks=landmarks, transform=transform)

    def close(self) -> None:
        self._landmarker.close()

"""
This file shows the camera image in the main window.
It can also draw the detection markers on the image: face points, irises, body
points, phone boxes and an arrow for the gaze direction (head and eyes).

It uses the detector results from app/vision/ and the FrameAnalysis from
app/features/feature_pipeline.py. It is used by app/ui/main_window.py.
"""

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy

from app.features.feature_pipeline import FrameAnalysis
from app.features.gaze_features import Attention
from app.features.eye_features import EyeState
from app.features.phone_features import NOSE_TIP
from app.ui.i18n import tr
from app.vision.face_detection import FaceResult
from app.vision.object_detection import PHONE_LABEL, ObjectResult
from app.vision.pose_detection import PoseLandmark, PoseResult

_FACE_COLOR = (113, 204, 46)  # BGR
_POSE_COLOR = (219, 152, 52)
_PHONE_COLOR = (18, 156, 243)
_PHONE_IN_USE_COLOR = (60, 76, 231)
_ON_SCREEN_COLOR = (113, 204, 46)
_OFF_SCREEN_COLOR = (18, 156, 243)
_IRIS_COLOR = (241, 196, 15)
_EYES_CLOSED_COLOR = (60, 76, 231)
_KEY_POSE_POINTS = [p.value for p in PoseLandmark]


def draw_overlays(
    frame: np.ndarray, face: FaceResult, pose: PoseResult, objects: ObjectResult, min_visibility: float,
    analysis: FrameAnalysis | None = None,
) -> np.ndarray:
    """Return a copy of the BGR frame with detections drawn on it."""
    canvas = frame.copy()
    height, width = canvas.shape[:2]

    if face.detected and face.landmarks is not None:
        # Every 8th landmark is enough to show tracking without cluttering the face.
        for x, y, _ in face.landmarks[::8]:
            cv2.circle(canvas, (int(x * width), int(y * height)), 1, _FACE_COLOR, -1)

    if pose.detected and pose.landmarks is not None:
        for idx in _KEY_POSE_POINTS:
            x, y, _, visibility = pose.landmarks[idx]
            if visibility >= min_visibility:
                cv2.circle(canvas, (int(x * width), int(y * height)), 5, _POSE_COLOR, -1)
        ls, rs = pose.landmarks[PoseLandmark.LEFT_SHOULDER], pose.landmarks[PoseLandmark.RIGHT_SHOULDER]
        if ls[3] >= min_visibility and rs[3] >= min_visibility:
            cv2.line(canvas, (int(ls[0] * width), int(ls[1] * height)),
                     (int(rs[0] * width), int(rs[1] * height)), _POSE_COLOR, 2)

    if analysis is not None:
        eyes = analysis.features.eyes
        color = _EYES_CLOSED_COLOR if eyes.state is EyeState.CLOSED else _IRIS_COLOR
        for x, y in eyes.irises:
            cv2.circle(canvas, (int(x), int(y)), 3, color, -1)

    in_use = analysis is not None and analysis.features.phone.looking_at_phone
    phone_color = _PHONE_IN_USE_COLOR if in_use else _PHONE_COLOR
    for det in objects.phone_candidates:
        x1, y1, x2, y2 = det.box
        # Remotes are only phone candidates; draw them thinner.
        cv2.rectangle(canvas, (x1, y1), (x2, y2), phone_color, 2 if det.label == PHONE_LABEL else 1)
        name = "phone" if det.label == PHONE_LABEL else "phone?"
        cv2.putText(canvas, f"{name} {det.confidence:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, phone_color, 1, cv2.LINE_AA)

    # Gaze direction (head and eyes together; the head alone without measurable eyes).
    head = (analysis.features.gaze_pose or analysis.features.head_pose) if analysis is not None else None
    if head is not None and face.landmarks is not None:
        # Head direction: yaw > 0 (user's left) is +x in the raw frame, pitch > 0 is up.
        nx, ny = face.landmarks[NOSE_TIP, :2]
        start = np.array([nx * width, ny * height])
        length = 0.25 * min(width, height)
        direction = np.array([np.sin(np.radians(head.yaw)), -np.sin(np.radians(head.pitch))])
        end = start + direction * length
        on_screen = analysis.features.attention in (Attention.ON_SCREEN, Attention.DESK)
        color = _ON_SCREEN_COLOR if on_screen else _OFF_SCREEN_COLOR
        cv2.arrowedLine(canvas, tuple(int(v) for v in start), tuple(int(v) for v in end), color, 2,
                        cv2.LINE_AA, tipLength=0.2)

    return canvas


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    """Convert an OpenCV image (BGR colours) into a QImage that Qt can show."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    # copy() detaches the QImage from the numpy buffer, which is freed after this call.
    return QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()


class CameraWidget(QLabel):
    """Shows the camera image (scaled to fit), or a text message when there is no image."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CameraView")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap: QPixmap | None = None
        self.show_message(tr("status.starting_camera"))

    def show_message(self, text: str) -> None:
        self._pixmap = None
        self.clear()
        self.setText(text)

    def set_frame(self, image: QImage) -> None:
        self._pixmap = QPixmap.fromImage(image)
        self._render()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render()

    def _render(self) -> None:
        if self._pixmap is None:
            return
        self.setPixmap(self._pixmap.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))

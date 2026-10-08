"""Thin wrapper around cv2.VideoCapture."""

import sys

import cv2
import numpy as np

from app.config.settings import CameraSettings


class Camera:
    def __init__(self, config: CameraSettings):
        self._config = config
        self._capture: cv2.VideoCapture | None = None

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def open(self) -> bool:
        self.release()
        # DirectShow opens much faster than the default MSMF backend on Windows.
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        capture = cv2.VideoCapture(self._config.index, backend)
        if not capture.isOpened():
            capture.release()
            return False
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        capture.set(cv2.CAP_PROP_FPS, self._config.fps)
        # Keep only the latest frame so slow processing does not accumulate lag.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._capture = capture
        return True

    def read(self) -> np.ndarray | None:
        """Return a BGR frame, or None if the camera is unavailable."""
        if not self.is_open:
            return None
        ok, frame = self._capture.read()
        return frame if ok and frame is not None else None

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

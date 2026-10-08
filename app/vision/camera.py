"""
This file opens the webcam and reads frames from it with OpenCV.
If the camera is not available, read() returns None and the caller tries again later.

It uses CameraSettings from app/config/settings.py. The background worker in
app/ui/main_window.py uses it.
"""

import logging
import sys

import cv2
import numpy as np

from app.config.settings import CameraSettings

logger = logging.getLogger(__name__)


class Camera:
    """Opens the webcam and reads frames. It can be opened again after an error."""
    def __init__(self, config: CameraSettings):
        self._config = config
        self._capture: cv2.VideoCapture | None = None
        self._failed_opens = 0

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
            self._failed_opens += 1
            # Log the first failure and then occasionally, not every retry.
            if self._failed_opens in (1, 10) or self._failed_opens % 100 == 0:
                logger.warning("Camera %d could not be opened (attempt %d)", self._config.index, self._failed_opens)
            return False
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        capture.set(cv2.CAP_PROP_FPS, self._config.fps)
        # Keep only the latest frame so slow processing does not accumulate lag.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._capture = capture
        logger.info("Camera %d opened (%dx%d) after %d failed attempt(s)", self._config.index,
                    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    self._failed_opens)
        self._failed_opens = 0
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
            logger.info("Camera %d released", self._config.index)

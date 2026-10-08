"""Cell phone and person detection with Ultralytics YOLO (COCO classes)."""

import logging
from dataclasses import dataclass, field

import numpy as np

from app.config.settings import DetectionSettings, ModelSettings

logger = logging.getLogger(__name__)

COCO_PERSON = 0
COCO_CELL_PHONE = 67


@dataclass
class Detection:
    label: str
    confidence: float
    # Pixel coordinates x1, y1, x2, y2.
    box: tuple[int, int, int, int]


@dataclass
class ObjectResult:
    phone_detected: bool = False
    person_detected: bool = False
    detections: list[Detection] = field(default_factory=list)


class ObjectDetector:
    """YOLO wrapper that degrades to a no-op if the model cannot be loaded."""

    def __init__(self, models: ModelSettings, detection: DetectionSettings):
        self._confidence = detection.yolo_confidence
        self._model = None
        self.error: str | None = None
        try:
            # Imported lazily: torch is heavy and may be unavailable on some machines.
            from ultralytics import YOLO

            models.yolo_weights_path.parent.mkdir(parents=True, exist_ok=True)
            self._model = YOLO(str(models.yolo_weights_path))
        except Exception as exc:  # noqa: BLE001 - any failure just disables this detector
            self.error = f"{type(exc).__name__}: {exc}"
            logger.warning("Object detection disabled: %s", self.error)

    @property
    def available(self) -> bool:
        return self._model is not None

    def detect(self, frame_bgr: np.ndarray) -> ObjectResult:
        if self._model is None:
            return ObjectResult()

        result = self._model.predict(
            frame_bgr,
            conf=self._confidence,
            classes=[COCO_PERSON, COCO_CELL_PHONE],
            verbose=False,
        )[0]

        detections = []
        for cls, conf, xyxy in zip(
            result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist()
        ):
            label = "cell phone" if int(cls) == COCO_CELL_PHONE else "person"
            box = tuple(int(v) for v in xyxy)
            detections.append(Detection(label=label, confidence=float(conf), box=box))

        return ObjectResult(
            phone_detected=any(d.label == "cell phone" for d in detections),
            person_detected=any(d.label == "person" for d in detections),
            detections=detections,
        )

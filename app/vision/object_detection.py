"""
This file finds phones and people in a camera frame with Ultralytics YOLO
(using the COCO object classes). If YOLO cannot be loaded (for example, torch is
missing), the detector turns itself off and the rest of the app keeps working
without phone detection.

Phones are hard for YOLO: they are small and the hand covers part of them, and
dark rectangles in the background (remotes, books, frames) are easily mistaken
for one. So:

* A second, cheaper pass runs YOLO on enlarged crops around the hands
  (hand_regions()), where a phone in use usually is.
* Every phone box that is not very confident is verified: it is cut out with
  some context, enlarged, and YOLO must find a phone there again. Background
  objects rarely pass this zoomed second look.
* Phones in a hand are often labelled "remote", so remotes are reported too
  (the feature layer only uses them when they are held).
* The feature layer (app/features/phone_features.py) adds plausibility checks
  (size and position relative to the user) and only trusts strong or repeated boxes.

The background worker in app/ui/main_window.py uses it. The results are used by
app/features/feature_pipeline.py and app/features/phone_features.py.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from app.config.settings import DetectionSettings, ModelSettings
from app.vision.pose_detection import PoseLandmark

logger = logging.getLogger(__name__)

COCO_PERSON = 0
COCO_REMOTE = 65
COCO_CELL_PHONE = 67

PHONE_LABEL = "cell phone"
REMOTE_LABEL = "remote"
PERSON_LABEL = "person"
_LABELS = {COCO_PERSON: PERSON_LABEL, COCO_REMOTE: REMOTE_LABEL, COCO_CELL_PHONE: PHONE_LABEL}
# Boxes from two passes that overlap this much are the same object.
_SAME_OBJECT_IOU = 0.4

Box = tuple[int, int, int, int]


@dataclass
class Detection:
    """One object found by YOLO."""
    label: str
    confidence: float
    # Pixel coordinates x1, y1, x2, y2.
    box: Box


@dataclass
class ObjectResult:
    """All objects found by YOLO in one frame."""
    phone_detected: bool = False
    person_detected: bool = False
    detections: list[Detection] = field(default_factory=list)

    @property
    def phone_candidates(self) -> list[Detection]:
        """Phones, and remotes (which may be phones in a hand)."""
        return [d for d in self.detections if d.label in (PHONE_LABEL, REMOTE_LABEL)]


def box_iou(a: Box, b: Box) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def hand_regions(
    pose_landmarks: np.ndarray | None, frame_size: tuple[int, int], cfg: DetectionSettings
) -> list[Box]:
    """Square crops around each visible hand, sized relative to the shoulders.

    The crop is centred a little beyond the wrist towards the index finger, where a
    held phone is. Overlapping crops (hands together, e.g. typing on a phone) are merged.
    """
    if pose_landmarks is None or cfg.hand_roi_imgsz <= 0:
        return []
    width, height = frame_size
    lm = pose_landmarks
    side = float(cfg.hand_roi_min_px)
    shoulders = lm[[PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER]]
    if np.all(shoulders[:, 3] >= cfg.min_landmark_visibility):
        span = np.hypot((shoulders[0, 0] - shoulders[1, 0]) * width, (shoulders[0, 1] - shoulders[1, 1]) * height)
        side = max(side, cfg.hand_roi_scale * float(span))
    side = min(side, float(min(width, height)))

    regions: list[Box] = []
    for wrist, index in ((PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_INDEX),
                         (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_INDEX)):
        if lm[wrist, 3] < cfg.min_hand_visibility:
            continue
        center = np.array([lm[wrist, 0] * width, lm[wrist, 1] * height])
        if lm[index, 3] >= cfg.min_hand_visibility:
            finger = np.array([lm[index, 0] * width, lm[index, 1] * height])
            center = center + 1.5 * (finger - center)
        if not (-side / 2 <= center[0] <= width + side / 2 and -side / 2 <= center[1] <= height + side / 2):
            continue  # hand far outside the image
        x1 = int(np.clip(center[0] - side / 2, 0, width - side))
        y1 = int(np.clip(center[1] - side / 2, 0, height - side))
        box = (x1, y1, int(x1 + side), int(y1 + side))
        merged = False
        for i, other in enumerate(regions):
            if box_iou(box, other) > 0.3:
                regions[i] = (min(box[0], other[0]), min(box[1], other[1]),
                              max(box[2], other[2]), max(box[3], other[3]))
                merged = True
        if not merged:
            regions.append(box)
    return regions


def merge_detections(detections: list[Detection]) -> list[Detection]:
    """Drop duplicates of the same object (same label family, overlapping boxes), keeping the strongest."""
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        phone_like = det.label != PERSON_LABEL
        if any((d.label != PERSON_LABEL) == phone_like and box_iou(d.box, det.box) > _SAME_OBJECT_IOU
               for d in kept):
            continue
        kept.append(det)
    return kept


class ObjectDetector:
    """YOLO wrapper that degrades to a no-op if the model cannot be loaded."""

    def __init__(self, models: ModelSettings, detection: DetectionSettings):
        self._cfg = detection
        self._thresholds = {
            COCO_PERSON: detection.yolo_confidence,
            COCO_REMOTE: detection.remote_confidence,
            COCO_CELL_PHONE: detection.phone_confidence,
        }
        self._model = None
        self.error: str | None = None
        try:
            # Imported lazily: torch is heavy and may be unavailable on some machines.
            from ultralytics import YOLO

            models.yolo_weights_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Loading YOLO weights from %s (exists=%s)",
                        models.yolo_weights_path, models.yolo_weights_path.exists())
            self._model = YOLO(str(models.yolo_weights_path))
            logger.info("YOLO ready: phone and person detection enabled")
        except Exception as exc:  # noqa: BLE001 - any failure just disables this detector
            self.error = f"{type(exc).__name__}: {exc}"
            logger.exception("Object detection disabled: %s", self.error)

    @property
    def available(self) -> bool:
        return self._model is not None

    def detect(self, frame_bgr: np.ndarray, hand_rois: list[Box] | None = None) -> ObjectResult:
        """Full-frame pass, plus a pass on each hand crop that does not already contain a phone."""
        if self._model is None:
            return ObjectResult()

        detections = self._predict([frame_bgr], [(0, 0)], imgsz=None)
        phones = [d.box for d in detections if d.label != PERSON_LABEL]
        crops, offsets = [], []
        for x1, y1, x2, y2 in hand_rois or []:
            if any(_inside(box, (x1, y1, x2, y2)) for box in phones):
                continue
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size:
                crops.append(crop)
                offsets.append((x1, y1))
        if crops:
            hand_dets = self._predict(crops, offsets, imgsz=self._cfg.hand_roi_imgsz)
            detections.extend(d for d in hand_dets if d.label != PERSON_LABEL)
        detections = merge_detections(detections)
        detections = self._verify_phones(frame_bgr, detections)

        return ObjectResult(
            phone_detected=any(d.label == PHONE_LABEL for d in detections),
            person_detected=any(d.label == PERSON_LABEL for d in detections),
            detections=detections,
        )

    def _verify_phones(self, frame_bgr: np.ndarray, detections: list[Detection]) -> list[Detection]:
        """Keep a less confident phone box only if YOLO finds a phone at the same place
        in an enlarged crop around it. Its confidence becomes the mean of both looks."""
        cfg = self._cfg
        if cfg.phone_verify_imgsz <= 0:
            return detections
        doubtful = [d for d in detections if d.label != PERSON_LABEL and d.confidence < cfg.phone_verify_skip]
        if not doubtful:
            return detections
        height, width = frame_bgr.shape[:2]
        crops, offsets = [], []
        for det in doubtful:
            x1, y1, x2, y2 = det.box
            side = min(max(cfg.phone_verify_context * max(x2 - x1, y2 - y1), 128.0), float(min(width, height)))
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            left = int(np.clip(cx - side / 2, 0, width - side))
            top = int(np.clip(cy - side / 2, 0, height - side))
            crops.append(frame_bgr[top:top + int(side), left:left + int(side)])
            offsets.append((left, top))
        second_looks = self._predict_each(crops, offsets, imgsz=cfg.phone_verify_imgsz)

        verified = {}
        for det, found in zip(doubtful, second_looks):
            matches = [d.confidence for d in found
                       if d.label != PERSON_LABEL and d.confidence >= cfg.phone_verify_confidence
                       and box_iou(d.box, det.box) >= 0.3]
            if matches:
                verified[id(det)] = Detection(det.label, (det.confidence + max(matches)) / 2.0, det.box)
            else:
                logger.debug("Phone candidate rejected by verification: %s %.2f %s", det.label, det.confidence, det.box)
        doubtful_ids = {id(d) for d in doubtful}
        return [verified[id(d)] if id(d) in verified else d
                for d in detections if id(d) not in doubtful_ids or id(d) in verified]

    def _predict(self, images: list[np.ndarray], offsets: list[tuple[int, int]], imgsz: int | None) -> list[Detection]:
        return [d for found in self._predict_each(images, offsets, imgsz) for d in found]

    def _predict_each(
        self, images: list[np.ndarray], offsets: list[tuple[int, int]], imgsz: int | None,
    ) -> list[list[Detection]]:
        """Detections per image, in frame coordinates."""
        kwargs = {"imgsz": imgsz} if imgsz else {}
        results = self._model.predict(
            images,
            conf=min(self._thresholds.values()),
            classes=list(self._thresholds),
            verbose=False,
            **kwargs,
        )
        per_image = []
        for result, (dx, dy) in zip(results, offsets):
            detections = []
            per_image.append(detections)
            for cls, conf, xyxy in zip(
                result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist()
            ):
                cls = int(cls)
                if conf < self._thresholds.get(cls, 1.0):
                    continue
                x1, y1, x2, y2 = (int(v) for v in xyxy)
                detections.append(Detection(_LABELS[cls], float(conf), (x1 + dx, y1 + dy, x2 + dx, y2 + dy)))
        return per_image


def _inside(box: Box, region: Box) -> bool:
    """Is the box centre inside the region?"""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]

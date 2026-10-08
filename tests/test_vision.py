"""
Tests for app/vision/: head angles from the face transform, and the camera
wrapper when no camera is open.
"""

import math

import numpy as np
import pytest

from app.config.settings import CameraSettings
from app.vision.camera import Camera
from app.vision.head_pose import estimate_head_pose, rotation_to_euler


def _rx(deg):
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def _ry(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def _rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def _transform(rotation, scale=1.0):
    transform = np.eye(4)
    transform[:3, :3] = rotation * scale
    transform[:3, 3] = [1.0, -2.0, -40.0]
    return transform


def test_identity_transform_is_facing_camera():
    pose = estimate_head_pose(np.eye(4))
    assert pose.yaw == pytest.approx(0.0)
    assert pose.pitch == pytest.approx(0.0)
    assert pose.roll == pytest.approx(0.0)


@pytest.mark.parametrize("yaw,pitch,roll", [(30, 0, 0), (0, 15, 0), (0, 0, -10), (20, -12, 5)])
def test_rotation_round_trip(yaw, pitch, roll):
    rotation = _rz(roll) @ _ry(yaw) @ _rx(pitch)
    assert rotation_to_euler(rotation) == pytest.approx((yaw, pitch, roll), abs=1e-6)


def test_head_pose_ignores_scale_and_reports_pitch_up_as_positive():
    # A negative X rotation in MediaPipe's camera frame tips the face upwards.
    pose = estimate_head_pose(_transform(_rx(-15), scale=1.7))
    assert pose.pitch == pytest.approx(15.0)
    assert pose.yaw == pytest.approx(0.0, abs=1e-6)


def test_head_pose_handles_missing_transform():
    assert estimate_head_pose(None) is None
    assert estimate_head_pose(np.zeros((4, 4))) is None


def test_camera_read_without_open_returns_none():
    camera = Camera(CameraSettings())
    assert not camera.is_open
    assert camera.read() is None
    camera.release()  # must be safe to call when never opened


# --- object detection: hand crops and the two-pass detector -------------------------------

from app.config.settings import DetectionSettings  # noqa: E402
from app.vision.object_detection import (  # noqa: E402
    COCO_CELL_PHONE,
    COCO_PERSON,
    COCO_REMOTE,
    Detection,
    ObjectDetector,
    hand_regions,
    merge_detections,
)
from app.vision.pose_detection import PoseLandmark  # noqa: E402

_FRAME_SIZE = (640, 480)


def _pose_with_hands(left=None, right=None, visibility=0.9):
    lm = np.zeros((33, 4), np.float32)
    lm[PoseLandmark.LEFT_SHOULDER] = (0.65, 0.75, 0, 0.99)
    lm[PoseLandmark.RIGHT_SHOULDER] = (0.35, 0.75, 0, 0.99)
    if left is not None:
        lm[PoseLandmark.LEFT_WRIST] = (*left, 0, visibility)
    if right is not None:
        lm[PoseLandmark.RIGHT_WRIST] = (*right, 0, visibility)
    return lm


def test_hand_regions_follow_the_visible_hands():
    regions = hand_regions(_pose_with_hands(left=(0.8, 0.6), right=(0.2, 0.6)), _FRAME_SIZE, DetectionSettings())
    assert len(regions) == 2
    for (x1, y1, x2, y2), (hx, hy) in zip(sorted(regions), [(128, 288), (512, 288)]):
        assert x1 <= hx <= x2 and y1 <= hy <= y2
        assert x2 - x1 >= DetectionSettings().hand_roi_min_px
        assert 0 <= x1 and x2 <= 640 and 0 <= y1 and y2 <= 480


def test_hand_regions_merge_hands_held_together_and_skip_hidden_ones():
    together = hand_regions(_pose_with_hands(left=(0.52, 0.8), right=(0.48, 0.8)), _FRAME_SIZE, DetectionSettings())
    assert len(together) == 1
    assert hand_regions(_pose_with_hands(left=(0.5, 0.8), visibility=0.1), _FRAME_SIZE, DetectionSettings()) == []
    assert hand_regions(None, _FRAME_SIZE, DetectionSettings()) == []


def test_merge_detections_keeps_the_strongest_duplicate():
    merged = merge_detections([
        Detection("cell phone", 0.3, (100, 100, 150, 180)),
        Detection("remote", 0.5, (102, 101, 151, 178)),
        Detection("person", 0.9, (90, 50, 400, 480)),
    ])
    assert [(d.label, d.confidence) for d in merged] == [("person", 0.9), ("remote", 0.5)]


class _Values(list):
    def tolist(self):
        return list(self)


class _Boxes:
    def __init__(self, items):
        self.cls = _Values(c for c, _, _ in items)
        self.conf = _Values(p for _, p, _ in items)
        self.xyxy = _Values(b for _, _, b in items)


class _Result:
    def __init__(self, items):
        self.boxes = _Boxes(items)


class _SceneYolo:
    """A fake YOLO that "sees" a fixed scene.

    `scene` lists (class, confidence at full frame, confidence when zoomed in, box in
    frame pixels); a confidence of None means YOLO misses it at that scale. Every pixel
    of the frame made by _scene_frame() encodes its own coordinates, so a crop tells
    where it was cut from, and the objects inside it are reported in crop coordinates.
    """

    def __init__(self, scene):
        self.scene, self.calls = scene, []

    def predict(self, images, conf, classes, verbose, imgsz=None):
        self.calls.append((len(images), imgsz))
        results = []
        for image in images:
            left, top = int(image[0, 0, 0]) * 4, int(image[0, 0, 1]) * 4
            height, width = image.shape[:2]
            items = []
            for cls, full, zoomed, (x1, y1, x2, y2) in self.scene:
                score = full if imgsz is None else zoomed
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if score is not None and left <= cx < left + width and top <= cy < top + height:
                    items.append((cls, score, [x1 - left, y1 - top, x2 - left, y2 - top]))
            results.append(_Result(items))
        return results


def _scene_frame():
    """640x480 frame whose pixels encode (x // 4, y // 4) in the first two channels."""
    ys, xs = np.mgrid[0:480, 0:640]
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[..., 0], frame[..., 1] = xs // 4, ys // 4
    return frame


def _detector(model, **changes):
    import dataclasses

    detector = ObjectDetector.__new__(ObjectDetector)
    cfg = dataclasses.replace(DetectionSettings(), **changes)
    detector._cfg = cfg
    detector._thresholds = {COCO_PERSON: cfg.yolo_confidence, COCO_REMOTE: cfg.remote_confidence,
                            COCO_CELL_PHONE: cfg.phone_confidence}
    detector._model, detector.error = model, None
    return detector


def test_detector_finds_a_small_phone_in_a_hand_crop():
    # Missed in the full frame, found in the hand crop and confirmed by the zoomed check.
    model = _SceneYolo([(COCO_PERSON, 0.9, None, (100, 50, 500, 480)),
                        (COCO_CELL_PHONE, None, 0.5, (210, 320, 240, 372))])
    result = _detector(model).detect(_scene_frame(), [(200, 300, 360, 460)])
    assert model.calls[:2] == [(1, None), (1, DetectionSettings().hand_roi_imgsz)]
    phone = [d for d in result.detections if d.label == "cell phone"]
    assert phone and phone[0].box == (210, 320, 240, 372)  # crop coordinates moved into the frame
    assert result.phone_detected and result.person_detected


def test_detector_skips_hand_crops_that_already_contain_a_phone():
    model = _SceneYolo([(COCO_CELL_PHONE, 0.8, 0.8, (250, 350, 300, 420))])
    _detector(model).detect(_scene_frame(), [(200, 300, 360, 460)])
    assert model.calls == [(1, None)]  # confident phone: no hand pass, no verification


def test_detector_applies_per_class_thresholds():
    cfg = DetectionSettings()
    model = _SceneYolo([
        (COCO_PERSON, cfg.yolo_confidence - 0.05, None, (0, 0, 100, 100)),  # weak person: dropped
        (COCO_CELL_PHONE, cfg.phone_confidence + 0.01, 0.6, (300, 300, 340, 360)),  # weak, verified: kept
        (COCO_REMOTE, cfg.remote_confidence - 0.01, 0.6, (400, 300, 440, 360)),  # weak remote: dropped
        (COCO_CELL_PHONE, cfg.phone_confidence - 0.05, 0.9, (500, 100, 540, 160)),  # below the floor: dropped
    ])
    result = _detector(model).detect(_scene_frame())
    assert [d.label for d in result.detections] == ["cell phone"]
    assert not result.person_detected


def test_background_object_that_fails_the_zoomed_check_is_dropped():
    # A dark rectangle on a shelf: YOLO's full-frame pass calls it a phone, the closer look does not.
    model = _SceneYolo([(COCO_CELL_PHONE, 0.5, None, (560, 40, 590, 90)),
                        (COCO_CELL_PHONE, 0.5, 0.2, (40, 40, 70, 90))])  # too weak when zoomed
    result = _detector(model).detect(_scene_frame())
    assert result.detections == [] and not result.phone_detected
    assert model.calls == [(1, None), (2, DetectionSettings().phone_verify_imgsz)]


def test_verified_phone_gets_the_mean_confidence_of_both_looks():
    model = _SceneYolo([(COCO_CELL_PHONE, 0.5, 0.9, (300, 300, 340, 370))])
    (phone,) = _detector(model).detect(_scene_frame()).detections
    assert phone.confidence == pytest.approx(0.7)


def test_verification_can_be_switched_off():
    model = _SceneYolo([(COCO_CELL_PHONE, 0.5, None, (300, 300, 340, 370))])
    result = _detector(model, phone_verify_imgsz=0).detect(_scene_frame())
    assert result.phone_detected and model.calls == [(1, None)]

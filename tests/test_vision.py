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

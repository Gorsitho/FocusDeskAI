"""Head orientation (yaw, pitch, roll) from the MediaPipe facial transform."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class HeadPose:
    """Angles in degrees. 0/0/0 means looking straight at the camera.

    yaw   > 0: head turned towards the user's left
    pitch > 0: head tilted up
    roll  > 0: head tilted towards the user's right shoulder
    """

    yaw: float
    pitch: float
    roll: float


def rotation_to_euler(rotation: np.ndarray) -> tuple[float, float, float]:
    """Decompose R = Rz(roll) @ Ry(yaw) @ Rx(pitch) into (yaw, pitch, roll) degrees."""
    r = rotation
    sy = math.hypot(r[0, 0], r[1, 0])
    if sy > 1e-6:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    else:  # gimbal lock: yaw at +/-90 degrees
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
        roll = 0.0
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def estimate_head_pose(transform: np.ndarray | None) -> HeadPose | None:
    if transform is None:
        return None
    rotation = np.asarray(transform, dtype=np.float64)[:3, :3]
    # The transform may include scale; normalise each column to get a pure rotation.
    norms = np.linalg.norm(rotation, axis=0)
    if np.any(norms < 1e-9):
        return None
    yaw, pitch, roll = rotation_to_euler(rotation / norms)
    # MediaPipe's camera looks down -Z with Y up, so a positive X rotation tips
    # the face downwards; flip it so that "looking up" reads as positive pitch.
    return HeadPose(yaw=yaw, pitch=-pitch, roll=roll)

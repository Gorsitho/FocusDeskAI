"""
This file measures behaviour over time, not in a single frame.
It answers questions like: How long has the user been looking away?
How long at the phone? How long have the eyes been closed? When was the face last seen?
Is anything moving?

Main parts:

* BehaviorTimer: measures how long a behaviour lasts. Very short breaks
  (shorter than a "grace" time) pause the timer instead of resetting it.
* ActivityTracker: uses these timers and the movement of the body and face points.

It is used by app/features/feature_pipeline.py. The state rules in that file
make their decisions with these durations (ActivityFeatures).
"""

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from app.config.settings import DetectionSettings, FeatureSettings

# Pose landmarks 0-12 cover the head and shoulders, which are visible at a desk.
_UPPER_BODY = slice(0, 13)


@dataclass
class ActivityFeatures:
    """Durations and movement values for one frame (made by ActivityTracker)."""
    # Time since the user was last present: since the face was last seen when the
    # caller passes `face_present`, otherwise since anybody was seen.
    seconds_since_person_seen: float = 0.0
    # Time the person has been present but looking away from the monitors.
    seconds_looking_away: float = 0.0
    # Time the person has been looking at a visible phone.
    seconds_looking_at_phone: float = 0.0
    # Time the eyes have been closed (blinks do not count).
    seconds_eyes_closed: float = 0.0
    # Mean landmark speed over the history window, in normalised image units per second.
    motion_level: float | None = None
    # Continuous time motion_level has stayed below the stillness threshold.
    seconds_still: float = 0.0


class BehaviorTimer:
    """Measures how long a behaviour has been going on, tolerating short interruptions.

    Only time spent in the behaviour is counted, so detector glitches cannot
    inflate it. An interruption shorter than `grace_s` pauses the timer instead
    of resetting it, so a quick glance back at the screen does not restart a
    long look-away. While the behaviour is not active the timer reports 0.
    """

    def __init__(self, grace_s: float):
        self._grace = grace_s
        self._elapsed = 0.0
        self._last_time: float | None = None
        self._last_active = False
        self._last_active_time: float | None = None

    def update(self, timestamp: float, active: bool) -> float:
        if active:
            if self._last_active and self._last_time is not None:
                self._elapsed += max(timestamp - self._last_time, 0.0)
            elif self._last_active_time is None or timestamp - self._last_active_time > self._grace:
                self._elapsed = 0.0
            self._last_active_time = timestamp
        self._last_time, self._last_active = timestamp, active
        return self._elapsed if active else 0.0


class ActivityTracker:
    """Keeps the history that is needed to measure how long behaviours last and how much the user moves."""
    def __init__(self, feature_cfg: FeatureSettings, detection_cfg: DetectionSettings, still_threshold: float):
        self._window = feature_cfg.history_window_s
        self._min_visibility = detection_cfg.min_landmark_visibility
        self._still_threshold = still_threshold

        self._last_person_time: float | None = None
        self._look_away = BehaviorTimer(feature_cfg.behavior_grace_s)
        self._phone_gaze = BehaviorTimer(feature_cfg.behavior_grace_s)
        self._eyes_closed = BehaviorTimer(feature_cfg.eye_blink_grace_s)
        self._still_since: float | None = None
        self._prev_points: np.ndarray | None = None
        self._prev_time: float | None = None
        self._motion = deque()  # (timestamp, speed)

    def update(
        self,
        timestamp: float,
        person_present: bool,
        facing_screen: bool | None,
        looking_at_phone: bool,
        tracking_points: np.ndarray | None,
        eyes_closed: bool = False,
        face_present: bool | None = None,
    ) -> ActivityFeatures:
        """Update all timers with the newest frame and return the current values.

        `face_present` (if given) decides presence for seconds_since_person_seen;
        `person_present` still drives the other timers.
        """
        if person_present if face_present is None else face_present:
            self._last_person_time = timestamp

        looking_at_phone = person_present and looking_at_phone
        # A missing face while the body is visible usually means the head is turned away.
        # Phone use has its own timer and threshold, so it is excluded here.
        looking_away = person_present and facing_screen is not True and not looking_at_phone
        seconds_looking_away = self._look_away.update(timestamp, looking_away)
        seconds_looking_at_phone = self._phone_gaze.update(timestamp, looking_at_phone)
        seconds_eyes_closed = self._eyes_closed.update(timestamp, person_present and eyes_closed)

        self._update_motion(timestamp, tracking_points)
        self._trim(timestamp)

        motion_level = float(np.mean([s for _, s in self._motion])) if self._motion else None
        if person_present and motion_level is not None and motion_level < self._still_threshold:
            self._still_since = self._still_since if self._still_since is not None else timestamp
        else:
            self._still_since = None

        return ActivityFeatures(
            # Infinite until someone has been seen, so the app starts as AWAY rather than FOCUSED.
            seconds_since_person_seen=(
                math.inf if self._last_person_time is None else timestamp - self._last_person_time
            ),
            seconds_looking_away=seconds_looking_away,
            seconds_looking_at_phone=seconds_looking_at_phone,
            seconds_eyes_closed=seconds_eyes_closed,
            motion_level=motion_level,
            seconds_still=0.0 if self._still_since is None else timestamp - self._still_since,
        )

    @staticmethod
    def tracking_points_from(pose_landmarks: np.ndarray | None, face_landmarks: np.ndarray | None,
                             min_visibility: float) -> np.ndarray | None:
        """Pick the landmarks used to measure motion, preferring the upper-body pose."""
        if pose_landmarks is not None:
            upper = pose_landmarks[_UPPER_BODY]
            # Hidden keypoints are hallucinated by the model and jitter a lot; zero them out
            # rather than dropping them so arrays stay aligned between frames.
            points = upper[:, :2].copy()
            points[upper[:, 3] < min_visibility] = 0.0
            return points
        if face_landmarks is not None:
            return face_landmarks[:, :2]
        return None

    def _update_motion(self, timestamp: float, points: np.ndarray | None) -> None:
        prev_points, prev_time = self._prev_points, self._prev_time
        self._prev_points, self._prev_time = points, timestamp
        if points is None or prev_points is None or prev_points.shape != points.shape:
            return
        dt = timestamp - prev_time
        if dt <= 0:
            return
        valid = np.any(points != 0, axis=1) & np.any(prev_points != 0, axis=1)
        if not np.any(valid):
            return
        displacement = np.linalg.norm(points[valid] - prev_points[valid], axis=1).mean()
        self._motion.append((timestamp, float(displacement / dt)))

    def _trim(self, now: float) -> None:
        while self._motion and now - self._motion[0][0] > self._window:
            self._motion.popleft()

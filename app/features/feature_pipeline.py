"""Combines per-frame detections into features and a rule-based focus state."""

import math
from dataclasses import dataclass, field
from enum import Enum

from app.config.settings import Settings, StateSettings, StudyMethod
from app.features.activity_features import ActivityFeatures, ActivityTracker
from app.features.gaze_features import (
    Attention,
    GazeFeatures,
    HeadPoseSmoother,
    PitchCalibrator,
    ScreenZones,
    compute_screen_zones,
    extract_gaze_features,
    relative_to_camera_line,
)
from app.features.phone_features import (
    PhoneFeatures,
    PhoneTracker,
    face_anchor,
    is_looking_at_phone,
    phone_gaze_angle,
    phone_near_face,
)
from app.features.pose_features import PoseFeatures, extract_pose_features
from app.vision.face_detection import FaceResult
from app.vision.head_pose import HeadPose, estimate_head_pose
from app.vision.object_detection import ObjectResult
from app.vision.pose_detection import PoseResult


class FocusState(str, Enum):
    FOCUSED = "FOCUSED"
    DISTRACTED = "DISTRACTED"
    AWAY = "AWAY"
    # Set manually by the user from the UI; never produced by the rules.
    BREAK = "BREAK"


class ReasonCode(str, Enum):
    ABSENT = "absent"
    NO_MOVEMENT = "no_movement"
    PHONE = "phone"
    LOOKING_AWAY = "looking_away"
    ON_MONITOR = "on_monitor"
    ON_DESK = "on_desk"


@dataclass(frozen=True)
class Reason:
    """Language-neutral explanation of a state; the UI turns it into text."""

    code: ReasonCode
    seconds: float = 0.0
    attention: Attention | None = None
    monitor: int | None = None


@dataclass
class FrameFeatures:
    timestamp: float
    person_detected: bool = False
    face_detected: bool = False
    # A phone is (or was very recently) visible; this alone is not a distraction.
    phone_detected: bool = False
    # Smoothed head orientation relative to the line towards the camera, with the
    # learned pitch bias removed. Used for every attention decision.
    head_pose: HeadPose | None = None
    # Learned head-pitch bias (degrees) that has been subtracted.
    pitch_calibration: float = 0.0
    pose: PoseFeatures = field(default_factory=PoseFeatures)
    gaze: GazeFeatures = field(default_factory=GazeFeatures)
    phone: PhoneFeatures = field(default_factory=PhoneFeatures)
    attention: Attention = Attention.ABSENT
    # Monitor being looked at (0-based), if any.
    monitor: int | None = None
    activity: ActivityFeatures = field(default_factory=ActivityFeatures)


@dataclass
class FrameAnalysis:
    features: FrameFeatures
    state: FocusState
    reason: Reason | None = None


def classify(features: FrameFeatures, cfg: StateSettings) -> tuple[FocusState, Reason]:
    """Rule-based classification of observable behaviour, in priority order.

    Every rule depends on how long a behaviour has lasted, never on a single frame.
    """
    activity = features.activity
    if activity.seconds_since_person_seen >= cfg.away_after_s:
        return FocusState.AWAY, Reason(ReasonCode.ABSENT)
    if activity.seconds_still >= cfg.still_away_after_s:
        return FocusState.AWAY, Reason(ReasonCode.NO_MOVEMENT, activity.seconds_still)
    if activity.seconds_looking_at_phone >= cfg.phone_after_s:
        return FocusState.DISTRACTED, Reason(ReasonCode.PHONE, activity.seconds_looking_at_phone)
    if activity.seconds_looking_away >= cfg.look_away_after_s:
        return FocusState.DISTRACTED, Reason(
            ReasonCode.LOOKING_AWAY, activity.seconds_looking_away, attention=features.attention
        )
    if features.attention is Attention.DESK:
        return FocusState.FOCUSED, Reason(ReasonCode.ON_DESK)
    return FocusState.FOCUSED, Reason(ReasonCode.ON_MONITOR, monitor=features.monitor)


def classify_state(features: FrameFeatures, cfg: StateSettings) -> FocusState:
    return classify(features, cfg)[0]


class StateStabilizer:
    """Only switches state once a new candidate has persisted for a minimum time."""

    def __init__(self, min_duration_s: float):
        self._min_duration = min_duration_s
        self._state: FocusState | None = None
        self._candidate: FocusState | None = None
        self._candidate_since = 0.0

    def update(self, candidate: FocusState, timestamp: float) -> FocusState:
        if self._state is None:
            self._state = candidate
        if candidate == self._state:
            self._candidate = None
        elif candidate != self._candidate:
            self._candidate, self._candidate_since = candidate, timestamp
        elif timestamp - self._candidate_since >= self._min_duration:
            self._state, self._candidate = candidate, None
        return self._state


def resolve_attention(
    person: bool, gaze: GazeFeatures, looking_at_phone: bool, pose: PoseFeatures, zones: ScreenZones,
    torso_margin: float,
) -> Attention:
    if not person:
        return Attention.ABSENT
    if looking_at_phone:
        return Attention.PHONE
    if gaze.facing_screen is not None:
        return gaze.direction
    # No face: fall back on the body. Either way the user is not visibly facing a monitor.
    if pose.torso_yaw is not None and not (
        zones.yaw_min - torso_margin <= pose.torso_yaw <= zones.yaw_max + torso_margin
    ):
        return Attention.BODY_TURNED
    return Attention.NO_FACE


class FeaturePipeline:
    def __init__(self, config: Settings):
        self._activity = ActivityTracker(
            config.features, config.detection, config.state.still_motion_threshold
        )
        self._smoother = HeadPoseSmoother(config.features.head_smoothing_s)
        self._phone = PhoneTracker(config.features.phone_hold_s)
        self._stabilizer = StateStabilizer(config.state.min_state_duration_s)
        self._reasons: dict[FocusState, Reason] = {}
        # Monitor the user was looking at when the face was lost, if that monitor lies
        # beyond the angle the face landmarker can track.
        self._out_of_view_monitor: int | None = None
        self._last_monitor: int | None = None
        self._last_yaw = 0.0
        # The head pointed at the desk when the face was last seen (tablet / mixed study).
        self._last_on_desk = False
        self._last_phone_gaze = -math.inf
        self._calibrator = PitchCalibrator(config.features)
        self.reconfigure(config)

    def reconfigure(self, config: Settings) -> None:
        """Apply new thresholds and workspace layout without losing temporal history."""
        self._config = config
        self._zones = compute_screen_zones(config.features)
        self._last_monitor = self._out_of_view_monitor = None
        self._last_on_desk = False

    @property
    def screen_zones(self) -> ScreenZones:
        return self._zones

    def process(
        self,
        timestamp: float,
        frame_size: tuple[int, int],
        face: FaceResult,
        pose: PoseResult,
        objects: ObjectResult,
    ) -> FrameAnalysis:
        cfg = self._config
        tablet = cfg.features.study_method is StudyMethod.TABLET
        raw_head = estimate_head_pose(face.transform) if face.detected else None
        smoothed = self._smoother.update(timestamp, raw_head)
        anchor = face_anchor(face.landmarks, frame_size) if face.detected else None
        head_pose = phone_direction = None
        if smoothed is not None:
            towards_camera = smoothed
            if anchor is not None:
                towards_camera = relative_to_camera_line(
                    smoothed, anchor, frame_size, cfg.features.camera_vertical_fov
                )
            # Without monitors there is no reference to learn the pitch bias from.
            offset = 0.0 if tablet else self._calibrator.update(timestamp, towards_camera, self._zones)
            head_pose = HeadPose(towards_camera.yaw, towards_camera.pitch - offset, towards_camera.roll)
            # The phone is compared with its position in the image, i.e. relative to the
            # optical axis, so that check uses the uncorrected direction (minus the bias).
            phone_direction = HeadPose(smoothed.yaw, smoothed.pitch - offset, smoothed.roll)
        gaze = extract_gaze_features(head_pose, cfg.features, self._zones)
        pose_features = extract_pose_features(pose, frame_size, cfg.features, cfg.detection)
        person = pose.detected or face.detected or objects.person_detected

        phone_center = self._phone.update(timestamp, objects)
        near_face = phone_near_face(
            self._phone.box, face.landmarks if face.detected else None, frame_size, cfg.features
        )
        looking_at_phone = person and is_looking_at_phone(
            phone_center, head_pose, anchor, self._zones, pose_features, cfg.features, phone_direction, near_face,
            self._phone.box,
        )
        if looking_at_phone and head_pose is not None:
            self._last_phone_gaze = timestamp
        elif (person and head_pose is None and phone_center is not None
              and timestamp - self._last_phone_gaze <= cfg.features.phone_gaze_hold_s):
            # MediaPipe briefly lost the face (common when looking down at a phone):
            # keep the decision instead of restarting the phone timer.
            looking_at_phone = True
        gaze_angle = (
            phone_gaze_angle(phone_center, phone_direction, anchor, cfg.features)
            if phone_center is not None and phone_direction is not None and anchor is not None else None
        )
        attention = resolve_attention(
            person, gaze, looking_at_phone, pose_features, self._zones, cfg.features.torso_margin
        )
        monitor = gaze.monitor
        # Looking where the study method expects: a monitor, or the desk if allowed.
        facing_screen = True if gaze.on_desk else gaze.facing_screen
        if head_pose is not None:
            self._last_monitor, self._last_yaw = monitor, head_pose.yaw
            self._out_of_view_monitor = None
            self._last_on_desk = gaze.on_desk
        elif person and attention in (Attention.NO_FACE, Attention.BODY_TURNED):
            monitor = self._monitor_beyond_tracking()
            if monitor is not None:
                attention, facing_screen = Attention.ON_SCREEN, True
            elif attention is Attention.NO_FACE and self._last_on_desk:
                # Bending further over a notebook often hides the face; keep crediting the desk.
                attention, facing_screen = Attention.DESK, True
        else:
            self._last_monitor = self._out_of_view_monitor = None
            self._last_on_desk = False

        tracking_points = ActivityTracker.tracking_points_from(
            pose.landmarks, face.landmarks, cfg.detection.min_landmark_visibility
        )
        activity = self._activity.update(timestamp, person, facing_screen, looking_at_phone, tracking_points)

        features = FrameFeatures(
            timestamp=timestamp,
            person_detected=person,
            face_detected=face.detected,
            phone_detected=phone_center is not None,
            head_pose=head_pose,
            pitch_calibration=0.0 if tablet else self._calibrator.offset,
            pose=pose_features,
            gaze=gaze,
            phone=PhoneFeatures(
                visible=phone_center is not None, center=phone_center,
                looking_at_phone=looking_at_phone, gaze_angle=gaze_angle, near_face=near_face,
            ),
            attention=attention,
            monitor=monitor,
            activity=activity,
        )
        candidate, reason = classify(features, cfg.state)
        self._reasons[candidate] = reason
        state = self._stabilizer.update(candidate, timestamp)
        return FrameAnalysis(features=features, state=state, reason=self._reasons.get(state))

    def _monitor_beyond_tracking(self) -> int | None:
        """Keep crediting a far-side monitor after the face turned out of the camera's view."""
        if self._out_of_view_monitor is None and self._last_monitor is not None:
            zone = self._zones.monitors[self._last_monitor]
            limit = self._config.features.face_tracking_limit
            # Only when the monitor extends beyond the trackable range *and* the head was
            # already turned far towards it, i.e. the face was lost because of the turn.
            if (max(abs(zone.core_yaw[0]), abs(zone.core_yaw[1])) >= limit
                    and abs(self._last_yaw) >= limit - 15.0):
                self._out_of_view_monitor = self._last_monitor
            self._last_monitor = None
        return self._out_of_view_monitor

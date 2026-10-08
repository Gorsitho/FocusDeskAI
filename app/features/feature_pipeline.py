"""Combines per-frame detections into features and a rule-based focus state."""

from dataclasses import dataclass, field
from enum import Enum

from app.config.settings import Settings, StateSettings
from app.features.activity_features import ActivityFeatures, ActivityTracker
from app.features.gaze_features import (
    Attention,
    GazeFeatures,
    HeadPoseSmoother,
    ScreenZone,
    compute_screen_zone,
    extract_gaze_features,
)
from app.features.phone_features import (
    PhoneFeatures,
    PhoneTracker,
    face_anchor,
    is_looking_at_phone,
)
from app.features.pose_features import PoseFeatures, extract_pose_features
from app.vision.face_detection import FaceResult
from app.vision.head_pose import HeadPose, estimate_head_pose
from app.vision.object_detection import ObjectResult
from app.vision.pose_detection import PoseResult


class FocusState(str, Enum):
    FOCUSED = "FOCUSED"
    DISTRACTED = "DISTRACTED"
    IDLE = "IDLE"
    AWAY = "AWAY"
    # Set manually by the user from the UI; never produced by the rules.
    BREAK = "BREAK"


@dataclass
class FrameFeatures:
    timestamp: float
    person_detected: bool = False
    face_detected: bool = False
    # A phone is (or was very recently) visible; this alone is not a distraction.
    phone_detected: bool = False
    # Smoothed head orientation.
    head_pose: HeadPose | None = None
    pose: PoseFeatures = field(default_factory=PoseFeatures)
    gaze: GazeFeatures = field(default_factory=GazeFeatures)
    phone: PhoneFeatures = field(default_factory=PhoneFeatures)
    attention: Attention = Attention.ABSENT
    activity: ActivityFeatures = field(default_factory=ActivityFeatures)


@dataclass
class FrameAnalysis:
    features: FrameFeatures
    state: FocusState
    # Short human-readable explanation of the displayed state.
    reason: str = ""


def classify(features: FrameFeatures, cfg: StateSettings) -> tuple[FocusState, str]:
    """Rule-based classification of observable behaviour, in priority order.

    Every rule depends on how long a behaviour has lasted, never on a single frame.
    """
    activity = features.activity
    if activity.seconds_since_person_seen >= cfg.away_after_s:
        return FocusState.AWAY, "Nobody in front of the camera"
    if activity.seconds_looking_at_phone >= cfg.phone_after_s:
        return FocusState.DISTRACTED, f"Looking at phone for {activity.seconds_looking_at_phone:.0f} s"
    if activity.seconds_looking_away >= cfg.look_away_after_s:
        what = features.attention.value if features.attention is not Attention.ON_SCREEN else "Looking away"
        return FocusState.DISTRACTED, f"{what} for {activity.seconds_looking_away:.0f} s"
    if activity.seconds_still >= cfg.still_idle_after_s:
        return FocusState.IDLE, f"No movement for {activity.seconds_still:.0f} s"
    return FocusState.FOCUSED, "Attention on the monitors"


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
    person: bool, gaze: GazeFeatures, looking_at_phone: bool, pose: PoseFeatures, zone: ScreenZone, torso_margin: float
) -> Attention:
    if not person:
        return Attention.ABSENT
    if looking_at_phone:
        return Attention.PHONE
    if gaze.facing_screen is not None:
        return gaze.direction
    # No face: fall back on the body. Either way the user is not visibly facing a monitor.
    if pose.torso_yaw is not None and not (
        zone.yaw_min - torso_margin <= pose.torso_yaw <= zone.yaw_max + torso_margin
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
        self._reasons: dict[FocusState, str] = {}
        self.reconfigure(config)

    def reconfigure(self, config: Settings) -> None:
        """Apply new thresholds and workspace layout without losing temporal history."""
        self._config = config
        self._zone = compute_screen_zone(config.features)

    @property
    def screen_zone(self) -> ScreenZone:
        return self._zone

    def process(
        self,
        timestamp: float,
        frame_size: tuple[int, int],
        face: FaceResult,
        pose: PoseResult,
        objects: ObjectResult,
    ) -> FrameAnalysis:
        cfg = self._config
        raw_head = estimate_head_pose(face.transform) if face.detected else None
        head_pose = self._smoother.update(timestamp, raw_head)
        gaze = extract_gaze_features(head_pose, cfg.features, self._zone)
        pose_features = extract_pose_features(pose, frame_size, cfg.features, cfg.detection)
        person = pose.detected or face.detected or objects.person_detected

        phone_center = self._phone.update(timestamp, objects)
        looking_at_phone = person and is_looking_at_phone(
            phone_center, head_pose, face_anchor(face.landmarks, frame_size) if face.detected else None,
            self._zone, pose_features, cfg.features,
        )
        attention = resolve_attention(
            person, gaze, looking_at_phone, pose_features, self._zone, cfg.features.torso_margin
        )

        tracking_points = ActivityTracker.tracking_points_from(
            pose.landmarks, face.landmarks, cfg.detection.min_landmark_visibility
        )
        activity = self._activity.update(
            timestamp, person, gaze.facing_screen, looking_at_phone, tracking_points
        )

        features = FrameFeatures(
            timestamp=timestamp,
            person_detected=person,
            face_detected=face.detected,
            phone_detected=phone_center is not None,
            head_pose=head_pose,
            pose=pose_features,
            gaze=gaze,
            phone=PhoneFeatures(
                visible=phone_center is not None, center=phone_center, looking_at_phone=looking_at_phone
            ),
            attention=attention,
            activity=activity,
        )
        candidate, reason = classify(features, cfg.state)
        self._reasons[candidate] = reason
        state = self._stabilizer.update(candidate, timestamp)
        return FrameAnalysis(features=features, state=state, reason=self._reasons.get(state, ""))

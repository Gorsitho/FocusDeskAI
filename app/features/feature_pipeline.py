"""
This file is the "brain" of FocusDesk AI.
For every camera frame, it combines the detector results into features and
decides the focus state: FOCUSED, DISTRACTED or AWAY.
(BREAK is never decided here; the user sets it with the Break button.)

How it works:

1. FeaturePipeline.process() gets the face, pose and object results from app/vision/.
2. It calculates the eye state and eye direction (eye_features.py), the gaze from
   head and eyes together (gaze_features.py), phone use (phone_features.py),
   posture (pose_features.py) and how long each behaviour lasts (activity_features.py).
3. classify() applies the state rules in order of priority.
4. StateStabilizer shows a new state only after it has lasted for a short time,
   so the state does not flicker.

The background worker in app/ui/main_window.py calls it for every frame.
The result (FrameAnalysis) is shown by app/ui/dashboard.py and app/ui/camera_widget.py.
"""

import math
from dataclasses import dataclass, field
from enum import Enum

from app.config.settings import Settings, StateSettings, StudyMethod
from app.features.activity_features import ActivityFeatures, ActivityTracker
from app.features.eye_features import EyeAnalyzer, EyeFeatures, EyeState, measure_eyes
from app.features.gaze_features import (
    Attention,
    GazeFeatures,
    HeadPoseSmoother,
    PitchCalibrator,
    ScreenZones,
    combine_gaze,
    compute_screen_zones,
    extract_gaze_features,
    relative_to_camera_line,
)
from app.features.phone_features import (
    PhoneFeatures,
    PhoneTracker,
    face_anchor,
    hand_points,
    is_looking_at_phone,
    phone_at_ear,
    phone_gaze_angle,
    phone_in_hand,
    phone_near_face,
    plausible_phone,
)
from app.features.pose_features import PoseFeatures, extract_pose_features
from app.vision.face_detection import FaceResult
from app.vision.head_pose import HeadPose, estimate_head_pose
from app.vision.object_detection import PHONE_LABEL, Detection, ObjectResult
from app.vision.pose_detection import PoseResult


class FocusState(str, Enum):
    """The states that the app can show."""
    FOCUSED = "FOCUSED"
    DISTRACTED = "DISTRACTED"
    AWAY = "AWAY"
    # Set manually by the user from the UI; never produced by the rules.
    BREAK = "BREAK"


class ReasonCode(str, Enum):
    """Short codes that explain why a state was chosen."""
    ABSENT = "absent"
    # A body is visible, but no face: counted as away.
    FACE_MISSING = "face_missing"
    NO_MOVEMENT = "no_movement"
    PHONE = "phone"
    EYES_CLOSED = "eyes_closed"
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
    """Everything the pipeline found out about one frame."""
    timestamp: float
    # Anybody visible at all (face, body or a YOLO person box).
    person_detected: bool = False
    face_detected: bool = False
    # A phone is (or was very recently) visible; this alone is not a distraction.
    phone_detected: bool = False
    # Smoothed head orientation relative to the line towards the camera, with the
    # learned pitch bias removed.
    head_pose: HeadPose | None = None
    # The gaze: head_pose corrected by the eye direction (head-angle units, see
    # combine_gaze). Used for every attention decision; equals head_pose without eyes.
    gaze_pose: HeadPose | None = None
    eyes: EyeFeatures = field(default_factory=EyeFeatures)
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
    """Final result for one frame: the features, the state to show and the reason for it."""
    features: FrameFeatures
    state: FocusState
    reason: Reason | None = None


def classify(features: FrameFeatures, cfg: StateSettings) -> tuple[FocusState, Reason]:
    """Rule-based classification of observable behaviour, in priority order.

    Every rule depends on how long a behaviour has lasted, never on a single frame.
    """
    activity = features.activity
    # Presence means a visible face: a body alone (turned away, face hidden, someone
    # else in the background) is AWAY once the face has been gone for away_after_s.
    if activity.seconds_since_person_seen >= cfg.away_after_s:
        return FocusState.AWAY, Reason(ReasonCode.FACE_MISSING if features.person_detected else ReasonCode.ABSENT)
    if activity.seconds_still >= cfg.still_away_after_s:
        return FocusState.AWAY, Reason(ReasonCode.NO_MOVEMENT, activity.seconds_still)
    if activity.seconds_looking_at_phone >= cfg.phone_after_s:
        return FocusState.DISTRACTED, Reason(ReasonCode.PHONE, activity.seconds_looking_at_phone)
    if activity.seconds_eyes_closed >= cfg.eyes_closed_after_s:
        return FocusState.DISTRACTED, Reason(ReasonCode.EYES_CLOSED, activity.seconds_eyes_closed)
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
    """Decide where the user's attention is in this frame (monitor, phone, left, no face, ...)."""
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
    """Turns the detector results of each frame into features and a focus state.

    It remembers information between frames (smoothing, timers), so one instance
    is used for the whole video.
    """
    def __init__(self, config: Settings):
        self._activity = ActivityTracker(
            config.features, config.detection, config.state.still_motion_threshold
        )
        self._smoother = HeadPoseSmoother(config.features.head_smoothing_s)
        self._phone = PhoneTracker(config.features)
        self._eyes = EyeAnalyzer(config.features)
        # The last YOLO result seen; YOLO does not run on every frame, and a repeated
        # result must not count as new evidence for a phone.
        self._last_objects: ObjectResult | None = None
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
        self._last_face_time: float | None = None
        self._calibrator = PitchCalibrator(config.features)
        self.reconfigure(config)

    def reconfigure(self, config: Settings) -> None:
        """Apply new thresholds and workspace layout without losing temporal history."""
        self._config = config
        self._zones = compute_screen_zones(config.features)
        self._phone.reconfigure(config.features)
        self._eyes.reconfigure(config.features)
        # Head pitch for looking at the monitors; looking further down lowers the eyelids.
        self._reference_pitch = sum(z.center[1] for z in self._zones.monitors) / len(self._zones.monitors)
        self._last_monitor = self._out_of_view_monitor = None
        self._last_on_desk = False
        # Monitor looked at in the previous frame (for the exit hysteresis).
        self._gaze_monitor: int | None = None

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
        """Analyse one frame and return its features, the state to show and the reason."""
        cfg = self._config
        tablet = cfg.features.study_method is StudyMethod.TABLET
        # Step 1: head direction. Smooth it, measure it from the camera line and
        # remove the learned up/down error.
        raw_head = estimate_head_pose(face.transform) if face.detected else None
        smoothed = self._smoother.update(timestamp, raw_head)
        anchor = face_anchor(face.landmarks, frame_size) if face.detected else None
        head_pose = phone_direction = gaze_pose = None
        eyes = EyeFeatures()
        if smoothed is not None:
            towards_camera = smoothed
            if anchor is not None:
                towards_camera = relative_to_camera_line(
                    smoothed, anchor, frame_size, cfg.features.camera_vertical_fov
                )
            # Without monitors there is no reference to learn the pitch bias from.
            offset = 0.0 if tablet else self._calibrator.update(timestamp, towards_camera, self._zones)
            head_pose = HeadPose(towards_camera.yaw, towards_camera.pitch - offset, towards_camera.roll)

            # Step 2: the eyes. Their state (open / partially closed / closed) and their
            # direction inside the head, which is added to the head direction (the gaze).
            measurement = measure_eyes(face.landmarks, frame_size, cfg.features, face.blendshapes)
            head_monitor = None if tablet else self._zones.monitor_at(head_pose.yaw, head_pose.pitch)
            monitor_center = None if head_monitor is None else self._zones.monitors[head_monitor].center
            eyes = self._eyes.update(timestamp, measurement, head_pose, monitor_center, self._reference_pitch)
            gaze_pose = combine_gaze(head_pose, eyes.yaw, eyes.pitch, cfg.features)
            # The phone is compared with its position in the image, i.e. relative to the
            # optical axis, so that check uses the uncorrected direction (minus the bias),
            # turned further by the eyes: the real line of sight.
            phone_direction = HeadPose(
                smoothed.yaw + cfg.features.eye_yaw_weight * (eyes.yaw or 0.0),
                smoothed.pitch - offset + cfg.features.eye_pitch_weight * (eyes.pitch or 0.0),
                smoothed.roll,
            )
        else:
            self._eyes.update(timestamp, None, None)
        # Step 3: where the user looks (monitor, desk, left, ...), posture, and is anybody there.
        # Measured eyes make the direction more certain, so the tolerance shrinks on that axis.
        scale = cfg.features.eye_margin_scale
        gaze_zones = self._zones.scaled(
            scale if eyes.yaw is not None else 1.0, scale if eyes.pitch is not None else 1.0
        )
        gaze = extract_gaze_features(gaze_pose, cfg.features, gaze_zones, self._gaze_monitor)
        self._gaze_monitor = gaze.monitor
        pose_features = extract_pose_features(pose, frame_size, cfg.features, cfg.detection)
        # The face decides presence. A body without a face only bridges a short face
        # dropout (head bowed over a phone or a notebook, a quick turn); after
        # away_after_s without a face the state is AWAY whatever else is visible.
        body = pose.detected or objects.person_detected
        if face.detected:
            self._last_face_time = timestamp
        face_recent = (self._last_face_time is not None
                       and timestamp - self._last_face_time < cfg.state.away_after_s)
        person = face.detected or (body and face_recent)

        # Step 4: phone. Collect evidence over YOLO runs, find out if it is in the hand,
        # at the ear or in front of the face, and if the gaze points at it.
        face_landmarks = face.landmarks if face.detected else None
        hands = hand_points(pose.landmarks if pose.detected else None, frame_size, cfg.detection.min_hand_visibility)
        fresh = objects is not self._last_objects
        self._last_objects = objects
        def held(d: Detection) -> bool:
            return (phone_in_hand(d.box, hands, cfg.features)
                    or phone_near_face(d.box, face_landmarks, frame_size, cfg.features))

        candidates = [
            d for d in objects.phone_candidates
            # A "remote" is only a phone when the user holds it; a remote on the desk is ignored.
            if (d.label == PHONE_LABEL or held(d))
            and plausible_phone(d, face_landmarks, frame_size, held(d), cfg.features)
        ]
        phone_center = self._phone.update(timestamp, candidates, fresh, held)
        box = self._phone.box if phone_center is not None else None
        if box is not None and hands:
            self._phone.in_hand = phone_in_hand(box, hands, cfg.features)
        # Without visible hands the last answer is kept (hands often leave the image).
        in_hand = box is not None and self._phone.in_hand
        near_face = phone_near_face(box, face_landmarks, frame_size, cfg.features)
        at_ear = phone_at_ear(box, face_landmarks, frame_size, cfg.features)
        looking_at_phone = person and (at_ear or is_looking_at_phone(
            phone_center, gaze_pose, anchor, gaze_zones, pose_features, cfg.features, phone_direction, near_face,
            box, in_hand,
        ))
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
        # Step 5: combine everything into one attention value. When the face is lost,
        # use what the user was looking at just before (far monitor or desk).
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

        # Step 6: update the behaviour timers (looking away, phone, eyes closed, no movement).
        tracking_points = ActivityTracker.tracking_points_from(
            pose.landmarks, face.landmarks, cfg.detection.min_landmark_visibility
        )
        activity = self._activity.update(
            timestamp, person, facing_screen, looking_at_phone, tracking_points,
            eyes_closed=eyes.state is EyeState.CLOSED, face_present=face.detected,
        )

        features = FrameFeatures(
            timestamp=timestamp,
            person_detected=face.detected or body,
            face_detected=face.detected,
            phone_detected=phone_center is not None,
            head_pose=head_pose,
            gaze_pose=gaze_pose,
            eyes=eyes,
            pitch_calibration=0.0 if tablet else self._calibrator.offset,
            pose=pose_features,
            gaze=gaze,
            phone=PhoneFeatures(
                visible=phone_center is not None, center=phone_center, box=box,
                confidence=self._phone.confidence if box is not None else 0.0,
                looking_at_phone=looking_at_phone, gaze_angle=gaze_angle, near_face=near_face,
                in_hand=in_hand, at_ear=at_ear,
            ),
            attention=attention,
            monitor=monitor,
            activity=activity,
        )
        # Step 7: apply the state rules, then only switch the shown state once the
        # new state has lasted long enough (no flicker).
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

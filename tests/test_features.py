"""
Tests for the detection logic in app/features/.
They check posture, head direction, monitor zones, phone use, timers, the state
rules and the three study methods. Many tests run the full FeaturePipeline with
fake face, pose and object results, so no camera is needed.
"""

import dataclasses
import math

import numpy as np
import pytest

from app.config.settings import CameraEdge, MonitorPlacement, Settings, Workspace, arc_layout
from app.features.activity_features import ActivityFeatures, ActivityTracker, BehaviorTimer
from app.features.feature_pipeline import (
    FeaturePipeline,
    FocusState,
    FrameFeatures,
    ReasonCode,
    StateStabilizer,
    classify,
    classify_state,
)
from app.features.gaze_features import (
    Attention,
    HeadPoseSmoother,
    PitchCalibrator,
    compute_screen_zones,
    extract_gaze_features,
    relative_to_camera_line,
)
from app.features.phone_features import NOSE_TIP, PhoneTracker, is_looking_at_phone
from app.features.pose_features import PoseFeatures, Posture, extract_pose_features
from app.features.state_timers import StateTimers, format_duration
from app.vision.face_detection import FaceResult
from app.vision.head_pose import HeadPose
from app.vision.object_detection import Detection, ObjectResult
from app.vision.pose_detection import PoseLandmark, PoseResult

CFG = Settings()
FRAME = (640, 480)


def _features_cfg(**changes):
    return dataclasses.replace(CFG.features, **changes)


def _head_transform(yaw=0.0, pitch=0.0):
    """MediaPipe-style facial transform for a head pose in HeadPose convention."""
    a, b = math.radians(yaw), math.radians(-pitch)  # MediaPipe's X rotation is inverted
    ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    transform = np.eye(4)
    transform[:3, :3] = ry @ rx
    return transform


def _face(yaw=0.0, pitch=0.0, nose=(0.5, 0.4)):
    landmarks = np.full((478, 3), 0.5, np.float32)
    landmarks[NOSE_TIP, :2] = nose
    return FaceResult(True, landmarks=landmarks, transform=_head_transform(yaw, pitch))


def _phone_at(box):
    return ObjectResult(phone_detected=True, person_detected=True,
                        detections=[Detection("cell phone", 0.8, box)])


def _phone_below():
    # Phone in the lower part of the frame, below the face.
    return _phone_at((290, 400, 350, 470))


def _pose(nose, left_shoulder, right_shoulder, visibility=0.99):
    lm = np.zeros((33, 4), dtype=np.float32)
    lm[:, 3] = visibility
    lm[PoseLandmark.NOSE, :2] = nose
    lm[PoseLandmark.LEFT_SHOULDER, :2] = left_shoulder
    lm[PoseLandmark.RIGHT_SHOULDER, :2] = right_shoulder
    return PoseResult(detected=True, landmarks=lm)


# --- posture -----------------------------------------------------------------

def test_upright_posture():
    features = extract_pose_features(_pose((0.5, 0.35), (0.65, 0.65), (0.35, 0.65)), FRAME, CFG.features, CFG.detection)
    assert features.posture is Posture.UPRIGHT
    assert features.shoulder_tilt == pytest.approx(0.0)


def test_slouching_when_head_drops_towards_shoulders():
    features = extract_pose_features(_pose((0.5, 0.60), (0.65, 0.65), (0.35, 0.65)), FRAME, CFG.features, CFG.detection)
    assert features.posture is Posture.SLOUCHING


def test_leaning_when_shoulders_tilt():
    features = extract_pose_features(_pose((0.5, 0.30), (0.65, 0.75), (0.35, 0.60)), FRAME, CFG.features, CFG.detection)
    assert features.posture is Posture.LEANING


def test_unknown_posture_without_reliable_landmarks():
    hidden = _pose((0.5, 0.35), (0.65, 0.65), (0.35, 0.65), visibility=0.1)
    assert extract_pose_features(hidden, FRAME, CFG.features, CFG.detection).posture is Posture.UNKNOWN
    assert extract_pose_features(PoseResult(False), FRAME, CFG.features, CFG.detection).posture is Posture.UNKNOWN


# --- gaze --------------------------------------------------------------------

def test_gaze_facing_screen():
    assert extract_gaze_features(HeadPose(5, -3, 1), CFG.features).facing_screen is True


def test_gaze_looking_sideways_and_down():
    assert extract_gaze_features(HeadPose(40, 0, 0), CFG.features).looking_sideways
    gaze = extract_gaze_features(HeadPose(0, -35, 0), CFG.features)
    assert gaze.looking_down and gaze.facing_screen is False


def test_gaze_without_head_pose_is_unknown():
    assert extract_gaze_features(None, CFG.features).facing_screen is None


# --- activity ----------------------------------------------------------------

def _tracker():
    return ActivityTracker(CFG.features, CFG.detection, CFG.state.still_motion_threshold)


def test_look_away_timer_accumulates_and_resets():
    tracker = _tracker()
    tracker.update(0.0, True, False, False, None)
    assert tracker.update(2.5, True, False, False, None).seconds_looking_away == pytest.approx(2.5)
    assert tracker.update(3.0, True, True, False, None).seconds_looking_away == 0.0


def test_missing_face_with_person_present_counts_as_looking_away():
    tracker = _tracker()
    tracker.update(0.0, True, None, False, None)
    assert tracker.update(1.0, True, None, False, None).seconds_looking_away == pytest.approx(1.0)


def test_time_since_person_seen():
    tracker = _tracker()
    tracker.update(0.0, True, True, False, None)
    assert tracker.update(4.0, False, None, False, None).seconds_since_person_seen == pytest.approx(4.0)


def test_phone_gaze_has_its_own_timer_and_pauses_look_away():
    tracker = _tracker()
    tracker.update(0.0, True, False, True, None)
    result = tracker.update(2.0, True, False, True, None)
    assert result.seconds_looking_at_phone == pytest.approx(2.0)
    assert result.seconds_looking_away == 0.0


def test_motion_and_stillness():
    tracker = _tracker()
    points = np.full((13, 2), 0.5, dtype=np.float32)
    for i in range(10):
        result = tracker.update(i * 0.1, True, True, False, points)
    assert result.motion_level == pytest.approx(0.0)
    assert result.seconds_still == pytest.approx(0.8)

    moved = tracker.update(1.0, True, True, False, points + 0.05)
    assert moved.motion_level > CFG.state.still_motion_threshold
    assert moved.seconds_still == 0.0


# --- state rules -------------------------------------------------------------

def _features(**activity):
    return FrameFeatures(timestamp=0.0, activity=ActivityFeatures(**activity))


@pytest.mark.parametrize(
    "activity,expected",
    [
        ({}, FocusState.FOCUSED),
        ({"seconds_since_person_seen": 5.0}, FocusState.AWAY),
        ({"seconds_looking_at_phone": 1.5}, FocusState.DISTRACTED),
        ({"seconds_looking_at_phone": 0.5}, FocusState.FOCUSED),
        ({"seconds_looking_away": 1.5}, FocusState.DISTRACTED),
        ({"seconds_looking_away": 0.5}, FocusState.FOCUSED),
        ({"seconds_still": 35.0}, FocusState.AWAY),
        ({"seconds_still": 20.0}, FocusState.FOCUSED),
        ({"seconds_still": 75.0, "seconds_looking_at_phone": 10.0}, FocusState.AWAY),
        ({"seconds_since_person_seen": 5.0, "seconds_looking_at_phone": 10.0}, FocusState.AWAY),
    ],
)
def test_classify_state(activity, expected):
    assert classify_state(_features(**activity), CFG.state) is expected


def test_stabilizer_ignores_brief_flicker():
    stabilizer = StateStabilizer(min_duration_s=1.0)
    assert stabilizer.update(FocusState.FOCUSED, 0.0) is FocusState.FOCUSED
    assert stabilizer.update(FocusState.DISTRACTED, 0.2) is FocusState.FOCUSED
    assert stabilizer.update(FocusState.FOCUSED, 0.4) is FocusState.FOCUSED
    assert stabilizer.update(FocusState.DISTRACTED, 0.6) is FocusState.FOCUSED
    assert stabilizer.update(FocusState.DISTRACTED, 1.7) is FocusState.DISTRACTED


def test_pipeline_reports_away_when_nobody_is_visible():
    pipeline = FeaturePipeline(CFG)
    nothing = (FaceResult(False), PoseResult(False), ObjectResult())
    pipeline.process(0.0, FRAME, *nothing)
    analysis = None
    for t in np.arange(0.5, 6.0, 0.5):
        analysis = pipeline.process(float(t), FRAME, *nothing)
    assert analysis.features.person_detected is False
    assert analysis.state is FocusState.AWAY


def test_visible_phone_while_looking_at_monitor_stays_focused():
    pipeline = FeaturePipeline(CFG)
    analysis = None
    for t in np.arange(0.0, 8.0, 0.1):
        analysis = pipeline.process(float(t), FRAME, _face(), PoseResult(False), _phone_below())
    assert analysis.features.phone_detected and analysis.features.face_detected
    assert not analysis.features.phone.looking_at_phone
    assert analysis.state is FocusState.FOCUSED


def test_nobody_seen_yet_is_away_immediately():
    pipeline = FeaturePipeline(CFG)
    analysis = pipeline.process(0.0, FRAME, FaceResult(False), PoseResult(False), ObjectResult())
    assert analysis.state is FocusState.AWAY


# --- 3D workspace / monitor zones -------------------------------------------------

def _ws_cfg(**workspace):
    return _features_cfg(workspace=Workspace(**workspace))


def test_single_centered_monitor_zone_is_symmetric():
    zones = compute_screen_zones(CFG.features)
    (zone,) = zones.monitors
    assert zone.core_yaw[0] == pytest.approx(-zone.core_yaw[1])
    # Camera on top: the whole screen lies below the camera.
    assert zone.core_pitch[1] < 0


def test_arc_layout_is_a_semicircle_facing_the_user():
    for count in (1, 2, 3):
        monitors = arc_layout(count, distance=0.7)
        assert len(monitors) == count
        for m in monitors:
            assert math.hypot(m.x, m.z) == pytest.approx(0.7, abs=1e-3)
            # Facing the user: the rotation equals the bearing from the user.
            assert m.angle == pytest.approx(math.degrees(math.atan2(m.x, m.z)), abs=0.05)
    left, middle, right = arc_layout(3)
    assert left.x < middle.x == 0 < right.x


def test_three_monitors_each_get_their_own_zone():
    cfg = _ws_cfg(monitors=arc_layout(3), camera_monitor=1)
    zones = compute_screen_zones(cfg)
    m1, m2, m3 = zones.monitors
    # Monitor 1 is on the user's left, i.e. positive yaw (HeadPose convention).
    assert m1.center[0] > m2.center[0] > m3.center[0]
    for zone in zones.monitors:
        yaw, pitch = zone.center
        assert extract_gaze_features(HeadPose(yaw, pitch, 0), cfg).monitor == zone.index


@pytest.mark.parametrize("target,expected", [(0, 0), (1, 1), (2, 2)])
def test_looking_at_any_monitor_is_on_screen(target, expected):
    cfg = _ws_cfg(monitors=arc_layout(3), camera_monitor=1)
    yaw, pitch = compute_screen_zones(cfg).monitors[target].center
    gaze = extract_gaze_features(HeadPose(yaw, pitch, 0), cfg)
    assert gaze.facing_screen is True
    assert gaze.direction is Attention.ON_SCREEN
    assert gaze.monitor == expected


def test_looking_beyond_the_outer_monitor_is_off_screen():
    cfg = _ws_cfg(monitors=arc_layout(2), camera_monitor=0)
    zones = compute_screen_zones(cfg)
    beyond_right = zones.yaw_min - 10
    gaze = extract_gaze_features(HeadPose(beyond_right, -7, 0), cfg)
    assert gaze.facing_screen is False and gaze.direction is Attention.RIGHT


def test_gap_between_monitors_is_not_a_monitor():
    # Two monitors far to each side, nothing straight ahead.
    monitors = (MonitorPlacement(-0.6, 0.35, angle=-60), MonitorPlacement(0.6, 0.35, angle=60))
    cfg = dataclasses.replace(_ws_cfg(monitors=monitors), attention_margin=5.0)
    zones = compute_screen_zones(cfg)
    middle = (zones.monitors[0].core_yaw[0] + zones.monitors[1].core_yaw[1]) / 2
    gaze = extract_gaze_features(HeadPose(middle, -7, 0), cfg)
    assert gaze.monitor is None and gaze.direction is Attention.BETWEEN


def test_camera_on_another_monitor_shifts_the_zones():
    on_left = compute_screen_zones(_ws_cfg(monitors=arc_layout(2), camera_monitor=0))
    on_right = compute_screen_zones(_ws_cfg(monitors=arc_layout(2), camera_monitor=1))
    # The monitor carrying the camera is always around yaw 0.
    assert on_left.monitors[0].center[0] == pytest.approx(0, abs=1)
    assert on_right.monitors[1].center[0] == pytest.approx(0, abs=1)
    # The other monitor is to the camera's right (negative yaw) or left (positive yaw).
    assert on_left.monitors[1].center[0] < -10
    assert on_right.monitors[0].center[0] > 10


def test_camera_below_monitor_expects_looking_up():
    zones = compute_screen_zones(_ws_cfg(camera_edge=CameraEdge.BOTTOM))
    assert zones.monitors[0].core_pitch[0] > 0
    cfg = _ws_cfg(camera_edge=CameraEdge.BOTTOM)
    assert extract_gaze_features(HeadPose(0, 8, 0), cfg).facing_screen is True


def test_moving_closer_widens_the_monitor_zone():
    far = compute_screen_zones(_ws_cfg(monitors=arc_layout(1, distance=1.0))).monitors[0]
    near = compute_screen_zones(_ws_cfg(monitors=arc_layout(1, distance=0.4))).monitors[0]
    assert near.core_yaw[1] - near.core_yaw[0] > far.core_yaw[1] - far.core_yaw[0]


def test_sitting_off_center_shifts_the_zone():
    # The user sits 30 cm to the right of a single monitor: it is now to their left.
    cfg = _ws_cfg(person_x=0.3, monitors=(MonitorPlacement(0.0, 0.65),), camera_edge=CameraEdge.LEFT)
    centered = _ws_cfg(monitors=(MonitorPlacement(0.0, 0.65),), camera_edge=CameraEdge.LEFT)
    assert compute_screen_zones(cfg).monitors[0].center[0] > compute_screen_zones(centered).monitors[0].center[0]


def test_monitor_angle_changes_its_apparent_width():
    facing = compute_screen_zones(_ws_cfg(monitors=(MonitorPlacement(0.0, 0.65, angle=0),))).monitors[0]
    turned = compute_screen_zones(_ws_cfg(monitors=(MonitorPlacement(0.0, 0.65, angle=60),))).monitors[0]
    assert turned.core_yaw[1] - turned.core_yaw[0] < facing.core_yaw[1] - facing.core_yaw[0]


def test_strong_head_roll_is_off_screen():
    assert extract_gaze_features(HeadPose(0, -5, 50), CFG.features).direction is Attention.HEAD_TILTED


def test_far_side_monitor_keeps_focus_when_face_leaves_the_camera():
    # Camera on the left monitor; the right monitor needs a large head turn.
    pipeline = FeaturePipeline(dataclasses.replace(
        CFG, features=_ws_cfg(monitors=arc_layout(3), camera_monitor=0)))
    far_zone = pipeline.screen_zones.monitors[2]
    assert max(abs(v) for v in far_zone.core_yaw) >= CFG.features.face_tracking_limit
    yaw, pitch = far_zone.center
    person = ObjectResult(person_detected=True)
    _run(pipeline, 2.0, _face(yaw=yaw, pitch=pitch), person)
    no_face = FaceResult(False)
    analysis = _run(pipeline, 12.0, no_face, person, start=2.0)
    assert analysis.features.monitor == 2
    assert analysis.state is FocusState.FOCUSED


def test_lost_face_on_a_near_monitor_still_counts_as_looking_away():
    pipeline = FeaturePipeline(CFG)
    person = ObjectResult(person_detected=True)
    _run(pipeline, 2.0, _face(), person)
    analysis = _run(pipeline, 9.0, FaceResult(False), person, start=2.0)
    assert analysis.state is FocusState.DISTRACTED


def test_workspace_change_applies_live():
    pipeline = FeaturePipeline(CFG)
    person = ObjectResult(person_detected=True)
    # Head turned 40 deg: off-screen with one monitor ...
    assert _run(pipeline, 8.0, _face(yaw=-40), person).state is FocusState.DISTRACTED
    # ... but on monitor 2 once a second monitor is configured on that side.
    pipeline.reconfigure(dataclasses.replace(
        CFG, features=_ws_cfg(monitors=arc_layout(2), camera_monitor=0)))
    analysis = _run(pipeline, 11.0, _face(yaw=-40), person, start=8.0)
    assert analysis.features.monitor == 1
    assert analysis.state is FocusState.FOCUSED


# --- temporal smoothing ------------------------------------------------------------

def test_behavior_timer_survives_short_interruptions():
    timer = BehaviorTimer(grace_s=1.0)
    timer.update(0.0, True)
    timer.update(2.0, True)
    assert timer.update(2.5, False) == 0.0  # glance back at the screen
    timer.update(2.8, True)
    assert timer.update(3.8, True) == pytest.approx(3.0)  # 2 s before + 1 s after


def test_behavior_timer_resets_after_long_interruption():
    timer = BehaviorTimer(grace_s=1.0)
    timer.update(0.0, True)
    timer.update(2.0, True)
    timer.update(2.5, False)
    timer.update(4.0, True)
    assert timer.update(5.0, True) == pytest.approx(1.0)


def test_behavior_timer_ignores_isolated_glitch_frames():
    # One bad frame every 0.5 s while working must never add up to a distraction.
    timer = BehaviorTimer(grace_s=1.0)
    longest = 0.0
    for i in range(200):
        longest = max(longest, timer.update(i * 0.1, i % 5 == 0))
    assert longest < 0.5


def test_head_pose_smoother_damps_jitter():
    smoother = HeadPoseSmoother(time_constant_s=0.25)
    smoother.update(0.0, HeadPose(0, 0, 0))
    spike = smoother.update(0.033, HeadPose(40, 0, 0))
    assert 0 < spike.yaw < 10
    settled = spike
    for i in range(2, 60):
        settled = smoother.update(i * 0.033, HeadPose(40, 0, 0))
    assert settled.yaw == pytest.approx(40, abs=0.5)


def test_head_pose_smoother_restarts_after_face_loss():
    smoother = HeadPoseSmoother(time_constant_s=0.25)
    smoother.update(0.0, HeadPose(0, 0, 0))
    assert smoother.update(0.5, None) is None
    assert smoother.update(2.0, None) is None
    assert smoother.update(2.1, HeadPose(30, 0, 0)).yaw == pytest.approx(30)


def test_phone_tracker_holds_briefly():
    tracker = PhoneTracker(hold_s=1.0)
    assert tracker.update(0.0, _phone_below()) is not None
    assert tracker.update(0.5, ObjectResult()) is not None
    assert tracker.update(1.6, ObjectResult()) is None


# --- looking at the phone ----------------------------------------------------------

_ZONE = compute_screen_zones(CFG.features)
_ANCHOR = (320.0, 190.0)
_PHONE_BELOW = (320.0, 430.0)


def _looking_at_phone(phone, head, pose=None):
    anchor = _ANCHOR if head is not None else None
    return is_looking_at_phone(phone, head, anchor, _ZONE, pose or PoseFeatures(), CFG.features)


def test_phone_ignored_while_looking_at_monitor():
    assert not _looking_at_phone(_PHONE_BELOW, HeadPose(yaw=5, pitch=-8, roll=0))


def test_looking_down_at_phone_below():
    assert _looking_at_phone(_PHONE_BELOW, HeadPose(yaw=0, pitch=-40, roll=0))


def test_looking_away_from_phone_is_not_phone_use():
    # Phone below, but the head is turned up and to the side (e.g. talking to someone).
    assert not _looking_at_phone(_PHONE_BELOW, HeadPose(yaw=50, pitch=10, roll=0))


def test_phone_to_the_side_matches_head_turn():
    # Raw frame: the user's left is the image right; yaw > 0 turns towards the user's left.
    phone_on_users_left = (560.0, 300.0)
    assert _looking_at_phone(phone_on_users_left, HeadPose(yaw=45, pitch=-20, roll=0))
    assert not _looking_at_phone(phone_on_users_left, HeadPose(yaw=-45, pitch=-20, roll=0))


def test_no_phone_means_no_phone_use():
    assert not _looking_at_phone(None, HeadPose(0, -40, 0))


def test_phone_in_lap_without_face_uses_posture():
    assert _looking_at_phone(_PHONE_BELOW, None, PoseFeatures(posture=Posture.SLOUCHING))
    assert not _looking_at_phone(_PHONE_BELOW, None, PoseFeatures(posture=Posture.UPRIGHT))


# --- full pipeline scenarios --------------------------------------------------------

def _run(pipeline, until, face, objects, start=0.0, step=0.1):
    analysis = None
    for t in np.arange(start, until, step):
        analysis = pipeline.process(float(t), FRAME, face, PoseResult(False), objects)
    return analysis


def test_phone_distraction_requires_configured_duration():
    pipeline = FeaturePipeline(CFG)
    looking_down = _face(pitch=-40)
    # Phone threshold 1 s + 1 s stabiliser: still focused after 1.5 s.
    early = _run(pipeline, 1.5, looking_down, _phone_below())
    assert early.features.phone.looking_at_phone
    assert early.state is FocusState.FOCUSED
    late = _run(pipeline, 2.6, looking_down, _phone_below(), start=1.5)
    assert late.state is FocusState.DISTRACTED
    assert late.reason.code is ReasonCode.PHONE


def test_looking_down_without_phone_uses_general_distraction_time():
    pipeline = FeaturePipeline(CFG)
    looking_down = _face(pitch=-40)
    person = ObjectResult(person_detected=True)
    assert _run(pipeline, 1.5, looking_down, person).state is FocusState.FOCUSED
    late = _run(pipeline, 3.0, looking_down, person, start=1.5)
    assert late.state is FocusState.DISTRACTED
    assert late.reason.code is ReasonCode.LOOKING_AWAY
    assert late.reason.attention is Attention.DOWN


def test_returning_to_screen_recovers_focus():
    pipeline = FeaturePipeline(CFG)
    _run(pipeline, 7.0, _face(yaw=60), ObjectResult(person_detected=True))
    recovered = _run(pipeline, 10.0, _face(), ObjectResult(person_detected=True), start=7.0)
    assert recovered.state is FocusState.FOCUSED


def test_single_off_screen_frame_does_not_distract():
    pipeline = FeaturePipeline(CFG)
    person = ObjectResult(person_detected=True)
    _run(pipeline, 3.0, _face(), person)
    pipeline.process(3.0, FRAME, _face(yaw=70), PoseResult(False), person)
    analysis = _run(pipeline, 4.0, _face(), person, start=3.1)
    assert analysis.state is FocusState.FOCUSED


def test_reconfigure_applies_new_duration():
    pipeline = FeaturePipeline(CFG)
    pipeline.reconfigure(dataclasses.replace(CFG, state=dataclasses.replace(CFG.state, look_away_after_s=2.0)))
    analysis = _run(pipeline, 3.5, _face(yaw=60), ObjectResult(person_detected=True))
    assert analysis.state is FocusState.DISTRACTED


# --- state timers ---------------------------------------------------------------

def test_state_timers_accumulate_per_state():
    timers = StateTimers(now=0.0)
    timers.update(0.0, FocusState.FOCUSED)
    timers.update(10.0, FocusState.DISTRACTED)
    timers.update(13.0, None)  # camera lost: time is not credited to any state
    timers.update(20.0, FocusState.BREAK)
    totals = timers.totals(now=25.0)
    assert totals[FocusState.FOCUSED] == pytest.approx(10.0)
    assert totals[FocusState.DISTRACTED] == pytest.approx(3.0)
    assert totals[FocusState.BREAK] == pytest.approx(5.0)
    assert totals[FocusState.AWAY] == 0.0


def test_state_timers_reset():
    timers = StateTimers(now=0.0)
    timers.update(0.0, FocusState.FOCUSED)
    timers.reset(now=50.0)
    assert sum(timers.totals(now=60.0).values()) == 0.0
    assert timers.current is None


@pytest.mark.parametrize("seconds,text", [(0, "00:00"), (59.9, "00:59"), (2551, "42:31"), (3725, "1:02:05")])
def test_format_duration(seconds, text):
    assert format_duration(seconds) == text


# --- AWAY without movement ---------------------------------------------------------

def test_no_movement_reason_and_threshold():
    state, reason = classify(_features(seconds_still=61.0), CFG.state)
    assert state is FocusState.AWAY and reason.code is ReasonCode.NO_MOVEMENT
    short = dataclasses.replace(CFG.state, still_away_after_s=20.0)
    assert classify_state(_features(seconds_still=25.0), short) is FocusState.AWAY


def test_motionless_person_becomes_away_after_configured_time():
    config = dataclasses.replace(CFG, state=dataclasses.replace(CFG.state, still_away_after_s=10.0))
    pipeline = FeaturePipeline(config)
    frozen = _face()  # identical landmarks every frame: no movement at all
    person = ObjectResult(person_detected=True)
    assert _run(pipeline, 8.0, frozen, person).state is FocusState.FOCUSED
    analysis = _run(pipeline, 13.0, frozen, person, start=8.0)
    assert analysis.state is FocusState.AWAY
    assert analysis.reason.code is ReasonCode.NO_MOVEMENT


def test_there_is_no_idle_state():
    assert [s.value for s in FocusState] == ["FOCUSED", "DISTRACTED", "AWAY", "BREAK"]


# --- phone + gaze direction against the 3D layout --------------------------------------

def test_phone_ignored_while_looking_at_any_of_three_monitors():
    cfg = _ws_cfg(monitors=arc_layout(3), camera_monitor=1)
    zones = compute_screen_zones(cfg)
    for zone in zones.monitors:
        yaw, pitch = zone.center
        head = HeadPose(yaw, pitch, 0)
        assert not is_looking_at_phone(_PHONE_BELOW, head, _ANCHOR, zones, PoseFeatures(), cfg)


def test_pipeline_phone_on_users_left_while_head_turned_there():
    pipeline = FeaturePipeline(CFG)
    # Raw frame: the user's left is the image right.
    phone_left = _phone_at((560, 300, 620, 380))
    analysis = _run(pipeline, 5.0, _face(yaw=45, pitch=-25), phone_left)
    assert analysis.features.phone.looking_at_phone
    assert analysis.state is FocusState.DISTRACTED
    # Same head turn, phone on the other side: not phone use (just looking away, still short).
    pipeline = FeaturePipeline(CFG)
    phone_right = _phone_at((20, 300, 80, 380))
    analysis = _run(pipeline, 1.5, _face(yaw=45, pitch=-25), phone_right)
    assert not analysis.features.phone.looking_at_phone
    assert analysis.state is FocusState.FOCUSED


# --- perspective correction and pitch calibration ------------------------------------

def test_centered_face_needs_no_perspective_correction():
    pose = relative_to_camera_line(HeadPose(10, -5, 2), (320, 240), FRAME, 63.0)
    assert (pose.yaw, pose.pitch, pose.roll) == pytest.approx((10, -5, 2))


def test_off_center_face_looking_into_the_lens_reads_as_zero():
    # Face low and on the image right: MediaPipe reports it turned right/up to see the lens.
    focal = 240 / math.tan(math.radians(31.5))
    anchor = (320 + 100, 240 + 80)
    looking_at_lens = HeadPose(-math.degrees(math.atan(100 / focal)), math.degrees(math.atan(80 / focal)), 0)
    pose = relative_to_camera_line(looking_at_lens, anchor, FRAME, 63.0)
    assert pose.yaw == pytest.approx(0, abs=1e-6)
    assert pose.pitch == pytest.approx(0, abs=1e-6)


def _calibrate(calibrator, pitches, yaw=0.0, start=0.0):
    zones = compute_screen_zones(CFG.features)
    offset = 0.0
    for i, pitch in enumerate(pitches):
        offset = calibrator.update(start + i * 0.5, HeadPose(yaw, pitch, 0), zones)
    return offset


def test_calibrator_learns_a_constant_pitch_bias():
    expected = compute_screen_zones(CFG.features).monitors[0].center[1]
    calibrator = PitchCalibrator(CFG.features)
    # Before enough samples, nothing is applied.
    assert _calibrate(calibrator, [expected + 10] * 5) == 0.0
    assert _calibrate(calibrator, [expected + 10] * 40, start=3.0) == pytest.approx(10, abs=0.01)


def test_calibrator_ignores_phone_glances_and_is_capped():
    expected = compute_screen_zones(CFG.features).monitors[0].center[1]
    calibrator = PitchCalibrator(CFG.features)
    # 60 % of the time looking far down at a phone must not shift the calibration.
    pitches = [expected + 4 if i % 5 < 2 else expected - 35 for i in range(200)]
    assert _calibrate(calibrator, pitches) == pytest.approx(4, abs=0.01)
    capped = PitchCalibrator(CFG.features)
    assert _calibrate(capped, [expected + 19] * 40) == CFG.features.calibration_max_offset


def test_calibrator_ignores_samples_away_from_the_monitors():
    calibrator = PitchCalibrator(CFG.features)
    assert _calibrate(calibrator, [5.0] * 60, yaw=80) == 0.0


def test_biased_user_reading_the_screen_is_not_flagged_as_looking_up():
    # This user's head pitch reads ~16 deg higher than the model expects while working.
    expected = compute_screen_zones(CFG.features).monitors[0].center[1]
    strict = dataclasses.replace(CFG, features=dataclasses.replace(CFG.features, attention_margin=6.0))
    pipeline = FeaturePipeline(strict)
    person = ObjectResult(person_detected=True)
    biased = _face(pitch=expected + 16, nose=(0.5, 0.5))
    first = pipeline.process(0.0, FRAME, biased, PoseResult(False), person)
    assert first.features.attention is Attention.UP  # before calibration
    analysis = _run(pipeline, 30.0, biased, person, start=0.1)
    assert analysis.features.pitch_calibration == pytest.approx(15.0, abs=1.5)
    assert analysis.features.attention is Attention.ON_SCREEN
    assert analysis.state is FocusState.FOCUSED


def test_face_lost_at_small_yaw_does_not_credit_far_monitor():
    pipeline = FeaturePipeline(dataclasses.replace(
        CFG, features=_ws_cfg(monitors=arc_layout(3), camera_monitor=0)))
    person = ObjectResult(person_detected=True)
    # Like the live test: on monitor 2 at only -20 deg when the face disappears.
    pitch = pipeline.screen_zones.monitors[1].center[1]
    assert pipeline.screen_zones.monitors[1].core_yaw[0] < -20 < pipeline.screen_zones.monitors[1].core_yaw[1]
    _run(pipeline, 2.0, _face(yaw=-20, pitch=pitch, nose=(0.5, 0.5)), person)
    analysis = _run(pipeline, 9.0, FaceResult(False), person, start=2.0)
    assert analysis.features.monitor is None
    assert analysis.state is FocusState.DISTRACTED


# --- phone held up near the face (live-test regression) --------------------------------

def _real_face(yaw=0.0, pitch=0.0, center=(0.5, 0.45), size=0.35):
    """Face whose landmarks span `size` of the frame height, like a user ~60 cm away."""
    rng = np.random.default_rng(0)
    landmarks = np.zeros((478, 3), np.float32)
    landmarks[:, 0] = center[0] + rng.uniform(-0.5, 0.5, 478) * size * FRAME[1] / FRAME[0] * 0.8
    landmarks[:, 1] = center[1] + rng.uniform(-0.5, 0.5, 478) * size
    landmarks[NOSE_TIP, :2] = center
    return FaceResult(True, landmarks=landmarks, transform=_head_transform(yaw, pitch))


def _two_monitor_config():
    # The tester's desk: two monitors, camera on top of the left one.
    return dataclasses.replace(CFG, features=_ws_cfg(monitors=arc_layout(2), camera_monitor=0))


def test_phone_held_next_to_face_in_line_with_a_monitor_is_phone_use():
    # Live test: head turned ~30 deg to the right towards monitor 2, phone held up
    # beside the face on that side (raw image: the user's right is the image left).
    pipeline = FeaturePipeline(_two_monitor_config())
    face = _real_face(yaw=-30, pitch=-3)
    yaw, pitch = pipeline.screen_zones.monitors[1].center
    phone_in_hand = _phone_at((80, 120, 200, 330))  # large, right next to the face
    first = pipeline.process(0.0, FRAME, face, PoseResult(False), phone_in_hand)
    assert first.features.phone.near_face
    assert first.features.monitor == 1  # the head *is* in monitor 2's direction ...
    analysis = _run(pipeline, 5.0, face, phone_in_hand, start=0.1)
    assert analysis.features.phone.looking_at_phone  # ... but the phone is in the way
    assert analysis.state is FocusState.DISTRACTED
    assert analysis.reason.code is ReasonCode.PHONE


def test_small_phone_far_away_in_line_with_a_monitor_is_ignored():
    pipeline = FeaturePipeline(_two_monitor_config())
    face = _real_face(yaw=-30, pitch=-3)
    phone_on_stand = _phone_at((90, 300, 120, 330))  # small: far from the user
    analysis = _run(pipeline, 8.0, face, phone_on_stand)
    assert not analysis.features.phone.near_face
    assert not analysis.features.phone.looking_at_phone
    assert analysis.state is FocusState.FOCUSED


def test_phone_in_hand_while_looking_at_another_monitor_is_ignored():
    pipeline = FeaturePipeline(_two_monitor_config())
    face = _real_face(yaw=0, pitch=0)  # looking at monitor 1 (camera monitor)
    phone_in_hand = _phone_at((80, 300, 200, 470))
    analysis = _run(pipeline, 8.0, face, phone_in_hand)
    assert analysis.features.phone.near_face
    assert not analysis.features.phone.looking_at_phone
    assert analysis.state is FocusState.FOCUSED


def test_phone_gaze_survives_short_yolo_dropouts():
    # Live test: the hand covering the phone made YOLO miss it for ~2 s.
    pipeline = FeaturePipeline(_two_monitor_config())
    face = _real_face(yaw=-30, pitch=-3)
    phone = _phone_at((80, 120, 200, 330))
    analysis = None
    for i, t in enumerate(np.arange(0.0, 6.0, 0.1)):
        missed = 2.0 <= t < 4.0  # no detection for 2 s
        analysis = pipeline.process(float(t), FRAME, face, PoseResult(False),
                                    ObjectResult(person_detected=True) if missed else phone)
    assert analysis.features.activity.seconds_looking_at_phone > 5.0
    assert analysis.state is FocusState.DISTRACTED


# --- nose / head direction ------------------------------------------------------------

from app.features.phone_features import nose_forward, nose_ray_hits_box  # noqa: E402


def test_nose_forward_follows_head_pose():
    # Raw image: yaw > 0 (user's left) points to +x, pitch > 0 (up) to -y.
    assert nose_forward(HeadPose(0, 0, 0)) == pytest.approx((0.0, 0.0))
    x, y = nose_forward(HeadPose(30, 0, 0))
    assert x == pytest.approx(0.5) and y == pytest.approx(0.0)
    x, y = nose_forward(HeadPose(0, -30, 0))
    assert x == pytest.approx(0.0) and y == pytest.approx(0.5)
    x, y = nose_forward(HeadPose(-20, 20, 5))
    assert x < 0 and y < 0


def test_nose_ray_hits_a_box_in_its_direction_only():
    nose = (320.0, 200.0)
    box_below = (280, 380, 360, 470)
    assert nose_ray_hits_box(nose, HeadPose(0, -35, 0), box_below, CFG.features)
    assert not nose_ray_hits_box(nose, HeadPose(0, 35, 0), box_below, CFG.features)  # looking up
    assert not nose_ray_hits_box(nose, HeadPose(40, 0, 0), box_below, CFG.features)  # looking sideways
    # Straight into the camera: the nose direction is undefined.
    assert not nose_ray_hits_box(nose, HeadPose(1, -1, 0), box_below, CFG.features)


def test_ray_catches_a_large_close_phone_at_a_wide_angle():
    # Big phone held low and to the side: its centre is ~50 deg off the nose direction,
    # but the nose ray still passes through the box.
    nose = (320.0, 200.0)
    head = HeadPose(10, -30, 0)
    box = (60, 300, 360, 470)
    angle = _phone_angle(nose, head, box)
    assert angle > CFG.features.phone_gaze_max_angle
    assert nose_ray_hits_box(nose, head, box, CFG.features)
    zones = compute_screen_zones(CFG.features)
    center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    assert is_looking_at_phone(center, head, nose, zones, PoseFeatures(), CFG.features, box=box)


def _phone_angle(nose, head, box):
    from app.features.phone_features import phone_gaze_angle
    center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    return phone_gaze_angle(center, head, nose, CFG.features)


def test_scenario_looking_at_monitor_with_phone_visible_stays_focused():
    pipeline = FeaturePipeline(_two_monitor_config())
    phone_on_desk = _phone_at((420, 400, 480, 440))
    for monitor in range(2):
        yaw, pitch = pipeline.screen_zones.monitors[monitor].center
        analysis = _run(pipeline, 10.0 * (monitor + 1), _real_face(yaw=yaw, pitch=pitch, center=(0.5, 0.5)),
                        phone_on_desk, start=10.0 * monitor)
        assert analysis.features.phone.visible
        assert not analysis.features.phone.looking_at_phone
        assert analysis.state is FocusState.FOCUSED


def test_scenario_looking_at_phone_becomes_distracted():
    pipeline = FeaturePipeline(CFG)
    head_down = _real_face(pitch=-45, center=(0.5, 0.35))
    phone_in_lap = _phone_at((270, 380, 370, 470))
    early = _run(pipeline, 2.0, head_down, phone_in_lap)
    assert early.features.phone.looking_at_phone and early.state is FocusState.FOCUSED
    late = _run(pipeline, 5.0, head_down, phone_in_lap, start=2.0)
    assert late.state is FocusState.DISTRACTED and late.reason.code is ReasonCode.PHONE


def test_scenario_brief_face_loss_while_looking_at_phone():
    # MediaPipe often drops the face for a few frames when the head is bowed over a phone.
    pipeline = FeaturePipeline(CFG)
    head_down = _real_face(pitch=-45, center=(0.5, 0.35))
    phone = _phone_at((270, 380, 370, 470))
    before = _run(pipeline, 4.5, head_down, phone)
    assert before.state is FocusState.DISTRACTED
    during = _run(pipeline, 5.3, FaceResult(False), phone, start=4.5)  # 0.8 s without a face
    assert during.features.phone.looking_at_phone
    assert during.state is FocusState.DISTRACTED
    after = _run(pipeline, 6.0, head_down, phone, start=5.3)
    assert after.features.activity.seconds_looking_at_phone > before.features.activity.seconds_looking_at_phone
    assert after.state is FocusState.DISTRACTED


def test_scenario_long_face_loss_does_not_keep_phone_use_forever():
    pipeline = FeaturePipeline(CFG)
    phone = _phone_at((270, 380, 370, 470))
    _run(pipeline, 4.5, _real_face(pitch=-45, center=(0.5, 0.35)), phone)
    analysis = _run(pipeline, 8.0, FaceResult(False), phone, start=4.5)
    assert not analysis.features.phone.looking_at_phone


def test_phone_visible_alone_never_causes_phone_distraction():
    pipeline = FeaturePipeline(CFG)
    phone = _phone_at((420, 380, 500, 470))
    reasons = set()
    for t in np.arange(0.0, 30.0, 0.1):
        analysis = pipeline.process(float(t), FRAME, _real_face(center=(0.5, 0.5)), PoseResult(False), phone)
        reasons.add(analysis.reason.code)
    assert analysis.state is FocusState.FOCUSED
    assert ReasonCode.PHONE not in reasons


# --- study method -----------------------------------------------------------------

from app.config.settings import StudyMethod  # noqa: E402


def _method_cfg(method, **workspace):
    features = dataclasses.replace(CFG.features, study_method=method, workspace=Workspace(**workspace))
    return dataclasses.replace(CFG, features=features)


PERSON = ObjectResult(person_detected=True)


def test_computer_is_the_default_study_method():
    assert CFG.features.study_method is StudyMethod.COMPUTER


@pytest.mark.parametrize("head,attention", [
    (HeadPose(0, -35, 0), Attention.DOWN),
    (HeadPose(5, -3, 0), Attention.ON_SCREEN),
    (HeadPose(60, 0, 0), Attention.LEFT),
])
def test_computer_gaze_is_unchanged(head, attention):
    gaze = extract_gaze_features(head, _method_cfg(StudyMethod.COMPUTER).features)
    assert gaze.direction is attention
    assert not gaze.on_desk


def test_computer_looking_down_still_distracts():
    pipeline = FeaturePipeline(_method_cfg(StudyMethod.COMPUTER))
    analysis = _run(pipeline, 7.0, _face(pitch=-40), PERSON)
    assert analysis.state is FocusState.DISTRACTED
    assert analysis.reason.attention is Attention.DOWN


@pytest.mark.parametrize("head,on_desk,attention", [
    (HeadPose(0, -30, 0), True, Attention.DESK),
    (HeadPose(-20, -45, 0), True, Attention.DESK),
    (HeadPose(0, 0, 0), False, Attention.UP),  # straight ahead: no monitor to look at
    (HeadPose(60, 0, 0), False, Attention.LEFT),
    (HeadPose(-60, -30, 0), True, Attention.DESK),  # any downward direction
    (HeadPose(0, -3, 0), True, Attention.DESK),  # no minimum downward angle
    (HeadPose(0, -30, 50), False, Attention.HEAD_TILTED),
])
def test_tablet_gaze_counts_the_desk_only(head, on_desk, attention):
    gaze = extract_gaze_features(head, _method_cfg(StudyMethod.TABLET).features)
    assert gaze.on_desk is on_desk
    assert gaze.direction is attention
    assert gaze.monitor is None and not gaze.facing_screen


def test_tablet_looking_down_stays_focused():
    pipeline = FeaturePipeline(_method_cfg(StudyMethod.TABLET))
    analysis = _run(pipeline, 15.0, _face(pitch=-40), PERSON)
    assert analysis.state is FocusState.FOCUSED
    assert analysis.reason.code is ReasonCode.ON_DESK
    assert analysis.features.attention is Attention.DESK
    assert analysis.features.activity.seconds_looking_away == 0.0


@pytest.mark.parametrize("face", [_face(yaw=60), _face()])
def test_tablet_looking_up_or_away_distracts(face):
    pipeline = FeaturePipeline(_method_cfg(StudyMethod.TABLET))
    analysis = _run(pipeline, 7.0, face, PERSON)
    assert analysis.state is FocusState.DISTRACTED
    assert analysis.reason.code is ReasonCode.LOOKING_AWAY


def test_tablet_face_lost_while_bent_over_the_desk_stays_focused():
    pipeline = FeaturePipeline(_method_cfg(StudyMethod.TABLET))
    _run(pipeline, 3.0, _face(pitch=-40), PERSON)
    analysis = _run(pipeline, 12.0, FaceResult(False), PERSON, start=3.0)
    assert analysis.state is FocusState.FOCUSED
    assert analysis.features.attention is Attention.DESK


def test_face_lost_after_looking_away_is_not_credited_to_the_desk():
    pipeline = FeaturePipeline(_method_cfg(StudyMethod.TABLET))
    _run(pipeline, 3.0, _face(yaw=60), PERSON)
    analysis = _run(pipeline, 10.0, FaceResult(False), PERSON, start=3.0)
    assert analysis.state is FocusState.DISTRACTED


def test_tablet_phone_use_is_still_a_distraction():
    pipeline = FeaturePipeline(_method_cfg(StudyMethod.TABLET))
    analysis = _run(pipeline, 5.0, _face(pitch=-40), _phone_below())
    assert analysis.state is FocusState.DISTRACTED
    assert analysis.reason.code is ReasonCode.PHONE


def test_tablet_does_not_learn_a_pitch_bias_from_the_desk():
    pipeline = FeaturePipeline(_method_cfg(StudyMethod.TABLET))
    analysis = _run(pipeline, 20.0, _face(pitch=-20), PERSON)
    assert analysis.features.pitch_calibration == 0.0


@pytest.mark.parametrize("head,attention,on_desk", [
    (HeadPose(5, -3, 0), Attention.ON_SCREEN, False),
    (HeadPose(0, -35, 0), Attention.DESK, True),
    (HeadPose(60, 0, 0), Attention.LEFT, False),
    (HeadPose(0, 30, 0), Attention.UP, False),
])
def test_mixed_gaze_counts_monitors_and_desk(head, attention, on_desk):
    gaze = extract_gaze_features(head, _method_cfg(StudyMethod.MIXED).features)
    assert gaze.direction is attention
    assert gaze.on_desk is on_desk


@pytest.mark.parametrize("head", [HeadPose(70, -35, 0), HeadPose(-50, -10, 0), HeadPose(30, -60, 0)])
def test_mixed_any_downward_direction_is_the_desk(head):
    gaze = extract_gaze_features(head, _method_cfg(StudyMethod.MIXED).features)
    assert gaze.on_desk and gaze.direction is Attention.DESK


def test_computer_downward_off_the_monitors_is_not_the_desk():
    gaze = extract_gaze_features(HeadPose(70, -35, 0), _method_cfg(StudyMethod.COMPUTER).features)
    assert not gaze.on_desk and gaze.direction is not Attention.DESK


@pytest.mark.parametrize("face,reason", [
    (_face(pitch=-40), ReasonCode.ON_DESK),
    (_face(), ReasonCode.ON_MONITOR),
])
def test_mixed_monitor_and_desk_both_stay_focused(face, reason):
    pipeline = FeaturePipeline(_method_cfg(StudyMethod.MIXED))
    analysis = _run(pipeline, 15.0, face, PERSON)
    assert analysis.state is FocusState.FOCUSED
    assert analysis.reason.code is reason


def test_mixed_switching_between_monitor_and_desk_stays_focused():
    pipeline = FeaturePipeline(_method_cfg(StudyMethod.MIXED, monitors=arc_layout(2)))
    start = 0.0
    for face in (_face(), _face(pitch=-40), _face(yaw=-15), _face(pitch=-40)):
        analysis = _run(pipeline, start + 4.0, face, PERSON, start=start)
        start += 4.0
        assert analysis.state is FocusState.FOCUSED


def test_mixed_looking_sideways_still_distracts():
    pipeline = FeaturePipeline(_method_cfg(StudyMethod.MIXED))
    analysis = _run(pipeline, 7.0, _face(yaw=60), PERSON)
    assert analysis.state is FocusState.DISTRACTED
    assert analysis.reason.code is ReasonCode.LOOKING_AWAY


def test_study_method_change_applies_live():
    pipeline = FeaturePipeline(_method_cfg(StudyMethod.COMPUTER))
    assert _run(pipeline, 7.0, _face(pitch=-40), PERSON).state is FocusState.DISTRACTED
    pipeline.reconfigure(_method_cfg(StudyMethod.TABLET))
    assert _run(pipeline, 10.0, _face(pitch=-40), PERSON, start=7.0).state is FocusState.FOCUSED


# --- tolerance around the monitors ----------------------------------------------------

@pytest.mark.parametrize("method", [StudyMethod.COMPUTER, StudyMethod.MIXED])
def test_monitor_is_kept_for_small_moves_past_its_edge(method):
    cfg = _method_cfg(method).features
    zones = compute_screen_zones(cfg)
    edge = zones.monitors[0].core_yaw[1] + zones.margin
    just_out = HeadPose(edge + 0.5 * cfg.monitor_exit_margin, -5, 0)
    far_out = HeadPose(edge + cfg.monitor_exit_margin + 2, -5, 0)
    # Entering from outside needs the normal zone; leaving needs the extra exit margin.
    assert extract_gaze_features(just_out, cfg, zones).monitor is None
    assert extract_gaze_features(just_out, cfg, zones, previous_monitor=0).monitor == 0
    assert extract_gaze_features(far_out, cfg, zones, previous_monitor=0).monitor is None


def test_tilted_head_is_not_kept_on_the_monitor():
    gaze = extract_gaze_features(HeadPose(0, -5, 50), CFG.features, previous_monitor=0)
    assert gaze.monitor is None and gaze.direction is Attention.HEAD_TILTED


@pytest.mark.parametrize("method", [StudyMethod.COMPUTER, StudyMethod.MIXED])
def test_small_head_drift_past_the_monitor_edge_stays_focused(method):
    pipeline = FeaturePipeline(_method_cfg(method))
    zones = pipeline.screen_zones
    edge = zones.monitors[0].core_yaw[1] + zones.margin
    _run(pipeline, 3.0, _face(yaw=0), PERSON)
    analysis = _run(pipeline, 10.0, _face(yaw=edge + 3), PERSON, start=3.0)
    assert analysis.state is FocusState.FOCUSED
    assert analysis.reason.code is ReasonCode.ON_MONITOR


def test_looking_clearly_away_still_distracts_after_the_short_default():
    pipeline = FeaturePipeline(CFG)
    _run(pipeline, 3.0, _face(), PERSON)
    analysis = _run(pipeline, 6.0, _face(yaw=70), PERSON, start=3.0)
    assert analysis.state is FocusState.DISTRACTED


def test_brief_glance_away_does_not_distract():
    pipeline = FeaturePipeline(CFG)
    _run(pipeline, 3.0, _face(), PERSON)
    _run(pipeline, 3.5, _face(yaw=70), PERSON, start=3.0)
    analysis = _run(pipeline, 6.0, _face(), PERSON, start=3.5)
    assert analysis.state is FocusState.FOCUSED

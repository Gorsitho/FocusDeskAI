import dataclasses
import math

import numpy as np
import pytest

from app.config.settings import CameraPosition, Settings
from app.features.activity_features import ActivityFeatures, ActivityTracker, BehaviorTimer
from app.features.feature_pipeline import (
    FeaturePipeline,
    FocusState,
    FrameFeatures,
    StateStabilizer,
    classify_state,
)
from app.features.gaze_features import (
    Attention,
    HeadPoseSmoother,
    compute_screen_zone,
    extract_gaze_features,
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
        ({"seconds_looking_at_phone": 3.5}, FocusState.DISTRACTED),
        ({"seconds_looking_at_phone": 2.0}, FocusState.FOCUSED),
        ({"seconds_looking_away": 6.0}, FocusState.DISTRACTED),
        ({"seconds_looking_away": 4.0}, FocusState.FOCUSED),
        ({"seconds_still": 75.0}, FocusState.IDLE),
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


# --- screen zone / workspace layout -------------------------------------------

def test_single_centered_monitor_zone_is_symmetric():
    zone = compute_screen_zone(CFG.features)
    assert zone.yaw_min == pytest.approx(-zone.yaw_max)
    assert zone.core_pitch == (-CFG.features.monitor_pitch_span, 0.0)


def test_more_monitors_widen_the_zone():
    zones = [compute_screen_zone(_features_cfg(monitor_count=n)) for n in (1, 2, 3)]
    widths = [z.yaw_max - z.yaw_min for z in zones]
    assert widths[0] < widths[1] < widths[2]


def test_side_camera_shifts_the_zone():
    # Camera on the user's left: working means looking to the camera's right (negative yaw).
    left = compute_screen_zone(_features_cfg(camera_position=CameraPosition.LEFT))
    right = compute_screen_zone(_features_cfg(camera_position=CameraPosition.RIGHT))
    assert left.yaw_center < 0 < right.yaw_center
    assert left.yaw_center == pytest.approx(-right.yaw_center)


def test_camera_position_changes_what_counts_as_on_screen():
    head = HeadPose(yaw=-35, pitch=-5, roll=0)  # turned to the user's right
    centered = _features_cfg(camera_position=CameraPosition.TOP_CENTER)
    camera_left = _features_cfg(camera_position=CameraPosition.LEFT)
    assert extract_gaze_features(head, centered).direction is Attention.RIGHT
    assert extract_gaze_features(head, camera_left).facing_screen is True


def test_camera_below_monitors_expects_looking_up():
    bottom = _features_cfg(camera_position=CameraPosition.BOTTOM_CENTER, attention_margin=5.0)
    top = _features_cfg(attention_margin=5.0)
    assert compute_screen_zone(bottom).core_pitch == (0.0, CFG.features.monitor_pitch_span)
    up = HeadPose(yaw=0, pitch=12, roll=0)
    assert extract_gaze_features(up, bottom).facing_screen is True
    assert extract_gaze_features(up, top).direction is Attention.UP


def test_top_left_camera_with_two_monitors():
    zone = compute_screen_zone(_features_cfg(monitor_count=2, camera_position=CameraPosition.TOP_LEFT))
    # Camera above the left monitor: the right monitor is to the camera's right (negative yaw).
    assert zone.core_yaw == pytest.approx((-45.0, 15.0))


def test_strong_head_roll_is_off_screen():
    assert extract_gaze_features(HeadPose(0, -5, 50), CFG.features).direction is Attention.HEAD_TILTED


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

_ZONE = compute_screen_zone(CFG.features)
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
    # Phone threshold 3 s + 1 s stabiliser: still focused after 2.5 s.
    early = _run(pipeline, 2.5, looking_down, _phone_below())
    assert early.features.phone.looking_at_phone
    assert early.state is FocusState.FOCUSED
    late = _run(pipeline, 4.6, looking_down, _phone_below(), start=2.5)
    assert late.state is FocusState.DISTRACTED
    assert "phone" in late.reason.lower()


def test_looking_down_without_phone_uses_general_distraction_time():
    pipeline = FeaturePipeline(CFG)
    looking_down = _face(pitch=-40)
    person = ObjectResult(person_detected=True)
    assert _run(pipeline, 4.5, looking_down, person).state is FocusState.FOCUSED
    late = _run(pipeline, 7.0, looking_down, person, start=4.5)
    assert late.state is FocusState.DISTRACTED
    assert "Looking down" in late.reason


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

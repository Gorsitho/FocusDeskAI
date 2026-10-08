import json
import math

import pytest

from app.config.settings import (
    MAX_MONITOR_DISTANCE,
    CameraEdge,
    MonitorPlacement,
    Settings,
    Workspace,
    arc_layout,
)
from app.config.user_settings import (
    UserSettings,
    apply_user_settings,
    load_user_settings,
    normalize_workspace,
    save_user_settings,
)


def test_defaults_match_pipeline_defaults():
    # The pipeline defaults and the default user preferences must describe the same behaviour.
    assert apply_user_settings(Settings(), UserSettings()) == Settings()


def test_default_workspace_is_the_single_monitor_arc():
    assert Workspace().monitors == arc_layout(1)


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "nested" / "user_settings.json"
    workspace = Workspace(person_x=0.1, person_z=-0.05, monitors=arc_layout(3, 0.8),
                          camera_monitor=2, camera_edge=CameraEdge.BOTTOM)
    user = UserSettings(workspace=workspace, distraction_after_s=8.5, phone_distraction_after_s=2.0,
                        no_movement_away_after_s=120, sensitivity=8, sound_enabled=False,
                        sound_volume=15, show_state_timers=False, language="de")
    assert save_user_settings(user, path)
    assert load_user_settings(path) == user


def test_save_failure_is_reported(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    # The parent "directory" is a file, so writing must fail.
    assert save_user_settings(UserSettings(), blocker / "user_settings.json") is False


def test_missing_or_corrupt_file_gives_defaults(tmp_path):
    assert load_user_settings(tmp_path / "missing.json") == UserSettings()
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert load_user_settings(corrupt) == UserSettings()


def test_invalid_values_are_ignored_or_clamped(tmp_path):
    path = tmp_path / "user_settings.json"
    path.write_text(json.dumps({
        "sensitivity": 99,
        "sound_volume": -5,
        "sound_enabled": "yes",
        "distraction_after_s": "abc",
        "language": "fr",
        "workspace": {"monitors": "nonsense"},
        "unknown_key": 1,
    }), encoding="utf-8")
    user = load_user_settings(path)
    assert user.sensitivity == 10
    assert user.sound_volume == 0
    assert user.sound_enabled is True
    assert user.distraction_after_s == UserSettings().distraction_after_s
    assert user.language == "en"
    assert user.workspace == Workspace()


def test_settings_from_previous_version_are_migrated(tmp_path):
    path = tmp_path / "user_settings.json"
    path.write_text(json.dumps({
        "monitor_count": 3, "camera_position": "top_left", "idle_after_s": 90, "sound_volume": 25,
    }), encoding="utf-8")
    user = load_user_settings(path)
    assert user.workspace.monitor_count == 3
    assert user.workspace.camera_monitor == 0  # "top_left" -> on top of monitor 1
    assert user.workspace.camera_edge is CameraEdge.TOP
    assert user.sound_volume == 25
    assert user.no_movement_away_after_s == UserSettings().no_movement_away_after_s


def test_monitors_are_kept_within_one_metre():
    far = Workspace(monitors=(MonitorPlacement(0.0, 2.5), MonitorPlacement(0.05, 0.05)))
    ws = normalize_workspace(far)
    for m in ws.monitors:
        distance = math.hypot(m.x - ws.person_x, m.z - ws.person_z)
        assert 0.3 - 1e-9 <= distance <= MAX_MONITOR_DISTANCE + 1e-9


def test_at_most_three_monitors_and_valid_camera():
    ws = normalize_workspace(Workspace(monitors=arc_layout(3) + arc_layout(1), camera_monitor=7))
    assert ws.monitor_count == 3
    assert ws.camera_monitor == 0


def test_apply_user_settings_maps_every_preference():
    workspace = Workspace(monitors=arc_layout(2), camera_monitor=1)
    config = apply_user_settings(Settings(), UserSettings(
        workspace=workspace, distraction_after_s=9, phone_distraction_after_s=4,
        no_movement_away_after_s=90, sensitivity=10,
    ))
    assert config.features.workspace == workspace
    assert config.state.look_away_after_s == 9
    assert config.state.phone_after_s == 4
    assert config.state.still_away_after_s == 90


def test_higher_sensitivity_is_stricter():
    low = apply_user_settings(Settings(), UserSettings(sensitivity=1)).features
    high = apply_user_settings(Settings(), UserSettings(sensitivity=10)).features
    assert high.attention_margin < low.attention_margin
    assert high.phone_gaze_max_angle > low.phone_gaze_max_angle
    assert high.attention_margin > 0


@pytest.mark.parametrize("count", [1, 2, 3])
def test_arc_layout_respects_distance_limits(count):
    for distance in (0.1, 0.65, 3.0):
        for m in arc_layout(count, distance):
            assert 0.3 - 1e-9 <= math.hypot(m.x, m.z) <= MAX_MONITOR_DISTANCE + 1e-9


def test_settings_file_with_byte_order_mark_is_read(tmp_path):
    # Notepad and PowerShell can save JSON with a UTF-8 BOM; it must not reset the settings.
    path = tmp_path / "user_settings.json"
    path.write_text('{"sound_volume": 33, "language": "es"}', encoding="utf-8-sig")
    user = load_user_settings(path)
    assert user.sound_volume == 33 and user.language == "es"


# --- study method -----------------------------------------------------------------

from app.config.settings import StudyMethod  # noqa: E402


def test_study_method_defaults_to_computer(tmp_path):
    assert UserSettings().study_method is StudyMethod.COMPUTER
    assert load_user_settings(tmp_path / "missing.json").study_method is StudyMethod.COMPUTER


@pytest.mark.parametrize("method", list(StudyMethod))
def test_study_method_persists(tmp_path, method):
    path = tmp_path / "user_settings.json"
    assert save_user_settings(UserSettings(study_method=method), path)
    assert json.loads(path.read_text(encoding="utf-8"))["study_method"] == method.value
    assert load_user_settings(path).study_method is method


@pytest.mark.parametrize("raw", ["phone", 3, None, ""])
def test_invalid_study_method_falls_back_to_computer(tmp_path, raw):
    path = tmp_path / "user_settings.json"
    path.write_text(json.dumps({"study_method": raw, "sensitivity": 7}), encoding="utf-8")
    loaded = load_user_settings(path)
    assert loaded.study_method is StudyMethod.COMPUTER
    assert loaded.sensitivity == 7


def test_settings_without_study_method_use_computer(tmp_path):
    path = tmp_path / "user_settings.json"
    path.write_text(json.dumps({"language": "es"}), encoding="utf-8")
    assert load_user_settings(path).study_method is StudyMethod.COMPUTER


@pytest.mark.parametrize("method", list(StudyMethod))
def test_apply_user_settings_maps_study_method(method):
    assert apply_user_settings(Settings(), UserSettings(study_method=method)).features.study_method is method


def test_default_durations():
    user, state = UserSettings(), Settings().state
    assert (user.distraction_after_s, user.phone_distraction_after_s, user.no_movement_away_after_s) == (
        1.0, 1.0, 30.0)
    assert (state.look_away_after_s, state.phone_after_s, state.still_away_after_s) == (1.0, 1.0, 30.0)


def test_default_sensitivity_is_more_tolerant_around_monitors():
    features = apply_user_settings(Settings(), UserSettings()).features
    assert features.attention_margin == pytest.approx(16.0)
    assert features.monitor_exit_margin > 0

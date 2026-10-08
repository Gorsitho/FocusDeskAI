import json

import pytest

from app.config.settings import CameraPosition, Settings
from app.config.user_settings import (
    UserSettings,
    apply_user_settings,
    load_user_settings,
    save_user_settings,
)


def test_defaults_match_pipeline_defaults():
    # The pipeline defaults and the default user preferences must describe the same behaviour.
    assert apply_user_settings(Settings(), UserSettings()) == Settings()


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "nested" / "user_settings.json"
    user = UserSettings(monitor_count=3, camera_position=CameraPosition.TOP_RIGHT, distraction_after_s=8.5,
                        phone_distraction_after_s=2.0, idle_after_s=120, sensitivity=8,
                        sound_enabled=False, sound_volume=15, show_state_timers=False)
    save_user_settings(user, path)
    assert load_user_settings(path) == user


def test_missing_or_corrupt_file_gives_defaults(tmp_path):
    assert load_user_settings(tmp_path / "missing.json") == UserSettings()
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert load_user_settings(corrupt) == UserSettings()


def test_invalid_values_are_ignored_or_clamped(tmp_path):
    path = tmp_path / "user_settings.json"
    path.write_text(json.dumps({
        "monitor_count": 7,
        "camera_position": "on_the_ceiling",
        "sensitivity": 99,
        "sound_volume": -5,
        "sound_enabled": "yes",
        "distraction_after_s": "abc",
        "unknown_key": 1,
    }), encoding="utf-8")
    user = load_user_settings(path)
    assert user.monitor_count == 1
    assert user.camera_position is CameraPosition.TOP_CENTER
    assert user.sensitivity == 10
    assert user.sound_volume == 0
    assert user.sound_enabled is True
    assert user.distraction_after_s == UserSettings().distraction_after_s


def test_side_monitor_camera_falls_back_with_one_monitor():
    user = UserSettings(monitor_count=1, camera_position=CameraPosition.TOP_LEFT).normalized()
    assert user.camera_position is CameraPosition.TOP_CENTER


def test_apply_user_settings_maps_every_preference():
    config = apply_user_settings(Settings(), UserSettings(
        monitor_count=2, camera_position=CameraPosition.LEFT, distraction_after_s=9,
        phone_distraction_after_s=4, idle_after_s=90, sensitivity=10,
    ))
    assert config.features.monitor_count == 2
    assert config.features.camera_position is CameraPosition.LEFT
    assert config.state.look_away_after_s == 9
    assert config.state.phone_after_s == 4
    assert config.state.still_idle_after_s == 90


def test_higher_sensitivity_is_stricter():
    low = apply_user_settings(Settings(), UserSettings(sensitivity=1)).features
    high = apply_user_settings(Settings(), UserSettings(sensitivity=10)).features
    assert high.attention_margin < low.attention_margin
    assert high.phone_gaze_max_angle > low.phone_gaze_max_angle
    assert high.attention_margin > 0


@pytest.mark.parametrize("monitors", [1, 2, 3])
def test_every_available_position_is_valid(monitors):
    for position in CameraPosition:
        user = UserSettings(monitor_count=monitors, camera_position=position).normalized()
        assert user.camera_position.available_for(monitors)

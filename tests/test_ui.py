"""Widget-level tests; they run without a display using Qt's offscreen platform."""

import io
import os
import wave

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.config.settings import CameraPosition  # noqa: E402
from app.config.user_settings import UserSettings  # noqa: E402
from app.features.feature_pipeline import FocusState, FrameAnalysis, FrameFeatures  # noqa: E402
from app.features.phone_features import PhoneFeatures  # noqa: E402
from app.ui.camera_widget import draw_overlays  # noqa: E402
from app.ui.dashboard import Dashboard  # noqa: E402
from app.ui.settings_dialog import SettingsDialog, SetupDialog  # noqa: E402
from app.ui.sound import LOOP_SECONDS, SAMPLE_RATE, build_chime_wav  # noqa: E402
from app.vision.face_detection import FaceResult  # noqa: E402
from app.vision.head_pose import HeadPose  # noqa: E402
from app.vision.object_detection import ObjectResult  # noqa: E402
from app.vision.pose_detection import PoseResult  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_chime_is_a_soft_looping_wav():
    with wave.open(io.BytesIO(build_chime_wav())) as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == SAMPLE_RATE
        assert wav.getnframes() == int(SAMPLE_RATE * LOOP_SECONDS)
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    peak = np.abs(samples).max() / 32767
    assert 0.1 < peak <= 0.35  # audible but never loud at full volume
    # The loop ends in silence, so repeats are spaced out rather than continuous.
    assert np.abs(samples[-SAMPLE_RATE:]).max() < 50


def test_setup_dialog_returns_workspace(qapp):
    dialog = SetupDialog(UserSettings(sound_volume=12))
    dialog._workspace._monitor_group.button(2).click()
    dialog._workspace._select_position(CameraPosition.TOP_RIGHT)
    result = dialog.result_settings()
    assert result.monitor_count == 2
    assert result.camera_position is CameraPosition.TOP_RIGHT
    assert result.sound_volume == 12  # untouched preferences are kept


def test_reducing_monitors_resets_unavailable_camera_position(qapp):
    dialog = SetupDialog(UserSettings(monitor_count=3, camera_position=CameraPosition.TOP_LEFT))
    dialog._workspace._monitor_group.button(1).click()
    assert dialog.result_settings().camera_position is CameraPosition.TOP_CENTER


def test_settings_dialog_edits_all_preferences(qapp):
    dialog = SettingsDialog(UserSettings())
    dialog._distraction.setValue(12)
    dialog._phone.setValue(6)
    dialog._idle.setValue(300)
    dialog._sensitivity.setValue(9)
    dialog._sound_enabled.setChecked(False)
    dialog._volume.setValue(70)
    result = dialog.result_settings()
    assert (result.distraction_after_s, result.phone_distraction_after_s, result.idle_after_s) == (12, 6, 300)
    assert result.sensitivity == 9
    assert result.sound_enabled is False and result.sound_volume == 70

    dialog._restore_defaults()
    assert dialog.result_settings() == UserSettings()


def test_dashboard_shows_phone_states_and_timers(qapp):
    dashboard = Dashboard()
    visible = FrameFeatures(timestamp=0.0, person_detected=True, phone=PhoneFeatures(visible=True))
    dashboard.update_analysis(FrameAnalysis(visible, FocusState.FOCUSED, "Attention on the monitors"))
    assert dashboard._phone.text() == "Visible, ignored"
    assert dashboard._state.text() == "FOCUSED"

    in_use = FrameFeatures(timestamp=0.0, person_detected=True,
                           phone=PhoneFeatures(visible=True, looking_at_phone=True))
    dashboard.update_analysis(FrameAnalysis(in_use, FocusState.DISTRACTED, "Looking at phone for 4 s"))
    assert dashboard._phone.text() == "Looking at it"

    dashboard.update_analysis(FrameAnalysis(in_use, FocusState.DISTRACTED), FocusState.BREAK)
    assert dashboard._state.text() == "BREAK"

    dashboard.update_timers({FocusState.FOCUSED: 2551, FocusState.BREAK: 310}, FocusState.FOCUSED)
    assert dashboard._timer_values[FocusState.FOCUSED].text() == "42:31"
    assert dashboard._timer_values[FocusState.BREAK].text() == "05:10"
    assert dashboard._timer_values[FocusState.AWAY].text() == "00:00"

    dashboard.set_timers_visible(False)
    assert dashboard._timers_card.isHidden()


def test_overlay_draws_head_direction():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    landmarks = np.full((478, 3), 0.5, np.float32)
    face = FaceResult(True, landmarks=landmarks, transform=np.eye(4))
    features = FrameFeatures(timestamp=0.0, head_pose=HeadPose(30, 0, 0))
    out = draw_overlays(frame, face, PoseResult(False), ObjectResult(), 0.5,
                        FrameAnalysis(features, FocusState.FOCUSED))
    # Yaw towards the user's left is drawn towards the image right of the nose.
    right, left = out[:, 330:].sum(), out[:, :310].sum()
    assert right > left

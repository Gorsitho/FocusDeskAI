"""
Tests for the user interface in app/ui/: the setup and Settings windows, the
desk layout editor, the dashboard, the translations, the sound and the camera overlay.
They run without a screen, using Qt's "offscreen" mode.
"""

import io
import math
import os
import re
import string
import wave

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QAbstractButton, QApplication, QLabel  # noqa: E402

from app.config.settings import MAX_MONITOR_DISTANCE, CameraEdge, Workspace, arc_layout  # noqa: E402
from app.config.user_settings import UserSettings, load_user_settings, save_user_settings  # noqa: E402
from app.features.feature_pipeline import (  # noqa: E402
    FocusState,
    FrameAnalysis,
    FrameFeatures,
    Reason,
    ReasonCode,
)
from app.features.gaze_features import Attention  # noqa: E402
from app.features.phone_features import PhoneFeatures  # noqa: E402
from app.ui import i18n  # noqa: E402
from app.ui.camera_widget import draw_overlays  # noqa: E402
from app.ui.dashboard import Dashboard, describe_reason  # noqa: E402
from app.ui.settings_dialog import SettingsDialog, SetupDialog, parse_seconds  # noqa: E402
from app.ui.sound import LOOP_SECONDS, SAMPLE_RATE, build_chime_wav  # noqa: E402
from app.ui.workspace_widget import WorkspaceEditor, move_person, nearest_distance, set_distance  # noqa: E402
from app.vision.face_detection import FaceResult  # noqa: E402
from app.vision.head_pose import HeadPose  # noqa: E402
from app.vision.object_detection import ObjectResult  # noqa: E402
from app.vision.pose_detection import PoseResult  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def english():
    i18n.set_language("en")
    yield
    i18n.set_language("en")


def _type(spin_box, text):
    spin_box.setFocus()
    spin_box.selectAll()
    QTest.keyClicks(spin_box, text)


# --- sound -------------------------------------------------------------------------

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


# --- settings: typed values are saved (regression) ------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("7.5", 7.5), ("7,5", 7.5), ("12", 12.0), ("12 s", 12.0), ("12s", 12.0), (" 8,0 s ", 8.0),
    ("", None), ("abc", None), (",", None),
])
def test_parse_seconds(text, expected):
    assert parse_seconds(text) == expected


@pytest.mark.parametrize("typed,expected", [("7,5", 7.5), ("7.5", 7.5), ("12", 12.0), ("45 s", 45.0)])
def test_typed_distraction_value_is_saved(qapp, typed, expected):
    dialog = SettingsDialog(UserSettings())
    dialog.show()
    _type(dialog._distraction.spin_box, typed)
    # Save is clicked straight away: the text in the field must be used.
    assert dialog.result_settings().distraction_after_s == expected


def test_typed_values_outside_the_range_are_clamped_not_dropped(qapp):
    dialog = SettingsDialog(UserSettings())
    dialog.show()
    _type(dialog._no_movement.spin_box, "2")  # minimum is 5 s
    _type(dialog._phone.spin_box, "9999")  # maximum is 600 s
    result = dialog.result_settings()
    assert result.no_movement_away_after_s == 5
    assert result.phone_distraction_after_s == 600


def test_step_buttons_start_from_the_typed_value(qapp):
    dialog = SettingsDialog(UserSettings())
    dialog.show()
    _type(dialog._distraction.spin_box, "10")
    dialog._distraction._step(1)
    assert dialog.result_settings().distraction_after_s == 10.5


def test_settings_persist_after_save_and_reopen(qapp, tmp_path):
    path = tmp_path / "user_settings.json"
    dialog = SettingsDialog(UserSettings())
    dialog.show()
    _type(dialog._distraction.spin_box, "8,5")
    _type(dialog._phone.spin_box, "4")
    _type(dialog._no_movement.spin_box, "150")
    dialog._sensitivity.setValue(7)
    dialog._sound_enabled.setChecked(False)
    dialog._volume.setValue(65)
    dialog._language.setCurrentIndex(dialog._language.findData("es"))
    dialog.workspace_editor._monitor_group.button(3).click()
    QTest.mouseClick(dialog._save, Qt.MouseButton.LeftButton)
    assert dialog.result() == SettingsDialog.DialogCode.Accepted
    save_user_settings(dialog.result_settings(), path)

    saved = load_user_settings(path)
    assert (saved.distraction_after_s, saved.phone_distraction_after_s, saved.no_movement_away_after_s) == (
        8.5, 4.0, 150.0)
    assert (saved.sensitivity, saved.sound_enabled, saved.sound_volume, saved.language) == (7, False, 65, "es")
    assert saved.workspace.monitor_count == 3

    reopened = SettingsDialog(saved)
    reopened.show()
    assert reopened._distraction.value() == 8.5
    assert reopened._distraction.spin_box.text() == "8,5 s"  # Spanish decimal comma
    assert reopened._no_movement.value() == 150
    assert reopened._volume.value() == 65
    assert reopened.result_settings() == saved


def test_restore_defaults_keeps_workspace_and_language(qapp):
    workspace = Workspace(monitors=arc_layout(2))
    dialog = SettingsDialog(UserSettings(workspace=workspace, language="de", sensitivity=9, sound_volume=5))
    dialog._restore_defaults()
    result = dialog.result_settings()
    assert result.workspace.monitor_count == 2 and result.language == "de"
    assert result.sensitivity == 5 and result.sound_volume == 40


# --- workspace editor ---------------------------------------------------------------

def test_setup_dialog_returns_workspace_and_language(qapp):
    dialog = SetupDialog(UserSettings(sound_volume=12))
    editor = dialog.workspace_editor
    editor._monitor_group.button(2).click()
    editor._camera_monitor.setCurrentIndex(1)
    editor._camera_edge.setCurrentIndex(editor._camera_edge.findData(CameraEdge.BOTTOM.value))
    dialog._language.setCurrentIndex(dialog._language.findData("de"))
    result = dialog.result_settings()
    assert result.workspace.monitor_count == 2
    assert result.workspace.camera_monitor == 1 and result.workspace.camera_edge is CameraEdge.BOTTOM
    assert result.language == "de"
    assert result.sound_volume == 12  # untouched preferences are kept


def test_distance_slider_moves_monitors_and_respects_one_metre(qapp):
    editor = WorkspaceEditor(Workspace(monitors=arc_layout(3)))
    editor._distance.setValue(90)
    assert nearest_distance(editor.workspace) == pytest.approx(0.9, abs=0.01)
    editor._distance.setValue(100)
    for m in editor.workspace.monitors:
        assert math.hypot(m.x, m.z) <= MAX_MONITOR_DISTANCE + 1e-6


def test_angle_slider_rotates_selected_monitor(qapp):
    editor = WorkspaceEditor(Workspace(monitors=arc_layout(3)))
    editor.plan.select(2)
    editor._angle.setValue(-30)
    assert editor.workspace.monitors[2].angle == -30
    editor._arrange.click()
    expected = arc_layout(3, nearest_distance(editor.workspace))
    for got, want in zip(editor.workspace.monitors, expected):
        assert (got.x, got.z, got.angle) == pytest.approx((want.x, want.z, want.angle), abs=1e-3)


def test_dragging_the_person_keeps_monitors_within_reach(qapp):
    ws = Workspace(monitors=arc_layout(3))
    moved = move_person(ws, -0.9, -0.35)
    for m in moved.monitors:
        assert math.hypot(m.x - moved.person_x, m.z - moved.person_z) <= MAX_MONITOR_DISTANCE + 1e-3


def test_drag_person_with_mouse(qapp):
    editor = WorkspaceEditor(Workspace(monitors=arc_layout(1)))
    plan = editor.plan
    plan.resize(500, 340)
    start = plan._to_screen(0.0, 0.0)
    end = plan._to_screen(0.2, 0.1)
    QTest.mousePress(plan, Qt.MouseButton.LeftButton, pos=start.toPoint())
    QTest.mouseMove(plan, end.toPoint())
    QTest.mouseRelease(plan, Qt.MouseButton.LeftButton, pos=end.toPoint())
    assert editor.workspace.person_x == pytest.approx(0.2, abs=0.02)
    assert editor.workspace.person_z == pytest.approx(0.1, abs=0.02)


def test_set_distance_scales_layout():
    ws = Workspace(monitors=arc_layout(2, 0.5))
    assert nearest_distance(set_distance(ws, 0.8)) == pytest.approx(0.8, abs=1e-3)


# --- languages --------------------------------------------------------------------

def _placeholders(text):
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def test_every_language_has_every_string():
    english = i18n._STRINGS["en"]
    for code in ("es", "de"):
        table = i18n._STRINGS[code]
        assert set(table) == set(english), code
        for key, text in english.items():
            assert _placeholders(table[key]) == _placeholders(text), (code, key)


def test_every_enum_value_is_translated():
    keys = set(i18n._STRINGS["en"])
    for state in FocusState:
        assert f"state.{state.value}" in keys
    for attention in Attention:
        assert f"attention.{attention.value}" in keys
    for edge in CameraEdge:
        assert f"ws.edge.{edge.value}" in keys


def _visible_texts(widget):
    texts = [w.text() for w in widget.findChildren(QLabel) + widget.findChildren(QAbstractButton)]
    return [t for t in texts if t and not re.fullmatch(r"[\d\W_]*", t)]


@pytest.mark.parametrize("code,expected", [
    ("en", ["Settings", "Save", "No movement before away", "Language"]),
    ("es", ["Ajustes", "Guardar", "Sin movimiento antes de ausente", "Idioma"]),
    ("de", ["Einstellungen", "Speichern", "Keine Bewegung bis abwesend", "Sprache"]),
])
def test_settings_dialog_in_each_language(qapp, code, expected):
    i18n.set_language(code)
    dialog = SettingsDialog(UserSettings(language=code))
    texts = _visible_texts(dialog) + [dialog.windowTitle()] + [
        box.title() for box in (dialog._detection_box, dialog._sound_box, dialog._language_box)]
    for word in expected:
        assert any(word in t for t in texts), (code, word)
    if code != "en":
        english = set(i18n._STRINGS["en"].values())
        untranslated = [t for t in texts if t in english and t not in i18n._STRINGS[code].values()]
        assert not untranslated, untranslated


def test_switching_language_in_settings_retranslates_live(qapp):
    dialog = SettingsDialog(UserSettings())
    dialog._language.setCurrentIndex(dialog._language.findData("de"))
    assert dialog.windowTitle() == "Einstellungen"
    assert dialog.workspace_editor._arrange.text() == "Im Halbkreis anordnen"


@pytest.mark.parametrize("code,focused,reason", [
    ("en", "FOCUSED", "Looking at monitor 2"),
    ("es", "CONCENTRADO", "Mirando el monitor 2"),
    ("de", "KONZENTRIERT", "Blick auf Monitor 2"),
])
def test_dashboard_in_each_language(qapp, code, focused, reason):
    i18n.set_language(code)
    dashboard = Dashboard()
    features = FrameFeatures(timestamp=0.0, person_detected=True, attention=Attention.ON_SCREEN, monitor=1)
    dashboard.update_analysis(FrameAnalysis(features, FocusState.FOCUSED, Reason(ReasonCode.ON_MONITOR, monitor=1)))
    assert dashboard._state.text() == focused
    assert dashboard._reason.text() == reason
    assert dashboard._monitor.text() == i18n.tr("monitor.n", n=2)
    names = [dashboard._timer_names[s].text() for s in FocusState]
    assert all(i18n.tr(f"state.{s.value}") in n for s, n in zip(FocusState, names))


def test_reasons_use_language_decimal_separator():
    i18n.set_language("de")
    text = describe_reason(Reason(ReasonCode.LOOKING_AWAY, 6.0, attention=Attention.DOWN))
    assert text == "Blick nach unten seit 6 s"


# --- dashboard --------------------------------------------------------------------

def test_dashboard_shows_phone_states_and_timers(qapp):
    dashboard = Dashboard()
    visible = FrameFeatures(timestamp=0.0, person_detected=True, phone=PhoneFeatures(visible=True))
    dashboard.update_analysis(FrameAnalysis(visible, FocusState.FOCUSED, Reason(ReasonCode.ON_MONITOR)))
    assert dashboard._phone.text() == "Visible, ignored"
    assert dashboard._state.text() == "FOCUSED"

    in_use = FrameFeatures(timestamp=0.0, person_detected=True,
                           phone=PhoneFeatures(visible=True, looking_at_phone=True))
    dashboard.update_analysis(FrameAnalysis(in_use, FocusState.DISTRACTED, Reason(ReasonCode.PHONE, 4.0)))
    assert dashboard._phone.text() == "Looking at it"
    assert dashboard._reason.text() == "Looking at the phone for 4 s"

    dashboard.update_analysis(FrameAnalysis(in_use, FocusState.DISTRACTED), FocusState.BREAK)
    assert dashboard._state.text() == "BREAK"

    dashboard.update_timers({FocusState.FOCUSED: 2551, FocusState.BREAK: 310}, FocusState.FOCUSED)
    assert dashboard._timer_values[FocusState.FOCUSED].text() == "42:31"
    assert dashboard._timer_values[FocusState.BREAK].text() == "05:10"
    assert dashboard._timer_values[FocusState.AWAY].text() == "00:00"
    assert len(dashboard._timer_values) == 4  # no IDLE

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


# --- detection landmarks setting -------------------------------------------------------

def test_landmarks_are_off_by_default_and_persist(tmp_path):
    assert UserSettings().show_landmarks is False
    path = tmp_path / "user_settings.json"
    save_user_settings(UserSettings(show_landmarks=True), path)
    assert load_user_settings(path).show_landmarks is True


@pytest.mark.parametrize("code,label", [
    ("en", "Show detection landmarks"), ("es", "Mostrar puntos de detección"), ("de", "Erkennungspunkte anzeigen"),
])
def test_landmark_checkbox_label_and_value(qapp, code, label):
    i18n.set_language(code)
    dialog = SettingsDialog(UserSettings(language=code))
    assert dialog._show_landmarks.text() == label
    assert dialog._show_landmarks.isChecked() is False
    dialog._show_landmarks.setChecked(True)
    assert dialog.result_settings().show_landmarks is True


def test_preview_hides_landmarks_when_disabled():
    from app.ui.main_window import compose_preview

    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    frame[:, :320] = 90  # asymmetric, so mirroring is visible
    landmarks = np.full((478, 3), 0.5, np.float32)
    landmarks[::2, :2] = 0.3
    face = FaceResult(True, landmarks=landmarks, transform=np.eye(4))
    features = FrameFeatures(timestamp=0.0, head_pose=HeadPose(30, 0, 0))
    analysis = FrameAnalysis(features, FocusState.FOCUSED)

    hidden = compose_preview(frame, face, PoseResult(False), ObjectResult(), 0.5, analysis, False)
    assert np.array_equal(hidden, frame[:, ::-1])  # just the mirrored camera image
    shown = compose_preview(frame, face, PoseResult(False), ObjectResult(), 0.5, analysis, True)
    assert not np.array_equal(shown, frame[:, ::-1])


def test_main_window_applies_landmark_setting_to_worker(qapp, monkeypatch):
    from app.ui import main_window

    monkeypatch.setattr(main_window.AnalysisWorker, "start", lambda self: None)
    window = main_window.MainWindow(user_settings=UserSettings())
    assert window._worker.show_landmarks is False
    window.apply_user_settings(UserSettings(show_landmarks=True))
    assert window._worker.show_landmarks is True
    window.close()


# --- study method -----------------------------------------------------------------

from app.config.settings import StudyMethod  # noqa: E402


def _click(button):
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)


def test_setup_starts_with_the_study_method_and_defaults_to_computer(qapp):
    dialog = SetupDialog(UserSettings())
    assert dialog.page == SetupDialog.METHOD_PAGE
    assert dialog.method_picker.method is StudyMethod.COMPUTER
    assert dialog.method_picker.card(StudyMethod.COMPUTER).isChecked()
    assert dialog._start.text() == "Next"


@pytest.mark.parametrize("method", [StudyMethod.COMPUTER, StudyMethod.MIXED])
def test_setup_shows_monitor_configuration_after_the_method(qapp, method):
    dialog = SetupDialog(UserSettings())
    dialog.show()
    _click(dialog.method_picker.card(method))
    _click(dialog._start)
    assert dialog.page == SetupDialog.WORKSPACE_PAGE
    assert dialog.result() != SetupDialog.DialogCode.Accepted
    assert dialog._start.text() == "Start monitoring" and dialog._back.isVisible()
    dialog.workspace_editor._monitor_group.button(2).click()
    _click(dialog._start)
    assert dialog.result() == SetupDialog.DialogCode.Accepted
    result = dialog.result_settings()
    assert result.study_method is method and result.workspace.monitor_count == 2


def test_setup_back_returns_to_the_method(qapp):
    dialog = SetupDialog(UserSettings())
    dialog.show()
    _click(dialog._start)
    _click(dialog._back)
    assert dialog.page == SetupDialog.METHOD_PAGE
    assert not dialog._back.isVisible()


def test_setup_skips_monitor_configuration_for_tablet(qapp):
    workspace = Workspace(monitors=arc_layout(3))
    dialog = SetupDialog(UserSettings(workspace=workspace))
    dialog.show()
    _click(dialog.method_picker.card(StudyMethod.TABLET))
    assert dialog._start.text() == "Start monitoring"
    _click(dialog._start)
    assert dialog.result() == SetupDialog.DialogCode.Accepted
    assert dialog.page == SetupDialog.METHOD_PAGE
    result = dialog.result_settings()
    assert result.study_method is StudyMethod.TABLET
    assert result.workspace == workspace  # kept for switching back later


def test_setup_preselects_the_saved_method(qapp):
    dialog = SetupDialog(UserSettings(study_method=StudyMethod.MIXED))
    assert dialog.method_picker.method is StudyMethod.MIXED


@pytest.mark.parametrize("code,expected", [
    ("en", ["Computer", "Tablet / Notebook", "Mixed", "How do you study?", "Next"]),
    ("es", ["Ordenador", "Tableta / Cuaderno", "Mixto", "¿Cómo estudias?", "Siguiente"]),
    ("de", ["Computer", "Tablet / Heft", "Gemischt", "Wie lernst du?", "Weiter"]),
])
def test_study_method_step_in_each_language(qapp, code, expected):
    dialog = SetupDialog(UserSettings(language=code))
    texts = _visible_texts(dialog)
    for word in expected:
        assert word in texts, (code, word)


def test_study_method_can_be_changed_in_settings_and_persists(qapp, tmp_path):
    path = tmp_path / "user_settings.json"
    dialog = SettingsDialog(UserSettings())
    dialog.show()
    assert dialog.method_picker.method is StudyMethod.COMPUTER
    _click(dialog.method_picker.card(StudyMethod.MIXED))
    _click(dialog._save)
    save_user_settings(dialog.result_settings(), path)
    saved = load_user_settings(path)
    assert saved.study_method is StudyMethod.MIXED
    reopened = SettingsDialog(saved)
    assert reopened.method_picker.method is StudyMethod.MIXED
    assert reopened.result_settings() == saved


def test_settings_marks_monitor_layout_unneeded_for_tablet(qapp):
    dialog = SettingsDialog(UserSettings(workspace=Workspace(monitors=arc_layout(2))))
    dialog.show()
    assert dialog.workspace_editor.isEnabled()
    assert not dialog._workspace_unneeded.isVisible()
    _click(dialog.method_picker.card(StudyMethod.TABLET))
    # The whole section is disabled and greyed out, and screens cannot be configured.
    assert not dialog._workspace_box.isEnabled()
    assert not dialog.workspace_editor.isEnabled()
    assert dialog.workspace_editor.graphicsEffect() is not None
    assert dialog._workspace_unneeded.isVisible()
    _click(dialog.workspace_editor._monitor_group.button(3))
    assert dialog.result_settings().workspace.monitor_count == 2
    _click(dialog.method_picker.card(StudyMethod.COMPUTER))
    assert dialog._workspace_box.isEnabled() and dialog.workspace_editor.isEnabled()
    assert dialog.workspace_editor.graphicsEffect() is None
    _click(dialog.workspace_editor._monitor_group.button(3))
    assert dialog.result_settings().workspace.monitor_count == 3


def test_restore_defaults_keeps_the_study_method(qapp):
    dialog = SettingsDialog(UserSettings(study_method=StudyMethod.TABLET))
    dialog._restore_defaults()
    assert dialog.result_settings().study_method is StudyMethod.TABLET


def test_dashboard_shows_desk_focus(qapp):
    dashboard = Dashboard()
    features = FrameFeatures(timestamp=0.0, person_detected=True, attention=Attention.DESK)
    dashboard.update_analysis(FrameAnalysis(features, FocusState.FOCUSED, Reason(ReasonCode.ON_DESK)))
    assert dashboard._reason.text() == "Studying at the desk"


def test_settings_dialog_shows_new_default_durations(qapp):
    dialog = SettingsDialog(UserSettings())
    assert (dialog._distraction.value(), dialog._phone.value(), dialog._no_movement.value()) == (1.0, 1.0, 30.0)
    dialog._distraction.setValue(9)
    dialog._restore_defaults()
    assert dialog.result_settings().distraction_after_s == 1.0


def test_settings_opened_in_tablet_mode_starts_greyed_out(qapp):
    dialog = SettingsDialog(UserSettings(study_method=StudyMethod.TABLET))
    assert not dialog._workspace_box.isEnabled()
    assert dialog.workspace_editor.graphicsEffect() is not None


# --- eyes and phone details on the dashboard ----------------------------------------------

from app.features.eye_features import EyeFeatures, EyeState  # noqa: E402


def test_every_eye_state_is_translated():
    keys = set(i18n._STRINGS["en"])
    for state in EyeState:
        assert f"eyes.{state.value}" in keys


def test_dashboard_shows_eyes_and_phone_details(qapp):
    i18n.set_language("en")
    dashboard = Dashboard()
    features = FrameFeatures(
        timestamp=0.0, person_detected=True, face_detected=True,
        eyes=EyeFeatures(state=EyeState.PARTIAL, openness=0.6, yaw=12.0, pitch=None),
        phone=PhoneFeatures(visible=True, in_hand=True),
    )
    dashboard.update_analysis(FrameAnalysis(features, FocusState.FOCUSED, Reason(ReasonCode.ON_MONITOR)))
    assert dashboard._eyes.text() == "Partly closed (60%)"
    assert dashboard._eye_gaze.text() == "+12°"
    assert dashboard._phone.text() == "In hand, ignored"

    call = FrameFeatures(timestamp=0.0, person_detected=True, face_detected=True,
                         eyes=EyeFeatures(state=EyeState.CLOSED, openness=0.1),
                         phone=PhoneFeatures(visible=True, looking_at_phone=True, at_ear=True))
    dashboard.update_analysis(FrameAnalysis(call, FocusState.DISTRACTED, Reason(ReasonCode.EYES_CLOSED, 3.0)))
    assert dashboard._phone.text() == "At the ear (call)"
    assert dashboard._reason.text() == "Eyes closed for 3 s"
    assert dashboard._eye_gaze.text() == "—"  # no eye direction while the eyes are closed


def test_reason_for_a_body_without_a_face():
    i18n.set_language("en")
    assert describe_reason(Reason(ReasonCode.FACE_MISSING)) == "Face not visible"

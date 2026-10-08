import os
import re
from datetime import datetime, timedelta, timezone

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.data.session_log import SessionLogStore, format_duration_precise, format_session  # noqa: E402
from app.features.feature_pipeline import FocusState, FrameAnalysis, FrameFeatures  # noqa: E402
from app.features.session import SessionController, SessionSummary  # noqa: E402


class FakeClock:
    """Monotonic and wall clocks that only move when told to."""

    def __init__(self):
        self.mono = 1000.0
        self.wall = datetime(2026, 10, 8, 9, 30, 0, 125000, tzinfo=timezone(timedelta(hours=2)))

    def monotonic(self):
        return self.mono

    def wall_clock(self):
        return self.wall

    def advance(self, seconds):
        self.mono += seconds
        self.wall += timedelta(seconds=seconds)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def controller(clock):
    return SessionController(clock.monotonic, clock.wall_clock)


# --- timing --------------------------------------------------------------------------

def test_no_time_is_counted_before_start(controller, clock):
    controller.set_state(FocusState.FOCUSED)
    clock.advance(30)
    assert not controller.active
    assert sum(controller.totals().values()) == 0
    assert controller.elapsed() == 0


def test_session_counts_each_state(controller, clock):
    controller.set_state(FocusState.AWAY)
    clock.advance(100)  # before START: ignored
    controller.start(0)
    clock.advance(10)
    controller.set_state(FocusState.FOCUSED)
    clock.advance(60)
    controller.set_state(FocusState.DISTRACTED)
    clock.advance(5)
    controller.set_state(FocusState.BREAK)
    clock.advance(20)
    controller.set_state(FocusState.FOCUSED)
    clock.advance(5)
    summary = controller.stop()
    assert summary.number == 0
    assert summary.total_s == pytest.approx(100)
    assert summary.durations[FocusState.AWAY] == pytest.approx(10)
    assert summary.durations[FocusState.FOCUSED] == pytest.approx(65)
    assert summary.durations[FocusState.DISTRACTED] == pytest.approx(5)
    assert summary.durations[FocusState.BREAK] == pytest.approx(20)
    assert summary.share(summary.durations[FocusState.FOCUSED]) == pytest.approx(65)
    assert not controller.active


def test_durations_use_the_monotonic_clock_not_the_wall_clock(controller, clock):
    controller.set_state(FocusState.FOCUSED)
    controller.start(3)
    clock.advance(42)
    clock.wall -= timedelta(hours=1)  # e.g. daylight-saving change during the session
    summary = controller.stop()
    assert summary.total_s == pytest.approx(42)
    assert summary.durations[FocusState.FOCUSED] == pytest.approx(42)


def test_time_without_a_state_is_reported_as_untracked(controller, clock):
    controller.set_state(FocusState.FOCUSED)
    controller.start(0)
    clock.advance(30)
    controller.set_state(None)  # camera unavailable
    clock.advance(10)
    summary = controller.stop()
    assert summary.untracked_s == pytest.approx(10)


def test_new_session_starts_from_zero(controller, clock):
    controller.set_state(FocusState.FOCUSED)
    controller.start(0)
    clock.advance(50)
    controller.stop()
    clock.advance(20)
    controller.start(1)
    clock.advance(5)
    assert controller.totals()[FocusState.FOCUSED] == pytest.approx(5)
    assert controller.number == 1


def test_invalid_transitions(controller):
    with pytest.raises(RuntimeError):
        controller.stop()
    controller.start(0)
    with pytest.raises(RuntimeError):
        controller.start(1)


# --- session log files ------------------------------------------------------------------

def _summary(number=0, total=125.5):
    start = datetime(2026, 10, 8, 9, 30, 0, 125000, tzinfo=timezone(timedelta(hours=2)))
    return SessionSummary(number, start, start + timedelta(seconds=total), total, {
        FocusState.FOCUSED: 100.0, FocusState.DISTRACTED: 15.5, FocusState.AWAY: 4.0, FocusState.BREAK: 6.0,
    })


def test_numbering_starts_at_zero_and_is_sequential(tmp_path):
    store = SessionLogStore(tmp_path / "sessions")
    assert store.next_number() == 0
    for expected in range(3):
        path, summary = store.write(_summary(store.next_number()))
        assert path.name == f"session_{expected:04d}.log"
    assert sorted(p.name for p in (tmp_path / "sessions").iterdir()) == [
        "session_0000.log", "session_0001.log", "session_0002.log"]


def test_numbering_continues_after_the_highest_and_ignores_other_files(tmp_path):
    for name in ("session_0000.log", "session_0004.log", "notes.txt", "session_abc.log", "FocusDeskAI.log"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    assert SessionLogStore(tmp_path).next_number() == 5


def test_existing_session_file_is_never_overwritten(tmp_path):
    store = SessionLogStore(tmp_path)
    (tmp_path / "session_0000.log").write_text("previous session", encoding="utf-8")
    path, summary = store.write(_summary(0))
    assert path.name == "session_0001.log" and summary.number == 1
    assert (tmp_path / "session_0000.log").read_text(encoding="utf-8") == "previous session"
    assert "Session number : 0001" in path.read_text(encoding="utf-8")


def test_session_log_contents(tmp_path):
    path, _ = SessionLogStore(tmp_path).write(_summary(7))
    text = path.read_text(encoding="utf-8")
    assert "Session number : 0007" in text
    assert "Start          : 2026-10-08 09:30:00.125+02:00" in text
    assert "End            : 2026-10-08 09:32:05.625+02:00" in text
    assert "Total duration : 00:02:05.500 (125.500 s)" in text
    assert re.search(r"FOCUSED\s+00:01:40\.000\s+100\.000\s+79\.7%", text)
    assert re.search(r"DISTRACTED\s+00:00:15\.500\s+15\.500\s+12\.4%", text)
    assert re.search(r"AWAY\s+00:00:04\.000\s+4\.000\s+3\.2%", text)
    assert re.search(r"BREAK\s+00:00:06\.000\s+6\.000\s+4\.8%", text)
    assert "IDLE" not in text


def test_format_duration_precise():
    assert format_duration_precise(0) == "00:00:00.000"
    assert format_duration_precise(3725.4567) == "01:02:05.457"
    assert format_duration_precise(-1) == "00:00:00.000"


def test_format_session_has_one_line_per_state():
    text = format_session(_summary())
    for state in FocusState:
        assert sum(1 for line in text.splitlines() if line.startswith(state.value + " ")) == 1


# --- START / STOP in the real main window ---------------------------------------------

@pytest.fixture
def window(tmp_path, monkeypatch, clock):
    from PySide6.QtWidgets import QApplication

    from app.config.user_settings import UserSettings
    from app.ui import main_window

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(main_window.AnalysisWorker, "start", lambda self: None)  # no camera in tests
    win = main_window.MainWindow(user_settings=UserSettings(), sessions_path=tmp_path / "sessions")
    win._session = SessionController(clock.monotonic, clock.wall_clock)
    win._update_session_controls()
    yield win
    win.close()


def _analysis(state):
    return FrameAnalysis(FrameFeatures(timestamp=0.0, person_detected=True), state)


def test_app_starts_without_a_session(window, clock):
    window._on_analysis(_analysis(FocusState.FOCUSED))
    clock.advance(60)
    window._refresh_timers()
    assert not window.session.active
    assert window._session_button.text() == "▶  START"
    assert window._dashboard._timer_values[FocusState.FOCUSED].text() == "00:00"


def test_start_and_stop_session_writes_one_log(window, clock, tmp_path):
    window._on_analysis(_analysis(FocusState.FOCUSED))
    window._session_button.click()  # START
    assert window.session.active and window.session.number == 0
    assert window._session_button.text() == "■  STOP SESSION"
    clock.advance(30)
    window._on_analysis(_analysis(FocusState.DISTRACTED))
    clock.advance(10)
    window._break_button.click()
    clock.advance(5)
    window._refresh_timers()
    assert window._dashboard._timer_values[FocusState.FOCUSED].text() == "00:30"
    window._session_button.click()  # STOP
    assert not window.session.active
    files = sorted(p.name for p in (tmp_path / "sessions").iterdir())
    assert files == ["session_0000.log"]
    text = (tmp_path / "sessions" / "session_0000.log").read_text(encoding="utf-8")
    assert "Total duration : 00:00:45.000" in text
    assert re.search(r"FOCUSED\s+00:00:30\.000", text)
    assert re.search(r"DISTRACTED\s+00:00:10\.000", text)
    assert re.search(r"BREAK\s+00:00:05\.000", text)
    assert "session_0000.log" in window._status.text()


def test_next_session_gets_the_next_number(window, clock, tmp_path):
    window._on_analysis(_analysis(FocusState.FOCUSED))
    for _ in range(2):
        window.start_session()
        clock.advance(3)
        window.stop_session()
    window.start_session()
    clock.advance(4)
    window.stop_session()
    assert sorted(p.name for p in (tmp_path / "sessions").iterdir()) == [
        "session_0000.log", "session_0001.log", "session_0002.log"]


def test_closing_the_window_saves_a_running_session(window, clock, tmp_path):
    window._on_analysis(_analysis(FocusState.FOCUSED))
    window.start_session()
    clock.advance(12)
    window.close()
    assert (tmp_path / "sessions" / "session_0000.log").exists()


def test_session_controls_are_translated(window):
    from app.ui import i18n

    window.apply_user_settings(window._user.__class__(language="es"))
    assert window._session_button.text() == "▶  INICIAR"
    window.start_session()
    assert window._session_button.text() == "■  DETENER SESIÓN"
    window.apply_user_settings(window._user.__class__(language="de"))
    assert window._session_button.text() == "■  SITZUNG BEENDEN"
    i18n.set_language("en")


def test_close_waits_for_a_busy_worker_instead_of_destroying_it(window, monkeypatch):
    """A running QThread must never be destroyed (that aborts the whole process)."""
    from PySide6.QtWidgets import QApplication

    from app.ui import main_window

    monkeypatch.setattr(main_window, "WORKER_STOP_TIMEOUT_MS", 0)
    busy = {"running": True}
    monkeypatch.setattr(window._worker, "isRunning", lambda: busy["running"])
    monkeypatch.setattr(window._worker, "wait", lambda *_: not busy["running"])
    quits = []
    monkeypatch.setattr(QApplication, "quit", staticmethod(lambda: quits.append(True)))
    window.show()
    window.close()
    assert window.isHidden() and window._closing and not quits  # deferred, not destroyed
    busy["running"] = False
    window._worker.finished.emit()
    QApplication.processEvents()
    assert quits == [True]

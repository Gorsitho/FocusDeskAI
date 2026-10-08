import faulthandler
import logging
import os
import sys
import threading

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app import logging_setup  # noqa: E402
from app.config import paths  # noqa: E402
from app.logging_setup import StreamToLogger, setup_logging  # noqa: E402


@pytest.fixture
def isolated_logging(tmp_path, monkeypatch):
    """Run setup_logging into a temp folder and restore global state afterwards."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    saved = (sys.stdout, sys.stderr, sys.excepthook, threading.excepthook)
    root = logging.getLogger()
    saved_level = root.level
    yield tmp_path
    for handler in [h for h in root.handlers if getattr(h, "_focusdesk", False)]:
        root.removeHandler(handler)
        handler.close()
    root.setLevel(saved_level)
    sys.stdout, sys.stderr, sys.excepthook, threading.excepthook = saved
    faulthandler.disable()
    if logging_setup._native_file is not None:
        logging_setup._native_file.close()
        logging_setup._native_file = None
    logging.captureWarnings(False)


def _read(path):
    for handler in logging.getLogger().handlers:
        handler.flush()
    return path.read_text(encoding="utf-8")


def test_default_log_location_is_localappdata(isolated_logging):
    path = setup_logging(console=False)
    assert path == isolated_logging / "FocusDeskAI" / "FocusDeskAI.log"
    assert path.exists()
    assert paths.sessions_dir() == isolated_logging / "FocusDeskAI" / "sessions"


def test_messages_have_timestamps_and_thread_names(isolated_logging):
    path = setup_logging(console=False)
    logging.getLogger("app.test").info("hello log")
    line = next(line for line in _read(path).splitlines() if "hello log" in line)
    # e.g. "2026-10-08 14:03:12.345 INFO     [MainThread] app.test: hello log"
    assert line[:4].isdigit() and line[10] == " " and line[19] == "."
    assert "INFO" in line and "[MainThread]" in line and "app.test" in line


def test_exceptions_are_logged_with_traceback(isolated_logging):
    path = setup_logging(console=False)
    try:
        raise ValueError("boom in detector")
    except ValueError:
        logging.getLogger("app.vision").exception("Detector failed")
    text = _read(path)
    assert "Detector failed" in text
    assert "Traceback (most recent call last)" in text and "ValueError: boom in detector" in text


def test_windowed_build_without_console_streams(isolated_logging, monkeypatch):
    # PyInstaller console=False: sys.stdout / sys.stderr are None.
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    path = setup_logging()
    assert isinstance(sys.stdout, StreamToLogger) and isinstance(sys.stderr, StreamToLogger)
    # Libraries that print (Ultralytics, tqdm) must not crash and should end up in the log.
    print("Ultralytics says hi")
    sys.stderr.write("\r 10%|#  |\r 50%|#####  |\r100%|##########|\n")
    text = _read(path)
    assert "Ultralytics says hi" in text
    assert "100%|##########|" in text and "10%|#" not in text  # progress bar collapsed
    # No console handler may point at the replaced streams (that would recurse).
    assert not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
                   for h in logging.getLogger().handlers if getattr(h, "_focusdesk", False))


def test_uncaught_exceptions_are_logged(isolated_logging):
    path = setup_logging(console=False)
    try:
        raise RuntimeError("unexpected main-thread error")
    except RuntimeError:
        sys.excepthook(*sys.exc_info())
    text = _read(path)
    assert "CRITICAL" in text and "Unhandled exception" in text
    assert "RuntimeError: unexpected main-thread error" in text


def test_thread_exceptions_are_logged(isolated_logging):
    path = setup_logging(console=False)

    def fail():
        raise KeyError("worker thread failure")

    thread = threading.Thread(target=fail, name="YoloLoader")
    thread.start()
    thread.join()
    text = _read(path)
    assert "Unhandled exception in thread YoloLoader" in text
    assert "KeyError: 'worker thread failure'" in text


def test_setup_is_idempotent(isolated_logging):
    setup_logging(console=False)
    setup_logging(console=False)
    ours = [h for h in logging.getLogger().handlers if getattr(h, "_focusdesk", False)]
    assert len(ours) == 1


def test_analysis_worker_crash_is_logged_and_reported(isolated_logging, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from app.config.settings import Settings
    from app.ui.main_window import AnalysisWorker

    QApplication.instance() or QApplication([])
    path = setup_logging(console=False)
    worker = AnalysisWorker(Settings())

    def explode():
        raise RuntimeError("camera driver exploded")

    monkeypatch.setattr(worker, "_run", explode)
    statuses = []
    worker.status_changed.connect(statuses.append)
    worker.run()  # run synchronously; must not raise
    text = _read(path)
    assert "Analysis worker crashed" in text and "RuntimeError: camera driver exploded" in text
    assert statuses and statuses[-1][0] == "status.worker_error"


def test_failed_yolo_load_is_logged_with_traceback(isolated_logging, monkeypatch):
    import dataclasses

    from app.config.settings import DetectionSettings, ModelSettings
    from app.vision import object_detection

    path = setup_logging(console=False)

    class BrokenYOLO:
        def __init__(self, *_):
            raise OSError("weights unreadable")

    import ultralytics
    monkeypatch.setattr(ultralytics, "YOLO", BrokenYOLO)
    models = dataclasses.replace(ModelSettings(), yolo_weights_path=isolated_logging / "missing.pt")
    detector = object_detection.ObjectDetector(models, DetectionSettings())
    assert not detector.available
    text = _read(path)
    assert "Object detection disabled" in text and "OSError: weights unreadable" in text


def test_frozen_build_downloads_models_to_a_writable_folder(isolated_logging, monkeypatch, tmp_path):
    from app.config import settings as settings_module

    monkeypatch.setattr(paths, "is_frozen", lambda: True)
    monkeypatch.setattr(settings_module, "MODELS_DIR", tmp_path / "read_only_bundle")
    target = settings_module.model_path("yolo11n.pt")
    assert target == isolated_logging / "FocusDeskAI" / "models" / "yolo11n.pt"
    # A bundled model is used when present.
    (tmp_path / "read_only_bundle").mkdir()
    (tmp_path / "read_only_bundle" / "yolo11n.pt").write_bytes(b"x")
    assert settings_module.model_path("yolo11n.pt") == tmp_path / "read_only_bundle" / "yolo11n.pt"

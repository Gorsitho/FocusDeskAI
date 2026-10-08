"""Main window and the background thread that runs the vision pipeline."""

import dataclasses
import logging
import threading
import time

from pathlib import Path

import cv2
from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.config.settings import Settings, settings
from app.config.user_settings import UserSettings, apply_user_settings, save_user_settings
from app.features.feature_pipeline import FeaturePipeline, FocusState, FrameAnalysis
from app.features.state_timers import StateTimers
from app.ui.camera_widget import CameraWidget, bgr_to_qimage, draw_overlays
from app.ui.dashboard import Dashboard
from app.ui.settings_dialog import SettingsDialog
from app.ui.sound import DistractionSound
from app.vision.camera import Camera
from app.vision.face_detection import FaceDetector, FaceResult
from app.vision.object_detection import ObjectDetector, ObjectResult
from app.vision.pose_detection import PoseDetector, PoseResult

logger = logging.getLogger(__name__)


class AnalysisWorker(QThread):
    """Captures frames and runs detection + features off the GUI thread."""

    frame_ready = Signal(QImage)
    analysis_ready = Signal(object)  # FrameAnalysis
    camera_unavailable = Signal(str)
    status_changed = Signal(str)
    phone_detection_available = Signal(bool)

    def __init__(self, config: Settings, parent=None):
        super().__init__(parent)
        self._config = config
        self._pending_config: Settings | None = None
        self._object_detector: ObjectDetector | None = None
        self._unavailable: list[str] = []

    def apply_config(self, config: Settings) -> None:
        """Called from the GUI thread; picked up before the next frame is analysed."""
        self._pending_config = config

    def run(self) -> None:
        cfg = self._config
        self.status_changed.emit("Loading vision models…")
        face_detector = self._create(FaceDetector, "face")
        pose_detector = self._create(PoseDetector, "pose")

        pipeline = FeaturePipeline(cfg)
        camera = Camera(cfg.camera)
        objects = ObjectResult()
        frame_index = 0
        last_ts_ms = -1
        clock_start = time.monotonic()
        yolo_loader = threading.Thread(target=self._load_object_detector, daemon=True)

        try:
            while not self.isInterruptionRequested():
                if not camera.is_open:
                    if not camera.open():
                        self.camera_unavailable.emit(f"Camera {cfg.camera.index} not available. Retrying…")
                        self._sleep(cfg.camera.reconnect_interval_s)
                        continue
                    self._emit_running_status()

                frame = camera.read()
                if frame is None:
                    camera.release()
                    continue

                now = time.monotonic() - clock_start
                # MediaPipe VIDEO mode requires strictly increasing timestamps.
                ts_ms = max(int(now * 1000), last_ts_ms + 1)
                last_ts_ms = ts_ms

                face = face_detector.detect(frame, ts_ms) if face_detector else FaceResult(detected=False)
                pose = pose_detector.detect(frame, ts_ms) if pose_detector else PoseResult(detected=False)
                object_detector = self._object_detector
                if object_detector and frame_index % cfg.detection.yolo_every_n_frames == 0:
                    objects = object_detector.detect(frame)
                frame_index += 1
                if frame_index == 1:
                    # Importing torch for YOLO can take many seconds and holds the DLL loader
                    # lock on Windows, so it starts only once the camera is already streaming.
                    yolo_loader.start()

                pending, self._pending_config = self._pending_config, None
                if pending is not None:
                    pipeline.reconfigure(pending)

                height, width = frame.shape[:2]
                analysis = pipeline.process(now, (width, height), face, pose, objects)

                preview = draw_overlays(frame, face, pose, objects, cfg.detection.min_landmark_visibility, analysis)
                # Mirror the preview so it behaves like a mirror; detection uses the raw frame.
                self.frame_ready.emit(bgr_to_qimage(cv2.flip(preview, 1)))
                self.analysis_ready.emit(analysis)
        finally:
            camera.release()
            for detector in (face_detector, pose_detector):
                if detector is not None:
                    detector.close()

    def _create(self, detector_cls, name: str):
        try:
            return detector_cls(self._config.models, self._config.detection)
        except Exception:  # noqa: BLE001 - keep the app running without this detector
            logger.exception("%s detection unavailable", name)
            self._unavailable.append(name)
            return None

    def _load_object_detector(self) -> None:
        detector = ObjectDetector(self._config.models, self._config.detection)
        if detector.available:
            self._object_detector = detector
        else:
            self._unavailable.append("phone")
        self.phone_detection_available.emit(detector.available)
        self._emit_running_status()

    def _emit_running_status(self) -> None:
        if self._unavailable:
            self.status_changed.emit(f"Running — unavailable: {', '.join(self._unavailable)} detection")
        else:
            self.status_changed.emit("Running")

    def _sleep(self, seconds: float) -> None:
        # Sleep in short steps so the thread stops promptly when the window closes.
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and not self.isInterruptionRequested():
            self.msleep(50)


class MainWindow(QMainWindow):
    def __init__(
        self,
        config: Settings = settings,
        user_settings: UserSettings | None = None,
        settings_path: Path | None = None,
    ):
        super().__init__()
        self.setWindowTitle("FocusDesk AI")
        self.resize(1200, 760)
        self._base_config = config
        self._user = (user_settings or UserSettings()).normalized()
        self._settings_path = settings_path
        self._analysis_state: FocusState | None = None
        self._on_break = False
        self._timers = StateTimers(time.monotonic())
        self._sound = DistractionSound(self._user.sound_enabled, self._user.sound_volume, self)

        title = QLabel("FocusDesk AI")
        title.setObjectName("AppTitle")
        self._break_button = QPushButton("☕ Break")
        self._break_button.setCheckable(True)
        self._break_button.setToolTip("Pause distraction detection while you take a break")
        self._break_button.toggled.connect(self._set_break)
        new_session = QPushButton("↺ New session")
        new_session.setToolTip("Reset the session timers")
        new_session.clicked.connect(self._confirm_new_session)
        self._timers_button = QPushButton()
        self._timers_button.setCheckable(True)
        self._timers_button.toggled.connect(self._set_timers_visible)
        settings_button = QPushButton("⚙ Settings")
        settings_button.clicked.connect(self._open_settings)

        header = QHBoxLayout()
        header.setSpacing(8)
        header.addWidget(title)
        header.addStretch(1)
        for button in (self._break_button, new_session, self._timers_button, settings_button):
            header.addWidget(button)

        self._camera_view = CameraWidget()
        self._dashboard = Dashboard()
        dashboard_scroll = QScrollArea()
        dashboard_scroll.setWidget(self._dashboard)
        dashboard_scroll.setWidgetResizable(True)
        dashboard_scroll.setMinimumWidth(300)
        self._status = QLabel("")
        self._status.setObjectName("StatusBar")

        content = QHBoxLayout()
        content.setSpacing(16)
        content.addWidget(self._camera_view, stretch=3)
        content.addWidget(dashboard_scroll, stretch=1)

        root = QVBoxLayout()
        root.setContentsMargins(20, 16, 20, 12)
        root.setSpacing(12)
        root.addLayout(header)
        root.addLayout(content, stretch=1)
        root.addWidget(self._status)

        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        self._timers_button.setChecked(self._user.show_state_timers)
        self._set_timers_visible(self._user.show_state_timers)

        self._worker = AnalysisWorker(apply_user_settings(config, self._user), self)
        self._worker.frame_ready.connect(self._camera_view.set_frame)
        self._worker.analysis_ready.connect(self._on_analysis)
        self._worker.camera_unavailable.connect(self._on_camera_unavailable)
        self._worker.status_changed.connect(self._status.setText)
        self._worker.phone_detection_available.connect(self._dashboard.set_phone_detection_available)
        self._worker.start()

        # Timers tick on their own clock so they keep counting between frames.
        self._tick = QTimer(self)
        self._tick.setInterval(250)
        self._tick.timeout.connect(self._refresh_timers)
        self._tick.start()

    @property
    def displayed_state(self) -> FocusState | None:
        return FocusState.BREAK if self._on_break else self._analysis_state

    def _on_analysis(self, analysis: FrameAnalysis) -> None:
        self._analysis_state = analysis.state
        self._dashboard.update_analysis(analysis, FocusState.BREAK if self._on_break else None)
        self._sync_state()

    def _on_camera_unavailable(self, message: str) -> None:
        self._analysis_state = None
        self._camera_view.show_message(message)
        self._dashboard.clear()
        self._status.setText(message)
        self._sync_state()

    def _sync_state(self) -> None:
        state = self.displayed_state
        if state is not self._timers.current:
            self._timers.update(time.monotonic(), state)
            self._refresh_timers()
        self._sound.set_active(state is FocusState.DISTRACTED)

    def _refresh_timers(self) -> None:
        self._dashboard.update_timers(self._timers.totals(time.monotonic()), self._timers.current)

    def _set_break(self, on_break: bool) -> None:
        self._on_break = on_break
        self._break_button.setText("▶ Resume" if on_break else "☕ Break")
        if on_break:
            self._dashboard.show_state(FocusState.BREAK, "Monitoring paused while you take a break")
        elif self._analysis_state is not None:
            self._dashboard.show_state(self._analysis_state)
        self._sync_state()

    def _confirm_new_session(self) -> None:
        answer = QMessageBox.question(
            self, "New session", "Start a new session? All state timers will be reset to 00:00."
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.start_new_session()

    def start_new_session(self) -> None:
        now = time.monotonic()
        self._timers.reset(now)
        self._timers.update(now, self.displayed_state)
        self._refresh_timers()

    def _set_timers_visible(self, visible: bool) -> None:
        self._dashboard.set_timers_visible(visible)
        self._timers_button.setText("⏱ Hide timers" if visible else "⏱ Show timers")
        if visible != self._user.show_state_timers:
            self._user = dataclasses.replace(self._user, show_state_timers=visible)
            self._save_user_settings()

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self._user, self)
        dialog.preview_sound.connect(self._sound.preview)
        if dialog.exec() == SettingsDialog.DialogCode.Accepted:
            self._user = dialog.result_settings()
            self._save_user_settings()
            self._worker.apply_config(apply_user_settings(self._base_config, self._user))
            self._sound.set_enabled(self._user.sound_enabled)
        # The "Test" button may have changed the volume; apply the saved one.
        self._sound.set_volume(self._user.sound_volume)

    def _save_user_settings(self) -> None:
        if self._settings_path is not None:
            save_user_settings(self._user, self._settings_path)

    def closeEvent(self, event) -> None:
        self._tick.stop()
        self._sound.stop()
        self._worker.requestInterruption()
        if not self._worker.wait(5000):
            logger.warning("Analysis worker did not stop in time")
        super().closeEvent(event)

"""
This file contains the main window of FocusDesk AI and the background worker.

* AnalysisWorker runs in its own thread, so the window never freezes. For every
  camera frame it runs the detectors (app/vision/) and the FeaturePipeline
  (app/features/feature_pipeline.py), and sends the image and the result to the window.
* MainWindow shows the camera image (camera_widget.py) and the side panel
  (dashboard.py). It also handles the START / STOP SESSION button, the Break
  button, the Settings window, the distraction sound and the session log files.

app/main.py creates it after the setup window.
"""

import dataclasses
import logging
import threading
import time
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.config.paths import log_file_path, sessions_dir
from app.config.settings import Settings, settings
from app.config.user_settings import UserSettings, apply_user_settings, save_user_settings
from app.data.session_log import SessionLogStore
from app.features.feature_pipeline import FeaturePipeline, FocusState, FrameAnalysis
from app.features.session import SessionController
from app.features.state_timers import format_duration
from app.ui import i18n
from app.ui.camera_widget import CameraWidget, bgr_to_qimage, draw_overlays
from app.ui.dashboard import Dashboard, describe_reason
from app.ui.i18n import tr
from app.ui.settings_dialog import SettingsDialog
from app.ui.sound import DistractionSound
from app.vision.camera import Camera
from app.vision.face_detection import FaceDetector, FaceResult
from app.vision.object_detection import ObjectDetector, ObjectResult, hand_regions
from app.vision.pose_detection import PoseDetector, PoseResult

logger = logging.getLogger(__name__)

WORKER_STOP_TIMEOUT_MS = 10000


def compose_preview(frame, face, pose, objects, min_visibility, analysis, show_landmarks: bool):
    """Camera image for the UI: mirrored, with detection markers only when enabled."""
    if show_landmarks:
        frame = draw_overlays(frame, face, pose, objects, min_visibility, analysis)
    # Mirror the preview so it behaves like a mirror; detection uses the raw frame.
    return cv2.flip(frame, 1)


class AnalysisWorker(QThread):
    """Captures frames and runs detection + features off the GUI thread.

    Status messages are emitted as (translation key, params) so the GUI can show
    them in the current language.
    """

    frame_ready = Signal(QImage)
    analysis_ready = Signal(object)  # FrameAnalysis
    camera_unavailable = Signal(object)  # (key, params)
    status_changed = Signal(object)  # (key, params)
    phone_detection_available = Signal(bool)

    def __init__(self, config: Settings, parent=None, show_landmarks: bool = False):
        super().__init__(parent)
        self.setObjectName("AnalysisWorker")
        self._config = config
        self._pending_config: Settings | None = None
        self._object_detector: ObjectDetector | None = None
        self._unavailable: list[str] = []
        self._show_landmarks = show_landmarks
        self._last_logged: tuple | None = None

    def apply_config(self, config: Settings) -> None:
        """Called from the GUI thread; picked up before the next frame is analysed."""
        self._pending_config = config

    def set_show_landmarks(self, show: bool) -> None:
        """Only affects drawing; detection always runs."""
        self._show_landmarks = show

    @property
    def show_landmarks(self) -> bool:
        return self._show_landmarks

    def run(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            threading.current_thread().name = "AnalysisWorker"  # shown in the log
        logger.info("Analysis worker started")
        try:
            self._run()
        except Exception:  # noqa: BLE001 - report instead of dying silently
            logger.exception("Analysis worker crashed")
            self.status_changed.emit(("status.worker_error", {"path": str(log_file_path())}))
        finally:
            logger.info("Analysis worker stopped")

    def _run(self) -> None:
        """Main loop: read a frame, run the detectors and the pipeline, and send the results."""
        cfg = self._config
        self.status_changed.emit(("status.loading", {}))
        face_detector = self._create(FaceDetector, "face")
        pose_detector = self._create(PoseDetector, "pose")

        pipeline = FeaturePipeline(cfg)
        camera = Camera(cfg.camera)
        objects = ObjectResult()
        frame_index = 0
        frames_since_yolo = cfg.detection.yolo_every_n_frames
        phone_recent = False
        last_ts_ms = -1
        clock_start = time.monotonic()
        yolo_loader = threading.Thread(target=self._load_object_detector, name="YoloLoader", daemon=True)

        try:
            while not self.isInterruptionRequested():
                # 1. Make sure the camera is open; if not, wait and try again.
                if not camera.is_open:
                    if not camera.open():
                        self.camera_unavailable.emit(("status.camera_unavailable", {"index": cfg.camera.index}))
                        self._sleep(cfg.camera.reconnect_interval_s)
                        continue
                    self._emit_running_status()

                # 2. Read the newest frame.
                frame = camera.read()
                if frame is None:
                    logger.warning("Camera %d returned no frame; reopening", cfg.camera.index)
                    camera.release()
                    continue

                now = time.monotonic() - clock_start
                # MediaPipe VIDEO mode requires strictly increasing timestamps.
                ts_ms = max(int(now * 1000), last_ts_ms + 1)
                last_ts_ms = ts_ms

                # 3. Run the detectors. YOLO is slow, so it only runs on every n-th frame
                #    (more often while a phone is around); the frames in between reuse its
                #    last result. It also looks closer at the hands, where a phone in use is.
                face = face_detector.detect(frame, ts_ms) if face_detector else FaceResult(detected=False)
                pose = pose_detector.detect(frame, ts_ms) if pose_detector else PoseResult(detected=False)
                height, width = frame.shape[:2]
                object_detector = self._object_detector
                interval = (cfg.detection.yolo_every_n_frames_active if phone_recent
                            else cfg.detection.yolo_every_n_frames)
                if object_detector and frames_since_yolo >= interval:
                    rois = hand_regions(pose.landmarks if pose.detected else None, (width, height), cfg.detection)
                    objects = self._detect_objects(object_detector, frame, objects, rois)
                    frames_since_yolo = 0
                frames_since_yolo += 1
                frame_index += 1
                if frame_index == 1:
                    # Importing torch for YOLO can take many seconds and holds the DLL loader
                    # lock on Windows, so it starts only once the camera is already streaming.
                    yolo_loader.start()

                # 4. Use new settings from the Settings window, if there are any.
                pending, self._pending_config = self._pending_config, None
                if pending is not None:
                    pipeline.reconfigure(pending)
                    logger.info("Detection settings applied")

                # 5. Turn the detections into features and a focus state.
                analysis = pipeline.process(now, (width, height), face, pose, objects)
                phone_recent = analysis.features.phone.visible or bool(objects.phone_candidates)
                self._log_changes(analysis)

                # 6. Send the image and the result to the window (Qt signals are thread-safe).
                preview = compose_preview(frame, face, pose, objects, cfg.detection.min_landmark_visibility,
                                          analysis, self._show_landmarks)
                self.frame_ready.emit(bgr_to_qimage(preview))
                self.analysis_ready.emit(analysis)
        finally:
            camera.release()
            for detector in (face_detector, pose_detector):
                if detector is not None:
                    detector.close()

    def _detect_objects(self, detector: ObjectDetector, frame, previous: ObjectResult, rois) -> ObjectResult:
        try:
            return detector.detect(frame, rois)
        except Exception:  # noqa: BLE001 - lose phone detection, keep the rest running
            logger.exception("YOLO inference failed; phone detection disabled")
            self._object_detector = None
            self._unavailable.append("phone")
            self.phone_detection_available.emit(False)
            self._emit_running_status()
            return previous

    def _log_changes(self, analysis: FrameAnalysis) -> None:
        """Log behaviour changes (not frames) for diagnostics."""
        f = analysis.features
        snapshot = (analysis.state, f.phone.visible, f.phone.looking_at_phone, f.phone.in_hand, f.phone.at_ear,
                    f.face_detected, f.person_detected)
        if snapshot == self._last_logged:
            return
        previous, self._last_logged = self._last_logged, snapshot
        if previous is None or previous[0] is not analysis.state:
            reason = analysis.reason.code.value if analysis.reason else "-"
            logger.info("State %s (%s)", analysis.state.value, reason)
        if previous is None or previous[1:5] != snapshot[1:5]:
            logger.info("Phone visible=%s (conf %.2f), in use=%s, in hand=%s, at ear=%s, gaze-to-phone angle=%s",
                        f.phone.visible, f.phone.confidence, f.phone.looking_at_phone, f.phone.in_hand,
                        f.phone.at_ear, "-" if f.phone.gaze_angle is None else f"{f.phone.gaze_angle:.0f} deg")
        if previous is None or previous[5:] != snapshot[5:]:
            logger.debug("Face detected=%s, person detected=%s", f.face_detected, f.person_detected)

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
            logger.warning("Running without: %s", ", ".join(self._unavailable))
            self.status_changed.emit(("status.running_unavailable", {"detectors": list(self._unavailable)}))
        else:
            self.status_changed.emit(("status.running", {}))

    def _sleep(self, seconds: float) -> None:
        # Sleep in short steps so the thread stops promptly when the window closes.
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and not self.isInterruptionRequested():
            self.msleep(50)


def render_message(message: tuple[str, dict]) -> str:
    """Turn a status message (translation key, params) into text in the current language."""
    key, params = message
    if "detectors" in params:
        params = {"items": ", ".join(tr(f"detector.{name}") for name in params["detectors"])}
    return tr(key, **params)


class MainWindow(QMainWindow):
    """The main window: camera image, side panel and buttons.

    It starts the AnalysisWorker and reacts to its results.
    """
    def __init__(
        self,
        config: Settings = settings,
        user_settings: UserSettings | None = None,
        settings_path: Path | None = None,
        sessions_path: Path | None = None,
    ):
        super().__init__()
        self.setWindowTitle("FocusDesk AI")
        self.resize(1200, 800)
        self._base_config = config
        self._user = (user_settings or UserSettings()).normalized()
        i18n.set_language(self._user.language)
        self._settings_path = settings_path
        self._session_store = SessionLogStore(sessions_path if sessions_path is not None else sessions_dir())
        self._session = SessionController()
        self._analysis_state: FocusState | None = None
        self._last_analysis: FrameAnalysis | None = None
        self._on_break = False
        self._closing = False
        self._status_message: tuple[str, dict] | None = None
        self._camera_message: tuple[str, dict] | None = None
        self._sound = DistractionSound(self._user.sound_enabled, self._user.sound_volume, self)

        title = QLabel("FocusDesk AI")
        title.setObjectName("AppTitle")
        self._break_button = QPushButton()
        self._break_button.setCheckable(True)
        self._break_button.toggled.connect(self._set_break)
        self._timers_button = QPushButton()
        self._timers_button.setCheckable(True)
        self._timers_button.toggled.connect(self._set_timers_visible)
        self._settings_button = QPushButton()
        self._settings_button.clicked.connect(self._open_settings)

        header = QHBoxLayout()
        header.setSpacing(8)
        header.addWidget(title)
        header.addStretch(1)
        for button in (self._break_button, self._timers_button, self._settings_button):
            header.addWidget(button)

        # Session control: always visible above the dashboard.
        self._session_button = QPushButton()
        self._session_button.setObjectName("StartButton")
        self._session_button.clicked.connect(self._toggle_session)
        self._session_label = QLabel()
        self._session_label.setObjectName("SessionLabel")
        self._session_label.setWordWrap(True)
        session_panel = QFrame()
        session_panel.setObjectName("Card")
        session_layout = QVBoxLayout(session_panel)
        session_layout.setContentsMargins(12, 12, 12, 10)
        session_layout.addWidget(self._session_button)
        session_layout.addWidget(self._session_label)

        self._camera_view = CameraWidget()
        self._dashboard = Dashboard()
        dashboard_scroll = QScrollArea()
        dashboard_scroll.setWidget(self._dashboard)
        dashboard_scroll.setWidgetResizable(True)
        side = QVBoxLayout()
        side.setSpacing(12)
        side.addWidget(session_panel)
        side.addWidget(dashboard_scroll, 1)
        side_widget = QWidget()
        side_widget.setLayout(side)
        side_widget.setMinimumWidth(320)
        self._status = QLabel("")
        self._status.setObjectName("StatusBar")
        self._status.setWordWrap(True)

        content = QHBoxLayout()
        content.setSpacing(16)
        content.addWidget(self._camera_view, stretch=3)
        content.addWidget(side_widget, stretch=1)

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
        self._dashboard.set_timers_visible(self._user.show_state_timers)
        self.retranslate()

        self._worker = AnalysisWorker(apply_user_settings(config, self._user), self, self._user.show_landmarks)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.analysis_ready.connect(self._on_analysis)
        self._worker.camera_unavailable.connect(self._on_camera_unavailable)
        self._worker.status_changed.connect(self._on_status)
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

    @property
    def session(self) -> SessionController:
        return self._session

    def retranslate(self) -> None:
        self._break_button.setText(tr("main.resume") if self._on_break else tr("main.break"))
        self._break_button.setToolTip(tr("main.resume_tip") if self._on_break else tr("main.break_tip"))
        visible = self._timers_button.isChecked()
        self._timers_button.setText(tr("main.hide_timers") if visible else tr("main.show_timers"))
        self._timers_button.setToolTip(tr("main.timers_tip"))
        self._settings_button.setText(tr("main.settings"))
        self._settings_button.setToolTip(tr("main.settings_tip"))
        self._update_session_controls()
        self._dashboard.retranslate()
        if self._last_analysis is not None:
            self._dashboard.update_analysis(self._last_analysis, FocusState.BREAK if self._on_break else None)
        if self._status_message is not None:
            self._status.setText(render_message(self._status_message))
        if self._camera_message is not None:
            self._camera_view.show_message(render_message(self._camera_message))
        elif self._last_analysis is None:
            self._camera_view.show_message(tr("status.starting_camera"))

    # --- sessions ----------------------------------------------------------------------
    def _toggle_session(self) -> None:
        if self._session.active:
            self.stop_session()
        else:
            self.start_session()

    def start_session(self) -> None:
        if self._session.active:
            return
        number = self._session_store.next_number()
        self._session.set_state(self.displayed_state)
        self._session.start(number)
        logger.info("Session %04d started", number)
        self._update_session_controls()
        self._refresh_timers()

    def stop_session(self) -> Path | None:
        if not self._session.active:
            return None
        summary = self._session.stop()
        logger.info("Session %04d stopped after %.1f s", summary.number, summary.total_s)
        self._update_session_controls()
        self._refresh_timers()
        try:
            path, summary = self._session_store.write(summary)
        except OSError as exc:
            logger.exception("Could not save session %04d", summary.number)
            QMessageBox.warning(self, tr("error.title"), tr("error.save_session", error=exc))
            return None
        self._on_status(("session.saved", {"n": f"{summary.number:04d}", "path": str(path)}))
        return path

    def _update_session_controls(self) -> None:
        active = self._session.active
        self._session_button.setText(tr("main.stop") if active else tr("main.start"))
        self._session_button.setToolTip(tr("main.stop_tip") if active else tr("main.start_tip"))
        self._session_button.setProperty("running", active)
        self._session_button.style().unpolish(self._session_button)
        self._session_button.style().polish(self._session_button)
        if active:
            self._session_label.setText(tr("session.active", n=f"{self._session.number:04d}",
                                           elapsed=format_duration(self._session.elapsed())))
        else:
            self._session_label.setText(tr("session.none"))

    # --- analysis ----------------------------------------------------------------------
    def _on_frame(self, image: QImage) -> None:
        self._camera_message = None
        self._camera_view.set_frame(image)

    def _on_status(self, message: tuple[str, dict]) -> None:
        self._status_message = message
        self._status.setText(render_message(message))

    def _on_analysis(self, analysis: FrameAnalysis) -> None:
        self._last_analysis = analysis
        self._analysis_state = analysis.state
        self._dashboard.update_analysis(analysis, FocusState.BREAK if self._on_break else None)
        self._sync_state()

    def _on_camera_unavailable(self, message: tuple[str, dict]) -> None:
        self._analysis_state = None
        self._last_analysis = None
        self._camera_message = message
        self._camera_view.show_message(render_message(message))
        self._dashboard.clear()
        self._on_status(message)
        self._sync_state()

    def _sync_state(self) -> None:
        """Tell the session timers and the sound which state is shown now."""
        state = self.displayed_state
        changed = self._session.active and state is not self._session.current
        self._session.set_state(state)
        if changed:
            self._refresh_timers()
        self._sound.set_active(state is FocusState.DISTRACTED)

    def _refresh_timers(self) -> None:
        self._dashboard.update_timers(self._session.totals(), self._session.current)
        if self._session.active:
            self._update_session_controls()

    def _set_break(self, on_break: bool) -> None:
        self._on_break = on_break
        logger.info("Break %s", "started" if on_break else "ended")
        self.retranslate()
        if on_break:
            self._dashboard.show_state(FocusState.BREAK, tr("reason.break"))
        elif self._last_analysis is not None:
            self._dashboard.show_state(self._last_analysis.state, describe_reason(self._last_analysis.reason))
        self._sync_state()

    def _set_timers_visible(self, visible: bool) -> None:
        self._dashboard.set_timers_visible(visible)
        self._timers_button.setText(tr("main.hide_timers") if visible else tr("main.show_timers"))
        if visible != self._user.show_state_timers:
            self._user = dataclasses.replace(self._user, show_state_timers=visible)
            self._save_user_settings()

    # --- settings ----------------------------------------------------------------------
    def _open_settings(self) -> None:
        dialog = SettingsDialog(self._user, self)
        dialog.preview_sound.connect(self._sound.preview)
        if dialog.exec() == SettingsDialog.DialogCode.Accepted:
            self.apply_user_settings(dialog.result_settings())
        else:
            # The dialog may have previewed another language or volume; restore ours.
            i18n.set_language(self._user.language)
            self._sound.set_volume(self._user.sound_volume)

    def apply_user_settings(self, user: UserSettings) -> None:
        """Use new preferences everywhere: language, detection, sound, and save them to the file."""
        self._user = user.normalized()
        logger.info("Settings applied: %s", self._user)
        i18n.set_language(self._user.language)
        self.retranslate()
        self._save_user_settings()
        self._worker.apply_config(apply_user_settings(self._base_config, self._user))
        self._worker.set_show_landmarks(self._user.show_landmarks)
        self._sound.set_enabled(self._user.sound_enabled)
        self._sound.set_volume(self._user.sound_volume)

    def _save_user_settings(self) -> None:
        if self._settings_path is None:
            return
        if not save_user_settings(self._user, self._settings_path):
            QMessageBox.warning(self, tr("error.title"), tr("error.save_settings", path=self._settings_path))

    def closeEvent(self, event) -> None:
        """Save a running session and stop the worker thread before the window closes."""
        if not self._closing:
            self._closing = True
            # An open session is completed and saved rather than lost.
            self.stop_session()
            self._tick.stop()
            self._sound.stop()
            self._worker.requestInterruption()
        if self._worker.isRunning() and not self._worker.wait(WORKER_STOP_TIMEOUT_MS):
            # E.g. YOLO's first inference can take several seconds. Destroying a running
            # QThread aborts the process, so finish closing once the worker has stopped.
            logger.warning("Analysis worker still busy; closing when it finishes")
            self.hide()
            self._worker.finished.connect(self._finish_close, Qt.ConnectionType.QueuedConnection)
            event.ignore()
            return
        super().closeEvent(event)

    def _finish_close(self) -> None:
        logger.info("Analysis worker finished; closing")
        self.close()
        QApplication.quit()

"""
This file starts FocusDesk AI. It is the entry point of the application.

What it does, in this order:
1. It sets up logging first, so every later error is saved to the log file.
2. It loads the user's saved preferences (app/config/user_settings.py).
3. It shows the setup window (app/ui/settings_dialog.py): study method and desk layout.
4. It saves the choices and opens the main window (app/ui/main_window.py).

Run it from the project root with either:
    python -m app.main
    python app/main.py
"""

import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Launched as a script: make the `app` package importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Logging first: in a windowed (console=False) build nothing else may print safely before this.
from app.logging_setup import log_startup, setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger("focusdesk")

from PySide6.QtCore import QtMsgType, qInstallMessageHandler  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from app.config.settings import settings  # noqa: E402
from app.config.user_settings import default_settings_path, load_user_settings, save_user_settings  # noqa: E402
from app.ui.i18n import tr  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402
from app.ui.settings_dialog import SetupDialog  # noqa: E402
from app.ui.styles import APP_STYLESHEET  # noqa: E402

_QT_LEVELS = {
    QtMsgType.QtDebugMsg: logging.DEBUG,
    QtMsgType.QtInfoMsg: logging.INFO,
    QtMsgType.QtWarningMsg: logging.WARNING,
    QtMsgType.QtCriticalMsg: logging.ERROR,
    QtMsgType.QtFatalMsg: logging.CRITICAL,
}


def _qt_message_handler(msg_type, context, message) -> None:
    """Send Qt's own warnings and messages to our log."""
    logging.getLogger("qt").log(_QT_LEVELS.get(msg_type, logging.WARNING), message)


def main() -> int:
    """Start the app: setup window first, then the main window. Returns the exit code."""
    log_startup()
    qInstallMessageHandler(_qt_message_handler)
    exit_code = 1
    try:
        app = QApplication(sys.argv)
        app.setApplicationName("FocusDesk AI")
        app.setStyleSheet(APP_STYLESHEET)

        settings_path = default_settings_path()
        user = load_user_settings(settings_path)
        logger.info("User settings loaded from %s", settings_path)
        setup = SetupDialog(user)
        if setup.exec() != SetupDialog.DialogCode.Accepted:
            logger.info("Setup window closed; quitting")
            exit_code = 0
            return exit_code
        user_settings = setup.result_settings()
        if not save_user_settings(user_settings, settings_path):
            QMessageBox.warning(None, tr("error.title"), tr("error.save_settings", path=settings_path))

        window = MainWindow(settings, user_settings, settings_path)
        window.show()
        exit_code = app.exec()
        return exit_code
    except Exception:
        logger.critical("Fatal error during startup or run", exc_info=True)
        raise
    finally:
        logger.info("FocusDesk AI shutting down (exit code %s)", exit_code)
        logging.shutdown()


if __name__ == "__main__":
    sys.exit(main())

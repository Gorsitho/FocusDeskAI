"""FocusDesk AI entry point.

Run from the project root with either:
    python -m app.main
    python app/main.py
"""

import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Launched as a script: make the `app` package importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication

from app.config.settings import settings
from app.config.user_settings import default_settings_path, load_user_settings, save_user_settings
from app.ui.main_window import MainWindow
from app.ui.settings_dialog import SetupDialog
from app.ui.styles import APP_STYLESHEET


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = QApplication(sys.argv)
    app.setApplicationName("FocusDesk AI")
    app.setStyleSheet(APP_STYLESHEET)

    settings_path = default_settings_path()
    setup = SetupDialog(load_user_settings(settings_path))
    if setup.exec() != SetupDialog.DialogCode.Accepted:
        return 0
    user_settings = setup.result_settings()
    save_user_settings(user_settings, settings_path)

    window = MainWindow(settings, user_settings, settings_path)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

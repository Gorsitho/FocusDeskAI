"""Startup setup window and the Settings window."""

import dataclasses

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.config.user_settings import MAX_SENSITIVITY, MIN_SENSITIVITY, UserSettings
from app.ui.workspace_widget import WorkspaceEditor


class SecondsInput(QWidget):
    """Number of seconds with explicit −/+ buttons (styled spin arrows render poorly)."""

    def __init__(self, value: float, low: float, high: float, step: float, parent=None):
        super().__init__(parent)
        self._spin = QDoubleSpinBox()
        self._spin.setRange(low, high)
        self._spin.setSingleStep(step)
        self._spin.setDecimals(1 if step < 1 else 0)
        self._spin.setSuffix(" s")
        self._spin.setValue(value)
        self._spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self._spin.setFixedWidth(76)
        minus, plus = QPushButton("−"), QPushButton("+")
        for button, direction in ((minus, -1), (plus, 1)):
            button.setObjectName("StepButton")
            button.setAutoRepeat(True)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(lambda _=False, d=direction: self._spin.stepBy(d))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(minus)
        layout.addWidget(self._spin)
        layout.addWidget(plus)

    def value(self) -> float:
        return self._spin.value()

    def setValue(self, value: float) -> None:
        self._spin.setValue(value)


def _slider(value: int, low: int, high: int) -> QSlider:
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(low, high)
    slider.setValue(value)
    return slider


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("Hint")
    label.setWordWrap(True)
    return label


class SetupDialog(QDialog):
    """Shown at startup to describe the desk layout before monitoring begins."""

    def __init__(self, user: UserSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("FocusDesk AI — Setup")
        self._user = user

        title = QLabel("Welcome to FocusDesk AI")
        title.setObjectName("DialogTitle")
        intro = _hint(
            "Tell FocusDesk how your desk is arranged. Looking at any of your monitors counts as "
            "focused; the camera position tells it which head angles that means. "
            "You can change this later under ⚙ Settings."
        )
        self._workspace = WorkspaceEditor(user.monitor_count, user.camera_position)

        buttons = QDialogButtonBox()
        start = buttons.addButton("Start monitoring", QDialogButtonBox.ButtonRole.AcceptRole)
        start.setObjectName("Primary")
        start.setDefault(True)
        buttons.addButton("Quit", QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 18)
        root.setSpacing(12)
        root.addWidget(title)
        root.addWidget(intro)
        root.addWidget(self._workspace)
        root.addWidget(buttons)
        self.setMinimumWidth(500)

    def result_settings(self) -> UserSettings:
        return dataclasses.replace(
            self._user,
            monitor_count=self._workspace.monitor_count,
            camera_position=self._workspace.camera_position,
        ).normalized()


class SettingsDialog(QDialog):
    preview_sound = Signal(int)  # volume percent

    def __init__(self, user: UserSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self._user = user

        # Workspace.
        self._workspace = WorkspaceEditor(user.monitor_count, user.camera_position)
        workspace_box = QGroupBox("Workspace")
        QVBoxLayout(workspace_box).addWidget(self._workspace)

        # Detection.
        self._distraction = SecondsInput(user.distraction_after_s, 1, 600, 0.5)
        self._phone = SecondsInput(user.phone_distraction_after_s, 1, 600, 0.5)
        self._idle = SecondsInput(user.idle_after_s, 5, 3600, 5)
        self._sensitivity = _slider(user.sensitivity, MIN_SENSITIVITY, MAX_SENSITIVITY)
        self._sensitivity.setPageStep(1)
        self._sensitivity.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._sensitivity_value = QLabel()
        self._sensitivity_value.setMinimumWidth(64)
        self._sensitivity.valueChanged.connect(self._update_sensitivity_label)
        self._update_sensitivity_label(user.sensitivity)
        sensitivity_row = QHBoxLayout()
        sensitivity_row.addWidget(self._sensitivity, 1)
        sensitivity_row.addWidget(self._sensitivity_value)

        detection_form = QFormLayout()
        detection_form.addRow("Looking away before distracted", self._distraction)
        detection_form.addRow("Looking at phone before distracted", self._phone)
        detection_form.addRow("No movement before idle", self._idle)
        detection_form.addRow("Detection sensitivity", sensitivity_row)
        detection_box = QGroupBox("Detection")
        detection_layout = QVBoxLayout(detection_box)
        detection_layout.addLayout(detection_form)
        detection_layout.addWidget(_hint(
            "Higher sensitivity narrows the head angles that count as looking at a monitor "
            "and catches phone use sooner. A visible phone alone never counts as distracted."
        ))

        # Sound.
        self._sound_enabled = QCheckBox("Play a soft sound while distracted")
        self._sound_enabled.setChecked(user.sound_enabled)
        self._volume = _slider(user.sound_volume, 0, 100)
        self._volume_value = QLabel()
        self._volume_value.setMinimumWidth(40)
        self._volume.valueChanged.connect(lambda v: self._volume_value.setText(f"{v} %"))
        self._volume_value.setText(f"{user.sound_volume} %")
        test = QPushButton("▶ Test")
        test.clicked.connect(lambda: self.preview_sound.emit(self._volume.value()))
        volume_row = QHBoxLayout()
        volume_row.addWidget(self._volume, 1)
        volume_row.addWidget(self._volume_value)
        volume_row.addWidget(test)
        self._sound_enabled.toggled.connect(self._volume.setEnabled)
        self._volume.setEnabled(user.sound_enabled)

        sound_form = QFormLayout()
        sound_form.addRow(self._sound_enabled)
        sound_form.addRow("Volume", volume_row)
        sound_box = QGroupBox("Sound")
        sound_box.setLayout(sound_form)

        buttons = QDialogButtonBox()
        save = buttons.addButton("Save", QDialogButtonBox.ButtonRole.AcceptRole)
        save.setObjectName("Primary")
        save.setDefault(True)
        buttons.addButton("Cancel", QDialogButtonBox.ButtonRole.RejectRole)
        defaults = buttons.addButton("Restore defaults", QDialogButtonBox.ButtonRole.ResetRole)
        defaults.clicked.connect(self._restore_defaults)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        columns = QHBoxLayout()
        columns.setSpacing(14)
        columns.addWidget(workspace_box, 1)
        right = QVBoxLayout()
        right.setSpacing(14)
        right.addWidget(detection_box)
        right.addWidget(sound_box)
        right.addStretch(1)
        right_widget = QWidget()
        right_widget.setLayout(right)
        right.setContentsMargins(0, 0, 0, 0)
        columns.addWidget(right_widget, 1)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(14)
        root.addLayout(columns)
        root.addWidget(buttons)

    def _update_sensitivity_label(self, value: int) -> None:
        word = "Low" if value <= 3 else "High" if value >= 8 else "Medium"
        self._sensitivity_value.setText(f"{value} · {word}")

    def _restore_defaults(self) -> None:
        defaults = UserSettings()
        # Workspace stays as is: it describes the physical desk, not a preference.
        self._distraction.setValue(defaults.distraction_after_s)
        self._phone.setValue(defaults.phone_distraction_after_s)
        self._idle.setValue(defaults.idle_after_s)
        self._sensitivity.setValue(defaults.sensitivity)
        self._sound_enabled.setChecked(defaults.sound_enabled)
        self._volume.setValue(defaults.sound_volume)

    def result_settings(self) -> UserSettings:
        return dataclasses.replace(
            self._user,
            monitor_count=self._workspace.monitor_count,
            camera_position=self._workspace.camera_position,
            distraction_after_s=self._distraction.value(),
            phone_distraction_after_s=self._phone.value(),
            idle_after_s=self._idle.value(),
            sensitivity=self._sensitivity.value(),
            sound_enabled=self._sound_enabled.isChecked(),
            sound_volume=self._volume.value(),
        ).normalized()

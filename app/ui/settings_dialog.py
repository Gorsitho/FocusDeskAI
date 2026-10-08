"""
This file contains the setup window (shown at start) and the Settings window.

* SetupDialog: step 1 asks for the study method (Computer, Tablet / Notebook, Mixed).
  Step 2 shows the monitor setup. Step 2 is skipped for Tablet / Notebook.
* SettingsDialog: lets the user change the study method, desk layout, waiting
  times, sensitivity, sound, display and language later.
* StudyMethodPicker: the three large cards for choosing the study method.
* SecondsInput: a field for seconds that accepts both "7.5" and "7,5".

Both windows return a UserSettings object (app/config/user_settings.py).
The desk layout editor comes from app/ui/workspace_widget.py.
They are opened by app/main.py and app/ui/main_window.py.
"""

import dataclasses
import re

from PySide6.QtCore import QLocale, Qt, Signal
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGraphicsOpacityEffect,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.config.settings import StudyMethod
from app.config.user_settings import MAX_SENSITIVITY, MIN_SENSITIVITY, UserSettings
from app.ui import i18n
from app.ui.i18n import LANGUAGE_NAMES, tr
from app.ui.workspace_widget import WorkspaceEditor

_NUMBER = re.compile(r"^\s*(\d*)(?:[.,](\d*))?\s*(?:s|sec|seg|sek)?\.?\s*$", re.IGNORECASE)


def parse_seconds(text: str) -> float | None:
    """Parse what the user typed: "7.5", "7,5", "12 s", "12s" ... -> float, or None."""
    match = _NUMBER.match(text)
    if not match or not (match.group(1) or match.group(2)):
        return None
    return float(f"{match.group(1) or 0}.{match.group(2) or 0}")


class SecondsSpinBox(QDoubleSpinBox):
    """Seconds field that accepts both decimal separators and never drops typed input.

    The stock QDoubleSpinBox follows the system locale, where "7,5" can be read
    as 75 (thousands separator) and values outside the range are silently
    discarded. Here the typed text itself is the source of truth.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLocale(QLocale.c())
        self.setGroupSeparatorShown(False)
        self.setKeyboardTracking(True)
        self.setCorrectionMode(QDoubleSpinBox.CorrectionMode.CorrectToNearestValue)

    def validate(self, text: str, pos: int):
        if parse_seconds(text) is not None:
            return QValidator.State.Acceptable, text, pos
        if _NUMBER.match(text):  # empty or a lone separator while typing
            return QValidator.State.Intermediate, text, pos
        return QValidator.State.Invalid, text, pos

    def valueFromText(self, text: str) -> float:
        value = parse_seconds(text)
        return self.value() if value is None else self._bounded(value)

    def textFromValue(self, value: float) -> str:
        text = f"{value:.{self.decimals()}f}"
        return text.replace(".", i18n.decimal_separator())

    def fixup(self, text: str) -> str:
        value = parse_seconds(text)
        return self.textFromValue(self.value() if value is None else self._bounded(value))

    def typed_value(self) -> float:
        """The value currently shown in the field, clamped to the allowed range."""
        value = parse_seconds(self.lineEdit().text())
        return self.value() if value is None else self._bounded(value)

    def _bounded(self, value: float) -> float:
        return max(self.minimum(), min(self.maximum(), round(value, self.decimals())))


class SecondsInput(QWidget):
    """A seconds field with explicit −/+ buttons (styled spin arrows render poorly)."""

    def __init__(self, value: float, low: float, high: float, step: float, parent=None):
        super().__init__(parent)
        self._spin = SecondsSpinBox()
        self._spin.setRange(low, high)
        self._spin.setSingleStep(step)
        self._spin.setDecimals(1 if step < 1 else 0)
        self._spin.setSuffix(" s")
        self._spin.setValue(value)
        self._spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self._spin.setFixedWidth(82)
        minus, plus = QPushButton("−"), QPushButton("+")
        for button, direction in ((minus, -1), (plus, 1)):
            button.setObjectName("StepButton")
            button.setAutoRepeat(True)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(lambda _=False, d=direction: self._step(d))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(minus)
        layout.addWidget(self._spin)
        layout.addWidget(plus)

    @property
    def spin_box(self) -> SecondsSpinBox:
        return self._spin

    def _step(self, direction: int) -> None:
        # Start from what is typed, not from the last committed value.
        self._spin.setValue(self._spin.typed_value())
        self._spin.stepBy(direction)

    def value(self) -> float:
        return self._spin.typed_value()

    def setValue(self, value: float) -> None:
        self._spin.setValue(value)

    def refresh(self) -> None:
        """Re-render with the current language's decimal separator."""
        self._spin.setValue(self._spin.typed_value())
        self._spin.lineEdit().setText(self._spin.textFromValue(self._spin.value()) + self._spin.suffix())

    def setToolTip(self, text: str) -> None:
        self._spin.setToolTip(text)


def _slider(value: int, low: int, high: int) -> QSlider:
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(low, high)
    slider.setValue(value)
    return slider


def _hint() -> QLabel:
    label = QLabel()
    label.setObjectName("Hint")
    label.setWordWrap(True)
    return label


def _language_combo(selected: str) -> QComboBox:
    combo = QComboBox()
    for code, name in LANGUAGE_NAMES.items():
        combo.addItem(name, code)
    combo.setCurrentIndex(max(combo.findData(selected), 0))
    return combo


_METHOD_ICONS = {StudyMethod.COMPUTER: "🖥️", StudyMethod.TABLET: "📓", StudyMethod.MIXED: "🖥️ 📓"}


class StudyMethodPicker(QWidget):
    """Three large selectable cards: Computer, Tablet / Notebook, Mixed."""

    changed = Signal(object)  # StudyMethod

    def __init__(self, method: StudyMethod, compact: bool = False, parent=None):
        super().__init__(parent)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._texts: dict[StudyMethod, tuple[QLabel, QLabel]] = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        for index, option in enumerate(StudyMethod):
            card = QPushButton()
            card.setObjectName("MethodCard")
            card.setCheckable(True)
            card.setMinimumHeight(90 if compact else 200)
            icon, title, description = QLabel(_METHOD_ICONS[option]), QLabel(), _hint()
            icon.setObjectName("MethodIcon")
            title.setObjectName("MethodTitle")
            title.setWordWrap(True)
            card_layout = QVBoxLayout(card)
            card_layout.setSpacing(6)
            for label in (icon, title, description):
                label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
                # Clicks on the texts must reach the card.
                label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
                card_layout.addWidget(label)
            card_layout.addStretch(1)
            description.setVisible(not compact)
            self._group.addButton(card, index)
            self._texts[option] = (title, description)
            layout.addWidget(card, 1)
        self.set_method(method)
        self._group.idClicked.connect(lambda _: self.changed.emit(self.method))
        self.retranslate()

    @property
    def method(self) -> StudyMethod:
        return list(StudyMethod)[max(self._group.checkedId(), 0)]

    def set_method(self, method: StudyMethod) -> None:
        self.card(method).setChecked(True)

    def card(self, method: StudyMethod) -> QPushButton:
        return self._group.button(list(StudyMethod).index(method))

    def retranslate(self) -> None:
        for option, (title, description) in self._texts.items():
            title.setText(tr(f"method.{option.value}"))
            description.setText(tr(f"method.{option.value}_desc"))
            self.card(option).setToolTip(tr(f"method.{option.value}_desc"))


class SetupDialog(QDialog):
    """Shown at startup: the study method first, then the desk layout when it is needed."""

    METHOD_PAGE, WORKSPACE_PAGE = 0, 1

    def __init__(self, user: UserSettings, parent=None):
        super().__init__(parent)
        self._user = user
        i18n.set_language(user.language)

        self._title = QLabel()
        self._title.setObjectName("DialogTitle")
        self._language = _language_combo(user.language)
        self._language.currentIndexChanged.connect(self._on_language)
        header = QHBoxLayout()
        header.addWidget(self._title, 1)
        header.addWidget(self._language)

        # Step 1: study method.
        self._method_title = QLabel()
        self._method_title.setObjectName("MethodTitle")
        self._method_intro = _hint()
        self._picker = StudyMethodPicker(user.study_method)
        self._picker.changed.connect(lambda _: self._update_buttons())
        method_page = QWidget()
        method_layout = QVBoxLayout(method_page)
        method_layout.setContentsMargins(0, 0, 0, 0)
        method_layout.setSpacing(12)
        method_layout.addWidget(self._method_title)
        method_layout.addWidget(self._method_intro)
        method_layout.addWidget(self._picker)
        method_layout.addStretch(1)

        # Step 2: monitor layout (skipped for tablet / notebook study).
        self._intro = _hint()
        self._workspace = WorkspaceEditor(user.workspace)
        workspace_page = QWidget()
        workspace_layout = QVBoxLayout(workspace_page)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(12)
        workspace_layout.addWidget(self._intro)
        workspace_layout.addWidget(self._workspace, 1)

        self._pages = QStackedWidget()
        self._pages.addWidget(method_page)
        self._pages.addWidget(workspace_page)

        buttons = QDialogButtonBox()
        self._start = buttons.addButton("", QDialogButtonBox.ButtonRole.AcceptRole)
        self._start.setObjectName("Primary")
        self._start.setDefault(True)
        self._back = buttons.addButton("", QDialogButtonBox.ButtonRole.ActionRole)
        self._quit = buttons.addButton("", QDialogButtonBox.ButtonRole.RejectRole)
        self._start.clicked.connect(self._on_primary)
        self._back.clicked.connect(lambda: self._show_page(self.METHOD_PAGE))
        buttons.rejected.connect(self.reject)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 18)
        root.setSpacing(12)
        root.addLayout(header)
        root.addWidget(self._pages, 1)
        root.addWidget(buttons)
        self.setMinimumSize(560, 640)
        self.retranslate()

    @property
    def workspace_editor(self) -> WorkspaceEditor:
        return self._workspace

    @property
    def method_picker(self) -> StudyMethodPicker:
        return self._picker

    @property
    def page(self) -> int:
        return self._pages.currentIndex()

    def _needs_workspace(self) -> bool:
        return self._picker.method is not StudyMethod.TABLET

    def _on_primary(self) -> None:
        if self.page == self.METHOD_PAGE and self._needs_workspace():
            self._show_page(self.WORKSPACE_PAGE)
        else:
            self.accept()

    def _show_page(self, page: int) -> None:
        self._pages.setCurrentIndex(page)
        self._update_buttons()

    def _update_buttons(self) -> None:
        on_method_page = self.page == self.METHOD_PAGE
        next_step = on_method_page and self._needs_workspace()
        self._start.setText(tr("setup.next") if next_step else tr("setup.start"))
        self._back.setVisible(not on_method_page)

    def _on_language(self) -> None:
        i18n.set_language(self._language.currentData())
        self.retranslate()

    def retranslate(self) -> None:
        self.setWindowTitle(tr("setup.title"))
        self._title.setText(tr("setup.welcome"))
        self._method_title.setText(tr("setup.method_title"))
        self._method_intro.setText(tr("setup.method_intro"))
        self._intro.setText(tr("setup.intro"))
        self._language.setToolTip(tr("settings.language_label"))
        self._back.setText(tr("setup.back"))
        self._quit.setText(tr("setup.quit"))
        self._picker.retranslate()
        self._workspace.retranslate()
        self._update_buttons()

    def result_settings(self) -> UserSettings:
        return dataclasses.replace(
            self._user, study_method=self._picker.method, workspace=self._workspace.workspace,
            language=self._language.currentData(),
        ).normalized()


class SettingsDialog(QDialog):
    """The Settings window. Changes are only used when the user presses Save."""
    preview_sound = Signal(int)  # volume percent

    def __init__(self, user: UserSettings, parent=None):
        super().__init__(parent)
        self._user = user

        # Study method.
        self._picker = StudyMethodPicker(user.study_method, compact=True)
        self._picker.changed.connect(lambda _: self._update_workspace_state())
        self._method_box = QGroupBox()
        QVBoxLayout(self._method_box).addWidget(self._picker)

        # Workspace.
        self._workspace = WorkspaceEditor(user.workspace)
        self._workspace_unneeded = _hint()
        self._workspace_box = QGroupBox()
        workspace_layout = QVBoxLayout(self._workspace_box)
        workspace_layout.addWidget(self._workspace_unneeded)
        workspace_layout.addWidget(self._workspace)

        # Detection.
        self._distraction = SecondsInput(user.distraction_after_s, 1, 600, 0.5)
        self._phone = SecondsInput(user.phone_distraction_after_s, 1, 600, 0.5)
        self._no_movement = SecondsInput(user.no_movement_away_after_s, 5, 3600, 5)
        self._sensitivity = _slider(user.sensitivity, MIN_SENSITIVITY, MAX_SENSITIVITY)
        self._sensitivity.setPageStep(1)
        self._sensitivity.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._sensitivity_value = QLabel()
        self._sensitivity_value.setMinimumWidth(72)
        self._sensitivity.valueChanged.connect(self._update_sensitivity_label)
        sensitivity_row = QHBoxLayout()
        sensitivity_row.addWidget(self._sensitivity, 1)
        sensitivity_row.addWidget(self._sensitivity_value)

        self._labels = {key: QLabel() for key in (
            "settings.distraction", "settings.phone", "settings.no_movement_away", "settings.sensitivity",
            "settings.volume", "settings.language_label",
        )}
        detection_form = QFormLayout()
        detection_form.addRow(self._labels["settings.distraction"], self._distraction)
        detection_form.addRow(self._labels["settings.phone"], self._phone)
        detection_form.addRow(self._labels["settings.no_movement_away"], self._no_movement)
        detection_form.addRow(self._labels["settings.sensitivity"], sensitivity_row)
        self._detection_hint = _hint()
        self._detection_box = QGroupBox()
        detection_layout = QVBoxLayout(self._detection_box)
        detection_layout.addLayout(detection_form)
        detection_layout.addWidget(self._detection_hint)

        # Sound.
        self._sound_enabled = QCheckBox()
        self._sound_enabled.setChecked(user.sound_enabled)
        self._volume = _slider(user.sound_volume, 0, 100)
        self._volume_value = QLabel(f"{user.sound_volume} %")
        self._volume_value.setMinimumWidth(40)
        self._volume.valueChanged.connect(lambda v: self._volume_value.setText(f"{v} %"))
        self._test = QPushButton()
        self._test.clicked.connect(lambda: self.preview_sound.emit(self._volume.value()))
        volume_row = QHBoxLayout()
        volume_row.addWidget(self._volume, 1)
        volume_row.addWidget(self._volume_value)
        volume_row.addWidget(self._test)
        self._sound_enabled.toggled.connect(self._volume.setEnabled)
        self._volume.setEnabled(user.sound_enabled)
        sound_form = QFormLayout()
        sound_form.addRow(self._sound_enabled)
        sound_form.addRow(self._labels["settings.volume"], volume_row)
        self._sound_box = QGroupBox()
        self._sound_box.setLayout(sound_form)

        # Display.
        self._show_landmarks = QCheckBox()
        self._show_landmarks.setChecked(user.show_landmarks)
        self._display_box = QGroupBox()
        QVBoxLayout(self._display_box).addWidget(self._show_landmarks)

        # Language.
        self._language = _language_combo(user.language)
        self._language.currentIndexChanged.connect(self._on_language)
        language_form = QFormLayout()
        language_form.addRow(self._labels["settings.language_label"], self._language)
        self._language_box = QGroupBox()
        self._language_box.setLayout(language_form)

        buttons = QDialogButtonBox()
        self._save = buttons.addButton("", QDialogButtonBox.ButtonRole.AcceptRole)
        self._save.setObjectName("Primary")
        self._save.setDefault(True)
        self._cancel = buttons.addButton("", QDialogButtonBox.ButtonRole.RejectRole)
        self._defaults = buttons.addButton("", QDialogButtonBox.ButtonRole.ResetRole)
        self._defaults.clicked.connect(self._restore_defaults)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(14)
        right.addWidget(self._detection_box)
        right.addWidget(self._sound_box)
        right.addWidget(self._display_box)
        right.addWidget(self._language_box)
        right.addStretch(1)
        right_widget = QWidget()
        right_widget.setLayout(right)
        columns = QHBoxLayout()
        columns.setSpacing(14)
        columns.addWidget(self._workspace_box, 6)
        columns.addWidget(right_widget, 5)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(14)
        root.addWidget(self._method_box)
        root.addLayout(columns, 1)
        root.addWidget(buttons)
        self._update_workspace_state()
        self.retranslate()

    @property
    def workspace_editor(self) -> WorkspaceEditor:
        return self._workspace

    @property
    def method_picker(self) -> StudyMethodPicker:
        return self._picker

    def _update_workspace_state(self) -> None:
        # The layout is kept for switching back, but tablet study does not use it.
        unneeded = self._picker.method is StudyMethod.TABLET
        # Disabling the whole section blocks every control; the fade also greys the
        # custom-drawn desk plan, which does not render a disabled look by itself.
        self._workspace_box.setEnabled(not unneeded)
        fade = QGraphicsOpacityEffect(self._workspace)
        fade.setOpacity(0.35)
        self._workspace.setGraphicsEffect(fade if unneeded else None)
        self._workspace_unneeded.setVisible(unneeded)

    def _on_language(self) -> None:
        # Preview the language in this window; the caller applies or reverts it afterwards.
        i18n.set_language(self._language.currentData())
        self.retranslate()

    def retranslate(self) -> None:
        self.setWindowTitle(tr("settings.title"))
        self._method_box.setTitle(tr("settings.study_method"))
        self._picker.retranslate()
        self._workspace_box.setTitle(tr("settings.workspace"))
        self._workspace_unneeded.setText(tr("settings.workspace_unneeded"))
        self._detection_box.setTitle(tr("settings.detection"))
        self._sound_box.setTitle(tr("settings.sound"))
        self._language_box.setTitle(tr("settings.language"))
        self._display_box.setTitle(tr("settings.display"))
        self._show_landmarks.setText(tr("settings.show_landmarks"))
        self._show_landmarks.setToolTip(tr("settings.show_landmarks_tip"))
        for key, label in self._labels.items():
            label.setText(tr(key))
        for field in (self._distraction, self._phone, self._no_movement):
            field.setToolTip(tr("settings.seconds_tip"))
            field.refresh()
        self._update_sensitivity_label(self._sensitivity.value())
        self._detection_hint.setText(tr("settings.detection_hint"))
        self._sound_enabled.setText(tr("settings.sound_enabled"))
        self._test.setText(tr("settings.test"))
        self._test.setToolTip(tr("settings.test_tip"))
        self._save.setText(tr("settings.save"))
        self._cancel.setText(tr("settings.cancel"))
        self._defaults.setText(tr("settings.defaults"))
        self._defaults.setToolTip(tr("settings.defaults_tip"))
        self._workspace.retranslate()

    def _update_sensitivity_label(self, value: int) -> None:
        word = "low" if value <= 3 else "high" if value >= 8 else "medium"
        self._sensitivity_value.setText(f"{value} · {tr(f'settings.sensitivity_{word}')}")

    def _restore_defaults(self) -> None:
        defaults = UserSettings()
        # Study method, workspace and language stay as they are: they describe the desk and the user.
        self._distraction.setValue(defaults.distraction_after_s)
        self._phone.setValue(defaults.phone_distraction_after_s)
        self._no_movement.setValue(defaults.no_movement_away_after_s)
        self._sensitivity.setValue(defaults.sensitivity)
        self._sound_enabled.setChecked(defaults.sound_enabled)
        self._volume.setValue(defaults.sound_volume)

    def result_settings(self) -> UserSettings:
        """Read every value from the widgets as they are shown right now."""
        return dataclasses.replace(
            self._user,
            study_method=self._picker.method,
            workspace=self._workspace.workspace,
            distraction_after_s=self._distraction.value(),
            phone_distraction_after_s=self._phone.value(),
            no_movement_away_after_s=self._no_movement.value(),
            sensitivity=self._sensitivity.value(),
            sound_enabled=self._sound_enabled.isChecked(),
            sound_volume=self._volume.value(),
            language=self._language.currentData(),
            show_landmarks=self._show_landmarks.isChecked(),
        ).normalized()

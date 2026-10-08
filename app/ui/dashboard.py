"""Side panel showing the current state, behaviour durations and detected signals."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget

from app.features.feature_pipeline import FocusState, FrameAnalysis
from app.features.gaze_features import Attention
from app.features.state_timers import format_duration
from app.ui import styles

_NO_VALUE = "—"


def _field(name: str) -> tuple[QLabel, QLabel]:
    label = QLabel(name)
    label.setObjectName("FieldName")
    value = QLabel(_NO_VALUE)
    value.setObjectName("FieldValue")
    return label, value


def _card(title: str) -> tuple[QFrame, QVBoxLayout]:
    card = QFrame()
    card.setObjectName("Card")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(16, 12, 16, 14)
    heading = QLabel(title.upper())
    heading.setObjectName("SectionTitle")
    layout.addWidget(heading)
    return card, layout


def _seconds(value: float) -> str:
    return f"{value:.1f} s" if value < 60 else format_duration(value)


class Dashboard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(270)
        self._phone_available = True

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        state_card, state_layout = _card("State")
        self._state = QLabel(_NO_VALUE)
        self._state.setObjectName("StateValue")
        self._reason = QLabel("")
        self._reason.setObjectName("StateReason")
        self._reason.setWordWrap(True)
        state_layout.addWidget(self._state)
        state_layout.addWidget(self._reason)
        root.addWidget(state_card)

        self._timers_card, timers_layout = _card("Session time")
        timers_grid = QGridLayout()
        timers_grid.setVerticalSpacing(4)
        self._timer_names: dict[FocusState, QLabel] = {}
        self._timer_values: dict[FocusState, QLabel] = {}
        for row, state in enumerate(FocusState):
            name = QLabel(f"●  {state.value}")
            value = QLabel("00:00")
            value.setObjectName("TimerValue")
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            timers_grid.addWidget(name, row, 0)
            timers_grid.addWidget(value, row, 1)
            self._timer_names[state], self._timer_values[state] = name, value
        timers_grid.setColumnStretch(1, 1)
        timers_layout.addLayout(timers_grid)
        root.addWidget(self._timers_card)
        self.update_timers({}, None)

        behaviour_card, behaviour_layout = _card("Behaviour")
        behaviour_grid = QGridLayout()
        self._attention = self._add_row(behaviour_grid, 0, "Attention")
        self._away_time = self._add_row(behaviour_grid, 1, "Looking away")
        self._phone_time = self._add_row(behaviour_grid, 2, "Looking at phone")
        self._still_time = self._add_row(behaviour_grid, 3, "Without moving")
        behaviour_layout.addLayout(behaviour_grid)
        root.addWidget(behaviour_card)

        head_card, head_layout = _card("Head")
        head_grid = QGridLayout()
        self._yaw = self._add_row(head_grid, 0, "Yaw")
        self._pitch = self._add_row(head_grid, 1, "Pitch")
        self._roll = self._add_row(head_grid, 2, "Roll")
        head_layout.addLayout(head_grid)
        root.addWidget(head_card)

        signals_card, signals_layout = _card("Signals")
        signals_grid = QGridLayout()
        self._posture = self._add_row(signals_grid, 0, "Posture")
        self._phone = self._add_row(signals_grid, 1, "Phone")
        self._face = self._add_row(signals_grid, 2, "Face")
        self._person = self._add_row(signals_grid, 3, "Person")
        signals_layout.addLayout(signals_grid)
        root.addWidget(signals_card)

        root.addStretch(1)

    @staticmethod
    def _add_row(grid: QGridLayout, row: int, name: str) -> QLabel:
        label, value = _field(name)
        grid.addWidget(label, row, 0)
        grid.addWidget(value, row, 1)
        grid.setColumnStretch(1, 1)
        return value

    def set_timers_visible(self, visible: bool) -> None:
        self._timers_card.setVisible(visible)

    def set_phone_detection_available(self, available: bool) -> None:
        self._phone_available = available
        if not available:
            self._set_flag(self._phone, None, "Unavailable")

    def show_state(self, state: FocusState | None, reason: str = "") -> None:
        if state is None:
            self._state.setText(_NO_VALUE)
            self._state.setStyleSheet("")
        else:
            self._state.setText(state.value)
            self._state.setStyleSheet(f"color: {styles.STATE_COLORS[state.value]};")
        self._reason.setText(reason)

    def update_timers(self, totals: dict[FocusState, float], current: FocusState | None) -> None:
        for state in FocusState:
            color = styles.STATE_COLORS[state.value]
            weight = "700" if state is current else "400"
            text_color = styles.TEXT if state is current else styles.TEXT_MUTED
            self._timer_names[state].setText(
                f"<span style='color:{color}'>●</span>&nbsp;&nbsp;"
                f"<span style='color:{text_color}; font-weight:{weight}'>{state.value}</span>"
            )
            value = self._timer_values[state]
            value.setText(format_duration(totals.get(state, 0.0)))
            value.setStyleSheet(f"color: {text_color}; font-weight: {weight};")

    def update_analysis(self, analysis: FrameAnalysis, state_override: FocusState | None = None) -> None:
        features = analysis.features
        if state_override is None:
            self.show_state(analysis.state, analysis.reason)
        else:
            self.show_state(state_override, "Monitoring paused while you take a break")

        attention = features.attention
        self._attention.setText(attention.value)
        good = {Attention.ON_SCREEN: True, Attention.ABSENT: None}.get(attention, False)
        self._set_flag(self._attention, good, attention.value)
        activity = features.activity
        self._away_time.setText(_seconds(activity.seconds_looking_away))
        self._phone_time.setText(_seconds(activity.seconds_looking_at_phone))
        self._still_time.setText(_seconds(activity.seconds_still))

        head = features.head_pose
        for label, angle in ((self._yaw, head and head.yaw), (self._pitch, head and head.pitch),
                             (self._roll, head and head.roll)):
            label.setText(_NO_VALUE if head is None else f"{angle:.1f}°")

        self._posture.setText(features.pose.posture.value)
        if self._phone_available:
            phone = features.phone
            if phone.looking_at_phone:
                self._set_flag(self._phone, False, "Looking at it")
            elif phone.visible:
                # A visible phone that is not being looked at is fine.
                self._phone.setText("Visible, ignored")
                self._phone.setStyleSheet(f"color: {styles.WARNING};")
            else:
                self._set_flag(self._phone, None, "Not detected")
        self._set_flag(self._face, features.face_detected, "Detected" if features.face_detected else "Not detected")
        self._set_flag(self._person, features.person_detected,
                       "Detected" if features.person_detected else "Not detected")

    def clear(self) -> None:
        self.show_state(None)
        for label in (self._yaw, self._pitch, self._roll, self._posture, self._face, self._person,
                      self._attention, self._away_time, self._phone_time, self._still_time):
            label.setText(_NO_VALUE)
            label.setStyleSheet("")
        if self._phone_available:
            self._set_flag(self._phone, None, _NO_VALUE)

    @staticmethod
    def _set_flag(label: QLabel, good: bool | None, text: str) -> None:
        label.setText(text)
        if good is None:
            label.setStyleSheet(f"color: {styles.TEXT_MUTED};")
        else:
            label.setStyleSheet(f"color: {styles.POSITIVE if good else styles.NEGATIVE};")

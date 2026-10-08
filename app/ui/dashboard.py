"""Side panel showing the current state, behaviour durations and detected signals."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget

from app.features.feature_pipeline import FocusState, FrameAnalysis, Reason, ReasonCode
from app.features.state_timers import format_duration
from app.ui import styles
from app.ui.i18n import format_seconds, tr

_NO_VALUE = "—"


def state_name(state: FocusState) -> str:
    return tr(f"state.{state.value}")


def describe_reason(reason: Reason | None) -> str:
    if reason is None:
        return ""
    seconds = format_seconds(reason.seconds, 0)
    if reason.code is ReasonCode.ABSENT:
        return tr("reason.absent")
    if reason.code is ReasonCode.NO_MOVEMENT:
        return tr("reason.no_movement", seconds=seconds)
    if reason.code is ReasonCode.PHONE:
        return tr("reason.phone", seconds=seconds)
    if reason.code is ReasonCode.LOOKING_AWAY:
        what = tr(f"attention.{reason.attention.value}") if reason.attention else tr("field.looking_away")
        return tr("reason.looking_away", what=what, seconds=seconds)
    if reason.monitor is not None:
        return tr("reason.on_monitor", n=reason.monitor + 1)
    return tr("reason.on_monitors")


def _duration(value: float) -> str:
    return format_seconds(value) if value < 60 else format_duration(value)


class Dashboard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(280)
        self._phone_available = True
        self._titles: dict[str, QLabel] = {}
        self._names: dict[str, QLabel] = {}
        self._last_state: tuple[FocusState | None, str] = (None, "")
        self._last_timers: tuple[dict, FocusState | None] = ({}, None)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        state_card, state_layout = self._card("card.state")
        self._state = QLabel(_NO_VALUE)
        self._state.setObjectName("StateValue")
        self._reason = QLabel("")
        self._reason.setObjectName("StateReason")
        self._reason.setWordWrap(True)
        state_layout.addWidget(self._state)
        state_layout.addWidget(self._reason)
        root.addWidget(state_card)

        self._timers_card, timers_layout = self._card("card.session")
        timers_grid = QGridLayout()
        timers_grid.setVerticalSpacing(4)
        self._timer_names: dict[FocusState, QLabel] = {}
        self._timer_values: dict[FocusState, QLabel] = {}
        for row, state in enumerate(FocusState):
            name = QLabel()
            value = QLabel("00:00")
            value.setObjectName("TimerValue")
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            timers_grid.addWidget(name, row, 0)
            timers_grid.addWidget(value, row, 1)
            self._timer_names[state], self._timer_values[state] = name, value
        timers_grid.setColumnStretch(1, 1)
        timers_layout.addLayout(timers_grid)
        root.addWidget(self._timers_card)

        behaviour_card, behaviour_layout = self._card("card.behaviour")
        grid = QGridLayout()
        self._attention = self._add_row(grid, 0, "field.attention")
        self._monitor = self._add_row(grid, 1, "field.monitor")
        self._away_time = self._add_row(grid, 2, "field.looking_away")
        self._phone_time = self._add_row(grid, 3, "field.looking_at_phone")
        self._still_time = self._add_row(grid, 4, "field.no_movement")
        behaviour_layout.addLayout(grid)
        root.addWidget(behaviour_card)

        head_card, head_layout = self._card("card.head")
        grid = QGridLayout()
        self._yaw = self._add_row(grid, 0, "field.yaw")
        self._pitch = self._add_row(grid, 1, "field.pitch")
        self._roll = self._add_row(grid, 2, "field.roll")
        self._calibration = self._add_row(grid, 3, "field.calibration")
        head_layout.addLayout(grid)
        root.addWidget(head_card)

        signals_card, signals_layout = self._card("card.signals")
        grid = QGridLayout()
        self._posture = self._add_row(grid, 0, "field.posture")
        self._phone = self._add_row(grid, 1, "field.phone")
        self._face = self._add_row(grid, 2, "field.face")
        self._person = self._add_row(grid, 3, "field.person")
        signals_layout.addLayout(grid)
        root.addWidget(signals_card)

        root.addStretch(1)
        self.retranslate()

    def _card(self, title_key: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 12, 16, 14)
        heading = QLabel()
        heading.setObjectName("SectionTitle")
        self._titles[title_key] = heading
        layout.addWidget(heading)
        return card, layout

    def _add_row(self, grid: QGridLayout, row: int, name_key: str) -> QLabel:
        label = QLabel()
        label.setObjectName("FieldName")
        value = QLabel(_NO_VALUE)
        value.setObjectName("FieldValue")
        self._names[name_key] = label
        grid.addWidget(label, row, 0)
        grid.addWidget(value, row, 1)
        grid.setColumnStretch(1, 1)
        return value

    def retranslate(self) -> None:
        for key, label in self._titles.items():
            label.setText(tr(key).upper())
        for key, label in self._names.items():
            label.setText(tr(key))
        state, reason = self._last_state
        self.show_state(state, reason)
        self.update_timers(*self._last_timers)
        if not self._phone_available:
            self._set_flag(self._phone, None, tr("value.unavailable"))

    def set_timers_visible(self, visible: bool) -> None:
        self._timers_card.setVisible(visible)

    def set_phone_detection_available(self, available: bool) -> None:
        self._phone_available = available
        if not available:
            self._set_flag(self._phone, None, tr("value.unavailable"))

    def show_state(self, state: FocusState | None, reason: str = "") -> None:
        self._last_state = (state, reason)
        if state is None:
            self._state.setText(_NO_VALUE)
            self._state.setStyleSheet("")
        else:
            self._state.setText(state_name(state))
            self._state.setStyleSheet(f"color: {styles.STATE_COLORS[state.value]};")
        self._reason.setText(reason)

    def update_timers(self, totals: dict[FocusState, float], current: FocusState | None) -> None:
        self._last_timers = (totals, current)
        for state in FocusState:
            color = styles.STATE_COLORS[state.value]
            weight = "700" if state is current else "400"
            text_color = styles.TEXT if state is current else styles.TEXT_MUTED
            self._timer_names[state].setText(
                f"<span style='color:{color}'>●</span>&nbsp;&nbsp;"
                f"<span style='color:{text_color}; font-weight:{weight}'>{state_name(state)}</span>"
            )
            value = self._timer_values[state]
            value.setText(format_duration(totals.get(state, 0.0)))
            value.setStyleSheet(f"color: {text_color}; font-weight: {weight};")

    def update_analysis(self, analysis: FrameAnalysis, state_override: FocusState | None = None) -> None:
        features = analysis.features
        if state_override is None:
            self.show_state(analysis.state, describe_reason(analysis.reason))
        else:
            self.show_state(state_override, tr("reason.break"))

        attention = features.attention
        good = {"on_screen": True, "absent": None}.get(attention.value, False)
        self._set_flag(self._attention, good, tr(f"attention.{attention.value}"))
        self._monitor.setText(_NO_VALUE if features.monitor is None else tr("monitor.n", n=features.monitor + 1))
        activity = features.activity
        self._away_time.setText(_duration(activity.seconds_looking_away))
        self._phone_time.setText(_duration(activity.seconds_looking_at_phone))
        self._still_time.setText(_duration(activity.seconds_still))

        head = features.head_pose
        for label, angle in ((self._yaw, head and head.yaw), (self._pitch, head and head.pitch),
                             (self._roll, head and head.roll)):
            label.setText(_NO_VALUE if head is None else f"{angle:.1f}°")
        self._calibration.setText(f"{features.pitch_calibration:+.1f}°")

        self._posture.setText(tr(f"posture.{features.pose.posture.value}"))
        if self._phone_available:
            phone = features.phone
            if phone.looking_at_phone:
                self._set_flag(self._phone, False, tr("value.phone_looking"))
            elif phone.visible:
                # A visible phone that is not being looked at is fine.
                self._phone.setText(tr("value.phone_ignored"))
                self._phone.setStyleSheet(f"color: {styles.WARNING};")
            else:
                self._set_flag(self._phone, None, tr("value.not_detected"))
        for label, ok in ((self._face, features.face_detected), (self._person, features.person_detected)):
            self._set_flag(label, ok, tr("value.detected") if ok else tr("value.not_detected"))

    def clear(self) -> None:
        self.show_state(None)
        for label in (self._yaw, self._pitch, self._roll, self._calibration, self._posture, self._face, self._person,
                      self._attention, self._monitor, self._away_time, self._phone_time, self._still_time):
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

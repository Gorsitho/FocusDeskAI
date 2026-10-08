"""Monitor count and camera position picker with a clickable desk diagram."""

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.config.settings import CameraPosition
from app.config.user_settings import MONITOR_COUNTS
from app.ui import styles

_MONITOR_W, _MONITOR_H, _MONITOR_GAP = 96.0, 60.0, 8.0
_SPOT_RADIUS = 9.0


class WorkspacePreview(QWidget):
    """Draws the monitors seen from behind the user; click a dot to move the camera."""

    position_clicked = Signal(object)  # CameraPosition

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(380, 190)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._monitors = 1
        self._position = CameraPosition.TOP_CENTER
        self._hover: CameraPosition | None = None

    def set_layout(self, monitors: int, position: CameraPosition) -> None:
        self._monitors, self._position = monitors, position
        self.update()

    def _block_rect(self) -> QRectF:
        width = self._monitors * _MONITOR_W + (self._monitors - 1) * _MONITOR_GAP
        # Centre the drawing (monitors, camera spots and the user, ~160 px tall) vertically.
        top = max((self.height() - 160.0) / 2.0, 0.0) + 24.0
        return QRectF((self.width() - width) / 2.0, top, width, _MONITOR_H)

    def _spots(self) -> dict[CameraPosition, QPointF]:
        block = self._block_rect()
        top, mid_y, bottom = block.top() - 13, block.center().y(), block.bottom() + 13
        first_center = block.left() + _MONITOR_W / 2.0
        last_center = block.right() - _MONITOR_W / 2.0
        spots = {
            CameraPosition.TOP_CENTER: QPointF(block.center().x(), top),
            CameraPosition.TOP_LEFT: QPointF(first_center, top),
            CameraPosition.TOP_RIGHT: QPointF(last_center, top),
            CameraPosition.LEFT: QPointF(block.left() - 18, mid_y),
            CameraPosition.RIGHT: QPointF(block.right() + 18, mid_y),
            CameraPosition.BOTTOM_CENTER: QPointF(block.center().x(), bottom),
        }
        return {pos: pt for pos, pt in spots.items() if pos.available_for(self._monitors)}

    def _spot_at(self, point: QPointF) -> CameraPosition | None:
        for position, center in self._spots().items():
            if (center - point).manhattanLength() <= _SPOT_RADIUS * 2:
                return position
        return None

    def mouseMoveEvent(self, event) -> None:
        hover = self._spot_at(event.position())
        if hover != self._hover:
            self._hover = hover
            self.update()

    def leaveEvent(self, event) -> None:
        self._hover = None
        self.update()

    def mousePressEvent(self, event) -> None:
        position = self._spot_at(event.position())
        if position is not None and event.button() == Qt.MouseButton.LeftButton:
            self.position_clicked.emit(position)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        block = self._block_rect()
        muted, accent = QColor(styles.TEXT_MUTED), QColor(styles.ACCENT)

        for i in range(self._monitors):
            x = block.left() + i * (_MONITOR_W + _MONITOR_GAP)
            screen = QRectF(x, block.top(), _MONITOR_W, _MONITOR_H)
            painter.setPen(QPen(QColor(styles.BORDER_STRONG), 2))
            painter.setBrush(QColor(styles.SURFACE_RAISED))
            painter.drawRoundedRect(screen, 5, 5)
            painter.setPen(muted)
            painter.drawText(screen, Qt.AlignmentFlag.AlignCenter, f"Monitor {i + 1}")
            # Stand.
            stand_x = screen.center().x()
            painter.drawLine(QPointF(stand_x, screen.bottom()), QPointF(stand_x, screen.bottom() + 8))

        for position, center in self._spots().items():
            selected = position is self._position
            if selected:
                painter.setPen(QPen(accent, 2))
                painter.setBrush(accent)
                painter.drawEllipse(center, _SPOT_RADIUS, _SPOT_RADIUS)
                painter.setBrush(QColor(styles.BACKGROUND))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(center, 3.5, 3.5)
            else:
                color = QColor(styles.TEXT) if position is self._hover else muted
                painter.setPen(QPen(color, 1.5, Qt.PenStyle.DashLine))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(center, _SPOT_RADIUS - 2, _SPOT_RADIUS - 2)

        # The user, seen from behind, so "left" in the diagram is the user's left.
        user_center = QPointF(self.width() / 2.0, block.bottom() + 52)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(styles.BORDER_STRONG))
        painter.drawEllipse(user_center, 11, 11)
        painter.drawRoundedRect(QRectF(user_center.x() - 24, user_center.y() + 12, 48, 18), 9, 9)
        painter.setPen(muted)
        font = QFont(painter.font())
        font.setPointSizeF(max(font.pointSizeF() - 1.5, 7.0))
        painter.setFont(font)
        painter.drawText(QRectF(0, user_center.y() - 11, user_center.x() - 34, 22),
                         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, "You (seen from behind)")
        painter.end()


class WorkspaceEditor(QWidget):
    changed = Signal()

    def __init__(self, monitors: int, position: CameraPosition, parent=None):
        super().__init__(parent)
        self._monitor_group = QButtonGroup(self)
        monitor_row = QHBoxLayout()
        monitor_row.setSpacing(6)
        for count in MONITOR_COUNTS:
            button = QPushButton(f"{count} monitor{'s' if count > 1 else ''}")
            button.setObjectName("Segment")
            button.setCheckable(True)
            self._monitor_group.addButton(button, count)
            monitor_row.addWidget(button)
        monitor_row.addStretch(1)

        self._position_combo = QComboBox()
        self._preview = WorkspacePreview()

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.addRow("Monitors", monitor_row)
        form.addRow("Camera position", self._position_combo)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addLayout(form)
        root.addWidget(self._preview)

        self._monitor_group.button(monitors).setChecked(True)
        self._fill_positions(monitors, position)
        self._monitor_group.idClicked.connect(self._on_monitors_changed)
        self._position_combo.currentIndexChanged.connect(self._on_position_changed)
        self._preview.position_clicked.connect(self._select_position)
        self._preview.set_layout(monitors, self.camera_position)

    @property
    def monitor_count(self) -> int:
        return self._monitor_group.checkedId()

    @property
    def camera_position(self) -> CameraPosition:
        # Qt hands str-based enums back as plain strings.
        return CameraPosition(self._position_combo.currentData())

    def _fill_positions(self, monitors: int, selected: CameraPosition) -> None:
        if not selected.available_for(monitors):
            selected = CameraPosition.TOP_CENTER
        self._position_combo.blockSignals(True)
        self._position_combo.clear()
        for position in CameraPosition:
            if position.available_for(monitors):
                self._position_combo.addItem(position.label, position.value)
        self._position_combo.setCurrentIndex(self._position_combo.findData(selected.value))
        self._position_combo.blockSignals(False)

    def _on_monitors_changed(self, monitors: int) -> None:
        self._fill_positions(monitors, self.camera_position)
        self._preview.set_layout(monitors, self.camera_position)
        self.changed.emit()

    def _select_position(self, position: CameraPosition) -> None:
        self._position_combo.setCurrentIndex(self._position_combo.findData(position.value))

    def _on_position_changed(self) -> None:
        self._preview.set_layout(self.monitor_count, self.camera_position)
        self.changed.emit()

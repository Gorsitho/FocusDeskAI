"""
This file contains the visual editor for the desk layout.
The user can place themselves and up to three monitors, and choose where the
webcam is mounted.

* WorkspacePlan: the top view drawing, in metres (x goes to the user's right,
  z goes forward). The user and the monitors can be dragged, and monitors can be rotated.
* WorkspaceEditor: the drawing plus the buttons, lists and sliders around it.

The result is a Workspace (app/config/settings.py). app/features/gaze_features.py
uses it to know where the monitors are. The editor is shown in the setup and
Settings windows (app/ui/settings_dialog.py).
"""

import dataclasses
import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.config.settings import (
    DEFAULT_MONITOR_DISTANCE,
    MAX_MONITOR_DISTANCE,
    MIN_MONITOR_DISTANCE,
    CameraEdge,
    MonitorPlacement,
    Workspace,
    arc_layout,
)
from app.config.user_settings import MONITOR_COUNTS, clamp_monitor, normalize_workspace
from app.ui import styles
from app.ui.i18n import tr

# Visible area of the plan, metres.
_X_RANGE = (-1.2, 1.2)
_Z_RANGE = (-0.45, 1.25)
_PERSON_X = (-0.9, 0.9)
_PERSON_Z = (-0.35, 0.8)
_HANDLE_OFFSET = 0.09  # rotation handle distance behind a monitor, metres
_PICK_RADIUS_PX = 12.0


def nearest_distance(ws: Workspace) -> float:
    return min(math.hypot(m.x - ws.person_x, m.z - ws.person_z) for m in ws.monitors)


def face_person(monitor: MonitorPlacement, ws: Workspace) -> MonitorPlacement:
    """Rotate a monitor so that its screen faces the user."""
    angle = math.degrees(math.atan2(monitor.x - ws.person_x, monitor.z - ws.person_z))
    return dataclasses.replace(monitor, angle=round(angle, 1))


def set_distance(ws: Workspace, distance: float) -> Workspace:
    """Scale the layout around the user so the nearest monitor is `distance` away."""
    current = nearest_distance(ws)
    if current < 1e-6:
        return ws
    scale = distance / current
    monitors = tuple(
        clamp_monitor(dataclasses.replace(
            m, x=ws.person_x + (m.x - ws.person_x) * scale, z=ws.person_z + (m.z - ws.person_z) * scale,
        ), ws.person_x, ws.person_z)
        for m in ws.monitors
    )
    return dataclasses.replace(ws, monitors=monitors)


def move_person(ws: Workspace, x: float, z: float) -> Workspace:
    """Move the user, staying within 0.3-1 m of every monitor."""
    x, z = _clamp(x, *_PERSON_X), _clamp(z, *_PERSON_Z)
    for _ in range(8):  # alternate projections onto each monitor's allowed ring
        for m in ws.monitors:
            dx, dz = x - m.x, z - m.z
            d = math.hypot(dx, dz) or 1e-6
            target = _clamp(d, MIN_MONITOR_DISTANCE, MAX_MONITOR_DISTANCE)
            x, z = m.x + dx * target / d, m.z + dz * target / d
    return dataclasses.replace(ws, person_x=round(x, 4), person_z=round(z, 4))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class WorkspacePlan(QWidget):
    """Top-view canvas of the workspace."""

    changed = Signal()
    selection_changed = Signal(int)

    def __init__(self, workspace: Workspace, parent=None):
        super().__init__(parent)
        self.setMinimumSize(440, 300)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._ws = normalize_workspace(workspace)
        self._selected = 0
        self._drag: tuple[str, int] | None = None  # ("person"|"monitor"|"handle", index)
        self._drag_offset = (0.0, 0.0)
        self._hover: tuple[str, int] | None = None

    # --- model -------------------------------------------------------------------
    @property
    def workspace(self) -> Workspace:
        return self._ws

    def set_workspace(self, workspace: Workspace, emit: bool = True) -> None:
        self._ws = normalize_workspace(workspace)
        self._selected = min(self._selected, self._ws.monitor_count - 1)
        self.update()
        if emit:
            self.changed.emit()

    @property
    def selected(self) -> int:
        return self._selected

    def select(self, index: int) -> None:
        if 0 <= index < self._ws.monitor_count and index != self._selected:
            self._selected = index
            self.selection_changed.emit(index)
            self.update()

    def _replace_monitor(self, index: int, monitor: MonitorPlacement) -> None:
        monitors = list(self._ws.monitors)
        monitors[index] = monitor
        self.set_workspace(dataclasses.replace(self._ws, monitors=tuple(monitors)))

    # --- coordinates -------------------------------------------------------------------
    def _scale(self) -> float:
        return min((self.width() - 24) / (_X_RANGE[1] - _X_RANGE[0]),
                   (self.height() - 24) / (_Z_RANGE[1] - _Z_RANGE[0]))

    def _to_screen(self, x: float, z: float) -> QPointF:
        s = self._scale()
        cx = self.width() / 2.0 - (_X_RANGE[0] + _X_RANGE[1]) / 2.0 * s
        cy = self.height() / 2.0 + (_Z_RANGE[0] + _Z_RANGE[1]) / 2.0 * s
        return QPointF(cx + x * s, cy - z * s)

    def _to_world(self, point: QPointF) -> tuple[float, float]:
        origin = self._to_screen(0.0, 0.0)
        s = self._scale()
        return (point.x() - origin.x()) / s, (origin.y() - point.y()) / s

    @staticmethod
    def _endpoints(m: MonitorPlacement) -> tuple[tuple[float, float], tuple[float, float]]:
        rad = math.radians(m.angle)
        rx, rz = math.cos(rad) * m.width / 2, -math.sin(rad) * m.width / 2
        return (m.x - rx, m.z - rz), (m.x + rx, m.z + rz)

    @staticmethod
    def _handle(m: MonitorPlacement) -> tuple[float, float]:
        rad = math.radians(m.angle)
        return m.x + math.sin(rad) * _HANDLE_OFFSET, m.z + math.cos(rad) * _HANDLE_OFFSET

    def _hit(self, point: QPointF) -> tuple[str, int] | None:
        sel = self._ws.monitors[self._selected]
        if _dist(point, self._to_screen(*self._handle(sel))) <= _PICK_RADIUS_PX:
            return "handle", self._selected
        if _dist(point, self._to_screen(self._ws.person_x, self._ws.person_z)) <= 22:
            return "person", 0
        for i, m in enumerate(self._ws.monitors):
            a, b = (self._to_screen(*p) for p in self._endpoints(m))
            if _dist_to_segment(point, a, b) <= _PICK_RADIUS_PX:
                return "monitor", i
        return None

    # --- interaction -------------------------------------------------------------
    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        hit = self._hit(event.position())
        if hit is None:
            return
        kind, index = hit
        if kind == "monitor":
            self.select(index)
            m = self._ws.monitors[index]
            wx, wz = self._to_world(event.position())
            self._drag_offset = (m.x - wx, m.z - wz)
        elif kind == "person":
            wx, wz = self._to_world(event.position())
            self._drag_offset = (self._ws.person_x - wx, self._ws.person_z - wz)
        self._drag = hit

    def mouseMoveEvent(self, event) -> None:
        if self._drag is None:
            hover = self._hit(event.position())
            if hover != self._hover:
                self._hover = hover
                cursor = Qt.CursorShape.ArrowCursor if hover is None else Qt.CursorShape.OpenHandCursor
                self.setCursor(cursor)
                self.update()
            return
        wx, wz = self._to_world(event.position())
        kind, index = self._drag
        if kind == "person":
            self.set_workspace(move_person(self._ws, wx + self._drag_offset[0], wz + self._drag_offset[1]))
        elif kind == "monitor":
            m = self._ws.monitors[index]
            moved = dataclasses.replace(
                m, x=_clamp(wx + self._drag_offset[0], _X_RANGE[0] + 0.1, _X_RANGE[1] - 0.1),
                z=_clamp(wz + self._drag_offset[1], _Z_RANGE[0] + 0.05, _Z_RANGE[1] - 0.05),
            )
            self._replace_monitor(index, clamp_monitor(moved, self._ws.person_x, self._ws.person_z))
        else:
            m = self._ws.monitors[index]
            angle = math.degrees(math.atan2(wx - m.x, wz - m.z))
            self._replace_monitor(index, dataclasses.replace(m, angle=round(angle)))

    def mouseReleaseEvent(self, event) -> None:
        self._drag = None

    def mouseDoubleClickEvent(self, event) -> None:
        hit = self._hit(event.position())
        if hit is not None and hit[0] == "monitor":
            self._replace_monitor(hit[1], face_person(self._ws.monitors[hit[1]], self._ws))

    def wheelEvent(self, event) -> None:
        hit = self._hit(event.position())
        index = hit[1] if hit is not None and hit[0] in ("monitor", "handle") else None
        if index is None:
            event.ignore()
            return
        self.select(index)
        m = self._ws.monitors[index]
        step = 3 if event.angleDelta().y() > 0 else -3
        self._replace_monitor(index, dataclasses.replace(m, angle=round(m.angle + step)))
        event.accept()

    def leaveEvent(self, event) -> None:
        self._hover = None
        self.update()

    # --- painting ----------------------------------------------------------------
    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(styles.SURFACE))
        s = self._scale()
        ws = self._ws
        person = self._to_screen(ws.person_x, ws.person_z)

        # Grid every 10 cm, stronger every 50 cm.
        for i in range(int(_X_RANGE[0] * 10), int(_X_RANGE[1] * 10) + 1):
            color = styles.BORDER_STRONG if i % 5 == 0 else styles.BORDER
            p.setPen(QPen(QColor(color), 1))
            p.drawLine(self._to_screen(i / 10, _Z_RANGE[0]), self._to_screen(i / 10, _Z_RANGE[1]))
        for i in range(int(_Z_RANGE[0] * 10), int(_Z_RANGE[1] * 10) + 1):
            color = styles.BORDER_STRONG if i % 5 == 0 else styles.BORDER
            p.setPen(QPen(QColor(color), 1))
            p.drawLine(self._to_screen(_X_RANGE[0], i / 10), self._to_screen(_X_RANGE[1], i / 10))

        # Maximum distance (1 m).
        p.setPen(QPen(QColor(styles.TEXT_MUTED), 1, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(person, MAX_MONITOR_DISTANCE * s, MAX_MONITOR_DISTANCE * s)
        p.setPen(QColor(styles.TEXT_MUTED))
        p.drawText(person + QPointF(MAX_MONITOR_DISTANCE * s * 0.71 + 4, -MAX_MONITOR_DISTANCE * s * 0.71), "1 m")

        # Viewing wedges: where looking counts as looking at a monitor.
        accent = QColor(styles.ACCENT)
        for m in ws.monitors:
            a, b = (self._to_screen(*pt) for pt in self._endpoints(m))
            path = QPainterPath(person)
            path.lineTo(a)
            path.lineTo(b)
            path.closeSubpath()
            fill = QColor(accent)
            fill.setAlpha(38)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(fill)
            p.drawPath(path)

        # Distance labels.
        small = QFont(self.font())
        small.setPointSizeF(max(small.pointSizeF() - 1.5, 7.0))
        p.setFont(small)
        for m in ws.monitors:
            center = self._to_screen(m.x, m.z)
            p.setPen(QPen(QColor(styles.TEXT_MUTED), 1, Qt.PenStyle.DotLine))
            p.drawLine(person, center)
            mid = (person + center) / 2.0
            distance = math.hypot(m.x - ws.person_x, m.z - ws.person_z)
            p.setPen(QColor(styles.TEXT))
            p.drawText(QRectF(mid.x() - 30, mid.y() - 9, 60, 18), Qt.AlignmentFlag.AlignCenter,
                       f"{distance * 100:.0f} cm")

        # Monitors.
        for i, m in enumerate(ws.monitors):
            a, b = (self._to_screen(*pt) for pt in self._endpoints(m))
            selected = i == self._selected
            hovered = self._hover is not None and self._hover[0] == "monitor" and self._hover[1] == i
            body = QColor(styles.ACCENT if selected else styles.TEXT if hovered else styles.TEXT_MUTED)
            pen = QPen(body, 7)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.drawLine(a, b)
            # Screen side (towards the viewer).
            rad = math.radians(m.angle)
            front = QPointF(-math.sin(rad), math.cos(rad)) * 5.0  # screen y grows downwards
            p.setPen(QPen(QColor("#e8f0ff"), 2))
            p.drawLine(a + front, b + front)
            # Number.
            back = self._to_screen(m.x + math.sin(rad) * 0.045, m.z + math.cos(rad) * 0.045)
            p.setPen(QColor(styles.TEXT))
            p.drawText(QRectF(back.x() - 10, back.y() - 9, 20, 18), Qt.AlignmentFlag.AlignCenter, str(i + 1))
            if selected:
                handle = self._to_screen(*self._handle(m))
                p.setPen(QPen(accent, 1.5))
                p.drawLine(self._to_screen(m.x, m.z), handle)
                p.setBrush(QColor(styles.BACKGROUND))
                p.drawEllipse(handle, 6, 6)

        # Camera.
        cam = ws.monitors[ws.camera_monitor]
        a, b = self._endpoints(cam)
        rad = math.radians(cam.angle)
        toward_user = (-math.sin(rad) * 0.03, -math.cos(rad) * 0.03)
        position = {
            CameraEdge.LEFT: a, CameraEdge.RIGHT: b,
        }.get(ws.camera_edge, (cam.x, cam.z))
        cam_pt = self._to_screen(position[0] + toward_user[0], position[1] + toward_user[1])
        p.setPen(QPen(QColor(styles.BACKGROUND), 1.5))
        p.setBrush(QColor(styles.WARNING))
        p.drawRoundedRect(QRectF(cam_pt.x() - 7, cam_pt.y() - 5, 14, 10), 3, 3)
        p.setBrush(QColor(styles.BACKGROUND))
        p.drawEllipse(cam_pt, 2.5, 2.5)

        # The user (top view: shoulders and head, facing the monitors).
        hovered = self._hover is not None and self._hover[0] == "person"
        color = QColor(styles.TEXT if hovered or (self._drag and self._drag[0] == "person") else "#c9ced6")
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(color.darker(140)))
        p.drawRoundedRect(QRectF(person.x() - 20, person.y() - 2, 40, 14), 7, 7)
        p.setBrush(QBrush(color))
        p.drawEllipse(person, 9, 9)
        p.setBrush(QColor(styles.BACKGROUND))
        p.drawEllipse(person + QPointF(0, -6), 2, 2)  # nose: the user faces forward
        p.setPen(QColor(styles.TEXT))
        p.drawText(QRectF(person.x() - 40, person.y() + 13, 80, 16), Qt.AlignmentFlag.AlignCenter, tr("ws.you"))
        p.end()


def _dist(a: QPointF, b: QPointF) -> float:
    return math.hypot(a.x() - b.x(), a.y() - b.y())


def _dist_to_segment(p: QPointF, a: QPointF, b: QPointF) -> float:
    ax, ay, bx, by = a.x(), a.y(), b.x(), b.y()
    length2 = (bx - ax) ** 2 + (by - ay) ** 2
    if length2 < 1e-9:
        return _dist(p, a)
    t = max(0.0, min(1.0, ((p.x() - ax) * (bx - ax) + (p.y() - ay) * (by - ay)) / length2))
    return math.hypot(p.x() - (ax + t * (bx - ax)), p.y() - (ay + t * (by - ay)))


class WorkspaceEditor(QWidget):
    """Monitor count, camera mount, the plan, and distance/angle controls."""

    changed = Signal()

    def __init__(self, workspace: Workspace, parent=None):
        super().__init__(parent)
        self._plan = WorkspacePlan(workspace)

        self._monitor_group = QButtonGroup(self)
        self._monitor_buttons = []
        monitor_row = QHBoxLayout()
        monitor_row.setSpacing(6)
        for count in MONITOR_COUNTS:
            button = QPushButton()
            button.setObjectName("Segment")
            button.setCheckable(True)
            self._monitor_group.addButton(button, count)
            self._monitor_buttons.append(button)
            monitor_row.addWidget(button)
        monitor_row.addStretch(1)

        self._camera_monitor = QComboBox()
        self._camera_edge = QComboBox()
        camera_row = QHBoxLayout()
        camera_row.setSpacing(6)
        camera_row.addWidget(self._camera_monitor, 1)
        camera_row.addWidget(self._camera_edge, 1)

        self._distance = QSlider(Qt.Orientation.Horizontal)
        self._distance.setRange(int(MIN_MONITOR_DISTANCE * 100), int(MAX_MONITOR_DISTANCE * 100))
        self._distance_value = QLabel()
        self._distance_value.setMinimumWidth(52)
        self._angle = QSlider(Qt.Orientation.Horizontal)
        self._angle.setRange(-180, 180)
        self._angle_value = QLabel()
        self._angle_value.setMinimumWidth(52)
        self._arrange = QPushButton()
        self._hint = QLabel()
        self._hint.setObjectName("Hint")
        self._hint.setWordWrap(True)

        self._monitors_label, self._camera_label = QLabel(), QLabel()
        self._distance_label, self._angle_label = QLabel(), QLabel()
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.addWidget(self._monitors_label, 0, 0)
        grid.addLayout(monitor_row, 0, 1, 1, 2)
        grid.addWidget(self._camera_label, 1, 0)
        grid.addLayout(camera_row, 1, 1, 1, 2)

        sliders = QGridLayout()
        sliders.setHorizontalSpacing(10)
        sliders.addWidget(self._distance_label, 0, 0)
        sliders.addWidget(self._distance, 0, 1)
        sliders.addWidget(self._distance_value, 0, 2)
        sliders.addWidget(self._angle_label, 1, 0)
        sliders.addWidget(self._angle, 1, 1)
        sliders.addWidget(self._angle_value, 1, 2)
        sliders.setColumnStretch(1, 1)

        bottom = QHBoxLayout()
        bottom.addWidget(self._hint, 1)
        bottom.addWidget(self._arrange, 0, Qt.AlignmentFlag.AlignTop)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)
        root.addLayout(grid)
        root.addWidget(self._plan, 1)
        root.addLayout(sliders)
        root.addLayout(bottom)

        self._monitor_group.idClicked.connect(self._on_monitor_count)
        self._camera_monitor.currentIndexChanged.connect(self._on_camera_changed)
        self._camera_edge.currentIndexChanged.connect(self._on_camera_changed)
        self._distance.valueChanged.connect(self._on_distance)
        self._angle.valueChanged.connect(self._on_angle)
        self._arrange.clicked.connect(self._on_arrange)
        self._plan.changed.connect(self._sync_controls)
        self._plan.changed.connect(self.changed)
        self._plan.selection_changed.connect(lambda _: self._sync_controls())

        self.retranslate()

    @property
    def workspace(self) -> Workspace:
        return self._plan.workspace

    @property
    def plan(self) -> WorkspacePlan:
        return self._plan

    def set_workspace(self, workspace: Workspace) -> None:
        self._plan.set_workspace(workspace)

    def retranslate(self) -> None:
        self._monitors_label.setText(tr("ws.monitors"))
        self._camera_label.setText(tr("ws.camera"))
        self._distance_label.setText(tr("ws.distance"))
        for count, button in zip(MONITOR_COUNTS, self._monitor_buttons):
            button.setText(tr("ws.monitors_1") if count == 1 else tr("ws.monitors_n", n=count))
        self._camera_monitor.setToolTip(tr("ws.camera_tip"))
        self._camera_edge.setToolTip(tr("ws.camera_tip"))
        self._distance.setToolTip(tr("ws.distance_tip"))
        self._angle.setToolTip(tr("ws.angle_tip"))
        self._arrange.setText(tr("ws.arrange"))
        self._arrange.setToolTip(tr("ws.arrange_tip"))
        self._hint.setText(tr("ws.hint"))
        self._plan.setToolTip(tr("ws.plan_tip"))
        self._camera_edge.blockSignals(True)
        current_edge = self._camera_edge.currentData()
        self._camera_edge.clear()
        for edge in CameraEdge:
            self._camera_edge.addItem(tr(f"ws.edge.{edge.value}"), edge.value)
        self._camera_edge.blockSignals(False)
        if current_edge is None:
            current_edge = self.workspace.camera_edge.value
        self._camera_edge.setCurrentIndex(max(self._camera_edge.findData(current_edge), 0))
        self._sync_controls()
        self._plan.update()

    def _sync_controls(self) -> None:
        ws = self.workspace
        for widget in (self._camera_monitor, self._camera_edge, self._distance, self._angle):
            widget.blockSignals(True)
        self._monitor_group.button(ws.monitor_count).setChecked(True)
        self._camera_monitor.clear()
        for i in range(ws.monitor_count):
            self._camera_monitor.addItem(tr("monitor.n", n=i + 1), i)
        self._camera_monitor.setCurrentIndex(ws.camera_monitor)
        self._camera_edge.setCurrentIndex(max(self._camera_edge.findData(ws.camera_edge.value), 0))
        distance = nearest_distance(ws)
        self._distance.setValue(round(distance * 100))
        self._distance_value.setText(f"{distance * 100:.0f} cm")
        selected = self._plan.selected
        angle = ws.monitors[selected].angle
        self._angle.setValue(round(angle))
        self._angle_value.setText(f"{angle:.0f}°")
        self._angle_label.setText(tr("ws.angle", n=selected + 1))
        for widget in (self._camera_monitor, self._camera_edge, self._distance, self._angle):
            widget.blockSignals(False)

    def _on_monitor_count(self, count: int) -> None:
        ws = self.workspace
        if count == ws.monitor_count:
            return
        distance = nearest_distance(ws) if ws.monitors else DEFAULT_MONITOR_DISTANCE
        monitors = arc_layout(count, distance, (ws.person_x, ws.person_z))
        camera = ws.camera_monitor if ws.camera_monitor < count else (count - 1) // 2
        self._plan.set_workspace(dataclasses.replace(ws, monitors=monitors, camera_monitor=camera))

    def _on_camera_changed(self) -> None:
        index = self._camera_monitor.currentData()
        edge = self._camera_edge.currentData()
        if index is None or edge is None:
            return
        self._plan.set_workspace(dataclasses.replace(
            self.workspace, camera_monitor=int(index), camera_edge=CameraEdge(edge)
        ))

    def _on_distance(self, value: int) -> None:
        self._plan.set_workspace(set_distance(self.workspace, value / 100.0))

    def _on_angle(self, value: int) -> None:
        ws = self.workspace
        index = self._plan.selected
        monitors = list(ws.monitors)
        monitors[index] = dataclasses.replace(monitors[index], angle=float(value))
        self._plan.set_workspace(dataclasses.replace(ws, monitors=tuple(monitors)))

    def _on_arrange(self) -> None:
        ws = self.workspace
        monitors = arc_layout(ws.monitor_count, nearest_distance(ws), (ws.person_x, ws.person_z))
        self._plan.set_workspace(dataclasses.replace(ws, monitors=monitors))

"""Qt stylesheet and colour palette."""

BACKGROUND = "#14161a"
SURFACE = "#1d2026"
SURFACE_RAISED = "#262a32"
BORDER = "#2b2f37"
BORDER_STRONG = "#3a3f4a"
TEXT = "#e6e8eb"
TEXT_MUTED = "#8b919c"
ACCENT = "#4c8dff"

STATE_COLORS = {
    "FOCUSED": "#2ecc71",
    "DISTRACTED": "#f39c12",
    "AWAY": "#7f8c8d",
    "BREAK": "#a37cf0",
}
POSITIVE = "#2ecc71"
NEGATIVE = "#e74c3c"
WARNING = "#f1c40f"

APP_STYLESHEET = f"""
QMainWindow, QWidget, QDialog {{
    background-color: {BACKGROUND};
    color: {TEXT};
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 14px;
}}
QLabel {{
    background: transparent;
}}
QLabel#AppTitle {{
    font-size: 22px;
    font-weight: 600;
}}
QLabel#DialogTitle {{
    font-size: 20px;
    font-weight: 600;
}}
QLabel#Hint {{
    color: {TEXT_MUTED};
    font-size: 12px;
}}
QFrame#Card {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}
QLabel#CameraView {{
    background-color: #000000;
    border-radius: 10px;
    color: {TEXT_MUTED};
}}
QLabel#SectionTitle {{
    color: {TEXT_MUTED};
    font-size: 12px;
    font-weight: 600;
}}
QLabel#StateValue {{
    font-size: 30px;
    font-weight: 700;
}}
QLabel#StateReason {{
    color: {TEXT_MUTED};
    font-size: 12px;
}}
QLabel#FieldName {{
    color: {TEXT_MUTED};
}}
QLabel#FieldValue {{
    font-weight: 600;
}}
QLabel#TimerValue {{
    font-family: "Cascadia Mono", "Consolas", "Menlo", monospace;
    font-size: 15px;
}}
QLabel#StatusBar {{
    color: {TEXT_MUTED};
    font-size: 12px;
}}
QScrollArea {{
    border: none;
}}
QPushButton {{
    background-color: {SURFACE_RAISED};
    border: 1px solid {BORDER_STRONG};
    border-radius: 7px;
    padding: 6px 14px;
}}
QPushButton:hover {{
    border-color: {TEXT_MUTED};
}}
QPushButton:pressed {{
    background-color: {BORDER};
}}
QPushButton:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
    color: #ffffff;
}}
QPushButton:focus {{
    border-color: {ACCENT};
}}
QPushButton#Primary {{
    background-color: {ACCENT};
    border-color: {ACCENT};
    color: #ffffff;
    font-weight: 600;
}}
QPushButton#Primary:hover {{
    background-color: #6aa1ff;
}}
QPushButton#MethodCard {{
    background-color: {SURFACE};
    border: 2px solid {BORDER_STRONG};
    border-radius: 12px;
    padding: 14px;
}}
QPushButton#MethodCard:hover {{
    border-color: {TEXT_MUTED};
}}
QPushButton#MethodCard:checked {{
    background-color: {SURFACE_RAISED};
    border-color: {ACCENT};
}}
QLabel#MethodIcon {{
    font-size: 30px;
}}
QLabel#MethodTitle {{
    font-size: 16px;
    font-weight: 600;
}}
QPushButton#Segment {{
    padding: 6px 12px;
}}
QPushButton#StartButton {{
    background-color: {POSITIVE};
    border: none;
    border-radius: 10px;
    color: #0b1f12;
    font-size: 20px;
    font-weight: 800;
    letter-spacing: 1px;
    min-height: 54px;
}}
QPushButton#StartButton:hover {{
    background-color: #4fe08c;
}}
QPushButton#StartButton[running="true"] {{
    background-color: {NEGATIVE};
    color: #ffffff;
}}
QPushButton#StartButton[running="true"]:hover {{
    background-color: #f06a5d;
}}
QLabel#SessionLabel {{
    color: {TEXT_MUTED};
    font-size: 12px;
}}
QPushButton#StepButton {{
    padding: 0;
    min-width: 28px;
    max-width: 28px;
    min-height: 28px;
    font-size: 16px;
}}
QGroupBox {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    margin-top: 22px;
    padding: 12px 12px 10px 12px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 4px;
    padding: 0 4px;
    color: {TEXT_MUTED};
}}
QGroupBox QWidget {{
    background-color: transparent;
    font-weight: normal;
}}
QGroupBox:disabled {{
    background-color: {BACKGROUND};
    border-style: dashed;
}}
QComboBox, QDoubleSpinBox, QSpinBox {{
    background-color: {SURFACE_RAISED};
    border: 1px solid {BORDER_STRONG};
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 22px;
}}
QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus {{
    border-color: {ACCENT};
}}
QComboBox QAbstractItemView {{
    background-color: {SURFACE_RAISED};
    border: 1px solid {BORDER_STRONG};
    selection-background-color: {ACCENT};
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: {BORDER_STRONG};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {TEXT};
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal:disabled {{
    background: {BORDER_STRONG};
}}
QSlider::handle:horizontal:disabled {{
    background: {TEXT_MUTED};
}}
QCheckBox {{
    spacing: 8px;
}}
"""

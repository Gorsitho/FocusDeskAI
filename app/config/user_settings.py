"""Preferences the user edits in the setup and Settings windows, persisted as JSON."""

import dataclasses
import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from app.config.settings import (
    MAX_MONITOR_DISTANCE,
    MAX_MONITORS,
    MIN_MONITOR_DISTANCE,
    CameraEdge,
    MonitorPlacement,
    Settings,
    StudyMethod,
    Workspace,
    arc_layout,
)

logger = logging.getLogger(__name__)

MIN_SENSITIVITY, MAX_SENSITIVITY = 1, 10
MONITOR_COUNTS = (1, 2, 3)
LANGUAGES = ("en", "es", "de")


@dataclass(frozen=True)
class UserSettings:
    study_method: StudyMethod = StudyMethod.COMPUTER
    workspace: Workspace = field(default_factory=Workspace)
    distraction_after_s: float = 5.0
    phone_distraction_after_s: float = 3.0
    # Time without any detected movement before the user counts as AWAY.
    no_movement_away_after_s: float = 60.0
    # 1 = tolerant (wide monitor zones), 10 = strict.
    sensitivity: int = 5
    sound_enabled: bool = True
    sound_volume: int = 40  # percent
    show_state_timers: bool = True
    language: str = "en"
    # Draw face/pose/phone markers on the camera preview (detection runs either way).
    show_landmarks: bool = False

    def normalized(self) -> "UserSettings":
        """Clamp every field into its valid range."""
        return dataclasses.replace(
            self,
            study_method=_study_method(self.study_method),
            workspace=normalize_workspace(self.workspace),
            distraction_after_s=_clamp(self.distraction_after_s, 1.0, 600.0),
            phone_distraction_after_s=_clamp(self.phone_distraction_after_s, 1.0, 600.0),
            no_movement_away_after_s=_clamp(self.no_movement_away_after_s, 5.0, 3600.0),
            sensitivity=int(_clamp(self.sensitivity, MIN_SENSITIVITY, MAX_SENSITIVITY)),
            sound_volume=int(_clamp(self.sound_volume, 0, 100)),
            language=self.language if self.language in LANGUAGES else "en",
        )

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        data["study_method"] = self.study_method.value
        ws = self.workspace
        data["workspace"] = {
            "person_x": ws.person_x,
            "person_z": ws.person_z,
            "monitors": [dataclasses.asdict(m) for m in ws.monitors],
            "camera_monitor": ws.camera_monitor,
            "camera_edge": ws.camera_edge.value,
        }
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "UserSettings":
        """Build settings from JSON data, ignoring unknown keys and bad values."""
        defaults = cls()
        values = {}
        for f in dataclasses.fields(cls):
            if f.name == "workspace" or f.name not in data:
                continue
            default = getattr(defaults, f.name)
            raw = data[f.name]
            try:
                if isinstance(default, bool):
                    if not isinstance(raw, bool):
                        raise ValueError(raw)
                    values[f.name] = raw
                elif isinstance(default, (int, float)):
                    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
                        raise ValueError(raw)
                    values[f.name] = type(default)(raw)
                else:
                    values[f.name] = str(raw)
            except (TypeError, ValueError):
                logger.warning("Ignoring invalid setting %s=%r", f.name, raw)
        values["workspace"] = _workspace_from_dict(data)
        return cls(**values).normalized()


def _workspace_from_dict(data: dict) -> Workspace:
    raw = data.get("workspace")
    if not isinstance(raw, dict):
        # Settings saved by older versions only stored a monitor count.
        count = data.get("monitor_count", 1)
        count = count if count in MONITOR_COUNTS else 1
        middle, last = (count - 1) // 2, count - 1
        camera_monitor, camera_edge = {
            "top_left": (0, CameraEdge.TOP),
            "top_right": (last, CameraEdge.TOP),
            "left": (0, CameraEdge.LEFT),
            "right": (last, CameraEdge.RIGHT),
            "bottom_center": (middle, CameraEdge.BOTTOM),
        }.get(data.get("camera_position"), (middle, CameraEdge.TOP))
        return Workspace(monitors=arc_layout(count), camera_monitor=camera_monitor, camera_edge=camera_edge)
    try:
        monitors = tuple(
            MonitorPlacement(**{k: float(m[k]) for k in ("x", "z", "angle", "width", "height") if k in m})
            for m in raw.get("monitors", [])
        )
        return Workspace(
            person_x=float(raw.get("person_x", 0.0)),
            person_z=float(raw.get("person_z", 0.0)),
            monitors=monitors or Workspace().monitors,
            camera_monitor=int(raw.get("camera_monitor", 0)),
            camera_edge=CameraEdge(raw.get("camera_edge", CameraEdge.TOP.value)),
        )
    except (TypeError, ValueError, KeyError, AttributeError):
        logger.warning("Ignoring invalid workspace %r", raw)
        return Workspace()


def normalize_workspace(ws: Workspace) -> Workspace:
    """Keep 1-3 monitors, each 0.3-1 m from the person, and a valid camera mount."""
    monitors = tuple(clamp_monitor(m, ws.person_x, ws.person_z) for m in ws.monitors[:MAX_MONITORS])
    if not monitors:
        monitors = arc_layout(1, person=(ws.person_x, ws.person_z))
    camera = ws.camera_monitor if 0 <= ws.camera_monitor < len(monitors) else 0
    return dataclasses.replace(ws, monitors=monitors, camera_monitor=camera)


def clamp_monitor(monitor: MonitorPlacement, person_x: float, person_z: float) -> MonitorPlacement:
    dx, dz = monitor.x - person_x, monitor.z - person_z
    distance = math.hypot(dx, dz)
    if distance < 1e-6:
        dx, dz, distance = 0.0, 1.0, 1.0
    clamped = _clamp(distance, MIN_MONITOR_DISTANCE, MAX_MONITOR_DISTANCE)
    angle = monitor.angle if -180.0 <= monitor.angle <= 180.0 else ((monitor.angle + 180.0) % 360.0) - 180.0
    if clamped == distance and angle == monitor.angle:
        return monitor
    scale = clamped / distance
    return dataclasses.replace(
        monitor, x=person_x + dx * scale, z=person_z + dz * scale, angle=angle
    )


def _study_method(value) -> StudyMethod:
    try:
        return StudyMethod(value)
    except ValueError:
        return StudyMethod.COMPUTER


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def default_settings_path() -> Path:
    base = os.environ.get("APPDATA") or os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "FocusDeskAI" / "user_settings.json"


def load_user_settings(path: Path) -> UserSettings:
    """Return the saved settings, or defaults if the file is missing or unreadable."""
    try:
        # utf-8-sig: tolerate a byte-order mark added by some Windows editors.
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return UserSettings()
    except (OSError, ValueError):
        logger.warning("Could not read %s; using default settings", path, exc_info=True)
        return UserSettings()
    if not isinstance(data, dict):
        return UserSettings()
    return UserSettings.from_dict(data)


def save_user_settings(user: UserSettings, path: Path) -> bool:
    """Write the settings atomically; returns False if the file could not be written."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(user.to_dict(), indent=2), encoding="utf-8")
        tmp_path.replace(path)
        return True
    except OSError:
        logger.exception("Could not save settings to %s", path)
        return False


def apply_user_settings(base: Settings, user: UserSettings) -> Settings:
    """Fold the user's preferences into the pipeline configuration."""
    user = user.normalized()
    s = user.sensitivity
    features = dataclasses.replace(
        base.features,
        workspace=user.workspace,
        study_method=user.study_method,
        # Sensitivity 1 -> 20.4 deg tolerance, 5 -> 14 deg, 10 -> 6 deg.
        attention_margin=22.0 - 1.6 * s,
        # Sensitivity 1 -> 37 deg, 5 -> 45 deg, 10 -> 55 deg.
        phone_gaze_max_angle=35.0 + 2.0 * s,
    )
    state = dataclasses.replace(
        base.state,
        look_away_after_s=user.distraction_after_s,
        phone_after_s=user.phone_distraction_after_s,
        still_away_after_s=user.no_movement_away_after_s,
    )
    return dataclasses.replace(base, features=features, state=state)

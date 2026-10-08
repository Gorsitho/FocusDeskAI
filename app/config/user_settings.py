"""Preferences the user edits in the setup and Settings windows, persisted as JSON."""

import dataclasses
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from app.config.settings import CameraPosition, Settings

logger = logging.getLogger(__name__)

MIN_SENSITIVITY, MAX_SENSITIVITY = 1, 10
MONITOR_COUNTS = (1, 2, 3)


@dataclass(frozen=True)
class UserSettings:
    monitor_count: int = 1
    camera_position: CameraPosition = CameraPosition.TOP_CENTER
    distraction_after_s: float = 5.0
    phone_distraction_after_s: float = 3.0
    idle_after_s: float = 60.0
    # 1 = tolerant (wide screen zone), 10 = strict.
    sensitivity: int = 5
    sound_enabled: bool = True
    sound_volume: int = 40  # percent
    show_state_timers: bool = True

    def normalized(self) -> "UserSettings":
        """Clamp every field into its valid range."""
        monitors = self.monitor_count if self.monitor_count in MONITOR_COUNTS else 1
        position = self.camera_position
        if not position.available_for(monitors):
            position = CameraPosition.TOP_CENTER
        return dataclasses.replace(
            self,
            monitor_count=monitors,
            camera_position=position,
            distraction_after_s=_clamp(self.distraction_after_s, 1.0, 600.0),
            phone_distraction_after_s=_clamp(self.phone_distraction_after_s, 1.0, 600.0),
            idle_after_s=_clamp(self.idle_after_s, 5.0, 3600.0),
            sensitivity=int(_clamp(self.sensitivity, MIN_SENSITIVITY, MAX_SENSITIVITY)),
            sound_volume=int(_clamp(self.sound_volume, 0, 100)),
        )

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        data["camera_position"] = self.camera_position.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "UserSettings":
        """Build settings from JSON data, ignoring unknown keys and bad values."""
        defaults = cls()
        values = {}
        for f in dataclasses.fields(cls):
            if f.name not in data:
                continue
            default = getattr(defaults, f.name)
            raw = data[f.name]
            try:
                if isinstance(default, CameraPosition):
                    values[f.name] = CameraPosition(raw)
                elif isinstance(default, bool):
                    if not isinstance(raw, bool):
                        raise ValueError(raw)
                    values[f.name] = raw
                else:
                    values[f.name] = type(default)(raw)
            except (TypeError, ValueError):
                logger.warning("Ignoring invalid setting %s=%r", f.name, raw)
        return cls(**values).normalized()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def default_settings_path() -> Path:
    base = os.environ.get("APPDATA") or os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "FocusDeskAI" / "user_settings.json"


def load_user_settings(path: Path) -> UserSettings:
    """Return the saved settings, or defaults if the file is missing or unreadable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return UserSettings()
    except (OSError, ValueError):
        logger.warning("Could not read %s; using default settings", path, exc_info=True)
        return UserSettings()
    if not isinstance(data, dict):
        return UserSettings()
    return UserSettings.from_dict(data)


def save_user_settings(user: UserSettings, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(user.to_dict(), indent=2), encoding="utf-8")
        tmp_path.replace(path)
    except OSError:
        logger.exception("Could not save settings to %s", path)


def apply_user_settings(base: Settings, user: UserSettings) -> Settings:
    """Fold the user's preferences into the pipeline configuration."""
    user = user.normalized()
    s = user.sensitivity
    features = dataclasses.replace(
        base.features,
        monitor_count=user.monitor_count,
        camera_position=user.camera_position,
        # Sensitivity 1 -> 20.4 deg tolerance, 5 -> 14 deg, 10 -> 6 deg.
        attention_margin=22.0 - 1.6 * s,
        # Sensitivity 1 -> 37 deg, 5 -> 45 deg, 10 -> 55 deg.
        phone_gaze_max_angle=35.0 + 2.0 * s,
    )
    state = dataclasses.replace(
        base.state,
        look_away_after_s=user.distraction_after_s,
        phone_after_s=user.phone_distraction_after_s,
        still_idle_after_s=user.idle_after_s,
    )
    return dataclasses.replace(base, features=features, state=state)

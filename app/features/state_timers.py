"""
This file adds up the time spent in each focus state (FOCUSED, DISTRACTED, AWAY, BREAK).
It also formats a duration for the screen as MM:SS or H:MM:SS.

It is used by app/features/session.py for the session totals, and by
app/ui/dashboard.py and app/ui/main_window.py to show the timers.
"""

from app.features.feature_pipeline import FocusState


class StateTimers:
    """Adds up how long each state was active. The caller reports every state change with update()."""
    def __init__(self, now: float):
        self.reset(now)

    def reset(self, now: float) -> None:
        self._totals = {state: 0.0 for state in FocusState}
        self._current: FocusState | None = None
        self._since = now

    def update(self, now: float, state: FocusState | None) -> None:
        """Credit the time since the last call to the previous state, then switch."""
        if self._current is not None:
            self._totals[self._current] += max(now - self._since, 0.0)
        self._current, self._since = state, now

    @property
    def current(self) -> FocusState | None:
        return self._current

    def totals(self, now: float) -> dict[FocusState, float]:
        totals = dict(self._totals)
        if self._current is not None:
            totals[self._current] += max(now - self._since, 0.0)
        return totals


def format_duration(seconds: float) -> str:
    """MM:SS, or H:MM:SS from one hour on."""
    total = int(max(seconds, 0.0))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"

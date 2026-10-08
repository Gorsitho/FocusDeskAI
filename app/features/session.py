"""
This file controls a work session. A session starts when the user presses
START and ends with STOP SESSION. During a session it adds up how long the
user was in each state.

Durations are measured with a monotonic clock, which does not jump when the
computer's clock changes. The normal clock is only used for the start and end
times that are written to the log.

It uses StateTimers from app/features/state_timers.py. app/ui/main_window.py
controls it, and app/data/session_log.py saves its result (SessionSummary).
"""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from app.features.feature_pipeline import FocusState
from app.features.state_timers import StateTimers


def _local_now() -> datetime:
    return datetime.now().astimezone()


@dataclass(frozen=True)
class SessionSummary:
    """Result of a finished session: number, start, end and the time spent in each state."""
    number: int
    started_at: datetime
    ended_at: datetime
    total_s: float
    durations: dict[FocusState, float]

    @property
    def untracked_s(self) -> float:
        """Time without any state (e.g. the camera was unavailable)."""
        return max(self.total_s - sum(self.durations.values()), 0.0)

    def share(self, seconds: float) -> float:
        return 100.0 * seconds / self.total_s if self.total_s > 0 else 0.0


class SessionController:
    """Accumulates per-state time only while a session is active."""

    def __init__(self, monotonic: Callable[[], float] = time.monotonic,
                 wall_clock: Callable[[], datetime] = _local_now):
        self._monotonic = monotonic
        self._wall_clock = wall_clock
        self._timers: StateTimers | None = None
        self._number: int | None = None
        self._started_mono = 0.0
        self._started_at: datetime | None = None
        self._state: FocusState | None = None

    @property
    def active(self) -> bool:
        return self._timers is not None

    @property
    def number(self) -> int | None:
        return self._number

    def set_state(self, state: FocusState | None) -> None:
        """Report the displayed state; it is credited only while a session runs."""
        self._state = state
        if self._timers is not None and state is not self._timers.current:
            self._timers.update(self._monotonic(), state)

    def start(self, number: int) -> None:
        if self.active:
            raise RuntimeError("A session is already running")
        now = self._monotonic()
        self._number = number
        self._started_mono = now
        self._started_at = self._wall_clock()
        self._timers = StateTimers(now)
        self._timers.update(now, self._state)

    def elapsed(self) -> float:
        return self._monotonic() - self._started_mono if self.active else 0.0

    def totals(self) -> dict[FocusState, float]:
        if self._timers is None:
            return {state: 0.0 for state in FocusState}
        return self._timers.totals(self._monotonic())

    @property
    def current(self) -> FocusState | None:
        return self._timers.current if self._timers is not None else None

    def stop(self) -> SessionSummary:
        if self._timers is None:
            raise RuntimeError("No session is running")
        now = self._monotonic()
        summary = SessionSummary(
            number=self._number,
            started_at=self._started_at,
            ended_at=self._wall_clock(),
            total_s=now - self._started_mono,
            durations=self._timers.totals(now),
        )
        self._timers = None
        self._number = None
        self._started_at = None
        return summary

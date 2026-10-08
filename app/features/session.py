"""A work session: started with START, ended with STOP SESSION.

Durations always come from a monotonic clock (immune to clock changes); the
wall clock is only used for the start/end timestamps written to the log.
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

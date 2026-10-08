"""
This file saves one text log file for every finished work session:
sessions/session_0000.log, session_0001.log, and so on.

Each file shows the start and end time of the session and how long the user
was in each state (FOCUSED, DISTRACTED, AWAY, BREAK). An existing file is never overwritten.

It gets the session result (SessionSummary) from app/features/session.py.
app/ui/main_window.py uses it when a session ends.
"""

import logging
import re
from pathlib import Path

from app.features.feature_pipeline import FocusState
from app.features.session import SessionSummary

logger = logging.getLogger(__name__)

_NAME = re.compile(r"^session_(\d{4,})\.log$")


def session_file_name(number: int) -> str:
    return f"session_{number:04d}.log"


def format_duration_precise(seconds: float) -> str:
    """HH:MM:SS.mmm"""
    millis = int(round(max(seconds, 0.0) * 1000))
    hours, rest = divmod(millis, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _timestamp(value) -> str:
    return value.isoformat(sep=" ", timespec="milliseconds")


def format_session(summary: SessionSummary) -> str:
    """Build the text of one session log file."""
    lines = [
        "FocusDesk AI - session log",
        "==========================",
        f"Session number : {summary.number:04d}",
        f"Start          : {_timestamp(summary.started_at)}",
        f"End            : {_timestamp(summary.ended_at)}",
        f"Total duration : {format_duration_precise(summary.total_s)} ({summary.total_s:.3f} s)",
        "",
        f"{'State':<12} {'Duration':>12} {'Seconds':>12} {'Share':>8}",
    ]
    rows = [(state.value, summary.durations.get(state, 0.0)) for state in FocusState]
    rows.append(("NO DATA", summary.untracked_s))  # camera unavailable during the session
    for name, seconds in rows:
        lines.append(f"{name:<12} {format_duration_precise(seconds):>12} {seconds:>12.3f} "
                     f"{summary.share(seconds):>7.1f}%")
    return "\n".join(lines) + "\n"


class SessionLogStore:
    """Writes session log files into one folder and finds the next free session number."""
    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def existing_numbers(self) -> list[int]:
        if not self.directory.is_dir():
            return []
        numbers = []
        for path in self.directory.iterdir():
            match = _NAME.match(path.name)
            if match:
                numbers.append(int(match.group(1)))
        return sorted(numbers)

    def next_number(self) -> int:
        numbers = self.existing_numbers()
        return numbers[-1] + 1 if numbers else 0

    def write(self, summary: SessionSummary) -> tuple[Path, SessionSummary]:
        """Write the session's single log file; never overwrites an existing one.

        If the number is already taken (e.g. a second app instance), the next free
        number is used and returned in the updated summary.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        number = summary.number
        while True:
            path = self.directory / session_file_name(number)
            try:
                with open(path, "x", encoding="utf-8") as handle:
                    if number != summary.number:
                        summary = SessionSummary(number, summary.started_at, summary.ended_at,
                                                 summary.total_s, summary.durations)
                    handle.write(format_session(summary))
                logger.info("Session %04d saved to %s (%.1f s)", number, path, summary.total_s)
                return path, summary
            except FileExistsError:
                logger.warning("%s already exists; using the next number", path.name)
                number = max(number + 1, self.next_number())

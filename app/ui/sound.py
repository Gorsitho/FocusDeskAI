"""Soft repeating chime played while the user is distracted."""

import io
import logging
import tempfile
import wave
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QUrl
from PySide6.QtMultimedia import QAudio, QSoundEffect

logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100
# One loop = a short two-note chime followed by silence, so it repeats every few seconds.
LOOP_SECONDS = 3.0
_NOTES = ((659.25, 0.0), (523.25, 0.16))  # E5 then C5: a gentle descending interval
_NOTE_SECONDS = 0.9
_PEAK = 0.3


def build_chime_wav(sample_rate: int = SAMPLE_RATE, loop_seconds: float = LOOP_SECONDS) -> bytes:
    """Return a mono 16-bit WAV file with a soft chime and trailing silence."""
    samples = np.zeros(int(sample_rate * loop_seconds), dtype=np.float64)
    t = np.arange(int(sample_rate * _NOTE_SECONDS)) / sample_rate
    # Fast fade-in avoids a click; exponential decay keeps it bell-like and soft.
    envelope = np.minimum(t / 0.015, 1.0) * np.exp(-t * 5.0)
    for freq, start in _NOTES:
        # A quiet octave partial makes the tone rounder than a bare sine.
        tone = np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(4 * np.pi * freq * t)
        begin = int(start * sample_rate)
        end = min(begin + len(t), len(samples))
        samples[begin:end] += (tone * envelope)[: end - begin]
    samples *= _PEAK / max(np.abs(samples).max(), 1e-9)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes((samples * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def _chime_file() -> Path:
    path = Path(tempfile.gettempdir()) / "focusdesk_ai_chime.wav"
    path.write_bytes(build_chime_wav())
    return path


class DistractionSound(QObject):
    """Loops the chime while active; also offers a one-shot preview."""

    def __init__(self, enabled: bool, volume_percent: int, parent=None):
        super().__init__(parent)
        self._enabled = enabled
        self._active = False
        try:
            url = QUrl.fromLocalFile(str(_chime_file()))
        except OSError:
            logger.exception("Could not create the alert sound; sound disabled")
            url = QUrl()
        self._loop = QSoundEffect(self)
        self._loop.setSource(url)
        self._loop.setLoopCount(QSoundEffect.Loop.Infinite.value)
        self._preview = QSoundEffect(self)
        self._preview.setSource(url)
        self.set_volume(volume_percent)

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self._sync()

    def set_volume(self, percent: int) -> None:
        # Perceptual (logarithmic) slider -> linear gain expected by QSoundEffect.
        linear = QAudio.convertVolume(
            max(0, min(100, percent)) / 100.0,
            QAudio.VolumeScale.LogarithmicVolumeScale,
            QAudio.VolumeScale.LinearVolumeScale,
        )
        self._loop.setVolume(linear)
        self._preview.setVolume(linear)

    def set_active(self, active: bool) -> None:
        if active != self._active:
            self._active = active
            self._sync()

    def preview(self, volume_percent: int) -> None:
        self.set_volume(volume_percent)
        self._preview.stop()
        self._preview.play()

    def stop(self) -> None:
        self._active = False
        self._sync()
        self._preview.stop()

    def _sync(self) -> None:
        should_play = self._enabled and self._active
        if should_play:
            if not self._loop.isPlaying():
                self._loop.play()
        else:
            # Unconditional: a play() requested while the file was still loading
            # does not report isPlaying() yet but would start later.
            self._loop.stop()

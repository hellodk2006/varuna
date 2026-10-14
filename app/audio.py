"""
Small sound-effect helper for MarineVision.

Two effects are used:
  - a short "sonar ping" while a detection is running
  - a sharper two-tone "alert" when a high-severity object is found

Both are tiny synthesized WAV files (no bundled audio assets needed), played
through PySide6's QtMultimedia if it's available. Audio backends vary a lot
across machines, so every step here is wrapped defensively: if anything
about sound playback fails, the app falls back to QApplication.beep() and,
failing even that, simply stays silent. Sound is a nice-to-have for the
demo, never something that should be able to crash or block detection.
"""

import math
import struct
import wave
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / "_sound_cache"


def _synthesize_wav(path: Path, freqs_and_durations, sample_rate=22050, volume=0.35):
    """Writes a simple mono 16-bit PCM WAV made of one or more sine tones
    played back to back, e.g. [(880, 0.12), (1320, 0.12)] for a two-note blip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for freq, duration in freqs_and_durations:
        n_samples = int(sample_rate * duration)
        for i in range(n_samples):
            # Gentle fade in/out to avoid clicky edges.
            fade = min(1.0, i / (sample_rate * 0.01), (n_samples - i) / (sample_rate * 0.01))
            sample = volume * fade * math.sin(2 * math.pi * freq * (i / sample_rate))
            frames.append(struct.pack("<h", int(sample * 32767)))

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))


def _ensure_sounds():
    ping_path = CACHE_DIR / "ping.wav"
    alert_path = CACHE_DIR / "alert.wav"
    if not ping_path.exists():
        _synthesize_wav(ping_path, [(1400, 0.09), (1000, 0.09)])
    if not alert_path.exists():
        _synthesize_wav(alert_path, [(1046, 0.11), (784, 0.11), (1046, 0.16)], volume=0.45)
    return ping_path, alert_path


class SoundPlayer:
    """Lazily initialised so importing this module never touches audio
    hardware until a sound is actually requested."""

    def __init__(self):
        self._ready = False
        self._effects = {}

    def _init(self):
        if self._ready:
            return
        self._ready = True
        try:
            from PySide6.QtMultimedia import QSoundEffect
            from PySide6.QtCore import QUrl

            ping_path, alert_path = _ensure_sounds()
            for name, path in (("ping", ping_path), ("alert", alert_path)):
                effect = QSoundEffect()
                effect.setSource(QUrl.fromLocalFile(str(path)))
                effect.setVolume(0.6)
                self._effects[name] = effect
        except Exception:
            # QtMultimedia missing, no audio device, unsupported platform, etc.
            # Playback falls back to QApplication.beep() below.
            self._effects = {}

    def play(self, name: str):
        self._init()
        effect = self._effects.get(name)
        try:
            if effect is not None:
                effect.play()
                return
        except Exception:
            pass
        # Fallback: a plain system beep still gives audible feedback.
        try:
            from PySide6.QtWidgets import QApplication
            QApplication.beep()
        except Exception:
            pass


_player = SoundPlayer()


def play_ping():
    """Short blip - used when a detection run starts/completes normally."""
    _player.play("ping")


def play_alert():
    """Sharper tone - used when a high-severity object is detected."""
    _player.play("alert")

from pathlib import Path

import pytest

from directorx.services import providers
from directorx.services.providers import EdgeSpeechTTS


def test_edge_tts_retries_transient_failures(monkeypatch, tmp_path: Path) -> None:
    attempts = 0
    tts = EdgeSpeechTTS(max_retries=2, retry_delay_s=0)

    def synthesize(_text: str, output_path: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("temporary outage")
        output_path.write_bytes(b"wav")

    monkeypatch.setattr(tts, "_edge_tts", synthesize)
    monkeypatch.setattr(providers, "_probe_duration", lambda _path: 1.25)

    duration = tts._synthesize_sync("hello", tmp_path / "speech.wav")

    assert attempts == 3
    assert duration == 1.25


def test_edge_tts_raises_after_retry_budget(monkeypatch, tmp_path: Path) -> None:
    attempts = 0
    tts = EdgeSpeechTTS(max_retries=1, retry_delay_s=0)

    def synthesize(_text: str, _output_path: Path) -> None:
        nonlocal attempts
        attempts += 1
        raise ConnectionError("still unavailable")

    monkeypatch.setattr(tts, "_edge_tts", synthesize)

    with pytest.raises(ConnectionError, match="still unavailable"):
        tts._synthesize_sync("hello", tmp_path / "speech.wav")

    assert attempts == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_retries": -1}, "max_retries"),
        ({"retry_delay_s": -1}, "retry_delay_s"),
    ],
)
def test_edge_tts_rejects_invalid_retry_settings(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EdgeSpeechTTS(**kwargs)

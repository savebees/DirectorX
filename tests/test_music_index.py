from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from directorx.core.models import MusicTrack
from directorx.services.sound import LocalMusicIndexBuilder
from tests.fakes import FixedAudioTextEmbeddingProvider, FixedMusicLibrary


def _tracks(tmp_path: Path) -> list[MusicTrack]:
    return [
        MusicTrack(
            path=tmp_path / "calm.mp3",
            title="Calm",
            tags=["calm"],
            duration_s=25,
        ),
        MusicTrack(
            path=tmp_path / "short.wav",
            title="Short",
            tags=["tense"],
            duration_s=4,
        ),
    ]


def test_builder_aggregates_windows_and_persists_index(tmp_path: Path) -> None:
    provider = FixedAudioTextEmbeddingProvider(
        {"calm.mp3": [[3.0, 0.0]], "short.wav": [[0.0, 4.0]]}
    )
    builder = LocalMusicIndexBuilder(
        FixedMusicLibrary(_tracks(tmp_path)),
        provider,
        model_name="fake-clap",
        analysis_window_s=10,
        analysis_windows_per_track=3,
    )

    index = asyncio.run(builder.build())
    path = asyncio.run(builder.build_and_persist(tmp_path / "artifacts"))

    assert index.model_name == "fake-clap"
    assert index.embedding_dimension == 2
    assert [entry.track.title for entry in index.entries] == ["Calm", "Short"]
    assert len(index.entries[0].analysis_windows) == 3
    assert len(index.entries[1].analysis_windows) == 1
    assert path.name == "music-index.json"
    assert path.is_file()


def test_builder_does_not_overwrite_existing_index(tmp_path: Path) -> None:
    provider = FixedAudioTextEmbeddingProvider({"calm.mp3": [[1.0, 0.0]]})
    builder = LocalMusicIndexBuilder(
        FixedMusicLibrary(_tracks(tmp_path)[:1]),
        provider,
        model_name="fake-clap",
    )
    destination = tmp_path / "artifacts"
    destination.mkdir()
    existing = destination / "music-index.json"
    existing.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        asyncio.run(builder.build_and_persist(destination))
    assert existing.read_text(encoding="utf-8") == "keep"
    assert list(destination.glob(".music-index.*")) == []


def test_builder_propagates_model_failure_without_partial_index(tmp_path: Path) -> None:
    tracks = _tracks(tmp_path)
    provider = FixedAudioTextEmbeddingProvider(
        {"calm.mp3": [[1.0, 0.0]], "short.wav": [[0.0, 1.0]]},
        failing_path=tracks[0].path,
    )
    builder = LocalMusicIndexBuilder(
        FixedMusicLibrary(tracks), provider, model_name="fake-clap"
    )

    with pytest.raises(RuntimeError, match="audio embedding unavailable"):
        asyncio.run(builder.build_and_persist(tmp_path / "artifacts"))
    assert not (tmp_path / "artifacts" / "music-index.json").exists()

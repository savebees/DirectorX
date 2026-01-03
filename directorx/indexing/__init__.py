"""Video indexing and scene-search infrastructure."""

from .backends import (
    AutoTranscriber,
    EmbeddedSubtitleTranscriber,
    FasterWhisperTranscriber,
    FFmpegKeyframeExtractor,
    HashingEmbeddingProvider,
    NullTranscriber,
    NoEmbeddedSubtitleError,
    PySceneDetectDetector,
    SentenceTransformerEmbeddingProvider,
    SidecarSubtitleTranscriber,
)
from .indexer import HybridVideoIndexer
from .store import SceneSearchStore
from .tools import SceneSearchTools
from .vlm import OpenAICompatibleDenseCaptioner

__all__ = [
    "EmbeddedSubtitleTranscriber",
    "AutoTranscriber",
    "FFmpegKeyframeExtractor",
    "FasterWhisperTranscriber",
    "HashingEmbeddingProvider",
    "HybridVideoIndexer",
    "OpenAICompatibleDenseCaptioner",
    "NullTranscriber",
    "NoEmbeddedSubtitleError",
    "PySceneDetectDetector",
    "SceneSearchStore",
    "SceneSearchTools",
    "SentenceTransformerEmbeddingProvider",
    "SidecarSubtitleTranscriber",
]

"""Video indexing and scene-search infrastructure."""

from .backends import (
    EmbeddedSubtitleTranscriber,
    FasterWhisperTranscriber,
    FFmpegKeyframeExtractor,
    HashingEmbeddingProvider,
    NullTranscriber,
    PySceneDetectDetector,
    SentenceTransformerEmbeddingProvider,
    SidecarSubtitleTranscriber,
)
from .indexer import HybridVideoIndexer
from .store import SceneSearchStore
from .tools import SceneSearchTools
from .vlm import OpenAICompatibleSceneAnnotator

__all__ = [
    "EmbeddedSubtitleTranscriber",
    "FFmpegKeyframeExtractor",
    "FasterWhisperTranscriber",
    "HashingEmbeddingProvider",
    "HybridVideoIndexer",
    "OpenAICompatibleSceneAnnotator",
    "NullTranscriber",
    "PySceneDetectDetector",
    "SceneSearchStore",
    "SceneSearchTools",
    "SentenceTransformerEmbeddingProvider",
    "SidecarSubtitleTranscriber",
]

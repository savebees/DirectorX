"""Video indexing and scene-search infrastructure."""

from .backends import (
    AutoTranscriber,
    ClipShotVisualEmbeddingProvider,
    EmbeddedSubtitleTranscriber,
    FasterWhisperTranscriber,
    HashingEmbeddingProvider,
    NoEmbeddedSubtitleError,
    NullTranscriber,
    PySceneDetectDetector,
    SentenceTransformerEmbeddingProvider,
    ShotKeyframeSelector,
    SidecarSubtitleTranscriber,
)
from .grouping import VisualSceneGrouper
from .indexer import HybridVideoIndexer
from .store import SceneSearchStore
from .tools import SceneSearchTools
from .vlm import OpenAICompatibleDenseCaptioner

__all__ = [
    "EmbeddedSubtitleTranscriber",
    "AutoTranscriber",
    "ClipShotVisualEmbeddingProvider",
    "ShotKeyframeSelector",
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
    "VisualSceneGrouper",
]

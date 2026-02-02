"""
Speech Feature Extraction Module v4.0
Research-grade feature extraction with parallel processing support.

Components:
- AcousticFeatureExtractor: Jitter, shimmer, HNR, CPPS, formants (Praat)
- ProsodicFeatureExtractor: Speech rate, pauses, rhythm, tremor
- CompositeFeatureExtractor: Novel biomarkers (NII, VFMT, ACE, FCR)
- EmbeddingExtractor: Deep learning embeddings (Wav2Vec2)
- UnifiedFeatureExtractor: Orchestrator with parallel processing
"""

from .acoustic import AcousticFeatureExtractor, AcousticFeatures
from .prosodic import ProsodicFeatureExtractor, ProsodicFeatures, TremorAnalysis
from .composite import CompositeFeatureExtractor, CompositeBiomarkers
from .embeddings import EmbeddingExtractor
from .pipeline import UnifiedFeatureExtractor, UnifiedFeatures

__all__ = [
    # Core extractors
    "AcousticFeatureExtractor",
    "ProsodicFeatureExtractor", 
    "CompositeFeatureExtractor",
    "EmbeddingExtractor",
    
    # Unified pipeline
    "UnifiedFeatureExtractor",
    
    # Feature dataclasses
    "AcousticFeatures",
    "ProsodicFeatures",
    "CompositeBiomarkers",
    "UnifiedFeatures",
    "TremorAnalysis",
]

"""
Retinal Pipeline - Features Module

Contains feature extraction components (matching speech/features/ structure):
- preprocessing: Image preprocessing (CLAHE, color normalization, artifacts)
- vessel: Vessel biomarker extraction (AVR, tortuosity, fractal dimension)
- optic_disc: Optic disc analysis (CDR, ISNT rule, rim area)
- lesions: Lesion detection (microaneurysms, hemorrhages, exudates)
- composite: Composite biomarkers (RHI, VRS, MHS)

Author: NeuraLens Medical AI Team
Version: 4.0.0
"""

# Preprocessing
from ..preprocessing import (
    ImagePreprocessor,
    PreprocessingResult,
    PreprocessingConfig,
    ColorNormalizer,
    ContrastEnhancer,
    ArtifactRemover,
    FundusDetector,
    QualityScorer,
    image_preprocessor,
)

# Vessel analysis (matching speech/features/acoustic.py)
from .vessel import (
    VesselMetrics,
    VesselFeatureExtractor,
    vessel_extractor,
)

# Optic disc analysis (matching speech/features/prosodic.py)
from .optic_disc import (
    OpticDiscMetrics,
    OpticDiscExtractor,
    optic_disc_extractor,
)

# Lesion detection
from .lesions import (
    LesionMetrics,
    LesionLocation,
    LesionDetector,
    lesion_detector,
)

# Composite biomarkers (matching speech/features/composite.py)
from .composite import (
    CompositeBiomarkers,
    CompositeFeatureExtractor,
    composite_extractor,
)

# Biomarker extraction (main extractor - local module)
from .biomarker_extractor import biomarker_extractor, BiomarkerExtractor

# v5.0 Deep vessel analysis
from .vessel_deep import (
    DeepVesselAnalyzer,
    DeepVesselMetrics,
    VesselSegment,
    BranchingPoint,
    AVClassifier,
    KnudtsonCalculator,
    TortuosityCalculator,
    RetinalZones,
    deep_vessel_analyzer,
)

__all__ = [
    # Preprocessing
    "ImagePreprocessor",
    "PreprocessingResult", 
    "PreprocessingConfig",
    "ColorNormalizer",
    "ContrastEnhancer",
    "ArtifactRemover",
    "FundusDetector",
    "QualityScorer",
    "image_preprocessor",
    
    # Vessel
    "VesselMetrics",
    "VesselFeatureExtractor",
    "vessel_extractor",
    
    # Optic Disc
    "OpticDiscMetrics",
    "OpticDiscExtractor",
    "optic_disc_extractor",
    
    # Lesions
    "LesionMetrics",
    "LesionLocation",
    "LesionDetector",
    "lesion_detector",
    
    # Composite
    "CompositeBiomarkers",
    "CompositeFeatureExtractor",
    "composite_extractor",
    
    # Main extractor
    "biomarker_extractor",
    "BiomarkerExtractor",
    
    # v5.0 Deep Vessel Analysis
    "DeepVesselAnalyzer",
    "DeepVesselMetrics",
    "VesselSegment",
    "BranchingPoint",
    "AVClassifier",
    "KnudtsonCalculator",
    "TortuosityCalculator",
    "RetinalZones",
    "deep_vessel_analyzer",
]

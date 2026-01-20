"""
Retinal Pipeline - Models Module

Contains model definitions and versioning:
- models.py: Model architecture definitions
- versioning.py: Model version tracking
- pretrained.py: Production-ready pretrained models (v5.0)

Author: NeuraLens Medical AI Team
Version: 5.0.0
"""

from .models import RetinalAssessment, RetinalAuditLog

# Backward compatibility alias
DRModel = RetinalAssessment

from .versioning import ModelVersionManager

# v5.0 Pretrained models
from .pretrained import (
    PretrainedVesselSegmenter,
    VesselUNet,
    DRClassifier,
    MultiTaskRetinalModel,
    ModelConfig,
    ModelCache,
    VESSEL_SEGMENTATION_CONFIG,
    DR_CLASSIFICATION_CONFIG,
    get_vessel_segmenter,
    get_dr_classifier,
    get_multitask_model,
    get_default_vessel_segmenter,
    get_default_dr_classifier,
)

__all__ = [
    # Database models
    "RetinalAssessment",
    "RetinalAuditLog",
    "DRModel",  # Backward compatibility
    "ModelVersionManager",
    
    # v5.0 Pretrained Models
    "PretrainedVesselSegmenter",
    "VesselUNet",
    "DRClassifier",
    "MultiTaskRetinalModel",
    "ModelConfig",
    "ModelCache",
    "VESSEL_SEGMENTATION_CONFIG",
    "DR_CLASSIFICATION_CONFIG",
    "get_vessel_segmenter",
    "get_dr_classifier",
    "get_multitask_model",
    "get_default_vessel_segmenter",
    "get_default_dr_classifier",
]

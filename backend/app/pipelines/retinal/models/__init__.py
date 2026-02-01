"""
Retinal Pipeline - Models Module

Contains model definitions and versioning:
- models.py: Model architecture definitions
- versioning.py: Model version tracking
- pretrained.py: Production-ready pretrained models (v5.0)
- weight_manager.py: v5.1 weight download and management

Author: NeuraLens Medical AI Team
Version: 5.1.0
"""

import logging
_logger = logging.getLogger(__name__)

# Database models (may require database setup)
try:
    from .models import RetinalAssessment, RetinalAuditLog
    # Backward compatibility alias
    DRModel = RetinalAssessment
except ImportError as e:
    _logger.warning(f"Database models not available: {e}")
    RetinalAssessment = None
    RetinalAuditLog = None
    DRModel = None

# Version manager
try:
    from .versioning import ModelVersionManager
except ImportError as e:
    _logger.warning(f"Version manager not available: {e}")
    ModelVersionManager = None

# v5.0 Pretrained models
try:
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
except ImportError as e:
    _logger.warning(f"v5.0 pretrained models not available: {e}")
    PretrainedVesselSegmenter = None
    VesselUNet = None
    DRClassifier = None
    MultiTaskRetinalModel = None
    ModelConfig = None
    ModelCache = None
    VESSEL_SEGMENTATION_CONFIG = None
    DR_CLASSIFICATION_CONFIG = None
    get_vessel_segmenter = None
    get_dr_classifier = None
    get_multitask_model = None
    get_default_vessel_segmenter = None
    get_default_dr_classifier = None


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
    
    # v5.1 Weight Manager
    "PretrainedWeightManager",
    "weight_manager",
    "UNetVesselSegmenter",
    "DRClassifierV51",
    "ModelType",
]

# v5.1 Weight manager with download support
try:
    from .weight_manager import (
        PretrainedWeightManager,
        weight_manager,
        UNetVesselSegmenter,
        DRClassifier as DRClassifierV51,
        ModelType,
        get_vessel_segmenter as get_vessel_segmenter_v51,
        get_dr_classifier as get_dr_classifier_v51,
    )
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"Weight manager not available: {e}")

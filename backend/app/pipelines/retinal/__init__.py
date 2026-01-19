"""
Retinal Analysis Pipeline v4.0
Medical-grade fundus image analysis for screening and clinical decision support.

Architecture: Follows standardized pipeline structure per ARCHITECTURE-GUIDE.md
- Root: __init__.py, config.py, schemas.py, router.py (max 4 files)
- core/: orchestrator.py, service.py
- input/: validator.py
- preprocessing/: normalizer.py
- features/: biomarker_extractor.py, vessel.py, optic_disc.py, lesions.py
- clinical/: clinical_assessment.py, risk_scorer.py, graders.py
- output/: report_generator.py, visualization.py
- errors/: codes.py
- utils/: constants.py
- explanation/: rules.py

Author: NeuraLens Medical AI Team
Version: 4.0.0
"""

__version__ = "4.0.0"

# Router for FastAPI (allowed in root)
from .router import router

# Configuration (allowed in root)
from .config import (
    INPUT_CONSTRAINTS,
    QUALITY_THRESHOLDS,
    BIOMARKER_NORMAL_RANGES,
    BIOMARKER_ABNORMAL_THRESHOLDS,
    RISK_WEIGHTS,
    RISK_CATEGORIES,
    RECOMMENDATIONS,
    ICD10_CODES,
    DRGrade,
    DR_GRADE_CRITERIA,
    CONDITION_PATTERNS,
    DISCLAIMERS,
    SUPPORTED_MIME_TYPES,
    PROCESSING_CONFIG,
)

# Schemas (allowed in root)
from .schemas import (
    PipelineStage,
    PipelineError,
    PipelineState,
    RetinalAnalysisResponse,
    ImageQuality,
    CompleteBiomarkers,
    DiabeticRetinopathyResult,
    RiskAssessment,
    ClinicalFinding,
)

# Core - orchestration and service (from core/ subfolder)
from .core.orchestrator import (
    ExecutionContext,
    ReceiptConfirmation,
    execute_with_retry,
    handle_hard_stop,
    AuditLogger,
    SAFETY_DISCLAIMERS,
    get_disclaimers,
    create_execution_context,
)

# Error handling (from errors/ subfolder)
from .errors.codes import (
    ErrorSeverity,
    ERROR_CODES,
    get_error,
    PipelineException,
)

# Preprocessing (from preprocessing/ subfolder)
from .preprocessing import (
    ImagePreprocessor,
    PreprocessingResult,
    PreprocessingConfig,
    image_preprocessor,
)

# Features - biomarker extraction (from features/ subfolder)
from .features import biomarker_extractor

# Clinical assessment (from clinical/ subfolder)
from .clinical import (
    DRGrader,
    DMEAssessor,
    RiskCalculator,
    ClinicalFindingsGenerator,
    DifferentialGenerator,
    RecommendationGenerator,
    ClinicalSummaryGenerator,
    dr_grader,
    dme_assessor,
    risk_calculator,
    findings_generator,
    differential_generator,
    recommendation_generator,
    summary_generator,
)

# Submodules
from . import features
from . import clinical
from . import monitoring
from . import output
from . import core
from . import input
from . import errors
from . import utils
from . import explanation

__all__ = [
    "__version__",
    "router",
    # Schemas
    "PipelineStage",
    "PipelineError",
    "PipelineState",
    "RetinalAnalysisResponse",
    "ImageQuality",
    "CompleteBiomarkers",
    "DiabeticRetinopathyResult",
    "RiskAssessment",
    "ClinicalFinding",
    # Error handling
    "ErrorSeverity",
    "ERROR_CODES",
    "get_error",
    "PipelineException",
    # Core
    "ExecutionContext",
    "ReceiptConfirmation",
    "AuditLogger",
    "get_disclaimers",
    "create_execution_context",
    # Preprocessing
    "ImagePreprocessor",
    "PreprocessingResult",
    "PreprocessingConfig",
    "image_preprocessor",
    # Features
    "biomarker_extractor",
    # Clinical
    "DRGrader",
    "DMEAssessor",
    "RiskCalculator",
    "ClinicalFindingsGenerator",
    "RecommendationGenerator",
    "dr_grader",
    "dme_assessor",
    "risk_calculator",
    "findings_generator",
    "recommendation_generator",
    # Submodules
    "features",
    "clinical",
    "monitoring",
    "output",
    "core",
    "input",
    "errors",
    "utils",
    "explanation",
]

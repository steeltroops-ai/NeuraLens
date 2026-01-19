"""
Retinal Pipeline - Monitoring Module

Contains monitoring and compliance components (matching speech/monitoring/ structure):
- audit_logger: HIPAA-compliant audit logging
- quality_checker: Image quality assessment
- drift_detector: Data/concept drift monitoring

Author: NeuraLens Medical AI Team
Version: 4.0.0
"""

# Audit logging (matching speech/monitoring/audit_logger.py)
from .audit_logger import (
    RetinalAuditLogger,
    AuditEntry,
    retinal_audit_logger,
)

# Quality checking (matching speech/monitoring/quality_checker.py)
from .quality_checker import (
    RetinalQualityChecker,
    QualityReport,
    retinal_quality_checker,
)

# Drift detection (matching speech/monitoring/drift_detector.py)
from .drift_detector import (
    DriftDetector,
    DriftReport,
    drift_detector,
)

# Orchestrator components (safety, disclaimers)
from ..core.orchestrator import (
    AuditLogger,
    SAFETY_DISCLAIMERS,
    get_disclaimers,
)

# Error handling
from ..errors.codes import (
    PipelineException,
    get_error,
    ERROR_CODES,
    ErrorSeverity,
)

__all__ = [
    # Audit
    "RetinalAuditLogger",
    "AuditEntry",
    "retinal_audit_logger",
    "AuditLogger",
    
    # Quality
    "RetinalQualityChecker",
    "QualityReport",
    "retinal_quality_checker",
    
    # Drift
    "DriftDetector",
    "DriftReport",
    "drift_detector",
    
    # Safety
    "SAFETY_DISCLAIMERS",
    "get_disclaimers",
    
    # Errors
    "PipelineException",
    "get_error",
    "ERROR_CODES",
    "ErrorSeverity",
]

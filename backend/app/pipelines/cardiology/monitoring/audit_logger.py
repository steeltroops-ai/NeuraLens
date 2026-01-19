"""
Cardiology Pipeline - Audit Logger
HIPAA-compliant logging for cardiology analysis.
"""

import logging
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger("cardiology.audit")


@dataclass
class AuditEntry:
    """Audit log entry."""
    timestamp: str
    request_id: str
    user_id_hash: Optional[str]
    session_id: Optional[str]
    
    # Input summary (no PHI)
    modalities: list
    ecg_duration_sec: Optional[float]
    metadata_provided: bool
    
    # Processing info
    processing_time_ms: int
    stages_completed: list
    
    # Results summary (no PHI)
    risk_category: str
    risk_score: float
    conditions_flagged: list
    confidence_overall: float
    
    # Quality metrics
    ecg_quality_score: Optional[float]
    snr_db: Optional[float]
    
    # System info
    model_versions: Dict[str, str]
    warnings_issued: list
    disclaimer_presented: bool


class CardiologyAuditLogger:
    """
    HIPAA-compliant audit logger for cardiology pipeline.
    
    Logs analysis events without storing PHI:
    - Request/response metadata
    - Processing times
    - Results summary (no raw data)
    - Quality metrics
    - Errors and warnings
    """
    
    def __init__(self, log_level: str = "INFO"):
        self.logger = logging.getLogger("cardiology.audit")
        self.logger.setLevel(getattr(logging, log_level))
    
    def log_analysis(
        self,
        request_id: str,
        modalities: list,
        processing_time_ms: int,
        stages: list,
        risk_category: str,
        risk_score: float,
        conditions: list,
        confidence: float,
        ecg_quality: Optional[float] = None,
        snr_db: Optional[float] = None,
        ecg_duration: Optional[float] = None,
        metadata_provided: bool = False,
        warnings: Optional[list] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Log an analysis event.
        
        Args:
            request_id: Unique request identifier
            modalities: Input modalities used
            processing_time_ms: Total processing time
            stages: Pipeline stages completed
            risk_category: Computed risk category
            risk_score: Numeric risk score
            conditions: Conditions flagged
            confidence: Overall confidence
            ecg_quality: ECG quality score
            snr_db: Signal-to-noise ratio
            ecg_duration: ECG duration in seconds
            metadata_provided: Whether metadata was provided
            warnings: List of warnings issued
            user_id: User ID (will be hashed)
            session_id: Session identifier
        """
        # Hash user ID for de-identification
        user_id_hash = None
        if user_id:
            user_id_hash = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        
        entry = AuditEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            request_id=request_id,
            user_id_hash=user_id_hash,
            session_id=session_id,
            modalities=modalities,
            ecg_duration_sec=ecg_duration,
            metadata_provided=metadata_provided,
            processing_time_ms=processing_time_ms,
            stages_completed=stages,
            risk_category=risk_category,
            risk_score=risk_score,
            conditions_flagged=conditions,
            confidence_overall=confidence,
            ecg_quality_score=ecg_quality,
            snr_db=snr_db,
            model_versions={
                "pipeline": "3.0.0",
                "rhythm_classifier": "1.0.0",
                "arrhythmia_detector": "1.0.0",
            },
            warnings_issued=warnings or [],
            disclaimer_presented=True,
        )
        
        self.logger.info(json.dumps(asdict(entry)))
    
    def log_error(
        self,
        request_id: str,
        error_code: str,
        error_message: str,
        stage: str,
        processing_time_ms: int,
    ) -> None:
        """Log an error event."""
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "request_id": request_id,
            "event_type": "error",
            "error_code": error_code,
            "error_message": error_message,
            "stage": stage,
            "processing_time_ms": processing_time_ms,
        }
        
        self.logger.error(json.dumps(entry))


# Convenience function
def log_analysis(
    request_id: str,
    response: Any,
    modalities: list,
    ecg_duration: Optional[float] = None,
) -> None:
    """Convenience function to log analysis."""
    audit_logger = CardiologyAuditLogger()
    
    audit_logger.log_analysis(
        request_id=request_id,
        modalities=modalities,
        processing_time_ms=response.processing_time_ms,
        stages=[s.stage for s in response.stages_completed],
        risk_category=response.risk_assessment.risk_category,
        risk_score=response.risk_assessment.risk_score,
        conditions=[f.title for f in response.findings if f.severity != "normal"],
        confidence=response.risk_assessment.confidence,
        ecg_quality=response.quality_assessment.ecg_quality.signal_quality_score if response.quality_assessment.ecg_quality else None,
        snr_db=response.quality_assessment.ecg_quality.snr_db if response.quality_assessment.ecg_quality else None,
        ecg_duration=ecg_duration,
    )

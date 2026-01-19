"""
Retinal Pipeline - Audit Logger
HIPAA-compliant audit logging for clinical deployments.

Matches speech/monitoring/audit_logger.py structure.

Author: NeuraLens Medical AI Team
Version: 4.0.0
"""

import logging
import hashlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """Single audit log entry."""
    # Session info
    session_id: str
    timestamp: str
    
    # Input info (no PHI - hashed)
    patient_id_hash: str
    image_hash: str
    image_size_bytes: int
    
    # Timing
    start_time: str
    end_time: str
    processing_duration_ms: int
    
    # Results (summary only)
    dr_grade: int
    risk_score: float
    risk_level: str
    confidence: float
    
    # Quality
    image_quality: float
    quality_issues: List[str]
    
    # Status
    success: bool
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    
    # Review flags
    requires_review: bool = False
    review_reason: Optional[str] = None
    
    # Model info
    model_version: str = "4.0.0"
    pipeline_version: str = "4.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "patient_id_hash": self.patient_id_hash,
            "image_hash": self.image_hash,
            "image_size_bytes": self.image_size_bytes,
            "processing_duration_ms": self.processing_duration_ms,
            "dr_grade": self.dr_grade,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "image_quality": self.image_quality,
            "quality_issues": self.quality_issues,
            "success": self.success,
            "error_code": self.error_code,
            "requires_review": self.requires_review,
            "review_reason": self.review_reason,
            "model_version": self.model_version,
            "pipeline_version": self.pipeline_version,
        }


class RetinalAuditLogger:
    """
    HIPAA-compliant audit logger for retinal analysis.
    
    Features:
    - De-identified logging (hashed IDs)
    - No image storage (hash only)
    - Result summary only
    - Structured for compliance queries
    """
    
    def __init__(self, log_to_file: bool = True, log_path: Optional[str] = None):
        self.log_to_file = log_to_file
        self.log_path = log_path or "logs/retinal_audit.jsonl"
        self.entries: List[AuditEntry] = []
    
    def create_entry(
        self,
        session_id: str,
        patient_id: Optional[str],
        image_bytes: bytes,
        start_time: datetime,
        end_time: datetime,
        dr_grade: int,
        risk_score: float,
        risk_level: str,
        confidence: float,
        image_quality: float,
        quality_issues: List[str],
        success: bool = True,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        requires_review: bool = False,
        review_reason: Optional[str] = None,
    ) -> AuditEntry:
        """Create an audit entry with de-identified data."""
        
        # Hash patient ID (never store raw)
        patient_id_hash = self._hash_id(patient_id or "ANONYMOUS")
        
        # Hash image content
        image_hash = self._hash_bytes(image_bytes)
        
        # Calculate duration
        duration_ms = int((end_time - start_time).total_seconds() * 1000)
        
        entry = AuditEntry(
            session_id=session_id,
            timestamp=datetime.utcnow().isoformat(),
            patient_id_hash=patient_id_hash,
            image_hash=image_hash,
            image_size_bytes=len(image_bytes),
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            processing_duration_ms=duration_ms,
            dr_grade=dr_grade,
            risk_score=risk_score,
            risk_level=risk_level,
            confidence=confidence,
            image_quality=image_quality,
            quality_issues=quality_issues,
            success=success,
            error_code=error_code,
            error_message=error_message,
            requires_review=requires_review,
            review_reason=review_reason,
        )
        
        return entry
    
    def log(self, entry: AuditEntry) -> None:
        """Log an audit entry."""
        self.entries.append(entry)
        
        # Log to logging system
        log_msg = (
            f"AUDIT|RETINAL|{entry.session_id}|"
            f"DR={entry.dr_grade}|"
            f"Risk={entry.risk_score:.1f}|"
            f"Quality={entry.image_quality:.2f}|"
            f"Success={entry.success}|"
            f"Duration={entry.processing_duration_ms}ms"
        )
        logger.info(log_msg)
        
        # Write to file if enabled
        if self.log_to_file:
            self._write_to_file(entry)
    
    def log_session_start(self, session_id: str, image_size: int) -> None:
        """Log session start event."""
        logger.info(f"AUDIT|RETINAL|SESSION_START|{session_id}|size={image_size}")
    
    def log_session_end(
        self, 
        session_id: str, 
        success: bool, 
        duration_ms: int,
        dr_grade: Optional[int] = None
    ) -> None:
        """Log session end event."""
        result = "SUCCESS" if success else "FAILURE"
        logger.info(
            f"AUDIT|RETINAL|SESSION_END|{session_id}|"
            f"{result}|DR={dr_grade}|duration={duration_ms}ms"
        )
    
    def log_error(self, session_id: str, error_code: str, message: str) -> None:
        """Log error event."""
        logger.error(f"AUDIT|RETINAL|ERROR|{session_id}|{error_code}|{message}")
    
    def log_review_required(self, session_id: str, reason: str) -> None:
        """Log review requirement."""
        logger.warning(f"AUDIT|RETINAL|REVIEW_REQUIRED|{session_id}|{reason}")
    
    def _hash_id(self, id_string: str) -> str:
        """Hash an identifier for de-identification."""
        return hashlib.sha256(id_string.encode()).hexdigest()[:16]
    
    def _hash_bytes(self, data: bytes) -> str:
        """Hash binary data."""
        return hashlib.sha256(data).hexdigest()[:32]
    
    def _write_to_file(self, entry: AuditEntry) -> None:
        """Write entry to audit log file."""
        try:
            import json
            import os
            
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def get_recent_entries(self, limit: int = 100) -> List[Dict]:
        """Get recent audit entries."""
        return [e.to_dict() for e in self.entries[-limit:]]


# Singleton instance
retinal_audit_logger = RetinalAuditLogger()

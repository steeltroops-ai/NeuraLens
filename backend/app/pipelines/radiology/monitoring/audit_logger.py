"""
Radiology Audit Logger

Logs all pipeline actions for audit trail.
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


logger = logging.getLogger("radiology.audit")


@dataclass
class AuditEntry:
    """Audit log entry."""
    timestamp: str
    request_id: str
    action: str
    stage: str
    success: bool
    duration_ms: float
    details: Optional[Dict] = None
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class AuditLogger:
    """
    Audit logger for radiology pipeline.
    
    Logs:
    - All requests with anonymized data
    - Stage transitions
    - Errors and warnings
    - Performance metrics
    """
    
    def __init__(self, log_to_file: bool = False):
        self.log_to_file = log_to_file
    
    def log_request(
        self,
        request_id: str,
        modality: str,
        file_size_mb: float,
        user_id: Optional[str] = None
    ):
        """Log incoming request."""
        entry = AuditEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            request_id=request_id,
            action="REQUEST_RECEIVED",
            stage="RECEIPT",
            success=True,
            duration_ms=0,
            details={
                "modality": modality,
                "file_size_mb": round(file_size_mb, 2)
            },
            user_id=user_id
        )
        self._log(entry)
    
    def log_stage(
        self,
        request_id: str,
        stage: str,
        success: bool,
        duration_ms: float,
        error: Optional[str] = None
    ):
        """Log stage transition."""
        entry = AuditEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            request_id=request_id,
            action=f"STAGE_{stage}",
            stage=stage,
            success=success,
            duration_ms=duration_ms,
            details={"error": error} if error else None
        )
        self._log(entry)
    
    def log_completion(
        self,
        request_id: str,
        success: bool,
        total_duration_ms: float,
        risk_level: Optional[str] = None,
        primary_finding: Optional[str] = None
    ):
        """Log request completion."""
        entry = AuditEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            request_id=request_id,
            action="REQUEST_COMPLETE",
            stage="COMPLETE" if success else "FAILED",
            success=success,
            duration_ms=total_duration_ms,
            details={
                "risk_level": risk_level,
                "primary_finding": primary_finding
            }
        )
        self._log(entry)
    
    def _log(self, entry: AuditEntry):
        """Write log entry."""
        log_data = json.dumps(entry.to_dict())
        
        if entry.success:
            logger.info(log_data)
        else:
            logger.warning(log_data)

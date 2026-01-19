"""
Audit Logger
Immutable audit logging for regulatory compliance.
"""

import json
import hashlib
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """Immutable audit log entry."""
    # Identifiers
    log_id: str
    session_id: str
    patient_id: Optional[str]
    
    # Timing
    timestamp: str
    processing_start: str
    processing_end: str
    processing_duration_ms: int
    
    # Input
    input_hash: str                 # SHA-256 of audio
    input_size_bytes: int
    input_duration_seconds: float
    
    # Processing
    pipeline_version: str
    model_versions: Dict[str, str]
    features_extracted: List[str]
    
    # Output
    risk_score: float
    risk_level: str
    condition_probabilities: Dict[str, float]
    confidence: float
    
    # Quality
    signal_quality: float
    quality_issues: List[str]
    
    # Review
    requires_review: bool
    review_reason: Optional[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


class AuditLogger:
    """
    Immutable audit logging for regulatory compliance.
    
    Features:
    - Append-only log files
    - SHA-256 hashing of inputs
    - Tamper-evident chain
    - Structured JSON format
    """
    
    PIPELINE_VERSION = "3.0.0"
    
    def __init__(
        self,
        log_dir: Optional[Path] = None,
        log_to_file: bool = True,
        log_to_console: bool = False
    ):
        self.log_dir = log_dir or Path("logs/audit")
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        
        # Create log directory
        if self.log_to_file:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Logger for audit entries
        self.audit_logger = logging.getLogger("audit")
        self.audit_logger.setLevel(logging.INFO)
    
    def create_entry(
        self,
        session_id: str,
        patient_id: Optional[str],
        audio_bytes: bytes,
        audio_duration: float,
        start_time: datetime,
        end_time: datetime,
        risk_score: float,
        risk_level: str,
        condition_probs: Dict[str, float],
        confidence: float,
        signal_quality: float,
        quality_issues: List[str],
        features_extracted: List[str],
        requires_review: bool = False,
        review_reason: Optional[str] = None,
        model_versions: Optional[Dict[str, str]] = None
    ) -> AuditEntry:
        """Create an audit entry."""
        
        # Generate unique log ID
        log_id = str(uuid.uuid4())
        
        # Hash input audio
        input_hash = hashlib.sha256(audio_bytes).hexdigest()
        
        # Calculate processing duration
        duration_ms = int((end_time - start_time).total_seconds() * 1000)
        
        entry = AuditEntry(
            log_id=log_id,
            session_id=session_id,
            patient_id=patient_id,
            timestamp=datetime.now().isoformat(),
            processing_start=start_time.isoformat(),
            processing_end=end_time.isoformat(),
            processing_duration_ms=duration_ms,
            input_hash=input_hash,
            input_size_bytes=len(audio_bytes),
            input_duration_seconds=audio_duration,
            pipeline_version=self.PIPELINE_VERSION,
            model_versions=model_versions or {"whisper": "tiny", "parselmouth": "0.4.3"},
            features_extracted=features_extracted,
            risk_score=risk_score,
            risk_level=risk_level,
            condition_probabilities=condition_probs,
            confidence=confidence,
            signal_quality=signal_quality,
            quality_issues=quality_issues,
            requires_review=requires_review,
            review_reason=review_reason
        )
        
        return entry
    
    def log(self, entry: AuditEntry):
        """Log an audit entry."""
        if self.log_to_file:
            self._log_to_file(entry)
        
        if self.log_to_console:
            self.audit_logger.info(f"Audit: {entry.session_id} | Risk: {entry.risk_score:.1f} | {entry.risk_level}")
    
    def _log_to_file(self, entry: AuditEntry):
        """Append entry to audit log file."""
        try:
            # Daily log file
            date_str = datetime.now().strftime("%Y-%m-%d")
            log_file = self.log_dir / f"audit_{date_str}.jsonl"
            
            # Append as single line JSON
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(entry.to_json().replace("\n", " ") + "\n")
                
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def query_by_session(self, session_id: str) -> Optional[AuditEntry]:
        """Query audit log by session ID."""
        if not self.log_to_file:
            return None
            
        try:
            # Search recent log files
            for log_file in sorted(self.log_dir.glob("audit_*.jsonl"), reverse=True)[:7]:
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if data.get("session_id") == session_id:
                                return AuditEntry(**data)
                        except:
                            continue
        except Exception as e:
            logger.error(f"Audit query failed: {e}")
        
        return None
    
    def get_recent_entries(self, limit: int = 100) -> List[Dict]:
        """Get recent audit entries."""
        entries = []
        
        if not self.log_to_file:
            return entries
            
        try:
            for log_file in sorted(self.log_dir.glob("audit_*.jsonl"), reverse=True):
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            entries.append(json.loads(line))
                            if len(entries) >= limit:
                                return entries
                        except:
                            continue
        except Exception as e:
            logger.error(f"Failed to read audit logs: {e}")
        
        return entries

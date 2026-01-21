"""
Cognitive Pipeline Error Codes - Production Grade
Layered error codes for traceability
"""

from enum import Enum
from typing import Dict, Optional
from dataclasses import dataclass


class ErrorLayer(str, Enum):
    """Pipeline layer where error originated"""
    HTTP = "HTTP"
    INPUT = "INP"
    PREPROCESSING = "PREP"
    FEATURE = "FEAT"
    CLINICAL = "CLIN"
    OUTPUT = "OUT"


@dataclass(frozen=True)
class ErrorDefinition:
    code: str
    message: str
    recoverable: bool
    http_status: int
    retry_after_ms: Optional[int] = None


class ErrorCode:
    """
    Standardized error codes: E_{LAYER}_{NUMBER}
    Each code maps to a definition with message, recoverability, and status.
    """
    
    # HTTP / Router Layer (400-499)
    E_HTTP_001 = ErrorDefinition("E_HTTP_001", "Invalid request format", True, 400)
    E_HTTP_002 = ErrorDefinition("E_HTTP_002", "Missing required field", True, 400)
    E_HTTP_003 = ErrorDefinition("E_HTTP_003", "Request body too large", True, 413)
    E_HTTP_004 = ErrorDefinition("E_HTTP_004", "Rate limit exceeded", True, 429, retry_after_ms=60000)
    
    # Input Validation Layer
    E_INP_001 = ErrorDefinition("E_INP_001", "Empty session data", True, 400)
    E_INP_002 = ErrorDefinition("E_INP_002", "Invalid task data structure", True, 400)
    E_INP_003 = ErrorDefinition("E_INP_003", "Session timestamp mismatch", True, 400)
    E_INP_004 = ErrorDefinition("E_INP_004", "Suspicious timing data detected", True, 400)
    E_INP_005 = ErrorDefinition("E_INP_005", "Invalid session_id format", True, 400)
    E_INP_006 = ErrorDefinition("E_INP_006", "Task events not monotonic", True, 400)
    E_INP_007 = ErrorDefinition("E_INP_007", "Insufficient events for analysis", True, 400)
    
    # Feature Extraction Layer
    E_FEAT_001 = ErrorDefinition("E_FEAT_001", "Feature extraction failed", False, 500)
    E_FEAT_002 = ErrorDefinition("E_FEAT_002", "Unknown task type", True, 400)
    E_FEAT_003 = ErrorDefinition("E_FEAT_003", "Insufficient valid trials", True, 400)
    E_FEAT_004 = ErrorDefinition("E_FEAT_004", "Numerical overflow in calculation", False, 500)
    
    # Clinical Scoring Layer
    E_CLIN_001 = ErrorDefinition("E_CLIN_001", "Risk model convergence failure", False, 500)
    E_CLIN_002 = ErrorDefinition("E_CLIN_002", "No valid domain scores", True, 400)
    E_CLIN_003 = ErrorDefinition("E_CLIN_003", "Confidence below threshold", True, 200)  # Partial success
    
    # Output Layer
    E_OUT_001 = ErrorDefinition("E_OUT_001", "Response serialization failed", False, 500)
    E_OUT_002 = ErrorDefinition("E_OUT_002", "Explainability generation failed", False, 500)
    
    @classmethod
    def get(cls, code: str) -> Optional[ErrorDefinition]:
        """Retrieve error definition by code string"""
        return getattr(cls, code, None)
    
    @classmethod
    def to_response(cls, error_def: ErrorDefinition, details: str = None) -> Dict:
        """Convert error definition to API response dict"""
        return {
            "error_code": error_def.code,
            "error_message": error_def.message + (f": {details}" if details else ""),
            "recoverable": error_def.recoverable,
            "retry_after_ms": error_def.retry_after_ms
        }


class PipelineError(Exception):
    """Custom exception with error code context"""
    
    def __init__(self, error_def: ErrorDefinition, details: str = None):
        self.error_def = error_def
        self.details = details
        super().__init__(f"{error_def.code}: {error_def.message}" + (f" - {details}" if details else ""))
    
    @property
    def http_status(self) -> int:
        return self.error_def.http_status
    
    @property
    def recoverable(self) -> bool:
        return self.error_def.recoverable
    
    def to_dict(self) -> Dict:
        return ErrorCode.to_response(self.error_def, self.details)

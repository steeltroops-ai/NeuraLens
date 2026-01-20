"""
Radiology Error Handlers

Handle and format pipeline errors for API response.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

from .codes import ErrorCode, get_error_message


@dataclass
class PipelineError(Exception):
    """Base exception for pipeline errors."""
    code: str
    message: str
    stage: str
    recoverable: bool = False
    details: Optional[Dict] = None
    
    def __str__(self):
        return f"[{self.code}] {self.message}"
    
    def to_dict(self) -> dict:
        user_message = get_error_message(self.code)
        return {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
            "user_message": user_message,
            "recoverable": self.recoverable,
            "details": self.details or {}
        }


class ValidationError(PipelineError):
    """Validation stage error."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(
            code=ErrorCode.E_VAL_003,
            message=message,
            stage="VALIDATION",
            recoverable=False,
            details=details
        )


class PreprocessingError(PipelineError):
    """Preprocessing stage error."""
    def __init__(self, code: str, message: str, details: dict = None):
        super().__init__(
            code=code,
            message=message,
            stage="PREPROCESSING",
            recoverable=True,
            details=details
        )


class InferenceError(PipelineError):
    """Model inference error."""
    def __init__(self, code: str, message: str, model_name: str = None):
        super().__init__(
            code=code,
            message=message,
            stage="INFERENCE",
            recoverable=True,
            details={"model_name": model_name}
        )


def handle_pipeline_error(
    error: Exception,
    orchestrator: Any
) -> Dict[str, Any]:
    """
    Handle pipeline error and format response.
    
    Args:
        error: The exception that occurred
        orchestrator: Pipeline orchestrator instance
    
    Returns:
        Formatted error response
    """
    # Determine error code
    if isinstance(error, PipelineError):
        error_code = error.code
        error_message = error.message
        stage = error.stage
        recoverable = error.recoverable
        details = error.details
    else:
        error_code = ErrorCode.E_SYS_001
        error_message = str(error)
        stage = orchestrator.current_stage.name if hasattr(orchestrator, 'current_stage') else "UNKNOWN"
        recoverable = True
        details = {}
    
    # Get user-friendly message
    user_message = get_error_message(error_code)
    
    return {
        "success": False,
        "request_id": getattr(orchestrator, 'request_id', None),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "processing_time_ms": int(orchestrator.get_total_duration()) if hasattr(orchestrator, 'get_total_duration') else 0,
        
        "error": {
            "code": error_code,
            "message": error_message,
            "stage": stage,
            "user_message": user_message,
            "technical_details": details,
            "recoverable": recoverable,
            "resubmission_hint": user_message.get("action", "Please try again")
        },
        
        "stages_completed": orchestrator.get_completed_stages() if hasattr(orchestrator, 'get_completed_stages') else [],
        "stages_failed": orchestrator.get_failed_stages() if hasattr(orchestrator, 'get_failed_stages') else []
    }

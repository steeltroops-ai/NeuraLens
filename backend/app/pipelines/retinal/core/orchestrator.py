"""
Pipeline Orchestrator for Retinal Analysis

Implements the pipeline orchestration from the specification:
- Stage-by-stage execution tracking
- Retry logic with exponential backoff
- Hard-stop conditions
- Receipt confirmation
- Comprehensive error handling

Author: NeuraLens Medical AI Team
Version: 4.0.0
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, Callable, TypeVar
from dataclasses import dataclass, field

from ..schemas import (
    PipelineStage,
    PipelineError,
    PipelineState,
)
from ..errors.codes import PipelineException, get_error, ErrorSeverity

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# STAGE CONFIGURATION
# =============================================================================

@dataclass
class StageConfig:
    """Configuration for a pipeline stage"""
    timeout_ms: int = 10000
    retryable: bool = False
    max_retries: int = 0
    required: bool = True
    fail_action: str = "hard_stop"  # hard_stop, warn, skip, fallback


STAGE_CONFIGS: Dict[str, StageConfig] = {
    PipelineStage.INPUT_VALIDATION: StageConfig(
        timeout_ms=5000,
        retryable=False,
        required=True,
        fail_action="hard_stop",
    ),
    PipelineStage.IMAGE_PREPROCESSING: StageConfig(
        timeout_ms=10000,
        retryable=True,
        max_retries=2,
        required=True,
        fail_action="hard_stop",
    ),
    PipelineStage.QUALITY_ASSESSMENT: StageConfig(
        timeout_ms=5000,
        retryable=False,
        required=True,
        fail_action="hard_stop",
    ),
    PipelineStage.VESSEL_ANALYSIS: StageConfig(
        timeout_ms=15000,
        retryable=True,
        max_retries=1,
        required=False,
        fail_action="fallback",
    ),
    PipelineStage.OPTIC_DISC_ANALYSIS: StageConfig(
        timeout_ms=10000,
        retryable=True,
        max_retries=1,
        required=False,
        fail_action="warn",
    ),
    PipelineStage.MACULAR_ANALYSIS: StageConfig(
        timeout_ms=10000,
        retryable=True,
        max_retries=1,
        required=False,
        fail_action="warn",
    ),
    PipelineStage.LESION_DETECTION: StageConfig(
        timeout_ms=15000,
        retryable=True,
        max_retries=1,
        required=True,
        fail_action="hard_stop",
    ),
    PipelineStage.DR_GRADING: StageConfig(
        timeout_ms=10000,
        retryable=True,
        max_retries=2,
        required=True,
        fail_action="hard_stop",
    ),
    PipelineStage.RISK_CALCULATION: StageConfig(
        timeout_ms=2000,
        retryable=False,
        required=True,
        fail_action="hard_stop",
    ),
    PipelineStage.HEATMAP_GENERATION: StageConfig(
        timeout_ms=10000,
        retryable=True,
        max_retries=1,
        required=False,
        fail_action="skip",
    ),
    PipelineStage.CLINICAL_ASSESSMENT: StageConfig(
        timeout_ms=5000,
        retryable=False,
        required=True,
        fail_action="hard_stop",
    ),
    PipelineStage.OUTPUT_FORMATTING: StageConfig(
        timeout_ms=2000,
        retryable=False,
        required=True,
        fail_action="hard_stop",
    ),
}


# =============================================================================
# RECEIPT CONFIRMATION
# =============================================================================

@dataclass
class ReceiptConfirmation:
    """Confirmation that image was received"""
    image_received: bool = True
    image_size_bytes: int = 0
    image_dimensions: tuple = (0, 0)
    received_at: str = ""
    filename: Optional[str] = None
    content_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_received": self.image_received,
            "image_size_bytes": self.image_size_bytes,
            "image_dimensions": list(self.image_dimensions),
            "received_at": self.received_at,
            "filename": self.filename,
            "content_type": self.content_type,
        }


# =============================================================================
# EXECUTION CONTEXT
# =============================================================================

@dataclass
class ExecutionContext:
    """Context passed through pipeline stages"""
    session_id: str
    state: PipelineState
    receipt: ReceiptConfirmation
    start_time: float = field(default_factory=time.time)
    
    # Intermediate results
    image_bytes: Optional[bytes] = None
    image_array: Optional[Any] = None
    quality_score: float = 0.0
    biomarkers: Optional[Any] = None
    dr_result: Optional[Any] = None
    dme_result: Optional[Any] = None
    risk_result: Optional[Any] = None
    heatmap_base64: str = ""
    
    def elapsed_ms(self) -> int:
        return int((time.time() - self.start_time) * 1000)


# =============================================================================
# RETRY EXECUTOR
# =============================================================================

async def execute_with_retry(
    stage: str,
    operation: Callable[..., T],
    context: ExecutionContext,
    *args,
    **kwargs
) -> T:
    """
    Execute a stage operation with retry logic.
    
    Args:
        stage: Pipeline stage name
        operation: The function to execute
        context: Execution context
        *args, **kwargs: Arguments for the operation
        
    Returns:
        Result from the operation
        
    Raises:
        PipelineException: If all retries fail
    """
    config = STAGE_CONFIGS.get(stage, StageConfig())
    max_attempts = config.max_retries + 1
    last_error = None
    
    for attempt in range(max_attempts):
        try:
            start_time = time.time()
            
            # Execute with timeout
            if asyncio.iscoroutinefunction(operation):
                result = await asyncio.wait_for(
                    operation(*args, **kwargs),
                    timeout=config.timeout_ms / 1000
                )
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: operation(*args, **kwargs)),
                    timeout=config.timeout_ms / 1000
                )
            
            # Record success
            duration_ms = (time.time() - start_time) * 1000
            context.state.stages_completed.append(stage)
            context.state.stages_timing_ms[stage] = round(duration_ms, 1)
            
            logger.info(f"[{context.session_id}] Stage {stage} completed in {duration_ms:.1f}ms")
            
            return result
            
        except asyncio.TimeoutError:
            last_error = PipelineException(
                "MOD_002",
                {"stage": stage, "timeout_ms": config.timeout_ms, "attempt": attempt + 1}
            )
            logger.warning(f"[{context.session_id}] Stage {stage} timeout (attempt {attempt + 1}/{max_attempts})")
            
        except PipelineException as e:
            last_error = e
            logger.warning(f"[{context.session_id}] Stage {stage} error: {e.code} (attempt {attempt + 1}/{max_attempts})")
            
        except Exception as e:
            last_error = PipelineException(
                "SYS_001",
                {"stage": stage, "error": str(e), "type": type(e).__name__},
                cause=e
            )
            logger.exception(f"[{context.session_id}] Stage {stage} unexpected error")
        
        # Retry logic
        if attempt < max_attempts - 1 and config.retryable:
            backoff = 0.5 * (2 ** attempt)  # Exponential backoff
            context.state.warnings.append(f"Retrying {stage} (attempt {attempt + 2}/{max_attempts})")
            await asyncio.sleep(backoff)
        else:
            break
    
    # All retries failed
    context.state.errors.append(PipelineError(
        stage=stage,
        error_type=last_error.code if isinstance(last_error, PipelineException) else "UNKNOWN",
        message=str(last_error)
    ))
    
    # Handle based on fail_action
    if config.fail_action == "hard_stop":
        raise last_error
    elif config.fail_action == "warn":
        context.state.warnings.append(f"Stage {stage} failed but continuing")
        return None
    elif config.fail_action == "skip":
        logger.info(f"[{context.session_id}] Skipping failed stage {stage}")
        return None
    elif config.fail_action == "fallback":
        context.state.warnings.append(f"Using fallback for {stage}")
        return None
    
    raise last_error


# =============================================================================
# HARD STOP HANDLER
# =============================================================================

def handle_hard_stop(
    context: ExecutionContext,
    error: PipelineException
) -> Dict[str, Any]:
    """
    Handle a hard stop condition.
    
    Args:
        context: Execution context
        error: The error that caused the stop
        
    Returns:
        Error response dictionary
    """
    context.state.current_stage = PipelineStage.FAILED
    context.state.completed_at = datetime.utcnow().isoformat()
    
    error_response = error.to_response()
    
    return {
        "success": False,
        "session_id": context.session_id,
        "timestamp": datetime.utcnow().isoformat(),
        "processing_time_ms": context.elapsed_ms(),
        
        "receipt_confirmation": context.receipt.to_dict(),
        
        "failure_stage": context.state.current_stage,
        "error": error_response,
        
        "stages_completed": context.state.stages_completed,
        "stages_failed": [context.state.current_stage],
        "stages_timing_ms": context.state.stages_timing_ms,
        
        "warnings": context.state.warnings,
        "partial_results": None,
    }


# =============================================================================
# AUDIT LOGGER
# =============================================================================

class AuditLogger:
    """Audit logging for compliance"""
    
    @staticmethod
    def log_session_start(context: ExecutionContext):
        """Log session start event"""
        logger.info(f"AUDIT|SESSION_START|{context.session_id}|{context.receipt.received_at}")
    
    @staticmethod
    def log_stage_complete(context: ExecutionContext, stage: str, duration_ms: float):
        """Log stage completion"""
        logger.info(f"AUDIT|STAGE_COMPLETE|{context.session_id}|{stage}|{duration_ms:.1f}ms")
    
    @staticmethod
    def log_session_end(context: ExecutionContext, success: bool, dr_grade: Optional[int] = None):
        """Log session end"""
        result = "SUCCESS" if success else "FAILURE"
        logger.info(f"AUDIT|SESSION_END|{context.session_id}|{result}|{context.elapsed_ms()}ms|DR={dr_grade}")
    
    @staticmethod
    def log_error(context: ExecutionContext, error: PipelineException):
        """Log error event"""
        logger.error(f"AUDIT|ERROR|{context.session_id}|{error.code}|{error.error['message']}")


# =============================================================================
# SAFETY DISCLAIMERS
# =============================================================================

SAFETY_DISCLAIMERS = {
    "screening_disclaimer": (
        "This AI system is intended for screening purposes only and does not "
        "provide a clinical diagnosis. All findings should be reviewed by a "
        "qualified ophthalmologist or optometrist before clinical decisions are made."
    ),
    "not_fda_cleared": (
        "This system has not been cleared or approved by the FDA for autonomous "
        "diagnostic use. It is intended as a clinical decision support tool only."
    ),
    "false_negative_warning": (
        "Negative results do not rule out the presence of disease. Patients with "
        "symptoms or risk factors should receive comprehensive eye examination "
        "regardless of AI screening results."
    ),
}


def get_disclaimers(result_type: str = "screening") -> list:
    """Get applicable disclaimers for result type"""
    disclaimers = [SAFETY_DISCLAIMERS["screening_disclaimer"]]
    
    if result_type == "screening":
        disclaimers.append(SAFETY_DISCLAIMERS["not_fda_cleared"])
    
    disclaimers.append(SAFETY_DISCLAIMERS["false_negative_warning"])
    
    return disclaimers


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_execution_context(
    session_id: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    filename: Optional[str] = None,
    content_type: Optional[str] = None
) -> ExecutionContext:
    """Create a new execution context"""
    
    if not session_id:
        session_id = str(uuid.uuid4())
    
    receipt = ReceiptConfirmation(
        image_received=image_bytes is not None,
        image_size_bytes=len(image_bytes) if image_bytes else 0,
        received_at=datetime.utcnow().isoformat(),
        filename=filename,
        content_type=content_type,
    )
    
    state = PipelineState(session_id=session_id)
    
    context = ExecutionContext(
        session_id=session_id,
        state=state,
        receipt=receipt,
        image_bytes=image_bytes,
    )
    
    return context


def validate_stage_transition(
    current: str,
    target: str,
    context: ExecutionContext
) -> bool:
    """Validate that a stage transition is allowed"""
    
    # Define valid transitions
    valid_transitions = {
        PipelineStage.PENDING: [PipelineStage.INPUT_VALIDATION],
        PipelineStage.INPUT_VALIDATION: [PipelineStage.IMAGE_PREPROCESSING, PipelineStage.QUALITY_ASSESSMENT],
        PipelineStage.IMAGE_PREPROCESSING: [PipelineStage.QUALITY_ASSESSMENT],
        PipelineStage.QUALITY_ASSESSMENT: [PipelineStage.VESSEL_ANALYSIS],
        PipelineStage.VESSEL_ANALYSIS: [PipelineStage.OPTIC_DISC_ANALYSIS],
        PipelineStage.OPTIC_DISC_ANALYSIS: [PipelineStage.MACULAR_ANALYSIS],
        PipelineStage.MACULAR_ANALYSIS: [PipelineStage.LESION_DETECTION],
        PipelineStage.LESION_DETECTION: [PipelineStage.DR_GRADING],
        PipelineStage.DR_GRADING: [PipelineStage.RISK_CALCULATION],
        PipelineStage.RISK_CALCULATION: [PipelineStage.HEATMAP_GENERATION, PipelineStage.CLINICAL_ASSESSMENT],
        PipelineStage.HEATMAP_GENERATION: [PipelineStage.CLINICAL_ASSESSMENT],
        PipelineStage.CLINICAL_ASSESSMENT: [PipelineStage.OUTPUT_FORMATTING],
        PipelineStage.OUTPUT_FORMATTING: [PipelineStage.COMPLETED],
    }
    
    allowed = valid_transitions.get(current, [])
    return target in allowed


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'StageConfig',
    'STAGE_CONFIGS',
    'ReceiptConfirmation',
    'ExecutionContext',
    'execute_with_retry',
    'handle_hard_stop',
    'AuditLogger',
    'SAFETY_DISCLAIMERS',
    'get_disclaimers',
    'create_execution_context',
    'validate_stage_transition',
]

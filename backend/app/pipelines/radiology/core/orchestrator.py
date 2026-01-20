"""
Radiology Pipeline Orchestrator

Manages pipeline execution flow and state transitions.
Implements the state machine for radiology analysis.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline execution stages."""
    START = auto()
    RECEIPT = auto()
    VALIDATION = auto()
    PREPROCESSING = auto()
    DETECTION = auto()
    ANALYSIS = auto()
    AGGREGATION = auto()
    SCORING = auto()
    FORMATTING = auto()
    COMPLETE = auto()
    FAILED = auto()


class PipelineState(Enum):
    """Pipeline layer states for error tracing."""
    L0_ROUTER = "L0_ROUTER"
    L1_INPUT = "L1_INPUT"
    L2_PREPROCESSING = "L2_PREPROCESSING"
    L3_DETECTION = "L3_DETECTION"
    L4_ANALYSIS = "L4_ANALYSIS"
    L5_CLINICAL = "L5_CLINICAL"
    L6_OUTPUT = "L6_OUTPUT"


@dataclass
class StageResult:
    """Result of a pipeline stage execution."""
    stage: PipelineStage
    success: bool
    duration_ms: float
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_code: Optional[str] = None


@dataclass
class StateTransition:
    """Record of a state transition."""
    from_stage: PipelineStage
    to_stage: PipelineStage
    timestamp: datetime
    duration_ms: float
    success: bool
    error: Optional[str] = None


@dataclass
class PipelineContext:
    """Context passed through pipeline stages."""
    request_id: str
    modality: str = "chest_xray"
    body_region: str = "chest"
    is_volumetric: bool = False
    image_bytes: Optional[bytes] = None
    preprocessed_data: Optional[Any] = None
    detection_results: Optional[Dict] = None
    analysis_results: Optional[Dict] = None
    clinical_results: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)


class RadiologyOrchestrator:
    """
    Orchestrate radiology analysis pipeline.
    
    Manages pipeline execution flow:
    1. Receipt - Acknowledge input received
    2. Validation - Validate all inputs
    3. Preprocessing - Normalize and enhance images
    4. Detection - Anatomical structure detection
    5. Analysis - Pathology analysis
    6. Aggregation - Multi-finding integration
    7. Scoring - Risk stratification
    8. Formatting - Output generation
    """
    
    def __init__(self):
        self.current_stage = PipelineStage.START
        self.transitions: List[StateTransition] = []
        self.stage_outputs: Dict[str, Any] = {}
        self.start_time: Optional[datetime] = None
        self.request_id: Optional[str] = None
    
    def initialize(self, request_id: Optional[str] = None) -> str:
        """Initialize orchestrator for new request."""
        self.request_id = request_id or f"rad_{uuid.uuid4().hex[:12]}"
        self.current_stage = PipelineStage.START
        self.transitions = []
        self.stage_outputs = {}
        self.start_time = datetime.utcnow()
        return self.request_id
    
    def transition(
        self,
        to_stage: PipelineStage,
        success: bool = True,
        error: Optional[str] = None,
        output: Optional[Dict] = None
    ) -> StageResult:
        """Record state transition."""
        now = datetime.utcnow()
        
        # Calculate duration
        if self.transitions:
            last_time = self.transitions[-1].timestamp
        else:
            last_time = self.start_time or now
        
        duration_ms = (now - last_time).total_seconds() * 1000
        
        # Create transition record
        transition = StateTransition(
            from_stage=self.current_stage,
            to_stage=to_stage,
            timestamp=now,
            duration_ms=duration_ms,
            success=success,
            error=error
        )
        self.transitions.append(transition)
        
        # Update current stage
        previous_stage = self.current_stage
        self.current_stage = to_stage
        
        # Store output
        if output:
            self.stage_outputs[to_stage.name] = output
        
        logger.debug(
            f"[{self.request_id}] {previous_stage.name} -> {to_stage.name} "
            f"({duration_ms:.1f}ms, success={success})"
        )
        
        return StageResult(
            stage=to_stage,
            success=success,
            duration_ms=duration_ms,
            output=output,
            error=error
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        total_duration = sum(t.duration_ms for t in self.transitions)
        
        return {
            "request_id": self.request_id,
            "current_stage": self.current_stage.name,
            "is_complete": self.current_stage in [PipelineStage.COMPLETE, PipelineStage.FAILED],
            "is_failed": self.current_stage == PipelineStage.FAILED,
            "stages_completed": [
                t.to_stage.name for t in self.transitions if t.success
            ],
            "total_duration_ms": total_duration,
            "last_error": next(
                (t.error for t in reversed(self.transitions) if t.error),
                None
            )
        }
    
    def get_stage_timings(self) -> Dict[str, float]:
        """Get timing breakdown by stage."""
        return {
            t.to_stage.name: t.duration_ms
            for t in self.transitions
        }
    
    def get_completed_stages(self) -> List[Dict]:
        """Get list of completed stages for response."""
        return [
            {
                "stage": t.to_stage.name,
                "status": "success" if t.success else "failed",
                "time_ms": round(t.duration_ms, 1),
                "error_code": None if t.success else "E_STAGE_FAIL"
            }
            for t in self.transitions
        ]
    
    def get_failed_stages(self) -> List[Dict]:
        """Get list of failed stages."""
        return [
            {
                "stage": t.to_stage.name,
                "status": "failed",
                "time_ms": round(t.duration_ms, 1),
                "error": t.error
            }
            for t in self.transitions if not t.success
        ]
    
    def get_total_duration(self) -> float:
        """Get total pipeline duration in milliseconds."""
        return sum(t.duration_ms for t in self.transitions)


@dataclass
class LayerError(Exception):
    """Error with layer context for tracing."""
    layer: PipelineState
    code: str
    message: str
    details: Optional[dict] = None
    cause: Optional[Exception] = None
    
    def __str__(self):
        return f"[{self.layer.value}] {self.code}: {self.message}"
    
    def to_dict(self) -> dict:
        return {
            "layer": self.layer.value,
            "code": self.code,
            "message": self.message,
            "details": self.details
        }

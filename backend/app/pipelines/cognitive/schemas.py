"""
Cognitive Pipeline Schemas - Production Grade
Version: 2.0.0
"""

from typing import List, Dict, Optional, Any, Literal, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
from enum import Enum

# =============================================================================
# ENUMS
# =============================================================================

class PipelineStage(str, Enum):
    PENDING = "pending"
    VALIDATING = "validating"
    EXTRACTING = "extracting"
    SCORING = "scoring"
    COMPLETE = "complete"
    FAILED = "failed"

class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class TaskCompletionStatus(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    INVALID = "invalid"
    UNKNOWN = "unknown"

# =============================================================================
# INPUT SCHEMAS
# =============================================================================

class TaskEvent(BaseModel):
    """Raw event from frontend task with strict validation"""
    timestamp: float = Field(..., ge=0, description="Milliseconds from task start")
    event_type: str = Field(..., min_length=1, max_length=50)
    payload: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('event_type')
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        allowed = {
            'test_start', 'stimulus_shown', 'response_received', 
            'response_early', 'trial_result', 'user_response', 'test_end'
        }
        if v not in allowed:
            # Allow but log unknown types for extensibility
            pass
        return v

class TaskSession(BaseModel):
    """Data for a single cognitive task"""
    task_id: str = Field(..., min_length=1, max_length=100, pattern=r'^[a-z0-9_]+$')
    start_time: datetime
    end_time: datetime
    events: List[TaskEvent] = Field(..., min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @model_validator(mode='after')
    def validate_time_order(self):
        if self.end_time < self.start_time:
            raise ValueError("end_time must be >= start_time")
        return self

class CognitiveSessionInput(BaseModel):
    """Full session payload from frontend - strict contract"""
    session_id: str = Field(..., min_length=5, max_length=100)
    patient_id: Optional[str] = Field(None, max_length=100)
    tasks: List[TaskSession] = Field(..., min_length=1, max_length=10)
    user_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        if not v.startswith('sess_'):
            raise ValueError("session_id must start with 'sess_'")
        return v

# =============================================================================
# INTERMEDIATE / FEATURE SCHEMAS
# =============================================================================

class TaskMetrics(BaseModel):
    """Calculated metrics for a single task"""
    task_id: str
    completion_status: TaskCompletionStatus
    performance_score: float = Field(..., ge=0, le=100)
    parameters: Dict[str, float] = Field(default_factory=dict)
    validity_flag: bool = True
    quality_warnings: List[str] = Field(default_factory=list)

class CognitiveFeatures(BaseModel):
    """Aggregated features across all tasks"""
    domain_scores: Dict[str, float] = Field(
        default_factory=dict, 
        description="Normalized scores (0-1) per domain"
    )
    raw_metrics: List[TaskMetrics] = Field(default_factory=list)
    fatigue_index: float = Field(0.0, ge=0, le=1)
    consistency_score: float = Field(0.0, ge=0, le=1)
    valid_task_count: int = 0
    total_task_count: int = 0

# =============================================================================
# STAGE PROGRESS SCHEMA
# =============================================================================

class StageProgress(BaseModel):
    """Real-time stage progress for frontend polling"""
    stage: PipelineStage
    stage_index: int = Field(..., ge=0, le=4)
    total_stages: int = 4
    message: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    error: Optional[str] = None

# =============================================================================
# OUTPUT SCHEMAS
# =============================================================================

class ClinicalRecommendation(BaseModel):
    """Individual recommendation with priority and category"""
    category: Literal["clinical", "lifestyle", "routine", "specific"]
    description: str = Field(..., min_length=10, max_length=500)
    priority: Literal["low", "medium", "high", "critical"]
    action_url: Optional[str] = None

class DomainRiskDetail(BaseModel):
    """Detailed risk info per cognitive domain"""
    score: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel
    percentile: Optional[int] = Field(None, ge=0, le=100)
    confidence: float = Field(..., ge=0, le=1)
    contributing_factors: List[str] = Field(default_factory=list)

class CognitiveRiskAssessment(BaseModel):
    """Complete risk assessment with uncertainty"""
    overall_risk_score: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel
    confidence_score: float = Field(..., ge=0, le=1)
    confidence_interval: Tuple[float, float] = Field(default=(0.0, 1.0))
    domain_risks: Dict[str, DomainRiskDetail] = Field(default_factory=dict)

class ExplainabilityArtifact(BaseModel):
    """Explainability data for clinical transparency"""
    summary: str
    key_factors: List[str] = Field(default_factory=list)
    domain_contributions: Dict[str, float] = Field(default_factory=dict)
    methodology_note: str = "Weighted multi-domain risk aggregation"

class CognitiveResponse(BaseModel):
    """Final API Response - Production Grade"""
    # Identity
    session_id: str
    pipeline_version: str = "2.0.0"
    
    # Timing
    timestamp: datetime
    processing_time_ms: float
    
    # Pipeline State
    status: Literal["success", "partial", "failed"]
    stages: List[StageProgress] = Field(default_factory=list)
    
    # Results (nullable on failure)
    risk_assessment: Optional[CognitiveRiskAssessment] = None
    features: Optional[CognitiveFeatures] = None
    recommendations: List[ClinicalRecommendation] = Field(default_factory=list)
    explainability: Optional[ExplainabilityArtifact] = None
    
    # Error Handling
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    recoverable: bool = True
    retry_after_ms: Optional[int] = None

# =============================================================================
# API REQUEST/RESPONSE WRAPPERS
# =============================================================================

class AnalyzeRequest(CognitiveSessionInput):
    """Alias for clarity in router"""
    pass

class HealthResponse(BaseModel):
    """Health check response"""
    status: Literal["ok", "degraded", "unhealthy"]
    service: str = "cognitive-pipeline"
    version: str = "2.0.0"
    uptime_seconds: Optional[float] = None
    last_request_at: Optional[datetime] = None

class ValidationErrorDetail(BaseModel):
    """Structured validation error"""
    field: str
    message: str
    code: str

class ErrorResponse(BaseModel):
    """Standard error response"""
    error_code: str
    error_message: str
    details: List[ValidationErrorDetail] = Field(default_factory=list)
    recoverable: bool = True
    retry_after_ms: Optional[int] = None

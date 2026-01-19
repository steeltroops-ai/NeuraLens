"""
Cardiology Pipeline - Error Codes and Exception Handling
Comprehensive error taxonomy with user-friendly messages.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


# ==============================================================================
# ERROR CATEGORIES
# ==============================================================================

class ErrorCategory(str, Enum):
    """Error category codes."""
    VALIDATION = "VAL"      # Input validation errors
    PREPROCESSING = "PREP"  # Signal/image processing errors
    DETECTION = "DET"       # Structure detection errors
    ANALYSIS = "ANAL"       # Functional analysis errors
    INFERENCE = "INF"       # Model inference errors
    SYSTEM = "SYS"          # System-level errors


# ==============================================================================
# ERROR CODE DEFINITIONS
# ==============================================================================

@dataclass
class ErrorDefinition:
    """Definition of an error code."""
    code: str
    message: str
    user_title: str
    user_explanation: str
    user_action: str
    recoverable: bool
    category: ErrorCategory


# Validation Errors (E_VAL_xxx)
VALIDATION_ERRORS: Dict[str, ErrorDefinition] = {
    "E_VAL_001": ErrorDefinition(
        code="E_VAL_001",
        message="No valid modality provided",
        user_title="No Input Provided",
        user_explanation="We didn't receive any analyzable data.",
        user_action="Please upload either an echocardiogram image/video or an ECG recording.",
        recoverable=False,
        category=ErrorCategory.VALIDATION,
    ),
    "E_VAL_002": ErrorDefinition(
        code="E_VAL_002",
        message="Invalid file format",
        user_title="Unsupported File Format",
        user_explanation="The uploaded file format is not supported.",
        user_action="Please use supported formats: ECG (CSV, JSON), Echo (JPG, PNG, MP4).",
        recoverable=False,
        category=ErrorCategory.VALIDATION,
    ),
    "E_VAL_003": ErrorDefinition(
        code="E_VAL_003",
        message="File size exceeds limit",
        user_title="File Too Large",
        user_explanation="The uploaded file exceeds the maximum allowed size.",
        user_action="Please reduce the file size and try again.",
        recoverable=False,
        category=ErrorCategory.VALIDATION,
    ),
    "E_VAL_004": ErrorDefinition(
        code="E_VAL_004",
        message="Invalid sample rate",
        user_title="Invalid ECG Sample Rate",
        user_explanation="The ECG sample rate is outside the acceptable range (100-1000 Hz).",
        user_action="Please ensure your ECG recording uses a sample rate between 100 and 1000 Hz.",
        recoverable=False,
        category=ErrorCategory.VALIDATION,
    ),
    "E_VAL_005": ErrorDefinition(
        code="E_VAL_005",
        message="Signal duration too short",
        user_title="ECG Recording Too Short",
        user_explanation="The ECG recording needs to be at least 5 seconds long for accurate analysis.",
        user_action="Please provide a longer ECG recording (recommended: 30 seconds or more).",
        recoverable=False,
        category=ErrorCategory.VALIDATION,
    ),
    "E_VAL_006": ErrorDefinition(
        code="E_VAL_006",
        message="Image resolution too low",
        user_title="Image Resolution Too Low",
        user_explanation="The echocardiogram image resolution is below the minimum requirement.",
        user_action="Please use images with at least 256x256 pixel resolution.",
        recoverable=False,
        category=ErrorCategory.VALIDATION,
    ),
    "E_VAL_007": ErrorDefinition(
        code="E_VAL_007",
        message="Corrupted file detected",
        user_title="File Corrupted",
        user_explanation="The uploaded file appears to be corrupted or incomplete.",
        user_action="Please check the file and upload again.",
        recoverable=False,
        category=ErrorCategory.VALIDATION,
    ),
    "E_VAL_008": ErrorDefinition(
        code="E_VAL_008",
        message="Invalid metadata schema",
        user_title="Invalid Metadata Format",
        user_explanation="The clinical metadata JSON format is invalid.",
        user_action="Please check the metadata format and correct any errors.",
        recoverable=True,
        category=ErrorCategory.VALIDATION,
    ),
    "E_VAL_009": ErrorDefinition(
        code="E_VAL_009",
        message="Signal duration too long",
        user_title="ECG Recording Too Long",
        user_explanation="The ECG recording exceeds the maximum duration (5 minutes).",
        user_action="Please provide a shorter ECG recording.",
        recoverable=False,
        category=ErrorCategory.VALIDATION,
    ),
    "E_VAL_010": ErrorDefinition(
        code="E_VAL_010",
        message="Empty signal detected",
        user_title="Empty ECG Signal",
        user_explanation="The ECG file contains no signal data.",
        user_action="Please upload a valid ECG recording.",
        recoverable=False,
        category=ErrorCategory.VALIDATION,
    ),
}

# Preprocessing Errors (E_PREP_xxx)
PREPROCESSING_ERRORS: Dict[str, ErrorDefinition] = {
    "E_PREP_001": ErrorDefinition(
        code="E_PREP_001",
        message="All frames below quality threshold",
        user_title="Image Quality Too Low",
        user_explanation="All echocardiogram frames are below the minimum quality threshold.",
        user_action="Please provide clearer echocardiogram images.",
        recoverable=False,
        category=ErrorCategory.PREPROCESSING,
    ),
    "E_PREP_002": ErrorDefinition(
        code="E_PREP_002",
        message="Video decode failed",
        user_title="Video Processing Failed",
        user_explanation="Unable to process the echocardiogram video file.",
        user_action="Please try converting the video to MP4 (H.264) format.",
        recoverable=False,
        category=ErrorCategory.PREPROCESSING,
    ),
    "E_PREP_003": ErrorDefinition(
        code="E_PREP_003",
        message="Insufficient cardiac cycles",
        user_title="Video Too Short",
        user_explanation="The video doesn't contain enough cardiac cycles for analysis.",
        user_action="Please provide a video with at least 3 complete heartbeats.",
        recoverable=False,
        category=ErrorCategory.PREPROCESSING,
    ),
    "E_PREP_004": ErrorDefinition(
        code="E_PREP_004",
        message="Signal flatline detected",
        user_title="No ECG Signal Detected",
        user_explanation="The ECG recording shows a flatline with no cardiac activity.",
        user_action="Please check the ECG electrodes and recording, then try again.",
        recoverable=False,
        category=ErrorCategory.PREPROCESSING,
    ),
    "E_PREP_005": ErrorDefinition(
        code="E_PREP_005",
        message="Excessive noise in signal",
        user_title="ECG Signal Quality Issue",
        user_explanation="The ECG signal has significant noise that affects analysis accuracy.",
        user_action="Try recording in a quieter environment with good electrode contact.",
        recoverable=True,
        category=ErrorCategory.PREPROCESSING,
    ),
    "E_PREP_006": ErrorDefinition(
        code="E_PREP_006",
        message="Baseline wander extreme",
        user_title="ECG Baseline Issue",
        user_explanation="The ECG signal has extreme baseline wander that couldn't be corrected.",
        user_action="Ensure the patient remains still during recording.",
        recoverable=True,
        category=ErrorCategory.PREPROCESSING,
    ),
}

# Detection Errors (E_DET_xxx)
DETECTION_ERRORS: Dict[str, ErrorDefinition] = {
    "E_DET_001": ErrorDefinition(
        code="E_DET_001",
        message="View not recognizable",
        user_title="Unrecognized Echo View",
        user_explanation="The echocardiogram view could not be identified.",
        user_action="Please use standard views: Apical 4-Chamber, Parasternal Long Axis, etc.",
        recoverable=True,
        category=ErrorCategory.DETECTION,
    ),
    "E_DET_002": ErrorDefinition(
        code="E_DET_002",
        message="LV not detected",
        user_title="Heart Structure Not Visible",
        user_explanation="The left ventricle could not be clearly identified in the images.",
        user_action="Ensure the echocardiogram clearly shows the left ventricle.",
        recoverable=True,
        category=ErrorCategory.DETECTION,
    ),
    "E_DET_003": ErrorDefinition(
        code="E_DET_003",
        message="Temporal inconsistency",
        user_title="Video Analysis Issue",
        user_explanation="Inconsistent detection across video frames.",
        user_action="This may resolve on retry. If not, try a different video segment.",
        recoverable=True,
        category=ErrorCategory.DETECTION,
    ),
    "E_DET_004": ErrorDefinition(
        code="E_DET_004",
        message="Implausible anatomy detected",
        user_title="Unusual Structure Detected",
        user_explanation="The detected cardiac structures have implausible dimensions.",
        user_action="Please verify the image is a valid echocardiogram.",
        recoverable=True,
        category=ErrorCategory.DETECTION,
    ),
}

# Analysis Errors (E_ANAL_xxx)
ANALYSIS_ERRORS: Dict[str, ErrorDefinition] = {
    "E_ANAL_001": ErrorDefinition(
        code="E_ANAL_001",
        message="EF calculation failed",
        user_title="Ejection Fraction Calculation Failed",
        user_explanation="Unable to calculate ejection fraction from the provided data.",
        user_action="Ensure clear visualization of the left ventricle in end-diastole and end-systole.",
        recoverable=True,
        category=ErrorCategory.ANALYSIS,
    ),
    "E_ANAL_002": ErrorDefinition(
        code="E_ANAL_002",
        message="R-peak detection failed",
        user_title="Heartbeat Detection Failed",
        user_explanation="Unable to reliably detect heartbeats in the ECG signal.",
        user_action="Check signal quality and ensure proper electrode placement.",
        recoverable=True,
        category=ErrorCategory.ANALYSIS,
    ),
    "E_ANAL_003": ErrorDefinition(
        code="E_ANAL_003",
        message="HRV computation failed",
        user_title="HRV Analysis Failed",
        user_explanation="Heart rate variability could not be calculated.",
        user_action="Ensure at least 5 clear heartbeats are present in the recording.",
        recoverable=True,
        category=ErrorCategory.ANALYSIS,
    ),
    "E_ANAL_004": ErrorDefinition(
        code="E_ANAL_004",
        message="Inconsistent metrics detected",
        user_title="Analysis Inconsistency",
        user_explanation="Some analysis results are inconsistent and have been flagged.",
        user_action="Results are provided but should be reviewed by a clinician.",
        recoverable=True,
        category=ErrorCategory.ANALYSIS,
    ),
}

# Inference Errors (E_INF_xxx)
INFERENCE_ERRORS: Dict[str, ErrorDefinition] = {
    "E_INF_001": ErrorDefinition(
        code="E_INF_001",
        message="Model timeout",
        user_title="Processing Timeout",
        user_explanation="The analysis took too long to complete.",
        user_action="Please try again. If the issue persists, try with a smaller file.",
        recoverable=True,
        category=ErrorCategory.INFERENCE,
    ),
    "E_INF_002": ErrorDefinition(
        code="E_INF_002",
        message="Model load failed",
        user_title="System Initialization Error",
        user_explanation="Failed to initialize the analysis system.",
        user_action="Please try again in a few moments.",
        recoverable=True,
        category=ErrorCategory.INFERENCE,
    ),
    "E_INF_003": ErrorDefinition(
        code="E_INF_003",
        message="Out of memory",
        user_title="Resource Limit Reached",
        user_explanation="The system ran out of resources processing your request.",
        user_action="Try with a smaller file or shorter recording.",
        recoverable=True,
        category=ErrorCategory.INFERENCE,
    ),
    "E_INF_004": ErrorDefinition(
        code="E_INF_004",
        message="Inference failed",
        user_title="Analysis Failed",
        user_explanation="The analysis could not be completed.",
        user_action="Please try again. If the issue persists, contact support.",
        recoverable=True,
        category=ErrorCategory.INFERENCE,
    ),
}

# System Errors (E_SYS_xxx)
SYSTEM_ERRORS: Dict[str, ErrorDefinition] = {
    "E_SYS_001": ErrorDefinition(
        code="E_SYS_001",
        message="Internal server error",
        user_title="System Error",
        user_explanation="An unexpected error occurred.",
        user_action="Please try again later.",
        recoverable=True,
        category=ErrorCategory.SYSTEM,
    ),
    "E_SYS_002": ErrorDefinition(
        code="E_SYS_002",
        message="Service unavailable",
        user_title="Service Temporarily Unavailable",
        user_explanation="The analysis service is temporarily unavailable.",
        user_action="Please try again in a few minutes.",
        recoverable=True,
        category=ErrorCategory.SYSTEM,
    ),
    "E_SYS_003": ErrorDefinition(
        code="E_SYS_003",
        message="Rate limit exceeded",
        user_title="Too Many Requests",
        user_explanation="You have exceeded the request limit.",
        user_action="Please wait a moment before trying again.",
        recoverable=True,
        category=ErrorCategory.SYSTEM,
    ),
}

# Combined error registry
ALL_ERRORS: Dict[str, ErrorDefinition] = {
    **VALIDATION_ERRORS,
    **PREPROCESSING_ERRORS,
    **DETECTION_ERRORS,
    **ANALYSIS_ERRORS,
    **INFERENCE_ERRORS,
    **SYSTEM_ERRORS,
}


# ==============================================================================
# EXCEPTION CLASSES
# ==============================================================================

@dataclass
class PipelineError(Exception):
    """Base exception for cardiology pipeline errors."""
    code: str
    message: str
    stage: str
    recoverable: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    cause: Optional[Exception] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        super().__init__(f"[{self.code}] {self.message}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        error_def = ALL_ERRORS.get(self.code)
        
        result = {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp.isoformat() + "Z",
            "details": self.details,
        }
        
        if error_def:
            result["user_message"] = {
                "title": error_def.user_title,
                "explanation": error_def.user_explanation,
                "action": error_def.user_action,
            }
        
        return result


class ValidationError(PipelineError):
    """Input validation error."""
    def __init__(self, code: str, message: str, details: Dict[str, Any] = None):
        super().__init__(
            code=code,
            message=message,
            stage="VALIDATION",
            recoverable=False,
            details=details or {},
        )


class PreprocessingError(PipelineError):
    """Signal/image preprocessing error."""
    def __init__(self, code: str, message: str, details: Dict[str, Any] = None):
        super().__init__(
            code=code,
            message=message,
            stage="PREPROCESSING",
            recoverable=True,
            details=details or {},
        )


class DetectionError(PipelineError):
    """Anatomical detection error."""
    def __init__(self, code: str, message: str, details: Dict[str, Any] = None):
        super().__init__(
            code=code,
            message=message,
            stage="DETECTION",
            recoverable=True,
            details=details or {},
        )


class AnalysisError(PipelineError):
    """Functional analysis error."""
    def __init__(self, code: str, message: str, details: Dict[str, Any] = None):
        super().__init__(
            code=code,
            message=message,
            stage="ANALYSIS",
            recoverable=True,
            details=details or {},
        )


class InferenceError(PipelineError):
    """Model inference error."""
    def __init__(self, code: str, message: str, details: Dict[str, Any] = None):
        super().__init__(
            code=code,
            message=message,
            stage="INFERENCE",
            recoverable=True,
            details=details or {},
        )


class SystemError(PipelineError):
    """System-level error."""
    def __init__(self, code: str, message: str, details: Dict[str, Any] = None):
        super().__init__(
            code=code,
            message=message,
            stage="SYSTEM",
            recoverable=True,
            details=details or {},
        )


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_error_definition(code: str) -> Optional[ErrorDefinition]:
    """Get error definition by code."""
    return ALL_ERRORS.get(code)


def get_user_message(code: str) -> Dict[str, str]:
    """Get user-friendly error message."""
    error_def = ALL_ERRORS.get(code)
    
    if error_def:
        return {
            "title": error_def.user_title,
            "explanation": error_def.user_explanation,
            "action": error_def.user_action,
        }
    
    return {
        "title": "Processing Error",
        "explanation": "An error occurred during analysis.",
        "action": "Please try again or contact support.",
    }


def create_error(
    code: str,
    stage: str = None,
    details: Dict[str, Any] = None,
    cause: Exception = None
) -> PipelineError:
    """Create appropriate error instance from code."""
    error_def = ALL_ERRORS.get(code)
    
    if not error_def:
        return PipelineError(
            code=code,
            message="Unknown error",
            stage=stage or "UNKNOWN",
            details=details or {},
            cause=cause,
        )
    
    return PipelineError(
        code=error_def.code,
        message=error_def.message,
        stage=stage or error_def.category.value,
        recoverable=error_def.recoverable,
        details=details or {},
        cause=cause,
    )

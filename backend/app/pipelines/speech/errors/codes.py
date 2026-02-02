"""
Speech Analysis Pipeline - Enhanced Error Handling v4.0
Research-grade error codes with clinical context and recovery strategies.

Error Format: E_{CATEGORY}_{NUMBER}
Categories:
- E_QUALITY_xxx: Audio quality and validation errors
- E_EXTRACTION_xxx: Feature extraction failures
- E_INFERENCE_xxx: ML model and prediction errors
- E_CLINICAL_xxx: Clinical assessment and risk scoring errors
- E_STREAMING_xxx: Real-time processing errors
- E_INTEGRATION_xxx: External system integration errors
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class PipelineLayer(str, Enum):
    """Pipeline execution layers for error tracing."""
    ROUTER = "L0_ROUTER"
    INPUT = "L1_INPUT"
    PREPROCESSING = "L2_PREPROCESSING"
    QUALITY_GATE = "L3_QUALITY_GATE"
    FEATURE_EXTRACTION = "L4_FEATURE_EXTRACTION"
    ML_INFERENCE = "L5_ML_INFERENCE"
    RISK_ASSESSMENT = "L6_RISK_ASSESSMENT"
    EXPLANATION = "L7_EXPLANATION"
    OUTPUT = "L8_OUTPUT"
    STREAMING = "L9_STREAMING"
    INTEGRATION = "L10_INTEGRATION"
    CORE = "CORE"


class PatientImpact(str, Enum):
    """Impact level on patient care."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(str, Enum):
    """Error recovery strategy types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    DEGRADE = "graceful_degradation"
    ABORT = "abort"
    USER_ACTION = "user_action"


# =============================================================================
# Error Codes
# =============================================================================

class ErrorCode:
    """Standardized error codes for the speech pipeline."""
    
    # Quality Gate Errors (E_QUALITY_xxx)
    E_QUALITY_SNR_LOW = "E_QUALITY_001"
    E_QUALITY_CLIPPING = "E_QUALITY_002"
    E_QUALITY_LOW_SPEECH = "E_QUALITY_003"
    E_QUALITY_FREQUENCY = "E_QUALITY_004"
    E_QUALITY_FORMAT = "E_QUALITY_005"
    E_QUALITY_DURATION = "E_QUALITY_006"
    E_QUALITY_CORRUPT = "E_QUALITY_007"
    E_QUALITY_EMPTY = "E_QUALITY_008"
    E_QUALITY_NOISE = "E_QUALITY_009"
    E_QUALITY_ASSESS_FAILED = "E_QUALITY_010"
    
    # Feature Extraction Errors (E_EXTRACTION_xxx)
    E_EXTRACTION_F0 = "E_EXTRACTION_001"
    E_EXTRACTION_FORMANTS = "E_EXTRACTION_002"
    E_EXTRACTION_JITTER = "E_EXTRACTION_003"
    E_EXTRACTION_SHIMMER = "E_EXTRACTION_004"
    E_EXTRACTION_HNR = "E_EXTRACTION_005"
    E_EXTRACTION_CPPS = "E_EXTRACTION_006"
    E_EXTRACTION_MFCC = "E_EXTRACTION_007"
    E_EXTRACTION_PROSODIC = "E_EXTRACTION_008"
    E_EXTRACTION_COMPOSITE = "E_EXTRACTION_009"
    E_EXTRACTION_TIMEOUT = "E_EXTRACTION_010"
    E_EXTRACTION_INSUFFICIENT_VOICED = "E_EXTRACTION_011"
    E_EXTRACTION_EMBEDDING = "E_EXTRACTION_012"
    
    # ML Inference Errors (E_INFERENCE_xxx)
    E_INFERENCE_MODEL_LOAD = "E_INFERENCE_001"
    E_INFERENCE_PREDICTION = "E_INFERENCE_002"
    E_INFERENCE_TIMEOUT = "E_INFERENCE_003"
    E_INFERENCE_VERSION = "E_INFERENCE_004"
    E_INFERENCE_GPU = "E_INFERENCE_005"
    E_INFERENCE_MEMORY = "E_INFERENCE_006"
    E_INFERENCE_ENSEMBLE = "E_INFERENCE_007"
    E_INFERENCE_UNCERTAINTY = "E_INFERENCE_008"
    
    # Clinical Errors (E_CLINICAL_xxx)
    E_CLINICAL_RISK_CALC = "E_CLINICAL_001"
    E_CLINICAL_CONDITION = "E_CLINICAL_002"
    E_CLINICAL_NORMALIZATION = "E_CLINICAL_003"
    E_CLINICAL_CONFIDENCE = "E_CLINICAL_004"
    E_CLINICAL_RECOMMENDATION = "E_CLINICAL_005"
    E_CLINICAL_EXPLANATION = "E_CLINICAL_006"
    E_CLINICAL_REVIEW_FLAG = "E_CLINICAL_007"
    
    # Streaming Errors (E_STREAMING_xxx)
    E_STREAMING_SESSION_INIT = "E_STREAMING_001"
    E_STREAMING_BUFFER = "E_STREAMING_002"
    E_STREAMING_LATENCY = "E_STREAMING_003"
    E_STREAMING_MEMORY = "E_STREAMING_004"
    E_STREAMING_TIMEOUT = "E_STREAMING_005"
    E_STREAMING_DISCONNECT = "E_STREAMING_006"
    E_STREAMING_SYNC = "E_STREAMING_007"
    
    # Integration Errors (E_INTEGRATION_xxx)
    E_INTEGRATION_DATABASE = "E_INTEGRATION_001"
    E_INTEGRATION_FHIR = "E_INTEGRATION_002"
    E_INTEGRATION_DICOM = "E_INTEGRATION_003"
    E_INTEGRATION_WEBHOOK = "E_INTEGRATION_004"
    E_INTEGRATION_EXPORT = "E_INTEGRATION_005"
    E_INTEGRATION_AUDIT = "E_INTEGRATION_006"
    
    # Input Errors (legacy compatibility)
    E_INP_001 = "E_QUALITY_EMPTY"
    E_INP_002 = "E_QUALITY_FORMAT"
    E_INP_003 = "E_QUALITY_SIZE"
    E_INP_004 = "E_QUALITY_VALIDATION"
    E_INP_005 = "E_QUALITY_DURATION_SHORT"
    E_INP_006 = "E_QUALITY_DURATION_LONG"
    E_INP_007 = "E_QUALITY_MIME"
    E_INP_008 = "E_QUALITY_CORRUPT"
    E_INP_009 = "E_QUALITY_EMPTY"
    
    # Preprocessing Errors (legacy compatibility)
    E_PREP_001 = "E_EXTRACTION_DECODE"
    E_PREP_002 = "E_EXTRACTION_RESAMPLE"
    E_PREP_003 = "E_EXTRACTION_NORMALIZE"
    E_PREP_004 = "E_EXTRACTION_TRIM"
    E_PREP_005 = "E_EXTRACTION_CHANNEL"
    E_PREP_006 = "E_QUALITY_LOW_SPEECH"
    E_PREP_007 = "E_QUALITY_SNR_LOW"
    
    # Analysis Errors (legacy compatibility)
    E_ANAL_001 = "E_EXTRACTION_001"
    E_ANAL_002 = "E_EXTRACTION_F0"
    E_ANAL_003 = "E_EXTRACTION_FORMANTS"
    E_ANAL_004 = "E_EXTRACTION_JITTER"
    E_ANAL_005 = "E_EXTRACTION_CPPS"
    E_ANAL_006 = "E_EXTRACTION_PROSODIC"
    E_ANAL_007 = "E_EXTRACTION_TIMEOUT"
    E_ANAL_008 = "E_EXTRACTION_INSUFFICIENT_VOICED"


# =============================================================================
# Error Messages and Clinical Context
# =============================================================================

ERROR_DETAILS: Dict[str, Dict[str, Any]] = {
    ErrorCode.E_QUALITY_SNR_LOW: {
        "message": "Signal-to-noise ratio below acceptable threshold",
        "clinical_context": "Low SNR may affect biomarker accuracy and clinical reliability",
        "recovery": RecoveryStrategy.USER_ACTION,
        "patient_impact": PatientImpact.MEDIUM,
        "suggestions": [
            "Record in a quieter environment",
            "Move closer to the microphone",
            "Use noise-canceling equipment",
            "Reduce background noise sources"
        ]
    },
    ErrorCode.E_QUALITY_CLIPPING: {
        "message": "Audio clipping detected exceeds acceptable threshold",
        "clinical_context": "Clipping distorts waveform peaks affecting voice quality measurements",
        "recovery": RecoveryStrategy.USER_ACTION,
        "patient_impact": PatientImpact.MEDIUM,
        "suggestions": [
            "Reduce microphone input volume",
            "Move further from the microphone",
            "Speak at a normal volume level",
            "Check microphone gain settings"
        ]
    },
    ErrorCode.E_QUALITY_LOW_SPEECH: {
        "message": "Insufficient speech content in recording",
        "clinical_context": "Inadequate speech data prevents reliable biomarker extraction",
        "recovery": RecoveryStrategy.USER_ACTION,
        "patient_impact": PatientImpact.HIGH,
        "suggestions": [
            "Ensure continuous speech during recording",
            "Reduce long pauses between phrases",
            "Follow the reading passage closely",
            "Record for the full required duration"
        ]
    },
    ErrorCode.E_QUALITY_FREQUENCY: {
        "message": "Frequency content outside acceptable range",
        "clinical_context": "Missing frequency bands may affect formant and harmonic analysis",
        "recovery": RecoveryStrategy.DEGRADE,
        "patient_impact": PatientImpact.LOW,
        "suggestions": [
            "Use a higher quality microphone",
            "Check audio recording settings",
            "Ensure sample rate is at least 16kHz"
        ]
    },
    ErrorCode.E_EXTRACTION_F0: {
        "message": "Fundamental frequency extraction failed",
        "clinical_context": "F0 is critical for tremor and prosody analysis",
        "recovery": RecoveryStrategy.FALLBACK,
        "patient_impact": PatientImpact.MEDIUM,
        "suggestions": [
            "Ensure clear voiced speech segments",
            "Reduce background noise",
            "Check for sustained vowel sounds"
        ]
    },
    ErrorCode.E_EXTRACTION_FORMANTS: {
        "message": "Formant extraction failed",
        "clinical_context": "Formants are essential for articulation clarity (FCR) calculation",
        "recovery": RecoveryStrategy.FALLBACK,
        "patient_impact": PatientImpact.MEDIUM,
        "suggestions": [
            "Ensure clear vowel pronunciation",
            "Reduce mumbling or unclear speech",
            "Check microphone quality"
        ]
    },
    ErrorCode.E_EXTRACTION_INSUFFICIENT_VOICED: {
        "message": "Insufficient voiced segments for analysis",
        "clinical_context": "Voice quality measures require adequate voiced speech",
        "recovery": RecoveryStrategy.USER_ACTION,
        "patient_impact": PatientImpact.HIGH,
        "suggestions": [
            "Include more sustained vowel sounds",
            "Reduce whispering or breathy speech",
            "Follow the standard reading passage"
        ]
    },
    ErrorCode.E_INFERENCE_MODEL_LOAD: {
        "message": "ML model failed to load",
        "clinical_context": "Advanced neural analysis unavailable",
        "recovery": RecoveryStrategy.FALLBACK,
        "patient_impact": PatientImpact.LOW,
        "suggestions": [
            "Analysis will proceed with traditional methods",
            "Contact support if issue persists"
        ]
    },
    ErrorCode.E_INFERENCE_TIMEOUT: {
        "message": "ML inference exceeded time limit",
        "clinical_context": "Complex analysis timed out",
        "recovery": RecoveryStrategy.RETRY,
        "patient_impact": PatientImpact.LOW,
        "suggestions": [
            "Try submitting the recording again",
            "Use a shorter audio sample",
            "System may be under high load"
        ]
    },
    ErrorCode.E_CLINICAL_RISK_CALC: {
        "message": "Risk calculation failed",
        "clinical_context": "Unable to compute clinical risk scores",
        "recovery": RecoveryStrategy.ABORT,
        "patient_impact": PatientImpact.CRITICAL,
        "suggestions": [
            "Submit recording again",
            "Contact clinical support",
            "Manual clinical review required"
        ]
    },
    ErrorCode.E_STREAMING_BUFFER: {
        "message": "Streaming buffer overflow",
        "clinical_context": "Real-time analysis data lost",
        "recovery": RecoveryStrategy.RETRY,
        "patient_impact": PatientImpact.LOW,
        "suggestions": [
            "Restart the recording session",
            "Use a stable network connection",
            "Close other applications"
        ]
    },
    ErrorCode.E_STREAMING_LATENCY: {
        "message": "Streaming latency exceeded threshold",
        "clinical_context": "Real-time feedback delayed",
        "recovery": RecoveryStrategy.DEGRADE,
        "patient_impact": PatientImpact.NONE,
        "suggestions": [
            "Analysis continues with reduced real-time feedback",
            "Check network connection",
            "Results will be available after recording"
        ]
    }
}


# =============================================================================
# Exception Classes
# =============================================================================

@dataclass
class ClinicalSpeechError(Exception):
    """
    Base exception for clinical speech analysis errors.
    
    Includes clinical context, recovery suggestions, and patient impact assessment.
    """
    code: str
    message: str
    clinical_context: str
    recovery_suggestions: List[str]
    patient_impact: PatientImpact = PatientImpact.LOW
    layer: PipelineLayer = PipelineLayer.CORE
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.ABORT
    details: Dict[str, Any] = field(default_factory=dict)
    cause: Optional[Exception] = None
    
    def __post_init__(self):
        super().__init__(f"{self.code}: {self.message}")
    
    def __str__(self):
        return f"[{self.layer.value}] {self.code}: {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "code": self.code,
            "message": self.message,
            "layer": self.layer.value,
            "clinical_context": self.clinical_context,
            "patient_impact": self.patient_impact.value,
            "recovery_strategy": self.recovery_strategy.value,
            "suggestions": self.recovery_suggestions,
            "details": self.details
        }
    
    def log(self):
        """Log the error with appropriate severity."""
        if self.patient_impact in (PatientImpact.HIGH, PatientImpact.CRITICAL):
            logger.error(str(self), exc_info=self.cause)
        elif self.patient_impact == PatientImpact.MEDIUM:
            logger.warning(str(self))
        else:
            logger.info(str(self))


@dataclass
class LayerError(Exception):
    """
    Legacy error class with layer context for backward compatibility.
    """
    layer: PipelineLayer
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    cause: Optional[Exception] = None
    
    def __post_init__(self):
        super().__init__(f"[{self.layer.value}] {self.code}: {self.message}")
    
    def __str__(self):
        return f"[{self.layer.value}] {self.code}: {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for API response."""
        return {
            "layer": self.layer.value,
            "code": self.code,
            "message": self.message,
            "details": self.details
        }


# =============================================================================
# Error Factory Functions
# =============================================================================

def create_clinical_error(
    code: str,
    message: Optional[str] = None,
    layer: PipelineLayer = PipelineLayer.CORE,
    details: Optional[Dict[str, Any]] = None,
    cause: Optional[Exception] = None
) -> ClinicalSpeechError:
    """
    Factory function to create clinical errors with full context.
    
    Args:
        code: Error code from ErrorCode class
        message: Optional override message
        layer: Pipeline layer where error occurred
        details: Additional error details
        cause: Original exception if any
        
    Returns:
        ClinicalSpeechError with full clinical context
    """
    error_info = ERROR_DETAILS.get(code, {})
    
    return ClinicalSpeechError(
        code=code,
        message=message or error_info.get("message", "Unknown error"),
        clinical_context=error_info.get("clinical_context", ""),
        recovery_suggestions=error_info.get("suggestions", []),
        patient_impact=error_info.get("patient_impact", PatientImpact.LOW),
        layer=layer,
        recovery_strategy=error_info.get("recovery", RecoveryStrategy.ABORT),
        details=details or {},
        cause=cause
    )


def raise_quality_error(
    code: str,
    message: str,
    snr_db: Optional[float] = None,
    clipping_ratio: Optional[float] = None,
    speech_ratio: Optional[float] = None
):
    """Raise a quality gate error with metrics."""
    details = {}
    if snr_db is not None:
        details["snr_db"] = snr_db
    if clipping_ratio is not None:
        details["clipping_ratio"] = clipping_ratio
    if speech_ratio is not None:
        details["speech_ratio"] = speech_ratio
    
    error = create_clinical_error(
        code=code,
        message=message,
        layer=PipelineLayer.QUALITY_GATE,
        details=details
    )
    error.log()
    raise error


def raise_extraction_error(
    code: str,
    message: str,
    feature_name: Optional[str] = None,
    details: Optional[Dict] = None
):
    """Raise a feature extraction error."""
    error_details = {"feature": feature_name} if feature_name else {}
    if details:
        error_details.update(details)
    
    error = create_clinical_error(
        code=code,
        message=message,
        layer=PipelineLayer.FEATURE_EXTRACTION,
        details=error_details
    )
    error.log()
    raise error


def raise_inference_error(
    code: str,
    message: str,
    model_name: Optional[str] = None,
    details: Optional[Dict] = None
):
    """Raise an ML inference error."""
    error_details = {"model": model_name} if model_name else {}
    if details:
        error_details.update(details)
    
    error = create_clinical_error(
        code=code,
        message=message,
        layer=PipelineLayer.ML_INFERENCE,
        details=error_details
    )
    error.log()
    raise error


def raise_clinical_error(
    code: str,
    message: str,
    details: Optional[Dict] = None
):
    """Raise a clinical assessment error."""
    error = create_clinical_error(
        code=code,
        message=message,
        layer=PipelineLayer.RISK_ASSESSMENT,
        details=details
    )
    error.log()
    raise error


def raise_streaming_error(
    code: str,
    message: str,
    session_id: Optional[str] = None,
    details: Optional[Dict] = None
):
    """Raise a streaming processing error."""
    error_details = {"session_id": session_id} if session_id else {}
    if details:
        error_details.update(details)
    
    error = create_clinical_error(
        code=code,
        message=message,
        layer=PipelineLayer.STREAMING,
        details=error_details
    )
    error.log()
    raise error


# =============================================================================
# Legacy Compatibility Functions
# =============================================================================

def raise_input_error(code: str, message: str, details: Dict = None):
    """Raise an input layer error (legacy)."""
    raise LayerError(
        layer=PipelineLayer.INPUT,
        code=code,
        message=message,
        details=details
    )


def raise_preprocessing_error(code: str, message: str, details: Dict = None):
    """Raise a preprocessing layer error (legacy)."""
    raise LayerError(
        layer=PipelineLayer.PREPROCESSING,
        code=code,
        message=message,
        details=details
    )


def raise_analysis_error(code: str, message: str, details: Dict = None):
    """Raise an analysis layer error (legacy)."""
    raise LayerError(
        layer=PipelineLayer.FEATURE_EXTRACTION,
        code=code,
        message=message,
        details=details
    )


def raise_output_error(code: str, message: str, details: Dict = None):
    """Raise an output layer error (legacy)."""
    raise LayerError(
        layer=PipelineLayer.OUTPUT,
        code=code,
        message=message,
        details=details
    )

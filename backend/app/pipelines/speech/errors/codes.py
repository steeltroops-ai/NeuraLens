"""
Speech Analysis Pipeline - Error Codes
Standardized error codes for each pipeline layer.

Error Format: E_{LAYER}_{NUMBER}
- E_HTTP_xxx: Router/HTTP errors
- E_INP_xxx: Input layer errors
- E_PREP_xxx: Preprocessing errors
- E_ANAL_xxx: Analysis errors
- E_CLIN_xxx: Clinical errors
- E_OUT_xxx: Output errors
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any


class PipelineLayer(Enum):
    """Pipeline execution layers for error tracing."""
    ROUTER = "L0_ROUTER"
    INPUT = "L1_INPUT"
    PREPROCESSING = "L2_PREPROCESSING"
    ANALYSIS = "L4_ANALYSIS"
    CLINICAL = "L5_CLINICAL"
    OUTPUT = "L6_OUTPUT"
    CORE = "CORE"
    MODEL = "MODEL"


class ErrorCode:
    """Standardized error codes for the speech pipeline."""
    
    # Layer 0: Router/HTTP
    E_HTTP_001 = "Invalid request format"
    E_HTTP_002 = "Missing required field"
    E_HTTP_003 = "File upload failed"
    E_HTTP_004 = "Unauthorized access"
    E_HTTP_005 = "Rate limit exceeded"
    
    # Layer 1: Input
    E_INP_001 = "No audio file received"
    E_INP_002 = "Invalid audio format"
    E_INP_003 = "File size exceeds limit"
    E_INP_004 = "Audio validation failed"
    E_INP_005 = "Audio duration too short"
    E_INP_006 = "Audio duration too long"
    E_INP_007 = "Unsupported MIME type"
    E_INP_008 = "Corrupted audio file"
    E_INP_009 = "Empty audio content"
    
    # Layer 2: Preprocessing
    E_PREP_001 = "Audio decode failed"
    E_PREP_002 = "Resampling failed"
    E_PREP_003 = "Normalization failed"
    E_PREP_004 = "Silence trimming failed"
    E_PREP_005 = "Channel conversion failed"
    E_PREP_006 = "Insufficient speech content"
    E_PREP_007 = "Audio quality too low"
    
    # Layer 4: Analysis
    E_ANAL_001 = "Feature extraction failed"
    E_ANAL_002 = "Pitch analysis failed"
    E_ANAL_003 = "Formant analysis failed"
    E_ANAL_004 = "Jitter/shimmer calculation failed"
    E_ANAL_005 = "CPPS calculation failed"
    E_ANAL_006 = "Prosodic analysis failed"
    E_ANAL_007 = "Analysis timeout"
    E_ANAL_008 = "Insufficient voiced segments"
    
    # Layer 5: Clinical
    E_CLIN_001 = "Risk calculation failed"
    E_CLIN_002 = "Condition scoring failed"
    E_CLIN_003 = "Recommendation generation failed"
    E_CLIN_004 = "Clinical notes generation failed"
    E_CLIN_005 = "Uncertainty estimation failed"
    
    # Layer 6: Output
    E_OUT_001 = "Response formatting failed"
    E_OUT_002 = "Biomarker mapping failed"
    E_OUT_003 = "Visualization generation failed"
    E_OUT_004 = "Serialization failed"
    
    # Model errors
    E_MODEL_001 = "Model loading failed"
    E_MODEL_002 = "Model inference failed"
    E_MODEL_003 = "Model timeout"
    E_MODEL_004 = "Model version mismatch"


@dataclass
class LayerError(Exception):
    """Error with layer context for tracing."""
    layer: PipelineLayer
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    cause: Optional[Exception] = None
    
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


def raise_input_error(code: str, message: str, details: Dict = None):
    """Raise an input layer error."""
    raise LayerError(
        layer=PipelineLayer.INPUT,
        code=code,
        message=message,
        details=details
    )


def raise_preprocessing_error(code: str, message: str, details: Dict = None):
    """Raise a preprocessing layer error."""
    raise LayerError(
        layer=PipelineLayer.PREPROCESSING,
        code=code,
        message=message,
        details=details
    )


def raise_analysis_error(code: str, message: str, details: Dict = None):
    """Raise an analysis layer error."""
    raise LayerError(
        layer=PipelineLayer.ANALYSIS,
        code=code,
        message=message,
        details=details
    )


def raise_clinical_error(code: str, message: str, details: Dict = None):
    """Raise a clinical layer error."""
    raise LayerError(
        layer=PipelineLayer.CLINICAL,
        code=code,
        message=message,
        details=details
    )


def raise_output_error(code: str, message: str, details: Dict = None):
    """Raise an output layer error."""
    raise LayerError(
        layer=PipelineLayer.OUTPUT,
        code=code,
        message=message,
        details=details
    )

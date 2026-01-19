"""
Error Codes and Messages for Retinal Analysis Pipeline

Implements the error taxonomy from the specification:
- VAL: Validation errors (001-099)
- PRE: Preprocessing errors (001-099)
- MOD: Model errors (001-099)
- ANA: Anatomical errors (001-099)
- CLI: Clinical errors (001-099)
- SYS: System errors (001-099)

Author: NeuraLens Medical AI Team
Version: 4.0.0
"""

from enum import Enum
from typing import Dict, Optional, Any
from dataclasses import dataclass


class ErrorSeverity(str, Enum):
    """Error severity levels"""
    FATAL = "fatal"      # Pipeline cannot continue
    ERROR = "error"      # Stage failed, may recover
    WARNING = "warning"  # Issue noted, continuing
    INFO = "info"        # Informational only


@dataclass
class ErrorDefinition:
    """Error code definition"""
    code: str
    title: str
    message: str
    guidance: str
    severity: ErrorSeverity
    recoverable: bool
    resubmission_recommended: bool
    retry_recommended: bool
    contact_support: bool = False


# =============================================================================
# ERROR CODE REGISTRY
# =============================================================================

ERROR_CODES: Dict[str, ErrorDefinition] = {
    # Validation Errors (VAL)
    "VAL_001": ErrorDefinition(
        code="VAL_001",
        title="Invalid File Format",
        message="The uploaded file is not a supported image format.",
        guidance="Please upload a JPEG, PNG, or TIFF fundus photograph.",
        severity=ErrorSeverity.FATAL,
        recoverable=False,
        resubmission_recommended=True,
        retry_recommended=False,
    ),
    "VAL_002": ErrorDefinition(
        code="VAL_002",
        title="Corrupted File Header",
        message="The file appears to be corrupted or has an invalid header.",
        guidance="Re-export or re-capture the image and try again.",
        severity=ErrorSeverity.FATAL,
        recoverable=False,
        resubmission_recommended=True,
        retry_recommended=False,
    ),
    "VAL_003": ErrorDefinition(
        code="VAL_003",
        title="Invalid File Extension",
        message="The file extension does not match the content type.",
        guidance="Rename the file with the correct extension.",
        severity=ErrorSeverity.ERROR,
        recoverable=False,
        resubmission_recommended=True,
        retry_recommended=False,
    ),
    "VAL_010": ErrorDefinition(
        code="VAL_010",
        title="Image Resolution Too Low",
        message="The image resolution is below the minimum required for accurate analysis.",
        guidance="Please upload an image with at least 512x512 pixels. 1024x1024 or higher is recommended.",
        severity=ErrorSeverity.FATAL,
        recoverable=False,
        resubmission_recommended=True,
        retry_recommended=False,
    ),
    "VAL_011": ErrorDefinition(
        code="VAL_011",
        title="Suboptimal Resolution",
        message="Image resolution is acceptable but below recommended.",
        guidance="1024x1024 or higher is recommended for best accuracy.",
        severity=ErrorSeverity.WARNING,
        recoverable=True,
        resubmission_recommended=False,
        retry_recommended=False,
    ),
    "VAL_012": ErrorDefinition(
        code="VAL_012",
        title="Image Resolution Too High",
        message="The image resolution exceeds maximum supported.",
        guidance="Maximum resolution is 8192x8192 pixels.",
        severity=ErrorSeverity.ERROR,
        recoverable=False,
        resubmission_recommended=True,
        retry_recommended=False,
    ),
    "VAL_020": ErrorDefinition(
        code="VAL_020",
        title="Poor Illumination",
        message="The image has poor illumination quality.",
        guidance="Recapture with proper lighting conditions.",
        severity=ErrorSeverity.ERROR,
        recoverable=False,
        resubmission_recommended=True,
        retry_recommended=False,
    ),
    "VAL_021": ErrorDefinition(
        code="VAL_021",
        title="Uneven Illumination",
        message="The image has uneven illumination across the field.",
        guidance="Adjust fundus camera settings for uniform lighting.",
        severity=ErrorSeverity.WARNING,
        recoverable=True,
        resubmission_recommended=False,
        retry_recommended=False,
    ),
    "VAL_022": ErrorDefinition(
        code="VAL_022",
        title="Overexposed Image",
        message="The image is overexposed with too many saturated pixels.",
        guidance="Reduce flash intensity and recapture.",
        severity=ErrorSeverity.ERROR,
        recoverable=False,
        resubmission_recommended=True,
        retry_recommended=False,
    ),
    "VAL_023": ErrorDefinition(
        code="VAL_023",
        title="Underexposed Image",
        message="The image is underexposed and too dark.",
        guidance="Increase flash intensity and recapture.",
        severity=ErrorSeverity.ERROR,
        recoverable=False,
        resubmission_recommended=True,
        retry_recommended=False,
    ),
    "VAL_030": ErrorDefinition(
        code="VAL_030",
        title="No Fundus Detected",
        message="Could not detect a fundus field of view in the image.",
        guidance="Ensure proper patient positioning and camera alignment.",
        severity=ErrorSeverity.FATAL,
        recoverable=False,
        resubmission_recommended=True,
        retry_recommended=False,
    ),
    "VAL_040": ErrorDefinition(
        code="VAL_040",
        title="Not a Retinal Image",
        message="The uploaded image does not appear to be a fundus photograph.",
        guidance="Please upload a retinal fundus image captured with a fundus camera.",
        severity=ErrorSeverity.FATAL,
        recoverable=False,
        resubmission_recommended=True,
        retry_recommended=False,
    ),
    "VAL_041": ErrorDefinition(
        code="VAL_041",
        title="No Vessel Pattern Detected",
        message="Could not detect retinal vessel patterns in the image.",
        guidance="Ensure the image is a fundus photograph with visible vessels.",
        severity=ErrorSeverity.ERROR,
        recoverable=False,
        resubmission_recommended=True,
        retry_recommended=False,
    ),
    "VAL_050": ErrorDefinition(
        code="VAL_050",
        title="Empty or Corrupted File",
        message="The uploaded file is empty or corrupted.",
        guidance="Upload a valid, non-empty image file.",
        severity=ErrorSeverity.FATAL,
        recoverable=False,
        resubmission_recommended=True,
        retry_recommended=False,
    ),
    "VAL_051": ErrorDefinition(
        code="VAL_051",
        title="Undecodable Image",
        message="Could not decode the image content.",
        guidance="The file may be corrupted. Try re-exporting from source.",
        severity=ErrorSeverity.FATAL,
        recoverable=False,
        resubmission_recommended=True,
        retry_recommended=False,
    ),
    "VAL_052": ErrorDefinition(
        code="VAL_052",
        title="File Too Large",
        message="The file exceeds the maximum allowed size.",
        guidance="Maximum file size is 15MB. Compress or resize the image.",
        severity=ErrorSeverity.ERROR,
        recoverable=False,
        resubmission_recommended=True,
        retry_recommended=False,
    ),
    
    # Preprocessing Errors (PRE)
    "PRE_001": ErrorDefinition(
        code="PRE_001",
        title="Color Normalization Failed",
        message="Could not normalize image colors.",
        guidance="Analysis will continue with original colors.",
        severity=ErrorSeverity.WARNING,
        recoverable=True,
        resubmission_recommended=False,
        retry_recommended=False,
    ),
    "PRE_002": ErrorDefinition(
        code="PRE_002",
        title="Contrast Enhancement Failed",
        message="CLAHE contrast enhancement failed.",
        guidance="Analysis will continue without enhancement.",
        severity=ErrorSeverity.WARNING,
        recoverable=True,
        resubmission_recommended=False,
        retry_recommended=False,
    ),
    "PRE_003": ErrorDefinition(
        code="PRE_003",
        title="Artifact Removal Failed",
        message="Could not remove image artifacts.",
        guidance="Analysis will continue with artifacts.",
        severity=ErrorSeverity.WARNING,
        recoverable=True,
        resubmission_recommended=False,
        retry_recommended=False,
    ),
    "PRE_010": ErrorDefinition(
        code="PRE_010",
        title="Image Quality Too Low",
        message="The image quality is too poor for reliable analysis.",
        guidance="Please recapture the fundus image with better focus, lighting, and patient positioning.",
        severity=ErrorSeverity.FATAL,
        recoverable=False,
        resubmission_recommended=True,
        retry_recommended=False,
    ),
    
    # Model Errors (MOD)
    "MOD_001": ErrorDefinition(
        code="MOD_001",
        title="Model Loading Failed",
        message="Could not load the analysis model.",
        guidance="System error. Please try again later.",
        severity=ErrorSeverity.FATAL,
        recoverable=False,
        resubmission_recommended=False,
        retry_recommended=True,
        contact_support=True,
    ),
    "MOD_002": ErrorDefinition(
        code="MOD_002",
        title="Analysis Timeout",
        message="The analysis took too long and was interrupted.",
        guidance="Please try again. If the problem persists, try with a smaller image.",
        severity=ErrorSeverity.ERROR,
        recoverable=True,
        resubmission_recommended=False,
        retry_recommended=True,
    ),
    "MOD_003": ErrorDefinition(
        code="MOD_003",
        title="Invalid Model Output",
        message="The model produced an invalid output.",
        guidance="System error. Please try again or contact support.",
        severity=ErrorSeverity.ERROR,
        recoverable=False,
        resubmission_recommended=False,
        retry_recommended=True,
        contact_support=True,
    ),
    "MOD_010": ErrorDefinition(
        code="MOD_010",
        title="Ensemble Disagreement",
        message="Model ensemble produced inconsistent predictions.",
        guidance="Results reported with high uncertainty.",
        severity=ErrorSeverity.WARNING,
        recoverable=True,
        resubmission_recommended=False,
        retry_recommended=False,
    ),
    
    # Anatomical Errors (ANA)
    "ANA_001": ErrorDefinition(
        code="ANA_001",
        title="Optic Disc Not Detected",
        message="The optic disc could not be reliably detected in the image.",
        guidance="Analysis will continue with reduced accuracy for glaucoma-related biomarkers.",
        severity=ErrorSeverity.WARNING,
        recoverable=True,
        resubmission_recommended=False,
        retry_recommended=False,
    ),
    "ANA_002": ErrorDefinition(
        code="ANA_002",
        title="Macula Not Detected",
        message="The macula could not be reliably detected in the image.",
        guidance="DME assessment may have reduced accuracy.",
        severity=ErrorSeverity.WARNING,
        recoverable=True,
        resubmission_recommended=False,
        retry_recommended=False,
    ),
    "ANA_003": ErrorDefinition(
        code="ANA_003",
        title="Vessel Segmentation Failed",
        message="Could not segment retinal vessels.",
        guidance="Vessel biomarkers will be unavailable.",
        severity=ErrorSeverity.WARNING,
        recoverable=True,
        resubmission_recommended=False,
        retry_recommended=False,
    ),
    "ANA_010": ErrorDefinition(
        code="ANA_010",
        title="Anatomical Inconsistency",
        message="Detected anatomy has inconsistent geometry.",
        guidance="Results flagged for manual review.",
        severity=ErrorSeverity.WARNING,
        recoverable=True,
        resubmission_recommended=False,
        retry_recommended=False,
    ),
    
    # Clinical Errors (CLI)
    "CLI_001": ErrorDefinition(
        code="CLI_001",
        title="DR Grading Failed",
        message="Could not determine DR grade.",
        guidance="Unable to grade - recommend clinical review.",
        severity=ErrorSeverity.ERROR,
        recoverable=False,
        resubmission_recommended=False,
        retry_recommended=True,
    ),
    "CLI_002": ErrorDefinition(
        code="CLI_002",
        title="Risk Calculation Error",
        message="Could not calculate risk score.",
        guidance="Using default risk level.",
        severity=ErrorSeverity.WARNING,
        recoverable=True,
        resubmission_recommended=False,
        retry_recommended=False,
    ),
    
    # System Errors (SYS)
    "SYS_001": ErrorDefinition(
        code="SYS_001",
        title="Internal Server Error",
        message="An internal error occurred.",
        guidance="Please try again. If the problem persists, contact support.",
        severity=ErrorSeverity.FATAL,
        recoverable=False,
        resubmission_recommended=False,
        retry_recommended=True,
        contact_support=True,
    ),
    "SYS_002": ErrorDefinition(
        code="SYS_002",
        title="Service Unavailable",
        message="The analysis service is temporarily unavailable.",
        guidance="Please try again in a few minutes.",
        severity=ErrorSeverity.FATAL,
        recoverable=False,
        resubmission_recommended=False,
        retry_recommended=True,
    ),
}


def get_error(code: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get formatted error response for a given error code.
    
    Args:
        code: Error code (e.g., "VAL_001")
        context: Optional context for message formatting
        
    Returns:
        Formatted error dictionary
    """
    error_def = ERROR_CODES.get(code)
    
    if not error_def:
        error_def = ErrorDefinition(
            code=code,
            title="Unknown Error",
            message="An unexpected error occurred.",
            guidance="Please try again or contact support.",
            severity=ErrorSeverity.ERROR,
            recoverable=False,
            resubmission_recommended=False,
            retry_recommended=True,
            contact_support=True,
        )
    
    # Format message with context if provided
    message = error_def.message
    guidance = error_def.guidance
    
    if context:
        try:
            message = message.format(**context)
            guidance = guidance.format(**context)
        except KeyError:
            pass  # Keep original if formatting fails
    
    return {
        "code": error_def.code,
        "category": error_def.code.split("_")[0].lower(),
        "title": error_def.title,
        "message": message,
        "guidance": guidance,
        "severity": error_def.severity.value,
        "recoverable": error_def.recoverable,
        "resubmission_recommended": error_def.resubmission_recommended,
        "retry_recommended": error_def.retry_recommended,
        "contact_support": error_def.contact_support,
        "technical_details": context,
    }


class PipelineException(Exception):
    """Pipeline-specific exception with error code"""
    
    def __init__(self, code: str, context: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        self.code = code
        self.context = context or {}
        self.cause = cause
        self.error = get_error(code, context)
        super().__init__(self.error["message"])
    
    def to_response(self) -> Dict[str, Any]:
        """Convert to API response format"""
        return self.error

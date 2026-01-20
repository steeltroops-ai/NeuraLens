"""
Dermatology Pipeline Error Codes

Comprehensive error taxonomy for the skin lesion analysis pipeline.
"""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass


class ErrorCategory(str, Enum):
    """Error categories."""
    VALIDATION = "validation"
    PREPROCESSING = "preprocessing"
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"
    SCORING = "scoring"
    EXPLANATION = "explanation"
    SYSTEM = "system"
    TIMEOUT = "timeout"


@dataclass
class ErrorDefinition:
    """Error definition with all metadata."""
    code: str
    category: ErrorCategory
    message: str
    user_title: str
    user_message: str
    user_action: str
    recoverable: bool
    tips: list


# =============================================================================
# ERROR CODE DEFINITIONS
# =============================================================================

ERROR_CODES: Dict[str, ErrorDefinition] = {
    # Validation - File errors
    "E_VAL_001": ErrorDefinition(
        code="E_VAL_001",
        category=ErrorCategory.VALIDATION,
        message="Invalid file type",
        user_title="Unsupported File Type",
        user_message="This file type is not supported. Please upload a JPEG, PNG, or HEIC image.",
        user_action="Upload a different file format",
        recoverable=True,
        tips=["Save your image as JPEG or PNG", "Check your camera settings"]
    ),
    "E_VAL_002": ErrorDefinition(
        code="E_VAL_002",
        category=ErrorCategory.VALIDATION,
        message="File too large",
        user_title="Image Too Large",
        user_message="Your image is too large (maximum 50MB). Please use a smaller image.",
        user_action="Compress or resize the image",
        recoverable=True,
        tips=["Reduce image resolution", "Use image compression"]
    ),
    "E_VAL_003": ErrorDefinition(
        code="E_VAL_003",
        category=ErrorCategory.VALIDATION,
        message="File too small",
        user_title="Image Too Small",
        user_message="The uploaded file is too small to be a valid image.",
        user_action="Upload a valid image file",
        recoverable=True,
        tips=["Ensure the file is not corrupted", "Try a different image"]
    ),
    "E_VAL_004": ErrorDefinition(
        code="E_VAL_004",
        category=ErrorCategory.VALIDATION,
        message="Corrupted file",
        user_title="Corrupted Image",
        user_message="The image file appears to be corrupted. Please try uploading again.",
        user_action="Re-upload the image",
        recoverable=True,
        tips=["Try saving the image again", "Take a new photo"]
    ),
    "E_VAL_005": ErrorDefinition(
        code="E_VAL_005",
        category=ErrorCategory.VALIDATION,
        message="Empty file",
        user_title="No Image Data",
        user_message="No image data was received. Please select an image and try again.",
        user_action="Select an image to upload",
        recoverable=True,
        tips=["Ensure you selected a file", "Check your internet connection"]
    ),
    
    # Validation - Quality errors
    "E_VAL_010": ErrorDefinition(
        code="E_VAL_010",
        category=ErrorCategory.VALIDATION,
        message="Resolution too low",
        user_title="Low Resolution",
        user_message="Image resolution is too low for reliable analysis. Please use at least 2MP.",
        user_action="Use a higher resolution image",
        recoverable=True,
        tips=["Use your camera's highest quality setting", "Get closer to the lesion"]
    ),
    "E_VAL_011": ErrorDefinition(
        code="E_VAL_011",
        category=ErrorCategory.VALIDATION,
        message="Image too blurry",
        user_title="Image Out of Focus",
        user_message="The image is too blurry for reliable analysis. Please retake with better focus.",
        user_action="Retake the photo with sharp focus",
        recoverable=True,
        tips=[
            "Hold your camera steady or use a tripod",
            "Tap on the lesion to focus before taking the photo",
            "Use macro mode if available",
            "Ensure good lighting"
        ]
    ),
    "E_VAL_012": ErrorDefinition(
        code="E_VAL_012",
        category=ErrorCategory.VALIDATION,
        message="Image overexposed",
        user_title="Image Too Bright",
        user_message="The image is overexposed (too bright). Please retake with less light.",
        user_action="Reduce lighting or adjust exposure",
        recoverable=True,
        tips=["Move away from direct light", "Reduce flash brightness", "Adjust exposure compensation"]
    ),
    "E_VAL_013": ErrorDefinition(
        code="E_VAL_013",
        category=ErrorCategory.VALIDATION,
        message="Image underexposed",
        user_title="Image Too Dark",
        user_message="The image is too dark. Please improve lighting and retake.",
        user_action="Increase lighting",
        recoverable=True,
        tips=["Use natural daylight", "Turn on room lights", "Use camera flash"]
    ),
    "E_VAL_014": ErrorDefinition(
        code="E_VAL_014",
        category=ErrorCategory.VALIDATION,
        message="Uneven lighting",
        user_title="Uneven Lighting",
        user_message="Uneven lighting detected. Please use diffuse, even lighting.",
        user_action="Improve lighting conditions",
        recoverable=True,
        tips=["Avoid direct sunlight", "Use diffuse lighting", "Remove shadows from the lesion"]
    ),
    "E_VAL_015": ErrorDefinition(
        code="E_VAL_015",
        category=ErrorCategory.VALIDATION,
        message="Severe color cast",
        user_title="Color Distortion",
        user_message="Strong color distortion detected. Please check lighting and camera settings.",
        user_action="Adjust white balance or lighting",
        recoverable=True,
        tips=["Use daylight or white LED lighting", "Disable color filters", "Check white balance"]
    ),
    
    # Validation - Content errors
    "E_VAL_020": ErrorDefinition(
        code="E_VAL_020",
        category=ErrorCategory.VALIDATION,
        message="No skin detected",
        user_title="No Skin Detected",
        user_message="This image does not appear to show skin. Please upload a skin lesion image.",
        user_action="Upload a skin lesion image",
        recoverable=True,
        tips=["Ensure the skin area is visible", "Take a close-up of the lesion"]
    ),
    "E_VAL_021": ErrorDefinition(
        code="E_VAL_021",
        category=ErrorCategory.VALIDATION,
        message="No lesion detected",
        user_title="No Lesion Detected",
        user_message="No lesion could be detected. Please ensure the lesion is clearly visible.",
        user_action="Center the lesion in the image",
        recoverable=True,
        tips=[
            "Center the lesion in the frame",
            "Get closer to the lesion",
            "Ensure good contrast with surrounding skin",
            "Use even lighting"
        ]
    ),
    "E_VAL_022": ErrorDefinition(
        code="E_VAL_022",
        category=ErrorCategory.VALIDATION,
        message="Multiple lesions detected",
        user_title="Multiple Lesions",
        user_message="Multiple lesions detected. Please submit one lesion at a time.",
        user_action="Crop to show single lesion",
        recoverable=True,
        tips=["Focus on the lesion of concern", "Crop the image to one lesion"]
    ),
    "E_VAL_023": ErrorDefinition(
        code="E_VAL_023",
        category=ErrorCategory.VALIDATION,
        message="Lesion at edge",
        user_title="Lesion Cut Off",
        user_message="The lesion is too close to the edge. Please center the lesion.",
        user_action="Retake with lesion centered",
        recoverable=True,
        tips=["Position the lesion in the center", "Include some surrounding skin"]
    ),
    "E_VAL_024": ErrorDefinition(
        code="E_VAL_024",
        category=ErrorCategory.VALIDATION,
        message="Excessive occlusion",
        user_title="Lesion Obscured",
        user_message="The lesion is partially hidden. Please remove hair or obstructions.",
        user_action="Clear the area and retake",
        recoverable=True,
        tips=["Remove or trim hair covering lesion", "Clean the area", "Ensure no shadows"]
    ),
    
    # Segmentation errors
    "E_SEG_001": ErrorDefinition(
        code="E_SEG_001",
        category=ErrorCategory.SEGMENTATION,
        message="No lesion detected",
        user_title="Lesion Not Found",
        user_message="Our AI could not identify a distinct lesion in the image.",
        user_action="Retake with clearer lesion visibility",
        recoverable=True,
        tips=["Get closer to the lesion", "Improve lighting", "Ensure contrast with skin"]
    ),
    "E_SEG_002": ErrorDefinition(
        code="E_SEG_002",
        category=ErrorCategory.SEGMENTATION,
        message="Low confidence segmentation",
        user_title="Uncertain Boundary",
        user_message="Lesion boundary could not be reliably determined.",
        user_action="Analysis proceeding with reduced accuracy",
        recoverable=True,
        tips=["Better image quality may improve results"]
    ),
    
    # Classification errors
    "E_CLS_001": ErrorDefinition(
        code="E_CLS_001",
        category=ErrorCategory.CLASSIFICATION,
        message="Model inference failed",
        user_title="Analysis Error",
        user_message="The analysis model encountered an error.",
        user_action="Please try again",
        recoverable=False,
        tips=["Wait a moment and retry", "Try a different image"]
    ),
    "E_CLS_003": ErrorDefinition(
        code="E_CLS_003",
        category=ErrorCategory.CLASSIFICATION,
        message="Low confidence",
        user_title="Uncertain Analysis",
        user_message="Our AI analysis has low confidence. Professional evaluation recommended.",
        user_action="Consult a dermatologist",
        recoverable=False,
        tips=["Do not delay seeking medical care", "Bring images to your appointment"]
    ),
    
    # System errors
    "E_SYS_001": ErrorDefinition(
        code="E_SYS_001",
        category=ErrorCategory.SYSTEM,
        message="Internal server error",
        user_title="Processing Error",
        user_message="An unexpected error occurred. Please try again.",
        user_action="Retry the analysis",
        recoverable=False,
        tips=["Wait a moment and try again", "Contact support if problem persists"]
    ),
    "E_SYS_002": ErrorDefinition(
        code="E_SYS_002",
        category=ErrorCategory.SYSTEM,
        message="Service unavailable",
        user_title="Service Unavailable",
        user_message="The analysis service is temporarily unavailable.",
        user_action="Please try again later",
        recoverable=False,
        tips=["Wait a few minutes", "Check service status"]
    ),
    
    # Timeout errors
    "E_TMO_001": ErrorDefinition(
        code="E_TMO_001",
        category=ErrorCategory.TIMEOUT,
        message="Processing timeout",
        user_title="Analysis Timeout",
        user_message="Analysis took too long. Please try again with a simpler image.",
        user_action="Retry with smaller image",
        recoverable=True,
        tips=["Use a smaller image", "Reduce image resolution"]
    ),
}


def get_error(code: str) -> Optional[ErrorDefinition]:
    """Get error definition by code."""
    return ERROR_CODES.get(code)


def get_error_response(code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get formatted error response."""
    error = get_error(code)
    if error is None:
        error = ERROR_CODES["E_SYS_001"]
    
    return {
        "code": error.code,
        "category": error.category.value,
        "title": error.user_title,
        "message": error.user_message,
        "action": error.user_action,
        "recoverable": error.recoverable,
        "tips": error.tips
    }

"""
Radiology Error Codes

Standardized error codes for the radiology pipeline.
Format: E_{LAYER}_{NUMBER}
"""


class ErrorCode:
    """
    Standardized error code definitions.
    
    Layers:
    - E_GEN: General errors
    - E_VAL: Validation errors
    - E_DCM: DICOM errors
    - E_VOL: Volume errors
    - E_PREP: Preprocessing errors
    - E_DET: Detection errors
    - E_ANAL: Analysis errors
    - E_INF: Inference errors
    - E_SYS: System errors
    - W_xxx: Warnings
    """
    
    # General Errors (E_GEN_xxx)
    E_GEN_001 = "E_GEN_001"  # No file received
    E_GEN_002 = "E_GEN_002"  # Invalid file format
    E_GEN_003 = "E_GEN_003"  # File size exceeds limit
    E_GEN_004 = "E_GEN_004"  # File decode failed
    E_GEN_005 = "E_GEN_005"  # Resolution out of range
    E_GEN_006 = "E_GEN_006"  # Image appears blank
    E_GEN_007 = "E_GEN_007"  # Non-medical image detected
    
    # Validation Errors (E_VAL_xxx)
    E_VAL_001 = "E_VAL_001"  # No valid input
    E_VAL_002 = "E_VAL_002"  # Invalid format
    E_VAL_003 = "E_VAL_003"  # Validation failed
    
    # DICOM Errors (E_DCM_xxx)
    E_DCM_001 = "E_DCM_001"  # Invalid DICOM structure
    E_DCM_002 = "E_DCM_002"  # Required tags missing
    E_DCM_003 = "E_DCM_003"  # Unsupported modality
    E_DCM_004 = "E_DCM_004"  # Pixel data decode error
    E_DCM_005 = "E_DCM_005"  # Slice ordering error
    E_DCM_006 = "E_DCM_006"  # Spatial inconsistency
    
    # Volume Errors (E_VOL_xxx)
    E_VOL_001 = "E_VOL_001"  # Insufficient slices
    E_VOL_002 = "E_VOL_002"  # Too many slices
    E_VOL_003 = "E_VOL_003"  # Missing slices detected
    E_VOL_004 = "E_VOL_004"  # Orientation mismatch
    E_VOL_005 = "E_VOL_005"  # Dimension mismatch
    
    # Preprocessing Errors (E_PREP_xxx)
    E_PREP_001 = "E_PREP_001"  # Critical decode failure
    E_PREP_002 = "E_PREP_002"  # Unsupported pixel format
    E_PREP_003 = "E_PREP_003"  # Normalization failed
    E_PREP_004 = "E_PREP_004"  # Zero dynamic range
    E_PREP_005 = "E_PREP_005"  # Bias correction diverged
    E_PREP_006 = "E_PREP_006"  # Memory overflow
    
    # Detection Errors (E_DET_xxx)
    E_DET_001 = "E_DET_001"  # Anatomy not found
    E_DET_002 = "E_DET_002"  # Segmentation failed
    E_DET_003 = "E_DET_003"  # Structure validation failed
    E_DET_004 = "E_DET_004"  # Implausible anatomy
    
    # Analysis Errors (E_ANAL_xxx)
    E_ANAL_001 = "E_ANAL_001"  # Pathology detection failed
    E_ANAL_002 = "E_ANAL_002"  # Confidence too low
    E_ANAL_003 = "E_ANAL_003"  # Inconsistent predictions
    E_ANAL_004 = "E_ANAL_004"  # Severity scoring failed
    
    # Inference Errors (E_INF_xxx)
    E_INF_001 = "E_INF_001"  # Model timeout
    E_INF_002 = "E_INF_002"  # Model load failed
    E_INF_003 = "E_INF_003"  # Out of memory
    E_INF_004 = "E_INF_004"  # Inference exception
    E_INF_005 = "E_INF_005"  # Model not available
    
    # System Errors (E_SYS_xxx)
    E_SYS_001 = "E_SYS_001"  # Internal server error
    E_SYS_002 = "E_SYS_002"  # Service unavailable
    E_SYS_003 = "E_SYS_003"  # Rate limit exceeded
    E_SYS_004 = "E_SYS_004"  # Authentication failed
    E_SYS_005 = "E_SYS_005"  # Storage error
    
    # Warnings (W_xxx_xxx)
    W_QUAL_001 = "W_QUAL_001"  # Image quality suboptimal
    W_DET_001 = "W_DET_001"   # Low segmentation confidence
    W_PREP_001 = "W_PREP_001" # Invalid HU values
    W_PREP_002 = "W_PREP_002" # Saturation detected


# Human-readable error messages
ERROR_MESSAGES = {
    ErrorCode.E_GEN_001: {
        "title": "No Image Uploaded",
        "explanation": "We didn't receive any image file. Please upload a medical image to analyze.",
        "action": "Upload a chest X-ray, CT scan, or MRI image."
    },
    ErrorCode.E_GEN_002: {
        "title": "Invalid File Type",
        "explanation": "The uploaded file type is not supported.",
        "action": "Please upload a JPEG, PNG, or DICOM file."
    },
    ErrorCode.E_GEN_003: {
        "title": "File Too Large",
        "explanation": "The uploaded file exceeds the maximum size limit.",
        "action": "Please upload a smaller file (max 10MB for images, 50MB for DICOM)."
    },
    ErrorCode.E_GEN_004: {
        "title": "File Decode Failed",
        "explanation": "The uploaded file could not be decoded.",
        "action": "Please ensure the file is not corrupted and try again."
    },
    ErrorCode.E_GEN_005: {
        "title": "Resolution Out of Range",
        "explanation": "The image resolution is outside acceptable limits.",
        "action": "Please upload an image between 224x224 and 4096x4096 pixels."
    },
    ErrorCode.E_GEN_006: {
        "title": "Blank Image",
        "explanation": "The uploaded image appears to be blank or nearly uniform.",
        "action": "Please upload a valid medical image."
    },
    ErrorCode.E_GEN_007: {
        "title": "Non-Medical Image",
        "explanation": "The uploaded image doesn't appear to be a medical radiological image.",
        "action": "Please upload a valid chest X-ray, CT scan, or MRI image."
    },
    ErrorCode.E_INF_001: {
        "title": "Analysis Timeout",
        "explanation": "The analysis took longer than expected and timed out.",
        "action": "Please try again. If the problem persists, try a smaller image."
    },
    ErrorCode.E_SYS_001: {
        "title": "Internal Error",
        "explanation": "An internal error occurred during processing.",
        "action": "Please try again. If the problem persists, contact support."
    }
}


def get_error_message(code: str) -> dict:
    """Get human-readable error message."""
    return ERROR_MESSAGES.get(code, {
        "title": "Analysis Error",
        "explanation": "An error occurred during image analysis.",
        "action": "Please try again or contact support if the problem persists."
    })

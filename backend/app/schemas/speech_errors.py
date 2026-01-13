"""
Speech Analysis Error Response Models
Structured error responses for the speech analysis pipeline.

Feature: speech-pipeline-fix
**Validates: Requirements 2.1, 2.5**
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid


# Error code definitions (SP001-SP999)
class SpeechErrorCodes:
    """
    Speech Analysis Error Codes
    
    SP001-SP099: Validation errors
    SP100-SP199: Processing errors
    SP200-SP299: Timeout errors
    SP900-SP999: Unknown/system errors
    """
    # Validation errors (SP001-SP099)
    INVALID_FORMAT = "SP001"
    DURATION_TOO_SHORT = "SP002"
    DURATION_TOO_LONG = "SP003"
    INVALID_SAMPLE_RATE = "SP004"
    FILE_TOO_LARGE = "SP005"
    EMPTY_AUDIO = "SP006"
    CORRUPTED_FILE = "SP007"
    
    # Processing errors (SP100-SP199)
    FEATURE_EXTRACTION_FAILED = "SP100"
    BIOMARKER_CALCULATION_FAILED = "SP101"
    MODEL_INFERENCE_FAILED = "SP102"
    AUDIO_LOADING_FAILED = "SP103"
    RESAMPLING_FAILED = "SP104"
    
    # Timeout errors (SP200-SP299)
    PROCESSING_TIMEOUT = "SP200"
    UPLOAD_TIMEOUT = "SP201"
    
    # Unknown/system errors (SP900-SP999)
    UNKNOWN_ERROR = "SP999"
    
    @classmethod
    def get_all_codes(cls) -> Dict[str, str]:
        """Get all error codes as a dictionary"""
        return {
            name: value for name, value in vars(cls).items()
            if isinstance(value, str) and value.startswith("SP")
        }


# Error code to HTTP status mapping
ERROR_CODE_TO_HTTP_STATUS: Dict[str, int] = {
    SpeechErrorCodes.INVALID_FORMAT: 400,
    SpeechErrorCodes.DURATION_TOO_SHORT: 400,
    SpeechErrorCodes.DURATION_TOO_LONG: 400,
    SpeechErrorCodes.INVALID_SAMPLE_RATE: 400,
    SpeechErrorCodes.FILE_TOO_LARGE: 413,
    SpeechErrorCodes.EMPTY_AUDIO: 400,
    SpeechErrorCodes.CORRUPTED_FILE: 400,
    SpeechErrorCodes.FEATURE_EXTRACTION_FAILED: 500,
    SpeechErrorCodes.BIOMARKER_CALCULATION_FAILED: 500,
    SpeechErrorCodes.MODEL_INFERENCE_FAILED: 500,
    SpeechErrorCodes.AUDIO_LOADING_FAILED: 500,
    SpeechErrorCodes.RESAMPLING_FAILED: 500,
    SpeechErrorCodes.PROCESSING_TIMEOUT: 408,
    SpeechErrorCodes.UPLOAD_TIMEOUT: 408,
    SpeechErrorCodes.UNKNOWN_ERROR: 500,
}


# User-friendly error messages
ERROR_CODE_TO_MESSAGE: Dict[str, str] = {
    SpeechErrorCodes.INVALID_FORMAT: "Unsupported audio format. Supported formats: WAV, MP3, M4A, WebM, OGG",
    SpeechErrorCodes.DURATION_TOO_SHORT: "Audio too short. Minimum duration is 3 seconds.",
    SpeechErrorCodes.DURATION_TOO_LONG: "Audio too long. Maximum duration is 60 seconds.",
    SpeechErrorCodes.INVALID_SAMPLE_RATE: "Invalid sample rate. Audio will be resampled to 16kHz.",
    SpeechErrorCodes.FILE_TOO_LARGE: "File too large. Please upload a smaller audio file.",
    SpeechErrorCodes.EMPTY_AUDIO: "Audio file appears to be empty or contains no audio data.",
    SpeechErrorCodes.CORRUPTED_FILE: "Audio file appears to be corrupted or unreadable.",
    SpeechErrorCodes.FEATURE_EXTRACTION_FAILED: "Failed to extract audio features. Please try again.",
    SpeechErrorCodes.BIOMARKER_CALCULATION_FAILED: "Failed to calculate biomarkers. Please try again.",
    SpeechErrorCodes.MODEL_INFERENCE_FAILED: "Model inference failed. Please try again.",
    SpeechErrorCodes.AUDIO_LOADING_FAILED: "Failed to load audio file. Please ensure the file is valid.",
    SpeechErrorCodes.RESAMPLING_FAILED: "Failed to resample audio. Please try a different file.",
    SpeechErrorCodes.PROCESSING_TIMEOUT: "Speech processing timeout. Please try with a shorter audio file.",
    SpeechErrorCodes.UPLOAD_TIMEOUT: "Upload timeout. Please check your connection and try again.",
    SpeechErrorCodes.UNKNOWN_ERROR: "An unexpected error occurred. Please try again.",
}


@dataclass
class SpeechAnalysisError:
    """
    Dataclass for internal error representation.
    
    Used for passing error information between components before
    converting to a response model.
    """
    error_code: str
    message: str
    error_reference_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate error code format"""
        if not self.error_code.startswith("SP"):
            raise ValueError(f"Invalid error code format: {self.error_code}")
    
    @classmethod
    def from_code(
        cls,
        error_code: str,
        details: Optional[Dict[str, Any]] = None,
        custom_message: Optional[str] = None
    ) -> "SpeechAnalysisError":
        """
        Create an error from an error code.
        
        Args:
            error_code: The error code (e.g., SP001)
            details: Optional additional details
            custom_message: Optional custom message (overrides default)
            
        Returns:
            SpeechAnalysisError instance
        """
        message = custom_message or ERROR_CODE_TO_MESSAGE.get(
            error_code, 
            ERROR_CODE_TO_MESSAGE[SpeechErrorCodes.UNKNOWN_ERROR]
        )
        return cls(
            error_code=error_code,
            message=message,
            details=details
        )
    
    def get_http_status(self) -> int:
        """Get the HTTP status code for this error"""
        return ERROR_CODE_TO_HTTP_STATUS.get(self.error_code, 500)


class SpeechAnalysisErrorResponse(BaseModel):
    """
    Pydantic model for API error responses.
    
    This is the structured JSON error response returned to clients.
    
    **Validates: Requirements 2.1, 2.5**
    """
    success: bool = Field(default=False, description="Always False for error responses")
    error_code: str = Field(..., description="Error code (SP001-SP999)")
    error_reference_id: str = Field(..., description="Unique error reference ID for support")
    message: str = Field(..., description="User-friendly error message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Additional error details for debugging"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error_code": "SP001",
                "error_reference_id": "550e8400-e29b-41d4-a716-446655440000",
                "message": "Unsupported audio format. Supported formats: WAV, MP3, M4A, WebM, OGG",
                "timestamp": "2024-01-15T10:30:00Z",
                "details": {
                    "detected_format": "video/mp4",
                    "supported_formats": ["audio/wav", "audio/mpeg", "audio/mp4"]
                }
            }
        }
    
    @classmethod
    def from_error(cls, error: SpeechAnalysisError) -> "SpeechAnalysisErrorResponse":
        """
        Create a response from a SpeechAnalysisError.
        
        Args:
            error: The SpeechAnalysisError instance
            
        Returns:
            SpeechAnalysisErrorResponse instance
        """
        return cls(
            error_code=error.error_code,
            error_reference_id=error.error_reference_id,
            message=error.message,
            timestamp=error.timestamp,
            details=error.details
        )
    
    @classmethod
    def from_code(
        cls,
        error_code: str,
        details: Optional[Dict[str, Any]] = None,
        custom_message: Optional[str] = None
    ) -> "SpeechAnalysisErrorResponse":
        """
        Create a response directly from an error code.
        
        Args:
            error_code: The error code (e.g., SP001)
            details: Optional additional details
            custom_message: Optional custom message
            
        Returns:
            SpeechAnalysisErrorResponse instance
        """
        error = SpeechAnalysisError.from_code(error_code, details, custom_message)
        return cls.from_error(error)
    
    def is_validation_error(self) -> bool:
        """Check if this is a validation error (SP001-SP099)"""
        if len(self.error_code) >= 3:
            code_num = int(self.error_code[2:])
            return 1 <= code_num <= 99
        return False
    
    def is_processing_error(self) -> bool:
        """Check if this is a processing error (SP100-SP199)"""
        if len(self.error_code) >= 3:
            code_num = int(self.error_code[2:])
            return 100 <= code_num <= 199
        return False
    
    def is_timeout_error(self) -> bool:
        """Check if this is a timeout error (SP200-SP299)"""
        if len(self.error_code) >= 3:
            code_num = int(self.error_code[2:])
            return 200 <= code_num <= 299
        return False

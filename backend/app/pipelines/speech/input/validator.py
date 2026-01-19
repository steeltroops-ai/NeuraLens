"""
Speech Analysis Pipeline - Input Validator
Validates all incoming audio data before processing.

Errors from this module have prefix: E_INP_
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
import mimetypes

from ..config import INPUT_CONSTRAINTS, SUPPORTED_MIME_TYPES
from ..errors.codes import ErrorCode, raise_input_error

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    file_info: Optional[dict] = None


class AudioValidator:
    """Validates audio input for speech analysis pipeline."""
    
    def __init__(self):
        self.max_file_size = INPUT_CONSTRAINTS["max_file_size_mb"] * 1024 * 1024
        self.min_duration = INPUT_CONSTRAINTS["min_duration_sec"]
        self.max_duration = INPUT_CONSTRAINTS["max_duration_sec"]
        self.supported_formats = INPUT_CONSTRAINTS["supported_formats"]
    
    def validate(
        self,
        audio_bytes: bytes,
        filename: str,
        content_type: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate audio input data.
        
        Args:
            audio_bytes: Raw audio file bytes
            filename: Original filename
            content_type: MIME type from upload
            
        Returns:
            ValidationResult with validation status
            
        Raises:
            LayerError: With E_INP_xxx code on critical failure
        """
        errors = []
        warnings = []
        
        # Check if file is empty
        if not audio_bytes or len(audio_bytes) == 0:
            raise_input_error(
                "E_INP_009",
                ErrorCode.E_INP_009,
                {"filename": filename}
            )
        
        # Check file size
        if len(audio_bytes) > self.max_file_size:
            raise_input_error(
                "E_INP_003",
                f"{ErrorCode.E_INP_003}: {len(audio_bytes) / (1024*1024):.1f}MB exceeds {INPUT_CONSTRAINTS['max_file_size_mb']}MB limit",
                {"size_bytes": len(audio_bytes), "max_bytes": self.max_file_size}
            )
        
        # Check file format
        if not self._is_valid_format(filename, content_type):
            raise_input_error(
                "E_INP_002",
                ErrorCode.E_INP_002,
                {"filename": filename, "content_type": content_type}
            )
        
        # Build file info
        file_info = {
            "filename": filename,
            "size_bytes": len(audio_bytes),
            "content_type": content_type or self._guess_mime_type(filename)
        }
        
        # Size warnings
        if len(audio_bytes) < 10 * 1024:  # Less than 10KB
            warnings.append("Audio file is very small - may affect analysis quality")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            file_info=file_info
        )
    
    def _is_valid_format(self, filename: str, content_type: Optional[str]) -> bool:
        """Check if audio format is supported."""
        # Check by extension
        if filename:
            ext = filename.lower().split('.')[-1] if '.' in filename else ''
            if ext in self.supported_formats:
                return True
        
        # Check by MIME type
        if content_type:
            # Handle content types with parameters (e.g., "audio/webm;codecs=opus")
            base_type = content_type.split(';')[0].strip().lower()
            if content_type.lower() in SUPPORTED_MIME_TYPES or base_type in SUPPORTED_MIME_TYPES:
                return True
        
        # If we have either a valid extension or MIME type, accept it
        return False
    
    def _guess_mime_type(self, filename: str) -> str:
        """Guess MIME type from filename."""
        if not filename:
            return "audio/wav"
        
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type or "audio/wav"
    
    def validate_duration(self, duration_seconds: float) -> None:
        """
        Validate audio duration after loading.
        
        Args:
            duration_seconds: Audio duration in seconds
            
        Raises:
            LayerError: If duration is outside acceptable range
        """
        if duration_seconds < self.min_duration:
            raise_input_error(
                "E_INP_005",
                f"{ErrorCode.E_INP_005}: {duration_seconds:.1f}s is less than minimum {self.min_duration}s",
                {"duration": duration_seconds, "minimum": self.min_duration}
            )
        
        if duration_seconds > self.max_duration:
            logger.warning(
                f"Audio duration {duration_seconds:.1f}s exceeds maximum {self.max_duration}s - will be truncated"
            )

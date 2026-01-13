"""
Error Handler Service for Speech Analysis Pipeline
Provides structured error responses and logging.

Feature: speech-pipeline-fix
**Validates: Requirements 2.2, 2.3, 2.4**
"""

import logging
import traceback
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi.responses import JSONResponse

from app.schemas.speech_errors import (
    SpeechAnalysisError,
    SpeechAnalysisErrorResponse,
    SpeechErrorCodes,
    ERROR_CODE_TO_HTTP_STATUS,
    ERROR_CODE_TO_MESSAGE,
)


# Configure structured logging
logger = logging.getLogger(__name__)


class SpeechAnalysisErrorHandler:
    """
    Centralized error handling for speech analysis pipeline.
    
    Provides:
    - Structured error response creation with unique reference IDs
    - Structured logging with context for debugging
    - HTTP status code mapping
    
    **Validates: Requirements 2.2, 2.3, 2.4**
    """
    
    def __init__(self):
        """Initialize the error handler"""
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def create_error_response(
        self,
        error_code: str,
        session_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        custom_message: Optional[str] = None
    ) -> SpeechAnalysisErrorResponse:
        """
        Create a structured error response with unique reference ID.
        
        Args:
            error_code: The error code (e.g., SP001)
            session_id: Optional session ID for tracking
            details: Optional additional details
            custom_message: Optional custom message (overrides default)
            
        Returns:
            SpeechAnalysisErrorResponse instance
        """
        # Generate unique error reference ID
        error_reference_id = str(uuid.uuid4())
        
        # Get message from mapping or use custom
        message = custom_message or ERROR_CODE_TO_MESSAGE.get(
            error_code,
            ERROR_CODE_TO_MESSAGE[SpeechErrorCodes.UNKNOWN_ERROR]
        )
        
        # Add session_id to details if provided
        if session_id and details is None:
            details = {"session_id": session_id}
        elif session_id and details is not None:
            details["session_id"] = session_id
        
        return SpeechAnalysisErrorResponse(
            error_code=error_code,
            error_reference_id=error_reference_id,
            message=message,
            timestamp=datetime.utcnow(),
            details=details
        )
    
    def create_json_response(
        self,
        error_code: str,
        session_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        custom_message: Optional[str] = None
    ) -> JSONResponse:
        """
        Create a FastAPI JSONResponse with proper HTTP status code.
        
        Args:
            error_code: The error code (e.g., SP001)
            session_id: Optional session ID for tracking
            details: Optional additional details
            custom_message: Optional custom message
            
        Returns:
            JSONResponse with appropriate status code
        """
        error_response = self.create_error_response(
            error_code=error_code,
            session_id=session_id,
            details=details,
            custom_message=custom_message
        )
        
        http_status = ERROR_CODE_TO_HTTP_STATUS.get(error_code, 500)
        
        return JSONResponse(
            status_code=http_status,
            content=error_response.model_dump(mode='json')
        )
    
    def log_error(
        self,
        error: Exception,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        error_reference_id: Optional[str] = None
    ) -> str:
        """
        Log error with full context for debugging.
        
        Args:
            error: The exception that occurred
            session_id: Optional session ID for tracking
            context: Optional additional context (e.g., audio metadata)
            error_reference_id: Optional pre-generated reference ID
            
        Returns:
            The error reference ID used for logging
        """
        if error_reference_id is None:
            error_reference_id = str(uuid.uuid4())
        
        # Build structured log context
        log_context = {
            "error_reference_id": error_reference_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        if session_id:
            log_context["session_id"] = session_id
        
        if context:
            log_context["context"] = context
        
        # Get stack trace
        stack_trace = traceback.format_exc()
        log_context["stack_trace"] = stack_trace
        
        # Log with structured format
        self._logger.error(
            f"Speech analysis error [{error_reference_id}]: {type(error).__name__}: {error}",
            extra=log_context
        )
        
        return error_reference_id
    
    def log_warning(
        self,
        message: str,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a warning with context.
        
        Args:
            message: Warning message
            session_id: Optional session ID
            context: Optional additional context
        """
        log_context = {
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        if session_id:
            log_context["session_id"] = session_id
        
        if context:
            log_context["context"] = context
        
        self._logger.warning(message, extra=log_context)
    
    def handle_validation_error(
        self,
        error_code: str,
        session_id: Optional[str] = None,
        audio_metadata: Optional[Dict[str, Any]] = None,
        custom_message: Optional[str] = None
    ) -> JSONResponse:
        """
        Handle audio validation errors.
        
        Args:
            error_code: The validation error code (SP001-SP099)
            session_id: Optional session ID
            audio_metadata: Optional audio file metadata
            custom_message: Optional custom error message
            
        Returns:
            JSONResponse with validation error details
        """
        details = {}
        if audio_metadata:
            details["audio_metadata"] = audio_metadata
        
        error_response = self.create_error_response(
            error_code=error_code,
            session_id=session_id,
            details=details if details else None,
            custom_message=custom_message
        )
        
        # Log the validation error
        self._logger.info(
            f"Validation error [{error_response.error_reference_id}]: {error_code}",
            extra={
                "error_reference_id": error_response.error_reference_id,
                "error_code": error_code,
                "session_id": session_id,
                "audio_metadata": audio_metadata
            }
        )
        
        http_status = ERROR_CODE_TO_HTTP_STATUS.get(error_code, 400)
        
        return JSONResponse(
            status_code=http_status,
            content=error_response.model_dump(mode='json')
        )
    
    def handle_processing_error(
        self,
        error: Exception,
        session_id: Optional[str] = None,
        processing_stage: Optional[str] = None
    ) -> JSONResponse:
        """
        Handle ML processing errors.
        
        Args:
            error: The exception that occurred
            session_id: Optional session ID
            processing_stage: Optional stage where error occurred
            
        Returns:
            JSONResponse with processing error details
        """
        # Determine error code based on processing stage
        error_code = SpeechErrorCodes.UNKNOWN_ERROR
        if processing_stage:
            stage_to_code = {
                "feature_extraction": SpeechErrorCodes.FEATURE_EXTRACTION_FAILED,
                "biomarker_calculation": SpeechErrorCodes.BIOMARKER_CALCULATION_FAILED,
                "model_inference": SpeechErrorCodes.MODEL_INFERENCE_FAILED,
                "audio_loading": SpeechErrorCodes.AUDIO_LOADING_FAILED,
                "resampling": SpeechErrorCodes.RESAMPLING_FAILED,
            }
            error_code = stage_to_code.get(processing_stage, SpeechErrorCodes.UNKNOWN_ERROR)
        
        # Create error response
        error_response = self.create_error_response(
            error_code=error_code,
            session_id=session_id,
            details={"processing_stage": processing_stage} if processing_stage else None
        )
        
        # Log the error with full context
        self.log_error(
            error=error,
            session_id=session_id,
            context={"processing_stage": processing_stage},
            error_reference_id=error_response.error_reference_id
        )
        
        http_status = ERROR_CODE_TO_HTTP_STATUS.get(error_code, 500)
        
        return JSONResponse(
            status_code=http_status,
            content=error_response.model_dump(mode='json')
        )
    
    def handle_timeout_error(
        self,
        session_id: Optional[str] = None,
        processing_duration: Optional[float] = None,
        timeout_limit: Optional[float] = None
    ) -> JSONResponse:
        """
        Handle processing timeout errors.
        
        Args:
            session_id: Optional session ID
            processing_duration: How long processing ran before timeout
            timeout_limit: The configured timeout limit
            
        Returns:
            JSONResponse with timeout error details
        """
        details = {}
        if processing_duration is not None:
            details["processing_duration"] = processing_duration
        if timeout_limit is not None:
            details["timeout_limit"] = timeout_limit
        
        error_response = self.create_error_response(
            error_code=SpeechErrorCodes.PROCESSING_TIMEOUT,
            session_id=session_id,
            details=details if details else None
        )
        
        # Log the timeout
        self._logger.warning(
            f"Processing timeout [{error_response.error_reference_id}]: "
            f"duration={processing_duration}s, limit={timeout_limit}s",
            extra={
                "error_reference_id": error_response.error_reference_id,
                "session_id": session_id,
                "processing_duration": processing_duration,
                "timeout_limit": timeout_limit
            }
        )
        
        return JSONResponse(
            status_code=408,
            content=error_response.model_dump(mode='json')
        )


# Singleton instance for use across the application
speech_error_handler = SpeechAnalysisErrorHandler()

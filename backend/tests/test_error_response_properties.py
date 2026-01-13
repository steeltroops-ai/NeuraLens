"""
Property-based tests for Speech Analysis Error Response Models
Uses hypothesis for property-based testing.

Feature: speech-pipeline-fix
Property 4: Error Response Structure Completeness
**Validates: Requirements 2.1, 2.5**
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from datetime import datetime
import uuid
import re

from app.schemas.speech_errors import (
    SpeechAnalysisError,
    SpeechAnalysisErrorResponse,
    SpeechErrorCodes,
    ERROR_CODE_TO_HTTP_STATUS,
    ERROR_CODE_TO_MESSAGE,
)


# Strategy for generating valid error codes
valid_error_codes = st.sampled_from([
    SpeechErrorCodes.INVALID_FORMAT,
    SpeechErrorCodes.DURATION_TOO_SHORT,
    SpeechErrorCodes.DURATION_TOO_LONG,
    SpeechErrorCodes.INVALID_SAMPLE_RATE,
    SpeechErrorCodes.FILE_TOO_LARGE,
    SpeechErrorCodes.EMPTY_AUDIO,
    SpeechErrorCodes.CORRUPTED_FILE,
    SpeechErrorCodes.FEATURE_EXTRACTION_FAILED,
    SpeechErrorCodes.BIOMARKER_CALCULATION_FAILED,
    SpeechErrorCodes.MODEL_INFERENCE_FAILED,
    SpeechErrorCodes.AUDIO_LOADING_FAILED,
    SpeechErrorCodes.RESAMPLING_FAILED,
    SpeechErrorCodes.PROCESSING_TIMEOUT,
    SpeechErrorCodes.UPLOAD_TIMEOUT,
    SpeechErrorCodes.UNKNOWN_ERROR,
])

# Strategy for generating optional details
optional_details = st.one_of(
    st.none(),
    st.dictionaries(
        keys=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('L', 'N'))),
        values=st.one_of(st.text(max_size=100), st.integers(), st.floats(allow_nan=False)),
        max_size=5
    )
)


class TestErrorResponseStructureProperty:
    """
    Property 4: Error Response Structure Completeness
    
    For any error returned by the Speech_Analyzer, the response SHALL contain:
    - error_code (non-empty string)
    - message (non-empty string)
    - error_reference_id (unique UUID)
    - timestamp (valid ISO datetime)
    
    **Validates: Requirements 2.1, 2.5**
    """
    
    @given(error_code=valid_error_codes, details=optional_details)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_error_response_contains_required_fields(self, error_code, details):
        """
        Property: For any error code, response contains all required fields
        
        Feature: speech-pipeline-fix, Property 4: Error Response Structure Completeness
        **Validates: Requirements 2.1, 2.5**
        """
        response = SpeechAnalysisErrorResponse.from_code(error_code, details)
        
        # Verify all required fields are present and non-empty
        assert response.error_code is not None
        assert len(response.error_code) > 0
        
        assert response.message is not None
        assert len(response.message) > 0
        
        assert response.error_reference_id is not None
        assert len(response.error_reference_id) > 0
        
        assert response.timestamp is not None
    
    @given(error_code=valid_error_codes, details=optional_details)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_error_reference_id_is_valid_uuid(self, error_code, details):
        """
        Property: For any error, error_reference_id is a valid UUID
        
        Feature: speech-pipeline-fix, Property 4: Error Response Structure Completeness
        **Validates: Requirements 2.1, 2.5**
        """
        response = SpeechAnalysisErrorResponse.from_code(error_code, details)
        
        # Verify error_reference_id is a valid UUID
        try:
            parsed_uuid = uuid.UUID(response.error_reference_id)
            assert str(parsed_uuid) == response.error_reference_id
        except ValueError:
            pytest.fail(f"error_reference_id is not a valid UUID: {response.error_reference_id}")
    
    @given(error_code=valid_error_codes, details=optional_details)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_timestamp_is_valid_datetime(self, error_code, details):
        """
        Property: For any error, timestamp is a valid datetime
        
        Feature: speech-pipeline-fix, Property 4: Error Response Structure Completeness
        **Validates: Requirements 2.1, 2.5**
        """
        response = SpeechAnalysisErrorResponse.from_code(error_code, details)
        
        # Verify timestamp is a valid datetime
        assert isinstance(response.timestamp, datetime)
        
        # Verify timestamp can be serialized to ISO format
        iso_string = response.timestamp.isoformat()
        assert len(iso_string) > 0
    
    @given(error_code=valid_error_codes, details=optional_details)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_error_code_follows_sp_format(self, error_code, details):
        """
        Property: For any error, error_code follows SP### format
        
        Feature: speech-pipeline-fix, Property 4: Error Response Structure Completeness
        **Validates: Requirements 2.1, 2.5**
        """
        response = SpeechAnalysisErrorResponse.from_code(error_code, details)
        
        # Verify error code format (SP followed by digits)
        assert response.error_code.startswith("SP")
        assert len(response.error_code) >= 3
        
        # The part after SP should be numeric
        code_number = response.error_code[2:]
        assert code_number.isdigit()
    
    @given(error_code=valid_error_codes)
    @settings(max_examples=100)
    def test_success_is_always_false_for_errors(self, error_code):
        """
        Property: For any error response, success field is always False
        
        Feature: speech-pipeline-fix, Property 4: Error Response Structure Completeness
        **Validates: Requirements 2.1, 2.5**
        """
        response = SpeechAnalysisErrorResponse.from_code(error_code)
        assert response.success is False
    
    @given(error_code=valid_error_codes, details=optional_details)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_error_reference_ids_are_unique(self, error_code, details):
        """
        Property: Each error response has a unique error_reference_id
        
        Feature: speech-pipeline-fix, Property 4: Error Response Structure Completeness
        **Validates: Requirements 2.1, 2.5**
        """
        response1 = SpeechAnalysisErrorResponse.from_code(error_code, details)
        response2 = SpeechAnalysisErrorResponse.from_code(error_code, details)
        
        # Each response should have a unique reference ID
        assert response1.error_reference_id != response2.error_reference_id
    
    @given(error_code=valid_error_codes)
    @settings(max_examples=100)
    def test_error_code_has_http_status_mapping(self, error_code):
        """
        Property: Every error code has a corresponding HTTP status code
        
        Feature: speech-pipeline-fix, Property 4: Error Response Structure Completeness
        **Validates: Requirements 2.1, 2.5**
        """
        assert error_code in ERROR_CODE_TO_HTTP_STATUS
        
        http_status = ERROR_CODE_TO_HTTP_STATUS[error_code]
        assert isinstance(http_status, int)
        assert 400 <= http_status < 600  # Valid HTTP error status codes
    
    @given(error_code=valid_error_codes)
    @settings(max_examples=100)
    def test_error_code_has_message_mapping(self, error_code):
        """
        Property: Every error code has a corresponding user-friendly message
        
        Feature: speech-pipeline-fix, Property 4: Error Response Structure Completeness
        **Validates: Requirements 2.1, 2.5**
        """
        assert error_code in ERROR_CODE_TO_MESSAGE
        
        message = ERROR_CODE_TO_MESSAGE[error_code]
        assert isinstance(message, str)
        assert len(message) > 0


class TestSpeechAnalysisErrorDataclass:
    """
    Tests for the SpeechAnalysisError dataclass
    """
    
    @given(error_code=valid_error_codes, details=optional_details)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_dataclass_creates_valid_error(self, error_code, details):
        """
        Property: SpeechAnalysisError.from_code creates valid error instances
        
        Feature: speech-pipeline-fix, Property 4: Error Response Structure Completeness
        **Validates: Requirements 2.1, 2.5**
        """
        error = SpeechAnalysisError.from_code(error_code, details)
        
        assert error.error_code == error_code
        assert error.message is not None
        assert len(error.message) > 0
        assert error.error_reference_id is not None
        assert error.timestamp is not None
    
    @given(error_code=valid_error_codes)
    @settings(max_examples=100)
    def test_dataclass_get_http_status(self, error_code):
        """
        Property: SpeechAnalysisError.get_http_status returns valid HTTP status
        
        Feature: speech-pipeline-fix, Property 4: Error Response Structure Completeness
        **Validates: Requirements 2.1, 2.5**
        """
        error = SpeechAnalysisError.from_code(error_code)
        http_status = error.get_http_status()
        
        assert isinstance(http_status, int)
        assert 400 <= http_status < 600
    
    @given(
        error_code=valid_error_codes,
        custom_message=st.text(min_size=1, max_size=200)
    )
    @settings(max_examples=100)
    def test_custom_message_overrides_default(self, error_code, custom_message):
        """
        Property: Custom message overrides default message
        
        Feature: speech-pipeline-fix, Property 4: Error Response Structure Completeness
        **Validates: Requirements 2.1, 2.5**
        """
        error = SpeechAnalysisError.from_code(error_code, custom_message=custom_message)
        assert error.message == custom_message


class TestErrorCategorization:
    """
    Tests for error categorization methods
    """
    
    # Validation error codes (SP001-SP099)
    validation_codes = st.sampled_from([
        SpeechErrorCodes.INVALID_FORMAT,
        SpeechErrorCodes.DURATION_TOO_SHORT,
        SpeechErrorCodes.DURATION_TOO_LONG,
        SpeechErrorCodes.INVALID_SAMPLE_RATE,
        SpeechErrorCodes.FILE_TOO_LARGE,
        SpeechErrorCodes.EMPTY_AUDIO,
        SpeechErrorCodes.CORRUPTED_FILE,
    ])
    
    # Processing error codes (SP100-SP199)
    processing_codes = st.sampled_from([
        SpeechErrorCodes.FEATURE_EXTRACTION_FAILED,
        SpeechErrorCodes.BIOMARKER_CALCULATION_FAILED,
        SpeechErrorCodes.MODEL_INFERENCE_FAILED,
        SpeechErrorCodes.AUDIO_LOADING_FAILED,
        SpeechErrorCodes.RESAMPLING_FAILED,
    ])
    
    # Timeout error codes (SP200-SP299)
    timeout_codes = st.sampled_from([
        SpeechErrorCodes.PROCESSING_TIMEOUT,
        SpeechErrorCodes.UPLOAD_TIMEOUT,
    ])
    
    @given(error_code=validation_codes)
    @settings(max_examples=100)
    def test_validation_errors_are_categorized_correctly(self, error_code):
        """
        Property: Validation errors (SP001-SP099) are correctly categorized
        
        Feature: speech-pipeline-fix, Property 4: Error Response Structure Completeness
        **Validates: Requirements 2.1, 2.5**
        """
        response = SpeechAnalysisErrorResponse.from_code(error_code)
        assert response.is_validation_error() is True
        assert response.is_processing_error() is False
        assert response.is_timeout_error() is False
    
    @given(error_code=processing_codes)
    @settings(max_examples=100)
    def test_processing_errors_are_categorized_correctly(self, error_code):
        """
        Property: Processing errors (SP100-SP199) are correctly categorized
        
        Feature: speech-pipeline-fix, Property 4: Error Response Structure Completeness
        **Validates: Requirements 2.1, 2.5**
        """
        response = SpeechAnalysisErrorResponse.from_code(error_code)
        assert response.is_validation_error() is False
        assert response.is_processing_error() is True
        assert response.is_timeout_error() is False
    
    @given(error_code=timeout_codes)
    @settings(max_examples=100)
    def test_timeout_errors_are_categorized_correctly(self, error_code):
        """
        Property: Timeout errors (SP200-SP299) are correctly categorized
        
        Feature: speech-pipeline-fix, Property 4: Error Response Structure Completeness
        **Validates: Requirements 2.1, 2.5**
        """
        response = SpeechAnalysisErrorResponse.from_code(error_code)
        assert response.is_validation_error() is False
        assert response.is_processing_error() is False
        assert response.is_timeout_error() is True

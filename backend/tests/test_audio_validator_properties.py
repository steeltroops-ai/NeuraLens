"""
Property-based tests for AudioValidator
Uses hypothesis for property-based testing.

Feature: speech-pipeline-fix
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

from app.services.audio_validator import AudioValidator


class TestMimeTypeValidationProperty:
    """
    Property 1: MIME Type Validation Correctness
    
    For any audio file upload, the Audio_Validator SHALL accept the file 
    if and only if its MIME type is in the set of supported types, and 
    SHALL return HTTP 400 with supported formats list for rejected files.
    
    **Validates: Requirements 1.1, 1.2**
    """
    
    # Valid MIME types that should be accepted
    VALID_MIME_TYPES = [
        'audio/wav', 'audio/wave', 'audio/x-wav',
        'audio/mpeg', 'audio/mp3',
        'audio/mp4', 'audio/x-m4a', 'audio/aac', 'audio/m4a',
        'audio/webm',
        'audio/ogg', 'audio/vorbis'
    ]
    
    # Invalid MIME types that should be rejected
    INVALID_MIME_TYPES = [
        'video/mp4', 'video/webm', 'video/avi',
        'image/png', 'image/jpeg', 'image/gif',
        'text/plain', 'text/html',
        'application/json', 'application/pdf', 'application/octet-stream',
        'audio/midi', 'audio/x-midi',  # Unsupported audio formats
    ]
    
    @given(mime_type=st.sampled_from(VALID_MIME_TYPES))
    @settings(max_examples=100)
    def test_valid_mime_types_are_accepted(self, mime_type):
        """
        Property: For any valid MIME type, is_supported_mime_type returns True
        
        Feature: speech-pipeline-fix, Property 1: MIME Type Validation Correctness
        **Validates: Requirements 1.1, 1.2**
        """
        validator = AudioValidator()
        assert validator.is_supported_mime_type(mime_type) is True
    
    @given(mime_type=st.sampled_from(INVALID_MIME_TYPES))
    @settings(max_examples=100)
    def test_invalid_mime_types_are_rejected(self, mime_type):
        """
        Property: For any invalid MIME type, is_supported_mime_type returns False
        
        Feature: speech-pipeline-fix, Property 1: MIME Type Validation Correctness
        **Validates: Requirements 1.1, 1.2**
        """
        validator = AudioValidator()
        assert validator.is_supported_mime_type(mime_type) is False
    
    @given(mime_type=st.sampled_from(VALID_MIME_TYPES + INVALID_MIME_TYPES))
    @settings(max_examples=100)
    def test_mime_type_validation_is_deterministic(self, mime_type):
        """
        Property: For any MIME type, validation result is consistent across calls
        
        Feature: speech-pipeline-fix, Property 1: MIME Type Validation Correctness
        **Validates: Requirements 1.1, 1.2**
        """
        validator = AudioValidator()
        result1 = validator.is_supported_mime_type(mime_type)
        result2 = validator.is_supported_mime_type(mime_type)
        assert result1 == result2
    
    @given(mime_type=st.sampled_from(VALID_MIME_TYPES + INVALID_MIME_TYPES))
    @settings(max_examples=100)
    def test_mime_type_acceptance_iff_in_supported_set(self, mime_type):
        """
        Property: A MIME type is accepted if and only if it's in SUPPORTED_MIME_TYPES
        
        Feature: speech-pipeline-fix, Property 1: MIME Type Validation Correctness
        **Validates: Requirements 1.1, 1.2**
        """
        validator = AudioValidator()
        is_accepted = validator.is_supported_mime_type(mime_type)
        is_in_supported = mime_type in AudioValidator.SUPPORTED_MIME_TYPES
        assert is_accepted == is_in_supported
    
    @given(mime_type=st.sampled_from(INVALID_MIME_TYPES))
    @settings(max_examples=100)
    def test_rejected_mime_types_produce_error_with_supported_formats(self, mime_type):
        """
        Property: For any rejected MIME type, error message lists supported formats
        
        Feature: speech-pipeline-fix, Property 1: MIME Type Validation Correctness
        **Validates: Requirements 1.1, 1.2**
        """
        validator = AudioValidator()
        # Create minimal bytes that won't match any magic bytes
        dummy_bytes = b'\x00' * 20
        
        is_valid, detected_type, error_message = validator.validate_mime_type(
            mime_type, dummy_bytes
        )
        
        assert is_valid is False
        assert error_message is not None
        # Error message should mention supported formats
        assert 'WAV' in error_message or 'MP3' in error_message or 'Supported' in error_message


class TestDurationValidationProperty:
    """
    Property 2: Duration Validation Range
    
    For any audio file with duration D seconds, the Audio_Validator SHALL 
    accept the file if and only if 3 ≤ D ≤ 60, returning appropriate error 
    messages for out-of-range durations.
    
    **Validates: Requirements 1.3, 1.4, 1.5**
    """
    
    SAMPLE_RATE = 16000  # Standard sample rate
    MIN_DURATION = 3.0
    MAX_DURATION = 60.0
    
    @given(duration=st.floats(min_value=3.0, max_value=60.0, allow_nan=False))
    @settings(max_examples=100)
    def test_valid_durations_are_accepted(self, duration):
        """
        Property: For any duration in [3, 60] seconds, validation succeeds
        
        Feature: speech-pipeline-fix, Property 2: Duration Validation Range
        **Validates: Requirements 1.3, 1.4, 1.5**
        """
        import numpy as np
        validator = AudioValidator()
        
        # Create audio data with the specified duration
        num_samples = int(duration * self.SAMPLE_RATE)
        audio_data = np.zeros(num_samples, dtype=np.float32)
        
        is_valid, actual_duration, error_message = validator.validate_duration(
            audio_data, self.SAMPLE_RATE
        )
        
        assert is_valid is True
        assert error_message is None
        assert abs(actual_duration - duration) < 0.001  # Allow small floating point error
    
    @given(duration=st.floats(min_value=0.0, max_value=2.99, allow_nan=False))
    @settings(max_examples=100)
    def test_too_short_durations_are_rejected(self, duration):
        """
        Property: For any duration < 3 seconds, validation fails with appropriate message
        
        Feature: speech-pipeline-fix, Property 2: Duration Validation Range
        **Validates: Requirements 1.3, 1.4, 1.5**
        """
        import numpy as np
        validator = AudioValidator()
        
        # Create audio data with the specified duration
        num_samples = max(1, int(duration * self.SAMPLE_RATE))
        audio_data = np.zeros(num_samples, dtype=np.float32)
        
        is_valid, actual_duration, error_message = validator.validate_duration(
            audio_data, self.SAMPLE_RATE
        )
        
        assert is_valid is False
        assert error_message is not None
        assert 'too short' in error_message.lower() or 'minimum' in error_message.lower()
    
    @given(duration=st.floats(min_value=60.01, max_value=300.0, allow_nan=False))
    @settings(max_examples=100)
    def test_too_long_durations_are_rejected(self, duration):
        """
        Property: For any duration > 60 seconds, validation fails with appropriate message
        
        Feature: speech-pipeline-fix, Property 2: Duration Validation Range
        **Validates: Requirements 1.3, 1.4, 1.5**
        """
        import numpy as np
        validator = AudioValidator()
        
        # Create audio data with the specified duration
        num_samples = int(duration * self.SAMPLE_RATE)
        audio_data = np.zeros(num_samples, dtype=np.float32)
        
        is_valid, actual_duration, error_message = validator.validate_duration(
            audio_data, self.SAMPLE_RATE
        )
        
        assert is_valid is False
        assert error_message is not None
        assert 'too long' in error_message.lower() or 'maximum' in error_message.lower()
    
    @given(duration=st.floats(min_value=0.0, max_value=120.0, allow_nan=False))
    @settings(max_examples=100)
    def test_duration_acceptance_iff_in_valid_range(self, duration):
        """
        Property: Duration is accepted if and only if 3 <= duration <= 60
        
        Feature: speech-pipeline-fix, Property 2: Duration Validation Range
        **Validates: Requirements 1.3, 1.4, 1.5**
        """
        import numpy as np
        validator = AudioValidator()
        
        # Create audio data with the specified duration
        num_samples = max(1, int(duration * self.SAMPLE_RATE))
        audio_data = np.zeros(num_samples, dtype=np.float32)
        
        is_valid, actual_duration, error_message = validator.validate_duration(
            audio_data, self.SAMPLE_RATE
        )
        
        expected_valid = self.MIN_DURATION <= duration <= self.MAX_DURATION
        assert is_valid == expected_valid
    
    @given(
        duration=st.floats(min_value=0.0, max_value=120.0, allow_nan=False),
        sample_rate=st.sampled_from([8000, 16000, 22050, 44100, 48000])
    )
    @settings(max_examples=100)
    def test_duration_validation_works_with_different_sample_rates(
        self, duration, sample_rate
    ):
        """
        Property: Duration validation works correctly regardless of sample rate
        
        Feature: speech-pipeline-fix, Property 2: Duration Validation Range
        **Validates: Requirements 1.3, 1.4, 1.5**
        """
        import numpy as np
        validator = AudioValidator()
        
        # Create audio data with the specified duration at given sample rate
        num_samples = max(1, int(duration * sample_rate))
        audio_data = np.zeros(num_samples, dtype=np.float32)
        
        is_valid, actual_duration, error_message = validator.validate_duration(
            audio_data, sample_rate
        )
        
        expected_valid = self.MIN_DURATION <= duration <= self.MAX_DURATION
        assert is_valid == expected_valid


class TestSampleRateNormalizationProperty:
    """
    Property 3: Sample Rate Normalization
    
    For any valid audio file with sample rate S, after processing by 
    Audio_Validator, the output audio data SHALL have sample rate exactly 16000 Hz.
    
    **Validates: Requirements 1.6, 1.7**
    """
    
    TARGET_SAMPLE_RATE = 16000
    SUPPORTED_SAMPLE_RATES = [8000, 16000, 22050, 44100, 48000]
    
    @given(
        original_sr=st.sampled_from(SUPPORTED_SAMPLE_RATES),
        duration=st.floats(min_value=0.1, max_value=1.0, allow_nan=False)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_output_sample_rate_is_always_16khz(self, original_sr, duration):
        """
        Property: For any input sample rate, output is always 16kHz
        
        Feature: speech-pipeline-fix, Property 3: Sample Rate Normalization
        **Validates: Requirements 1.6, 1.7**
        """
        import numpy as np
        validator = AudioValidator()
        
        # Create audio data at original sample rate
        num_samples = int(duration * original_sr)
        audio_data = np.random.randn(num_samples).astype(np.float32)
        
        resampled_audio, output_sr, was_resampled = validator.validate_and_resample(
            audio_data, original_sr
        )
        
        assert output_sr == self.TARGET_SAMPLE_RATE
    
    @given(duration=st.floats(min_value=0.1, max_value=1.0, allow_nan=False))
    @settings(max_examples=100)
    def test_16khz_input_is_not_resampled(self, duration):
        """
        Property: Audio already at 16kHz is not resampled
        
        Feature: speech-pipeline-fix, Property 3: Sample Rate Normalization
        **Validates: Requirements 1.6, 1.7**
        """
        import numpy as np
        validator = AudioValidator()
        
        # Create audio data at target sample rate
        num_samples = int(duration * self.TARGET_SAMPLE_RATE)
        audio_data = np.random.randn(num_samples).astype(np.float32)
        
        resampled_audio, output_sr, was_resampled = validator.validate_and_resample(
            audio_data, self.TARGET_SAMPLE_RATE
        )
        
        assert was_resampled is False
        assert output_sr == self.TARGET_SAMPLE_RATE
        assert np.array_equal(audio_data, resampled_audio)
    
    @given(
        original_sr=st.sampled_from([8000, 22050, 44100, 48000]),  # Exclude 16kHz
        duration=st.floats(min_value=0.1, max_value=1.0, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_non_16khz_input_is_resampled(self, original_sr, duration):
        """
        Property: Audio not at 16kHz is resampled
        
        Feature: speech-pipeline-fix, Property 3: Sample Rate Normalization
        **Validates: Requirements 1.6, 1.7**
        """
        import numpy as np
        validator = AudioValidator()
        
        # Create audio data at non-target sample rate
        num_samples = int(duration * original_sr)
        audio_data = np.random.randn(num_samples).astype(np.float32)
        
        resampled_audio, output_sr, was_resampled = validator.validate_and_resample(
            audio_data, original_sr
        )
        
        assert was_resampled is True
        assert output_sr == self.TARGET_SAMPLE_RATE
    
    @given(
        original_sr=st.sampled_from(SUPPORTED_SAMPLE_RATES),
        duration=st.floats(min_value=0.5, max_value=2.0, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_resampled_duration_is_preserved(self, original_sr, duration):
        """
        Property: Resampling preserves audio duration
        
        Feature: speech-pipeline-fix, Property 3: Sample Rate Normalization
        **Validates: Requirements 1.6, 1.7**
        """
        import numpy as np
        validator = AudioValidator()
        
        # Create audio data at original sample rate
        num_samples = int(duration * original_sr)
        audio_data = np.random.randn(num_samples).astype(np.float32)
        
        original_duration = len(audio_data) / original_sr
        
        resampled_audio, output_sr, was_resampled = validator.validate_and_resample(
            audio_data, original_sr
        )
        
        resampled_duration = len(resampled_audio) / output_sr
        
        # Duration should be preserved within small tolerance
        assert abs(original_duration - resampled_duration) < 0.01

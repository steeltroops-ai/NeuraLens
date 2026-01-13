"""
Audio Validator Service for Speech Analysis Pipeline
Validates audio files for format, duration, and sample rate.
"""

import io
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Set
import numpy as np

logger = logging.getLogger(__name__)


# Magic bytes for audio format detection
AUDIO_MAGIC_BYTES = {
    # WAV: RIFF....WAVE
    b'RIFF': 'audio/wav',
    # MP3: ID3 tag or frame sync
    b'ID3': 'audio/mpeg',
    b'\xff\xfb': 'audio/mpeg',
    b'\xff\xfa': 'audio/mpeg',
    b'\xff\xf3': 'audio/mpeg',
    b'\xff\xf2': 'audio/mpeg',
    # OGG: OggS
    b'OggS': 'audio/ogg',
    # FLAC: fLaC
    b'fLaC': 'audio/flac',
}

# M4A/AAC detection requires checking for ftyp box
M4A_FTYP_BRANDS = {b'M4A ', b'mp42', b'isom', b'mp41', b'M4B '}


@dataclass
class ValidationResult:
    """Result of audio validation"""
    is_valid: bool
    audio_data: Optional[np.ndarray] = None
    sample_rate: Optional[int] = None
    duration: Optional[float] = None
    original_format: Optional[str] = None
    resampled: bool = False
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class AudioValidator:
    """Validates audio files for format, duration, and sample rate"""
    
    SUPPORTED_MIME_TYPES: Set[str] = {
        'audio/wav', 'audio/wave', 'audio/x-wav',
        'audio/mpeg', 'audio/mp3',
        'audio/mp4', 'audio/x-m4a', 'audio/aac', 'audio/m4a',
        'audio/webm',
        'audio/ogg', 'audio/vorbis'
    }
    
    SUPPORTED_SAMPLE_RATES: Set[int] = {8000, 16000, 22050, 44100, 48000}
    TARGET_SAMPLE_RATE: int = 16000
    MIN_DURATION_SECONDS: float = 3.0
    MAX_DURATION_SECONDS: float = 60.0
    
    def __init__(self):
        self._librosa = None
    
    @property
    def librosa(self):
        """Lazy load librosa to avoid import overhead"""
        if self._librosa is None:
            import librosa
            self._librosa = librosa
        return self._librosa
    
    def detect_mime_type_from_bytes(self, audio_bytes: bytes) -> Optional[str]:
        """
        Detect MIME type using magic bytes detection.
        
        Args:
            audio_bytes: Raw audio file bytes
            
        Returns:
            Detected MIME type or None if unknown
        """
        if len(audio_bytes) < 12:
            return None
        
        # Check standard magic bytes
        for magic, mime_type in AUDIO_MAGIC_BYTES.items():
            if audio_bytes.startswith(magic):
                # Additional check for WAV to ensure it's actually WAVE format
                if magic == b'RIFF' and len(audio_bytes) >= 12:
                    if audio_bytes[8:12] != b'WAVE':
                        continue
                return mime_type
        
        # Check for M4A/AAC (ftyp box at offset 4)
        if len(audio_bytes) >= 12:
            if audio_bytes[4:8] == b'ftyp':
                brand = audio_bytes[8:12]
                if brand in M4A_FTYP_BRANDS:
                    return 'audio/mp4'
                # Generic MP4 container could be audio
                return 'audio/mp4'
        
        # Check for WebM (EBML header)
        if audio_bytes[:4] == b'\x1a\x45\xdf\xa3':
            return 'audio/webm'
        
        return None
    
    def validate_mime_type(
        self, 
        content_type: Optional[str], 
        audio_bytes: bytes
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate MIME type using both header and magic bytes.
        
        Args:
            content_type: MIME type from HTTP header
            audio_bytes: Raw audio file bytes
            
        Returns:
            Tuple of (is_valid, detected_mime_type, error_message)
        """
        # Detect from magic bytes
        detected_type = self.detect_mime_type_from_bytes(audio_bytes)
        
        # If we can detect from bytes, use that
        if detected_type:
            if detected_type in self.SUPPORTED_MIME_TYPES:
                return True, detected_type, None
            else:
                return False, detected_type, (
                    f"Unsupported audio format: {detected_type}. "
                    f"Supported formats: WAV, MP3, M4A, WebM, OGG"
                )
        
        # Fall back to content type header
        if content_type:
            # Normalize content type (remove parameters like charset)
            normalized_type = content_type.split(';')[0].strip().lower()
            if normalized_type in self.SUPPORTED_MIME_TYPES:
                return True, normalized_type, None
            else:
                return False, normalized_type, (
                    f"Unsupported audio format: {normalized_type}. "
                    f"Supported formats: WAV, MP3, M4A, WebM, OGG"
                )
        
        return False, None, (
            "Could not determine audio format. "
            "Supported formats: WAV, MP3, M4A, WebM, OGG"
        )
    
    def is_supported_mime_type(self, mime_type: str) -> bool:
        """
        Check if a MIME type is supported.
        
        Args:
            mime_type: MIME type string to check
            
        Returns:
            True if supported, False otherwise
        """
        if not mime_type:
            return False
        normalized = mime_type.split(';')[0].strip().lower()
        return normalized in self.SUPPORTED_MIME_TYPES

    def validate_duration(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Validate audio duration is within acceptable range.
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate in Hz
            
        Returns:
            Tuple of (is_valid, duration_seconds, error_message)
        """
        if sample_rate <= 0:
            return False, 0.0, "Invalid sample rate"
        
        duration = len(audio_data) / sample_rate
        
        if duration < self.MIN_DURATION_SECONDS:
            return False, duration, (
                f"Audio too short: {duration:.1f}s. "
                f"Minimum duration is {self.MIN_DURATION_SECONDS:.0f} seconds."
            )
        
        if duration > self.MAX_DURATION_SECONDS:
            return False, duration, (
                f"Audio too long: {duration:.1f}s. "
                f"Maximum duration is {self.MAX_DURATION_SECONDS:.0f} seconds."
            )
        
        return True, duration, None
    
    def validate_and_resample(
        self, 
        audio_data: np.ndarray, 
        original_sr: int
    ) -> Tuple[np.ndarray, int, bool]:
        """
        Validate sample rate and resample to target if needed.
        
        Args:
            audio_data: Audio samples as numpy array
            original_sr: Original sample rate in Hz
            
        Returns:
            Tuple of (resampled_audio, target_sample_rate, was_resampled)
        """
        if original_sr == self.TARGET_SAMPLE_RATE:
            return audio_data, self.TARGET_SAMPLE_RATE, False
        
        # Resample to target sample rate using librosa
        resampled = self.librosa.resample(
            audio_data, 
            orig_sr=original_sr, 
            target_sr=self.TARGET_SAMPLE_RATE
        )
        
        logger.info(
            f"Resampled audio from {original_sr}Hz to {self.TARGET_SAMPLE_RATE}Hz"
        )
        
        return resampled, self.TARGET_SAMPLE_RATE, True
    
    async def load_audio(
        self, 
        audio_bytes: bytes
    ) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
        """
        Load audio from bytes using librosa.
        
        Args:
            audio_bytes: Raw audio file bytes
            
        Returns:
            Tuple of (audio_data, sample_rate, error_message)
        """
        try:
            # Load audio using librosa (handles multiple formats)
            audio_data, sample_rate = self.librosa.load(
                io.BytesIO(audio_bytes),
                sr=None,  # Preserve original sample rate
                mono=True  # Convert to mono
            )
            return audio_data, sample_rate, None
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            return None, None, f"Failed to load audio file: {str(e)}"
    
    async def validate(
        self, 
        audio_bytes: bytes,
        content_type: Optional[str] = None
    ) -> ValidationResult:
        """
        Perform complete validation of audio file.
        
        Args:
            audio_bytes: Raw audio file bytes
            content_type: MIME type from HTTP header (optional)
            
        Returns:
            ValidationResult with validation status and processed audio
        """
        # Step 1: Validate MIME type
        mime_valid, detected_mime, mime_error = self.validate_mime_type(
            content_type, audio_bytes
        )
        
        if not mime_valid:
            return ValidationResult(
                is_valid=False,
                error_code='SP001',
                error_message=mime_error,
                original_format=detected_mime
            )
        
        # Step 2: Load audio
        audio_data, original_sr, load_error = await self.load_audio(audio_bytes)
        
        if load_error:
            return ValidationResult(
                is_valid=False,
                error_code='SP001',
                error_message=load_error,
                original_format=detected_mime
            )
        
        # Step 3: Validate duration
        duration_valid, duration, duration_error = self.validate_duration(
            audio_data, original_sr
        )
        
        if not duration_valid:
            error_code = 'SP002' if duration < self.MIN_DURATION_SECONDS else 'SP003'
            return ValidationResult(
                is_valid=False,
                error_code=error_code,
                error_message=duration_error,
                original_format=detected_mime,
                duration=duration,
                sample_rate=original_sr
            )
        
        # Step 4: Resample if needed
        resampled_audio, final_sr, was_resampled = self.validate_and_resample(
            audio_data, original_sr
        )
        
        # Recalculate duration after resampling
        final_duration = len(resampled_audio) / final_sr
        
        return ValidationResult(
            is_valid=True,
            audio_data=resampled_audio,
            sample_rate=final_sr,
            duration=final_duration,
            original_format=detected_mime,
            resampled=was_resampled
        )


# Singleton instance
audio_validator = AudioValidator()

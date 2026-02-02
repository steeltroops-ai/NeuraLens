"""
Format Validator v4.0
Multi-format audio validation with automatic conversion.

Supports: WAV, FLAC, MP3, M4A, WebM, OGG (Requirement 4.5)
"""

import os
import io
import tempfile
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, BinaryIO

import numpy as np
from pydub import AudioSegment

logger = logging.getLogger(__name__)


@dataclass
class FormatValidationResult:
    """Result of format validation."""
    is_valid: bool = False
    format_detected: str = ""
    mime_type: str = ""
    
    # Audio properties
    duration: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    bit_depth: int = 0
    
    # File properties
    file_size: int = 0
    codec: str = ""
    
    # Conversion info
    needs_conversion: bool = False
    conversion_applied: bool = False
    
    # Issues
    issues: List[str] = field(default_factory=list)
    
    # Converted audio (if applicable)
    converted_audio: Optional[np.ndarray] = None
    converted_sample_rate: int = 16000
    
    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "format_detected": self.format_detected,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "needs_conversion": self.needs_conversion,
            "issues": self.issues
        }


class FormatValidator:
    """
    Validates and converts audio file formats.
    
    Supports multiple formats and handles conversion to
    the target format for processing.
    """
    
    # Supported formats
    SUPPORTED_FORMATS = {
        "wav", "wave", "mp3", "m4a", "mp4", 
        "webm", "ogg", "opus", "flac"
    }
    
    SUPPORTED_MIME_TYPES = {
        "audio/wav", "audio/wave", "audio/x-wav",
        "audio/mpeg", "audio/mp3",
        "audio/webm", "audio/webm;codecs=opus",
        "audio/ogg", "audio/ogg;codecs=opus",
        "audio/mp4", "audio/m4a", "audio/x-m4a",
        "audio/flac", "audio/x-flac",
        "audio/opus", "video/webm"  # WebM can be video container
    }
    
    # Format magic bytes
    MAGIC_BYTES = {
        b'RIFF': 'wav',
        b'ID3': 'mp3',
        b'\xff\xfb': 'mp3',
        b'\xff\xfa': 'mp3',
        b'\xff\xf3': 'mp3',
        b'\xff\xf2': 'mp3',
        b'OggS': 'ogg',
        b'fLaC': 'flac',
        b'\x1aE\xdf\xa3': 'webm',  # EBML header
    }
    
    # Processing constraints
    TARGET_SAMPLE_RATE = 16000
    TARGET_CHANNELS = 1
    TARGET_BIT_DEPTH = 16
    
    MAX_FILE_SIZE_MB = 10
    MIN_DURATION_SEC = 3.0
    MAX_DURATION_SEC = 60.0
    
    def __init__(
        self,
        target_sample_rate: int = 16000,
        max_file_size_mb: float = 10.0,
        min_duration_sec: float = 3.0,
        max_duration_sec: float = 60.0
    ):
        self.target_sample_rate = target_sample_rate
        self.max_file_size_mb = max_file_size_mb
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
    
    def validate(
        self,
        audio_bytes: bytes,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
        convert: bool = True
    ) -> FormatValidationResult:
        """
        Validate audio format and optionally convert.
        
        Args:
            audio_bytes: Raw audio file bytes
            filename: Optional filename for format detection
            content_type: Optional MIME type
            convert: Whether to convert to target format
            
        Returns:
            FormatValidationResult with validation status and converted audio
        """
        result = FormatValidationResult()
        result.file_size = len(audio_bytes)
        
        # Check file size
        if not self._check_file_size(audio_bytes, result):
            return result
        
        # Detect format
        format_detected = self._detect_format(audio_bytes, filename, content_type)
        result.format_detected = format_detected
        result.mime_type = self._get_mime_type(format_detected)
        
        # Validate format is supported
        if not self._is_format_supported(format_detected, result):
            return result
        
        # Load and validate audio
        try:
            audio_segment = self._load_audio(audio_bytes, format_detected, filename)
            
            if audio_segment is None:
                result.issues.append("Failed to decode audio file")
                return result
            
            # Extract properties
            result.duration = audio_segment.duration_seconds
            result.sample_rate = audio_segment.frame_rate
            result.channels = audio_segment.channels
            result.bit_depth = audio_segment.sample_width * 8
            
            # Validate duration
            if not self._check_duration(result):
                return result
            
            # Check if conversion needed
            result.needs_conversion = (
                result.sample_rate != self.target_sample_rate or
                result.channels != self.TARGET_CHANNELS
            )
            
            # Convert if requested
            if convert:
                result.converted_audio = self._convert_audio(audio_segment)
                result.converted_sample_rate = self.target_sample_rate
                result.conversion_applied = True
            
            result.is_valid = True
            
        except Exception as e:
            logger.error(f"Audio validation failed: {e}")
            result.issues.append(f"Audio processing error: {str(e)}")
        
        return result
    
    def _check_file_size(self, audio_bytes: bytes, result: FormatValidationResult) -> bool:
        """Check if file size is within limits."""
        size_mb = len(audio_bytes) / (1024 * 1024)
        
        if size_mb > self.max_file_size_mb:
            result.issues.append(
                f"File too large ({size_mb:.1f} MB > {self.max_file_size_mb} MB)"
            )
            return False
        
        if len(audio_bytes) < 1000:  # Less than 1KB
            result.issues.append("File too small to contain valid audio")
            return False
        
        return True
    
    def _detect_format(
        self,
        audio_bytes: bytes,
        filename: Optional[str],
        content_type: Optional[str]
    ) -> str:
        """Detect audio format using multiple methods."""
        # Method 1: Magic bytes
        format_from_magic = self._detect_from_magic(audio_bytes)
        if format_from_magic:
            return format_from_magic
        
        # Method 2: File extension
        if filename:
            ext = os.path.splitext(filename)[1].lower().lstrip('.')
            if ext in self.SUPPORTED_FORMATS:
                return ext
        
        # Method 3: MIME type
        if content_type:
            format_from_mime = self._detect_from_mime(content_type)
            if format_from_mime:
                return format_from_mime
        
        # Method 4: Check for M4A (ftyp header)
        if len(audio_bytes) >= 12:
            if audio_bytes[4:8] == b'ftyp':
                return 'm4a'
        
        return "unknown"
    
    def _detect_from_magic(self, audio_bytes: bytes) -> Optional[str]:
        """Detect format from magic bytes."""
        if len(audio_bytes) < 4:
            return None
        
        header = audio_bytes[:4]
        
        # Check exact matches
        for magic, fmt in self.MAGIC_BYTES.items():
            if header.startswith(magic):
                return fmt
        
        # Check 2-byte patterns for MP3
        if header[:2] == b'\xff\xfb' or header[:2] == b'\xff\xfa':
            return 'mp3'
        
        return None
    
    def _detect_from_mime(self, content_type: str) -> Optional[str]:
        """Detect format from MIME type."""
        content_type = content_type.lower().split(';')[0].strip()
        
        mime_to_format = {
            'audio/wav': 'wav',
            'audio/wave': 'wav',
            'audio/x-wav': 'wav',
            'audio/mpeg': 'mp3',
            'audio/mp3': 'mp3',
            'audio/mp4': 'm4a',
            'audio/m4a': 'm4a',
            'audio/x-m4a': 'm4a',
            'audio/webm': 'webm',
            'video/webm': 'webm',
            'audio/ogg': 'ogg',
            'audio/opus': 'opus',
            'audio/flac': 'flac',
            'audio/x-flac': 'flac',
        }
        
        return mime_to_format.get(content_type)
    
    def _is_format_supported(self, format_detected: str, result: FormatValidationResult) -> bool:
        """Check if format is supported."""
        if format_detected == "unknown":
            result.issues.append("Unable to detect audio format")
            return False
        
        if format_detected not in self.SUPPORTED_FORMATS:
            result.issues.append(f"Unsupported format: {format_detected}")
            return False
        
        return True
    
    def _get_mime_type(self, format_detected: str) -> str:
        """Get MIME type for format."""
        format_to_mime = {
            'wav': 'audio/wav',
            'wave': 'audio/wav',
            'mp3': 'audio/mpeg',
            'm4a': 'audio/mp4',
            'mp4': 'audio/mp4',
            'webm': 'audio/webm',
            'ogg': 'audio/ogg',
            'opus': 'audio/opus',
            'flac': 'audio/flac',
        }
        return format_to_mime.get(format_detected, 'application/octet-stream')
    
    def _load_audio(
        self,
        audio_bytes: bytes,
        format_detected: str,
        filename: Optional[str]
    ) -> Optional[AudioSegment]:
        """Load audio using pydub."""
        # Determine file extension
        if filename:
            suffix = os.path.splitext(filename)[1]
        else:
            suffix = f".{format_detected}" if format_detected != "unknown" else ".wav"
        
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        
        # Write to temp file for robust loading
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(audio_bytes)
                temp_path = f.name
            
            audio = AudioSegment.from_file(temp_path)
            return audio
            
        except Exception as e:
            logger.warning(f"Failed to load with detected format: {e}")
            
            # Try generic loading
            try:
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
                return audio
            except Exception as e2:
                logger.error(f"All audio loading methods failed: {e2}")
                return None
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    
    def _check_duration(self, result: FormatValidationResult) -> bool:
        """Check if duration is within limits."""
        if result.duration < self.min_duration_sec:
            result.issues.append(
                f"Audio too short ({result.duration:.1f}s < {self.min_duration_sec}s)"
            )
            return False
        
        if result.duration > self.max_duration_sec:
            result.issues.append(
                f"Audio too long ({result.duration:.1f}s > {self.max_duration_sec}s) - will be truncated"
            )
            # Note: We still return True, just issue a warning
        
        return True
    
    def _convert_audio(self, audio_segment: AudioSegment) -> np.ndarray:
        """Convert audio segment to numpy array with target parameters."""
        # Truncate if too long
        if audio_segment.duration_seconds > self.max_duration_sec:
            audio_segment = audio_segment[:int(self.max_duration_sec * 1000)]
        
        # Convert to mono
        if audio_segment.channels > 1:
            audio_segment = audio_segment.set_channels(1)
        
        # Resample to target rate
        if audio_segment.frame_rate != self.target_sample_rate:
            audio_segment = audio_segment.set_frame_rate(self.target_sample_rate)
        
        # Get samples as numpy array
        samples = np.array(audio_segment.get_array_of_samples())
        
        # Normalize to float32 [-1, 1]
        if audio_segment.sample_width == 2:  # 16-bit
            audio_array = samples.astype(np.float32) / 32768.0
        elif audio_segment.sample_width == 4:  # 32-bit
            audio_array = samples.astype(np.float32) / 2147483648.0
        elif audio_segment.sample_width == 1:  # 8-bit
            audio_array = (samples.astype(np.float32) - 128) / 128.0
        else:
            audio_array = samples.astype(np.float32) / (2 ** (8 * audio_segment.sample_width - 1))
        
        return audio_array
    
    @classmethod
    def is_supported_format(cls, filename: str) -> bool:
        """Quick check if filename has supported extension."""
        ext = os.path.splitext(filename)[1].lower().lstrip('.')
        return ext in cls.SUPPORTED_FORMATS
    
    @classmethod
    def is_supported_mime(cls, content_type: str) -> bool:
        """Quick check if MIME type is supported."""
        content_type = content_type.lower().split(';')[0].strip()
        return content_type in cls.SUPPORTED_MIME_TYPES

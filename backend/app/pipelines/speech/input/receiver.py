"""
Speech Analysis Pipeline - Input Receiver
Handles file reception and initial processing.

Errors from this module have prefix: E_INP_
"""

import logging
import tempfile
import os
from typing import Tuple, Optional
from dataclasses import dataclass

from ..errors.codes import ErrorCode, raise_input_error

logger = logging.getLogger(__name__)


@dataclass
class ReceivedAudio:
    """Represents received audio data."""
    audio_bytes: bytes
    filename: str
    content_type: Optional[str]
    temp_path: Optional[str] = None


class AudioReceiver:
    """Handles audio file reception and temporary storage."""
    
    def __init__(self):
        self.temp_files = []
    
    async def receive(
        self,
        audio_bytes: bytes,
        filename: str,
        content_type: Optional[str] = None
    ) -> ReceivedAudio:
        """
        Receive and validate audio input.
        
        Args:
            audio_bytes: Raw audio file contents
            filename: Original filename
            content_type: MIME type
            
        Returns:
            ReceivedAudio object with file data
        """
        if not audio_bytes:
            raise_input_error(
                "E_INP_001",
                ErrorCode.E_INP_001,
                {"filename": filename}
            )
        
        return ReceivedAudio(
            audio_bytes=audio_bytes,
            filename=filename,
            content_type=content_type
        )
    
    def save_to_temp(self, received: ReceivedAudio) -> str:
        """
        Save received audio to temporary file.
        
        Args:
            received: ReceivedAudio object
            
        Returns:
            Path to temporary file
        """
        suffix = self._get_extension(received.filename)
        
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(received.audio_bytes)
                temp_path = f.name
            
            self.temp_files.append(temp_path)
            return temp_path
            
        except Exception as e:
            logger.error(f"Failed to save audio to temp file: {e}")
            raise_input_error(
                "E_INP_008",
                f"Could not process audio file: {str(e)}",
                {"filename": received.filename}
            )
    
    def cleanup(self) -> None:
        """Clean up temporary files."""
        for temp_path in self.temp_files:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_path}: {e}")
        
        self.temp_files.clear()
    
    def _get_extension(self, filename: str) -> str:
        """Extract file extension from filename."""
        if not filename:
            return ".wav"
        
        if '.' in filename:
            return '.' + filename.split('.')[-1].lower()
        
        return ".wav"
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        self.cleanup()

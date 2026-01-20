"""
Radiology Input Receiver

Handles input reception and file parsing.
"""

import hashlib
from typing import Dict, Any, Optional, Tuple
from io import BytesIO
from dataclasses import dataclass
from PIL import Image

from ..config import RadiologyConfig


@dataclass
class ReceivedInput:
    """Received input data."""
    file_bytes: bytes
    filename: str
    content_type: Optional[str]
    file_hash: str
    file_size_mb: float
    modality_detected: str
    body_region: Optional[str]
    is_volumetric: bool


class InputReceiver:
    """
    Receive and parse input files.
    
    Handles:
    - File reception
    - Hash calculation
    - Modality detection
    - Basic format parsing
    """
    
    def receive(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: Optional[str] = None
    ) -> ReceivedInput:
        """
        Receive and process input file.
        
        Args:
            file_bytes: Raw file data
            filename: Original filename
            content_type: MIME type
        
        Returns:
            ReceivedInput with metadata
        """
        # Calculate hash
        file_hash = f"sha256:{hashlib.sha256(file_bytes).hexdigest()[:16]}"
        
        # Calculate size
        file_size_mb = len(file_bytes) / (1024 * 1024)
        
        # Detect modality
        modality, body_region = self._detect_modality(file_bytes, filename, content_type)
        
        return ReceivedInput(
            file_bytes=file_bytes,
            filename=filename,
            content_type=content_type,
            file_hash=file_hash,
            file_size_mb=file_size_mb,
            modality_detected=modality,
            body_region=body_region,
            is_volumetric=False  # Single images are not volumetric
        )
    
    def _detect_modality(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: Optional[str]
    ) -> Tuple[str, Optional[str]]:
        """
        Detect imaging modality from file.
        
        Returns:
            Tuple of (modality, body_region)
        """
        # For standard images, assume chest X-ray
        # In a production system, this would use more sophisticated detection
        ext = filename.lower().split(".")[-1] if "." in filename else ""
        
        if ext in ["jpg", "jpeg", "png"]:
            # Analyze image characteristics
            try:
                img = Image.open(BytesIO(file_bytes))
                width, height = img.size
                
                # Aspect ratio can hint at modality
                aspect = width / height if height > 0 else 1
                
                # Chest X-rays are typically square or slightly portrait
                if 0.8 <= aspect <= 1.2:
                    return "chest_xray", "chest"
                else:
                    return "chest_xray", "chest"  # Default to chest
                    
            except Exception:
                pass
        
        return "chest_xray", "chest"
    
    def create_receipt(self, received: ReceivedInput) -> Dict[str, Any]:
        """Create receipt confirmation."""
        return {
            "acknowledged": True,
            "modality_received": received.modality_detected,
            "body_region": received.body_region,
            "is_volumetric": received.is_volumetric,
            "file_hash": received.file_hash,
            "file_size_mb": round(received.file_size_mb, 2)
        }

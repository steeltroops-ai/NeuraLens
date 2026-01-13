"""
Validation service for NeuraLens backend
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ValidationService:
    """Data validation service"""
    
    def __init__(self):
        pass
    
    def validate_audio_data(self, audio_data: bytes) -> Dict[str, Any]:
        """Validate audio data"""
        if not audio_data:
            return {"valid": False, "error": "No audio data provided"}
        
        if len(audio_data) < 1000:  # Minimum size check
            return {"valid": False, "error": "Audio data too small"}
            
        return {"valid": True, "message": "Audio data is valid"}
    
    def validate_image_data(self, image_data: bytes) -> Dict[str, Any]:
        """Validate image data"""
        if not image_data:
            return {"valid": False, "error": "No image data provided"}
        
        if len(image_data) < 1000:  # Minimum size check
            return {"valid": False, "error": "Image data too small"}
            
        return {"valid": True, "message": "Image data is valid"}
    
    def validate_assessment_data(self, assessment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate assessment data"""
        required_fields = ["session_id", "modality"]
        
        for field in required_fields:
            if field not in assessment_data:
                return {"valid": False, "error": f"Missing required field: {field}"}
        
        return {"valid": True, "message": "Assessment data is valid"}
"""
Radiology Input Validator

Validates all incoming image data before processing.
Errors from this module have prefix: E_INP_
"""

import numpy as np
from PIL import Image
from io import BytesIO
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from ..config import RadiologyConfig
from ..errors.codes import ErrorCode


class ValidationSeverity(Enum):
    """Validation result severity."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationCheck:
    """Result of a single validation check."""
    check_id: str
    passed: bool
    severity: ValidationSeverity
    message: str
    details: Optional[Dict] = None


@dataclass
class ValidationResult:
    """Complete validation report."""
    is_valid: bool
    errors: List[ValidationCheck] = field(default_factory=list)
    warnings: List[ValidationCheck] = field(default_factory=list)
    info: List[ValidationCheck] = field(default_factory=list)
    quality_score: float = 1.0
    image_info: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "errors": [
                {"check": e.check_id, "message": e.message}
                for e in self.errors
            ],
            "warnings": [
                {"check": w.check_id, "message": w.message}
                for w in self.warnings
            ],
            "quality_score": self.quality_score,
            "image_info": self.image_info
        }


class ImageValidator:
    """
    Validate standard image inputs (PNG/JPEG).
    
    Validates:
    - File type and extension
    - File size
    - Image resolution
    - Image content (not blank)
    - Medical content detection
    """
    
    ALLOWED_TYPES = {".jpg", ".jpeg", ".png"}
    MIN_RESOLUTION = RadiologyConfig.MIN_RESOLUTION
    MAX_RESOLUTION = RadiologyConfig.MAX_RESOLUTION
    MAX_FILE_SIZE_MB = RadiologyConfig.MAX_FILE_SIZE_MB
    MIN_INTENSITY_VARIANCE = RadiologyConfig.MIN_INTENSITY_VARIANCE
    
    def validate(
        self,
        file_path: str,
        file_bytes: bytes
    ) -> ValidationResult:
        """
        Validate input image data.
        
        Args:
            file_path: Original filename
            file_bytes: Raw image bytes
        
        Returns:
            ValidationResult with all checks
        """
        errors = []
        warnings = []
        info = []
        image_info = {}
        
        # Check file type
        ext = "." + file_path.lower().split(".")[-1] if "." in file_path else ""
        if ext not in self.ALLOWED_TYPES:
            errors.append(ValidationCheck(
                check_id=ErrorCode.E_GEN_002,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid file type: {ext}. Allowed: {self.ALLOWED_TYPES}"
            ))
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                info=info,
                image_info=image_info
            )
        
        # Check file size
        size_mb = len(file_bytes) / (1024 * 1024)
        image_info["file_size_mb"] = round(size_mb, 2)
        
        if size_mb > self.MAX_FILE_SIZE_MB:
            errors.append(ValidationCheck(
                check_id=ErrorCode.E_GEN_003,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"File size {size_mb:.1f}MB exceeds limit of {self.MAX_FILE_SIZE_MB}MB"
            ))
        
        # Decode and validate image
        try:
            img = Image.open(BytesIO(file_bytes))
            width, height = img.size
            image_info["width"] = width
            image_info["height"] = height
            image_info["resolution"] = f"{width}x{height}"
            image_info["mode"] = img.mode
            
            # Resolution check
            if width < self.MIN_RESOLUTION or height < self.MIN_RESOLUTION:
                errors.append(ValidationCheck(
                    check_id=ErrorCode.E_GEN_005,
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Resolution {width}x{height} below minimum {self.MIN_RESOLUTION}x{self.MIN_RESOLUTION}"
                ))
            elif width > self.MAX_RESOLUTION or height > self.MAX_RESOLUTION:
                errors.append(ValidationCheck(
                    check_id=ErrorCode.E_GEN_005,
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Resolution {width}x{height} exceeds maximum {self.MAX_RESOLUTION}x{self.MAX_RESOLUTION}"
                ))
            
            # Convert to grayscale for analysis
            img_gray = img.convert('L')
            img_array = np.array(img_gray)
            
            # Empty/blank image check
            variance = np.var(img_array)
            if variance < self.MIN_INTENSITY_VARIANCE:
                errors.append(ValidationCheck(
                    check_id=ErrorCode.E_GEN_006,
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message="Image appears to be blank or nearly uniform"
                ))
            
            # Medical content detection
            if not self._appears_medical(img_array):
                errors.append(ValidationCheck(
                    check_id=ErrorCode.E_GEN_007,
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message="Image does not appear to be a medical radiological image"
                ))
            
        except Exception as e:
            errors.append(ValidationCheck(
                check_id=ErrorCode.E_GEN_004,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Failed to decode image: {str(e)}"
            ))
        
        # Calculate quality score
        quality_score = 1.0 if len(errors) == 0 else 0.0
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            info=info,
            quality_score=quality_score,
            image_info=image_info
        )
    
    def _appears_medical(self, img_array: np.ndarray) -> bool:
        """
        Basic heuristic to check if image looks like medical imaging.
        
        Medical X-rays typically have:
        - Good dynamic range
        - Specific intensity distribution patterns
        """
        # Check intensity distribution
        hist, _ = np.histogram(img_array.flatten(), bins=256, range=(0, 255))
        
        low_intensity = np.sum(hist[:50])
        high_intensity = np.sum(hist[200:])
        total = np.sum(hist)
        
        # X-rays should have significant content in dark and light regions
        if (low_intensity / total > 0.1 or high_intensity / total > 0.1):
            return True
        
        # Accept if there's good dynamic range
        if np.ptp(img_array) > 100:
            return True
        
        return False
    
    def assess_quality(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Assess image quality for analysis.
        
        Returns quality metrics including:
        - Overall quality score
        - Resolution assessment
        - Contrast assessment
        - Issues detected
        """
        try:
            img = Image.open(BytesIO(image_bytes))
            img_array = np.array(img.convert('L'))
            
            h, w = img_array.shape
            
            # Resolution check
            resolution_ok = h >= RadiologyConfig.RECOMMENDED_RESOLUTION and \
                           w >= RadiologyConfig.RECOMMENDED_RESOLUTION
            
            # Contrast check (dynamic range)
            contrast = (np.max(img_array) - np.min(img_array)) / 255
            
            # Brightness check
            mean_brightness = np.mean(img_array) / 255
            
            # Overall quality score
            quality_score = (
                0.4 * (1 if resolution_ok else 0.6) +
                0.4 * contrast +
                0.2 * (1 - abs(mean_brightness - 0.5) * 2)
            )
            
            issues = []
            if not resolution_ok:
                issues.append("Low resolution - recommend higher quality image")
            if contrast < 0.5:
                issues.append("Low contrast - may affect analysis accuracy")
            if mean_brightness < 0.2 or mean_brightness > 0.8:
                issues.append("Suboptimal exposure")
            
            return {
                "quality": "good" if quality_score > 0.7 else ("adequate" if quality_score > 0.5 else "poor"),
                "quality_score": round(quality_score, 2),
                "resolution": f"{w}x{h}",
                "resolution_ok": resolution_ok,
                "contrast": round(contrast, 2),
                "brightness": round(mean_brightness, 2),
                "issues": issues,
                "usable": quality_score > 0.5
            }
            
        except Exception as e:
            return {
                "quality": "unknown",
                "quality_score": 0.5,
                "issues": [f"Quality assessment failed: {str(e)}"],
                "usable": True
            }

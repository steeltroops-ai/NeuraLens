"""
Dermatology Pipeline Input Validator

Validates skin lesion images for quality, content, and analysis readiness.
"""

import hashlib
import io
import logging
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
import cv2

from ..config import (
    ACCEPTED_MIME_TYPES,
    MAX_FILE_SIZE_BYTES,
    MIN_FILE_SIZE_BYTES,
    QUALITY_THRESHOLDS
)
from ..schemas import (
    ValidationCheck,
    QualityReport,
    ContentReport,
    ValidationResult
)
from ..errors import get_error_response

logger = logging.getLogger(__name__)


class FileValidator:
    """Validates file-level properties."""
    
    def validate(self, file_data: bytes, content_type: str = None) -> Tuple[bool, List[Dict]]:
        """Validate file properties."""
        errors = []
        
        # Check file size
        if len(file_data) < MIN_FILE_SIZE_BYTES:
            errors.append(get_error_response("E_VAL_003"))
        elif len(file_data) > MAX_FILE_SIZE_BYTES:
            errors.append(get_error_response("E_VAL_002"))
        
        # Check content type
        if content_type and content_type not in ACCEPTED_MIME_TYPES:
            errors.append(get_error_response("E_VAL_001"))
        
        # Try to open as image
        try:
            img = Image.open(io.BytesIO(file_data))
            img.verify()
        except Exception as e:
            errors.append(get_error_response("E_VAL_004"))
            return False, errors
        
        return len(errors) == 0, errors


class ImageQualityValidator:
    """Validates image quality for dermatological analysis."""
    
    def __init__(self):
        self.thresholds = QUALITY_THRESHOLDS
    
    def validate(self, image: np.ndarray) -> QualityReport:
        """Validate all quality aspects."""
        resolution = self._check_resolution(image)
        focus = self._check_focus(image)
        illumination = self._check_illumination(image)
        color_balance = self._check_color_balance(image)
        
        # Calculate overall quality score
        scores = [
            resolution.score or 0,
            focus.score or 0,
            illumination.score or 0,
            color_balance.score or 0
        ]
        overall = sum(scores) / len(scores) if scores else 0
        
        return QualityReport(
            resolution=resolution,
            focus=focus,
            illumination=illumination,
            color_balance=color_balance,
            overall_quality=overall
        )
    
    def _check_resolution(self, image: np.ndarray) -> ValidationCheck:
        """Check image resolution."""
        height, width = image.shape[:2]
        megapixels = (height * width) / 1_000_000
        
        if width < self.thresholds.min_width or height < self.thresholds.min_height:
            return ValidationCheck(
                name="resolution",
                passed=False,
                score=megapixels / self.thresholds.optimal_resolution_mp,
                message=f"Resolution too low: {width}x{height}"
            )
        
        if megapixels < self.thresholds.min_resolution_mp:
            return ValidationCheck(
                name="resolution",
                passed=False,
                score=megapixels / self.thresholds.optimal_resolution_mp,
                message=f"Resolution {megapixels:.1f}MP below minimum {self.thresholds.min_resolution_mp}MP"
            )
        
        score = min(megapixels / self.thresholds.optimal_resolution_mp, 1.0)
        warning = None
        if megapixels < self.thresholds.optimal_resolution_mp:
            warning = f"Resolution {megapixels:.1f}MP - optimal is {self.thresholds.optimal_resolution_mp}MP"
        
        return ValidationCheck(
            name="resolution",
            passed=True,
            score=score,
            warning=warning
        )
    
    def _check_focus(self, image: np.ndarray) -> ValidationCheck:
        """Check image sharpness using Laplacian variance."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < self.thresholds.min_focus_score:
            return ValidationCheck(
                name="focus",
                passed=False,
                score=laplacian_var / self.thresholds.optimal_focus_score,
                message=f"Image too blurry (sharpness: {laplacian_var:.1f})"
            )
        
        score = min(laplacian_var / self.thresholds.optimal_focus_score, 1.0)
        warning = None
        if laplacian_var < self.thresholds.optimal_focus_score * 0.5:
            warning = "Moderate blur detected"
        
        return ValidationCheck(
            name="focus",
            passed=True,
            score=score,
            warning=warning
        )
    
    def _check_illumination(self, image: np.ndarray) -> ValidationCheck:
        """Check brightness and exposure."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        value = hsv[:, :, 2]
        
        mean_brightness = np.mean(value)
        
        # Check overexposure
        overexposed = np.sum(value > 250) / value.size
        underexposed = np.sum(value < 10) / value.size
        
        issues = []
        if overexposed > self.thresholds.max_overexposed_ratio:
            issues.append("overexposed")
        if underexposed > self.thresholds.max_underexposed_ratio:
            issues.append("underexposed")
        if mean_brightness < self.thresholds.min_brightness:
            issues.append("too_dark")
        if mean_brightness > self.thresholds.max_brightness:
            issues.append("too_bright")
        
        if issues:
            return ValidationCheck(
                name="illumination",
                passed=False,
                score=0.3,
                message=f"Illumination issues: {', '.join(issues)}"
            )
        
        # Calculate uniformity
        std_brightness = np.std(value)
        uniformity = 1.0 - min(std_brightness / 80, 1.0)
        
        warning = None
        if uniformity < self.thresholds.min_uniformity:
            warning = "Uneven lighting detected"
        
        score = uniformity
        return ValidationCheck(
            name="illumination",
            passed=True,
            score=score,
            warning=warning
        )
    
    def _check_color_balance(self, image: np.ndarray) -> ValidationCheck:
        """Check for color cast."""
        r_mean = np.mean(image[:, :, 0])
        g_mean = np.mean(image[:, :, 1])
        b_mean = np.mean(image[:, :, 2])
        
        max_diff = max(
            abs(r_mean - g_mean),
            abs(g_mean - b_mean),
            abs(r_mean - b_mean)
        )
        
        if max_diff > self.thresholds.max_color_cast:
            return ValidationCheck(
                name="color_balance",
                passed=False,
                score=0.3,
                message="Severe color cast detected"
            )
        
        score = 1.0 - (max_diff / self.thresholds.max_color_cast)
        warning = None
        if max_diff > self.thresholds.warning_color_cast:
            warning = "Moderate color cast - will attempt correction"
        
        return ValidationCheck(
            name="color_balance",
            passed=True,
            score=score,
            warning=warning
        )


class ContentValidator:
    """Validates image content for skin lesion presence."""
    
    def __init__(self):
        self.thresholds = QUALITY_THRESHOLDS
    
    def validate(self, image: np.ndarray) -> ContentReport:
        """Validate image content."""
        # Detect skin
        skin_ratio = self._detect_skin(image)
        is_skin = skin_ratio >= self.thresholds.min_skin_ratio
        
        # Detect lesion
        lesion_detected, lesion_centered = self._detect_lesion(image)
        
        # Detect occlusions
        occlusions = self._detect_occlusions(image)
        
        warnings = []
        if not is_skin:
            warnings.append("No skin detected in image")
        if not lesion_detected:
            warnings.append("No distinct lesion detected")
        if not lesion_centered:
            warnings.append("Lesion not centered in frame")
        if occlusions:
            warnings.append(f"Occlusions detected: {', '.join([o['type'] for o in occlusions])}")
        
        return ContentReport(
            is_skin_image=is_skin,
            skin_ratio=skin_ratio,
            lesion_detected=lesion_detected,
            lesion_centered=lesion_centered,
            occlusions=occlusions,
            warnings=warnings
        )
    
    def _detect_skin(self, image: np.ndarray) -> float:
        """Detect skin using YCrCb color space."""
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        
        # Broad skin color range (all Fitzpatrick types)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        
        skin_mask = cv2.inRange(ycrcb, lower, upper)
        skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
        
        return skin_ratio
    
    def _detect_lesion(self, image: np.ndarray) -> Tuple[bool, bool]:
        """Quick lesion detection check."""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Otsu thresholding on L channel
        _, binary = cv2.threshold(
            lab[:, :, 0], 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return False, False
        
        # Check for reasonable sized contours
        h, w = image.shape[:2]
        img_area = h * w
        
        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if 0.001 * img_area < area < 0.8 * img_area:
                valid_contours.append(c)
        
        if len(valid_contours) == 0:
            return False, False
        
        # Check centering of largest contour
        largest = max(valid_contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        
        if M["m00"] == 0:
            return True, False
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Check if center is in middle 60%
        margin_x = w * 0.2
        margin_y = h * 0.2
        
        centered = (
            margin_x < cx < (w - margin_x) and
            margin_y < cy < (h - margin_y)
        )
        
        return True, centered
    
    def _detect_occlusions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect hair, shadows, and other occlusions."""
        occlusions = []
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Hair detection using black-hat transform
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 1))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        hair_ratio = np.sum(hair_mask > 0) / hair_mask.size
        
        if hair_ratio > 0.05:
            occlusions.append({
                "type": "hair",
                "severity": hair_ratio,
                "message": "Hair detected over lesion area"
            })
        
        return occlusions


class DermatologyInputValidator:
    """Main input validator combining all checks."""
    
    def __init__(self):
        self.file_validator = FileValidator()
        self.quality_validator = ImageQualityValidator()
        self.content_validator = ContentValidator()
    
    def validate(
        self,
        file_data: bytes,
        content_type: str = None
    ) -> Tuple[ValidationResult, Optional[np.ndarray], str]:
        """
        Validate input image.
        
        Returns:
            - ValidationResult
            - Loaded image as numpy array (if valid)
            - Image hash for tracking
        """
        # File validation
        file_valid, file_errors = self.file_validator.validate(file_data, content_type)
        
        if not file_valid:
            return ValidationResult(
                passed=False,
                quality=None,
                content=None,
                quality_score=0.0,
                errors=file_errors
            ), None, ""
        
        # Load image
        try:
            pil_image = Image.open(io.BytesIO(file_data))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            image = np.array(pil_image)
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return ValidationResult(
                passed=False,
                quality=None,
                content=None,
                quality_score=0.0,
                errors=[get_error_response("E_VAL_004")]
            ), None, ""
        
        # Compute hash
        image_hash = hashlib.sha256(file_data).hexdigest()
        
        # Quality validation
        quality_report = self.quality_validator.validate(image)
        
        # Content validation
        content_report = self.content_validator.validate(image)
        
        # Aggregate warnings
        warnings = []
        for check in [quality_report.resolution, quality_report.focus,
                      quality_report.illumination, quality_report.color_balance]:
            if check.warning:
                warnings.append(check.warning)
        warnings.extend(content_report.warnings)
        
        # Determine pass/fail
        errors = []
        
        if not quality_report.passed:
            if not quality_report.resolution.passed:
                errors.append(get_error_response("E_VAL_010"))
            if not quality_report.focus.passed:
                errors.append(get_error_response("E_VAL_011"))
            if not quality_report.illumination.passed:
                if "overexposed" in (quality_report.illumination.message or ""):
                    errors.append(get_error_response("E_VAL_012"))
                elif "dark" in (quality_report.illumination.message or ""):
                    errors.append(get_error_response("E_VAL_013"))
                else:
                    errors.append(get_error_response("E_VAL_014"))
            if not quality_report.color_balance.passed:
                errors.append(get_error_response("E_VAL_015"))
        
        if not content_report.is_skin_image:
            errors.append(get_error_response("E_VAL_020"))
        
        if not content_report.lesion_detected:
            errors.append(get_error_response("E_VAL_021"))
        
        passed = len(errors) == 0
        
        # Always return image so preprocessing can attempt to fix issues
        return ValidationResult(
            passed=passed,
            quality=quality_report,
            content=content_report,
            quality_score=quality_report.overall_quality,
            warnings=warnings,
            errors=errors
        ), image, image_hash

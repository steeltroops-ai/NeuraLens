"""
Retinal Pipeline - Quality Checker
Image quality assessment for fundus images.

Matches speech/monitoring/quality_checker.py structure.

Author: NeuraLens Medical AI Team  
Version: 4.0.0
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from ..config import QUALITY_THRESHOLDS

logger = logging.getLogger(__name__)

# Graceful cv2 import
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None


@dataclass
class QualityReport:
    """Quality assessment report."""
    quality_score: float          # 0-1
    is_acceptable: bool
    grade: str                    # excellent, good, fair, poor, ungradable
    
    # Individual metrics
    sharpness_score: float = 0.0
    illumination_score: float = 0.0
    contrast_score: float = 0.0
    snr_db: float = 0.0
    
    # Anatomical visibility
    disc_visible: bool = True
    macula_visible: bool = True
    vessels_visible: bool = True
    
    # Issues detected
    issues: List[str] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)


class RetinalQualityChecker:
    """
    Fundus image quality checker.
    
    Assesses image quality based on:
    - Sharpness (focus)
    - Illumination uniformity
    - Contrast
    - Anatomical visibility
    """
    
    def __init__(
        self,
        min_quality: float = 0.30,
        excellent_threshold: float = 0.80,
        good_threshold: float = 0.60,
        fair_threshold: float = 0.40,
        poor_threshold: float = 0.20,
    ):
        self.min_quality = min_quality
        self.thresholds = {
            "excellent": excellent_threshold,
            "good": good_threshold,
            "fair": fair_threshold,
            "poor": poor_threshold,
        }
    
    def check(
        self, 
        image: np.ndarray,
        check_anatomy: bool = True
    ) -> QualityReport:
        """
        Perform comprehensive quality check.
        
        Args:
            image: RGB image array [0-255] or [0-1]
            check_anatomy: Whether to check anatomical visibility
            
        Returns:
            QualityReport with scores and issues
        """
        issues = []
        recommendations = []
        
        # Normalize to 0-255
        if image.max() <= 1.0:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image.astype(np.uint8)
        
        # Calculate individual metrics
        sharpness = self._check_sharpness(img_uint8)
        illumination = self._check_illumination(img_uint8)
        contrast = self._check_contrast(img_uint8)
        snr = self._calculate_snr(img_uint8)
        
        # Issue detection
        if sharpness < 0.4:
            issues.append("Image appears out of focus")
            recommendations.append("Ensure proper focus before capture")
        
        if illumination < 0.4:
            issues.append("Uneven illumination detected")
            recommendations.append("Adjust flash and ambient lighting")
        
        if contrast < 0.4:
            issues.append("Low contrast image")
            recommendations.append("Check pupil dilation and exposure")
        
        if snr < 15:
            issues.append("High noise in image")
            recommendations.append("Use higher resolution or better sensor")
        
        # Anatomical checks
        disc_visible = True
        macula_visible = True
        vessels_visible = True
        
        if check_anatomy:
            disc_visible, macula_visible, vessels_visible = \
                self._check_anatomy_visibility(img_uint8)
            
            if not disc_visible:
                issues.append("Optic disc not clearly visible")
            if not macula_visible:
                issues.append("Macula region not visible")
            if not vessels_visible:
                issues.append("Vessel arcades not clearly visible")
        
        # Calculate overall quality
        anatomy_penalty = 0.0
        if not disc_visible:
            anatomy_penalty += 0.15
        if not macula_visible:
            anatomy_penalty += 0.10
        if not vessels_visible:
            anatomy_penalty += 0.10
        
        quality_score = (
            0.35 * sharpness +
            0.25 * illumination +
            0.20 * contrast +
            0.20 * min(1.0, snr / 30)
        ) - anatomy_penalty
        
        quality_score = max(0, min(1, quality_score))
        
        # Determine grade
        grade = self._get_grade(quality_score)
        
        # Acceptability
        is_acceptable = quality_score >= self.min_quality
        
        if not is_acceptable:
            recommendations.append("Image quality insufficient for analysis - please recapture")
        
        return QualityReport(
            quality_score=round(quality_score, 3),
            is_acceptable=is_acceptable,
            grade=grade,
            sharpness_score=round(sharpness, 3),
            illumination_score=round(illumination, 3),
            contrast_score=round(contrast, 3),
            snr_db=round(snr, 1),
            disc_visible=disc_visible,
            macula_visible=macula_visible,
            vessels_visible=vessels_visible,
            issues=issues,
            recommendations=recommendations,
        )
    
    def _check_sharpness(self, image: np.ndarray) -> float:
        """Calculate sharpness score using Laplacian variance."""
        if not CV2_AVAILABLE:
            return 0.7
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize (typical fundus images: 100-2000)
        normalized = min(1.0, variance / 500)
        return max(0, normalized)
    
    def _check_illumination(self, image: np.ndarray) -> float:
        """Check illumination uniformity."""
        if not CV2_AVAILABLE:
            return 0.7
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Divide into quadrants
        h, w = gray.shape
        quadrants = [
            gray[:h//2, :w//2],
            gray[:h//2, w//2:],
            gray[h//2:, :w//2],
            gray[h//2:, w//2:],
        ]
        
        # Calculate mean intensity of each quadrant
        means = [q.mean() for q in quadrants]
        
        # Uniformity = 1 - coefficient of variation
        if max(means) > 0:
            cv = np.std(means) / np.mean(means)
            uniformity = max(0, 1 - cv)
        else:
            uniformity = 0
        
        return uniformity
    
    def _check_contrast(self, image: np.ndarray) -> float:
        """Calculate contrast score."""
        if not CV2_AVAILABLE:
            return 0.7
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Standard deviation as contrast measure
        std = gray.std()
        
        # Normalize (good contrast: std > 50)
        normalized = min(1.0, std / 60)
        return max(0, normalized)
    
    def _calculate_snr(self, image: np.ndarray) -> float:
        """Estimate Signal-to-Noise Ratio."""
        if not CV2_AVAILABLE:
            return 20.0
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)
        
        # Blur to estimate signal
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Noise = difference
        noise = gray - blurred
        noise_std = noise.std()
        signal_mean = blurred.mean()
        
        if noise_std > 0:
            snr = 20 * np.log10(signal_mean / noise_std)
        else:
            snr = 40  # Perfect image
        
        return max(0, min(50, snr))
    
    def _check_anatomy_visibility(self, image: np.ndarray) -> tuple:
        """Check if key anatomical structures are visible."""
        disc_visible = True
        macula_visible = True
        vessels_visible = True
        
        if not CV2_AVAILABLE:
            return disc_visible, macula_visible, vessels_visible
        
        # Simple heuristics based on color and intensity patterns
        
        # Disc: look for bright yellowish region
        red = image[:, :, 0]
        bright_mask = red > 200
        disc_area = bright_mask.sum() / bright_mask.size
        disc_visible = 0.01 < disc_area < 0.15
        
        # Vessels: check green channel variance
        green = image[:, :, 1]
        vessel_contrast = green.std()
        vessels_visible = vessel_contrast > 30
        
        # Macula: check for darker central region (less reliable)
        h, w = image.shape[:2]
        center_region = image[h//3:2*h//3, w//3:2*w//3, 1]
        macula_visible = center_region.mean() < green.mean() + 20
        
        return disc_visible, macula_visible, vessels_visible
    
    def _get_grade(self, score: float) -> str:
        """Get quality grade from score."""
        if score >= self.thresholds["excellent"]:
            return "excellent"
        elif score >= self.thresholds["good"]:
            return "good"
        elif score >= self.thresholds["fair"]:
            return "fair"
        elif score >= self.thresholds["poor"]:
            return "poor"
        else:
            return "ungradable"


# Singleton instance
retinal_quality_checker = RetinalQualityChecker()

"""
Radiology Pipeline - Image Quality Assessment
Evaluates X-ray image quality for reliable analysis
"""

import numpy as np
from PIL import Image
from io import BytesIO
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class QualityResult:
    """Image quality assessment result"""
    quality_score: float  # 0-1
    image_quality: str  # good, adequate, poor
    positioning: str
    technical_factors: str
    resolution: str
    resolution_adequate: bool
    contrast: float
    brightness: float
    issues: List[str]
    usable: bool
    
    def to_dict(self) -> Dict:
        return {
            "quality_score": round(self.quality_score, 2),
            "image_quality": self.image_quality,
            "positioning": self.positioning,
            "technical_factors": self.technical_factors,
            "resolution": self.resolution,
            "contrast": round(self.contrast, 2),
            "issues": self.issues,
            "usable": self.usable
        }


class XRayQualityAssessor:
    """
    X-Ray Image Quality Assessment
    
    Evaluates:
    - Resolution (minimum 512x512 recommended)
    - Contrast (dynamic range)
    - Brightness/Exposure
    - Positioning (centered, rotation)
    """
    
    MIN_RESOLUTION = 512
    MIN_CONTRAST = 0.4
    OPTIMAL_BRIGHTNESS_RANGE = (0.3, 0.7)
    
    def assess(self, image_bytes: bytes) -> QualityResult:
        """
        Assess X-ray image quality
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            QualityResult with quality metrics
        """
        # Load image
        img = Image.open(BytesIO(image_bytes))
        img_np = np.array(img.convert('L')).astype(np.float32)
        
        # Get dimensions
        height, width = img_np.shape
        
        # Resolution check
        resolution_ok = height >= self.MIN_RESOLUTION and width >= self.MIN_RESOLUTION
        resolution_str = f"{width}x{height}"
        
        # Contrast check (dynamic range)
        min_val, max_val = np.min(img_np), np.max(img_np)
        contrast = (max_val - min_val) / 255.0
        
        # Brightness check
        brightness = np.mean(img_np) / 255.0
        
        # Positioning check (look for centering)
        center_y, center_x = height // 2, width // 2
        quadrant_means = [
            np.mean(img_np[:center_y, :center_x]),
            np.mean(img_np[:center_y, center_x:]),
            np.mean(img_np[center_y:, :center_x]),
            np.mean(img_np[center_y:, center_x:])
        ]
        positioning_variance = np.std(quadrant_means) / 255.0
        positioning_ok = positioning_variance < 0.15
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            resolution_ok, contrast, brightness, positioning_ok
        )
        
        # Identify issues
        issues = self._identify_issues(
            resolution_ok, contrast, brightness, positioning_ok, width, height
        )
        
        # Determine quality category
        if quality_score >= 0.8:
            image_quality = "good"
        elif quality_score >= 0.6:
            image_quality = "adequate"
        else:
            image_quality = "poor"
        
        # Determine positioning description
        if positioning_ok:
            positioning = "adequate"
        else:
            positioning = "suboptimal"
        
        # Technical factors
        if contrast >= 0.6 and self.OPTIMAL_BRIGHTNESS_RANGE[0] <= brightness <= self.OPTIMAL_BRIGHTNESS_RANGE[1]:
            technical_factors = "satisfactory"
        elif contrast >= self.MIN_CONTRAST:
            technical_factors = "acceptable"
        else:
            technical_factors = "suboptimal"
        
        return QualityResult(
            quality_score=quality_score,
            image_quality=image_quality,
            positioning=positioning,
            technical_factors=technical_factors,
            resolution=resolution_str,
            resolution_adequate=resolution_ok,
            contrast=contrast,
            brightness=brightness,
            issues=issues,
            usable=quality_score >= 0.4
        )
    
    def _calculate_quality_score(
        self,
        resolution_ok: bool,
        contrast: float,
        brightness: float,
        positioning_ok: bool
    ) -> float:
        """Calculate overall quality score (0-1)"""
        score = 0.0
        
        # Resolution (25% weight)
        score += 0.25 * (1.0 if resolution_ok else 0.6)
        
        # Contrast (35% weight)
        contrast_score = min(1.0, contrast / 0.7)  # Normalize to ideal contrast
        score += 0.35 * contrast_score
        
        # Brightness (25% weight) - optimal around 0.5
        brightness_deviation = abs(brightness - 0.5)
        brightness_score = max(0, 1.0 - brightness_deviation * 2)
        score += 0.25 * brightness_score
        
        # Positioning (15% weight)
        score += 0.15 * (1.0 if positioning_ok else 0.7)
        
        return min(1.0, score)
    
    def _identify_issues(
        self,
        resolution_ok: bool,
        contrast: float,
        brightness: float,
        positioning_ok: bool,
        width: int,
        height: int
    ) -> List[str]:
        """Identify quality issues"""
        issues = []
        
        if not resolution_ok:
            issues.append(f"Low resolution ({width}x{height}). Recommend at least {self.MIN_RESOLUTION}x{self.MIN_RESOLUTION}.")
        
        if contrast < self.MIN_CONTRAST:
            issues.append("Low contrast may affect detection accuracy.")
        elif contrast < 0.5:
            issues.append("Suboptimal contrast - consider image enhancement.")
        
        if brightness < self.OPTIMAL_BRIGHTNESS_RANGE[0]:
            issues.append("Image appears underexposed (too dark).")
        elif brightness > self.OPTIMAL_BRIGHTNESS_RANGE[1]:
            issues.append("Image appears overexposed (too bright).")
        
        if not positioning_ok:
            issues.append("Image may be poorly centered or rotated.")
        
        return issues


def assess_xray_quality(image_bytes: bytes) -> Dict:
    """
    Convenience function to assess X-ray quality
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Quality assessment dictionary
    """
    assessor = XRayQualityAssessor()
    result = assessor.assess(image_bytes)
    return result.to_dict()

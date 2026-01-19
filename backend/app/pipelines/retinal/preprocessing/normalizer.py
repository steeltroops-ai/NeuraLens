"""
Image Preprocessing Module for Retinal Analysis Pipeline

Implements preprocessing stages from the specification:
1. Color Normalization - LAB space standardization
2. Illumination Correction - MSRCR algorithm
3. Contrast Enhancement - CLAHE
4. Artifact Removal - Dust, reflections, eyelash shadows
5. Fundus Detection - Verify image is retinal fundus
6. Quality Scoring - Composite quality gate

Author: NeuraLens Medical AI Team
Version: 4.0.0
"""

import logging
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass, field
from PIL import Image
import io

# Graceful cv2 import
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

from ..errors.codes import PipelineException

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PreprocessingConfig:
    """Preprocessing configuration constants"""
    # Target values for color normalization
    TARGET_L_MEAN: float = 128.0
    TARGET_L_STD: float = 50.0
    
    # CLAHE parameters
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_SIZE: Tuple[int, int] = (8, 8)
    
    # Illumination correction
    RETINEX_SCALES: List[int] = field(default_factory=lambda: [15, 80, 250])
    
    # Quality thresholds
    MIN_QUALITY_SCORE: float = 0.3
    RECOMMENDED_QUALITY_SCORE: float = 0.6
    
    # Fundus detection
    MIN_FUNDUS_COVERAGE: float = 0.4
    MIN_VESSEL_DENSITY: float = 0.02
    RED_CHANNEL_DOMINANCE: float = 0.8
    
    # Artifact detection
    DUST_THRESHOLD: int = 30
    REFLECTION_THRESHOLD: int = 250
    
    # Size limits
    MODEL_INPUT_SIZE: int = 512


CONFIG = PreprocessingConfig()


# =============================================================================
# PREPROCESSING RESULT
# =============================================================================

@dataclass
class PreprocessingResult:
    """Result from preprocessing pipeline"""
    image: np.ndarray
    original_image: np.ndarray
    quality_score: float
    quality_grade: str  # excellent, good, fair, poor, ungradable
    is_fundus: bool
    fundus_confidence: float
    artifacts_removed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Component scores
    sharpness_score: float = 0.0
    illumination_score: float = 0.0
    contrast_score: float = 0.0
    snr_db: float = 0.0
    
    # Preprocessing applied
    color_normalized: bool = False
    clahe_applied: bool = False
    artifacts_cleaned: bool = False


# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================

class ColorNormalizer:
    """Normalize image colors to standard fundus appearance"""
    
    @staticmethod
    def normalize(image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Normalize color using LAB color space.
        
        Args:
            image: RGB image (H, W, 3) float32 [0, 1]
            
        Returns:
            Normalized image and success flag
        """
        try:
            if not CV2_AVAILABLE:
                return image, False
            
            # Convert to uint8 for cv2
            img_uint8 = (image * 255).astype(np.uint8)
            
            # Convert to LAB
            lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
            
            # Normalize L channel
            l_channel = lab[:, :, 0].astype(np.float32)
            l_mean = l_channel.mean()
            l_std = l_channel.std()
            
            if l_std > 1e-6:
                l_normalized = (l_channel - l_mean) / l_std
                l_normalized = l_normalized * CONFIG.TARGET_L_STD + CONFIG.TARGET_L_MEAN
                l_normalized = np.clip(l_normalized, 0, 255).astype(np.uint8)
                lab[:, :, 0] = l_normalized
            
            # Convert back to RGB
            normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return normalized.astype(np.float32) / 255.0, True
            
        except Exception as e:
            logger.warning(f"Color normalization failed: {e}")
            return image, False


class IlluminationCorrector:
    """Correct non-uniform illumination using Multi-Scale Retinex"""
    
    @staticmethod
    def correct(image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Apply Multi-Scale Retinex with Color Restoration (MSRCR).
        
        Args:
            image: RGB image (H, W, 3) float32 [0, 1]
            
        Returns:
            Corrected image and success flag
        """
        try:
            if not CV2_AVAILABLE:
                return image, False
            
            img_float = image.astype(np.float32) + 1e-6  # Avoid log(0)
            
            retinex = np.zeros_like(img_float)
            
            for scale in CONFIG.RETINEX_SCALES:
                blur = cv2.GaussianBlur(img_float, (0, 0), scale)
                retinex += np.log10(img_float) - np.log10(blur + 1e-6)
            
            retinex = retinex / len(CONFIG.RETINEX_SCALES)
            
            # Normalize to [0, 1]
            for c in range(3):
                channel = retinex[:, :, c]
                min_val, max_val = channel.min(), channel.max()
                if max_val - min_val > 1e-6:
                    retinex[:, :, c] = (channel - min_val) / (max_val - min_val)
            
            return retinex.astype(np.float32), True
            
        except Exception as e:
            logger.warning(f"Illumination correction failed: {e}")
            return image, False


class ContrastEnhancer:
    """Enhance contrast using CLAHE"""
    
    @staticmethod
    def enhance(image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Apply CLAHE contrast enhancement on L channel.
        
        Args:
            image: RGB image (H, W, 3) float32 [0, 1]
            
        Returns:
            Enhanced image and success flag
        """
        try:
            if not CV2_AVAILABLE:
                return image, False
            
            # Convert to uint8
            img_uint8 = (image * 255).astype(np.uint8)
            
            # Convert to LAB
            lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(
                clipLimit=CONFIG.CLAHE_CLIP_LIMIT,
                tileGridSize=CONFIG.CLAHE_TILE_SIZE
            )
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return enhanced.astype(np.float32) / 255.0, True
            
        except Exception as e:
            logger.warning(f"CLAHE enhancement failed: {e}")
            return image, False


class ArtifactRemover:
    """Remove common image artifacts"""
    
    @staticmethod
    def remove(image: np.ndarray) -> Tuple[np.ndarray, List[str], bool]:
        """
        Detect and remove artifacts (dust, reflections).
        
        Args:
            image: RGB image (H, W, 3) float32 [0, 1]
            
        Returns:
            Cleaned image, list of artifacts found, and success flag
        """
        artifacts_found = []
        
        try:
            if not CV2_AVAILABLE:
                return image, [], False
            
            img_uint8 = (image * 255).astype(np.uint8)
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
            
            # Detect dark spots (dust)
            dust_mask = gray < CONFIG.DUST_THRESHOLD
            dust_count = np.sum(dust_mask)
            
            if dust_count > 100:
                # Dilate to ensure coverage
                kernel = np.ones((5, 5), np.uint8)
                dust_mask_dilated = cv2.dilate(dust_mask.astype(np.uint8), kernel, iterations=1)
                img_uint8 = cv2.inpaint(img_uint8, dust_mask_dilated, 5, cv2.INPAINT_TELEA)
                artifacts_found.append("dust_spots")
            
            # Detect reflections (saturated areas)
            reflection_mask = gray > CONFIG.REFLECTION_THRESHOLD
            reflection_count = np.sum(reflection_mask)
            
            if reflection_count > 100:
                kernel = np.ones((3, 3), np.uint8)
                reflection_mask_dilated = cv2.dilate(reflection_mask.astype(np.uint8), kernel, iterations=1)
                img_uint8 = cv2.inpaint(img_uint8, reflection_mask_dilated, 5, cv2.INPAINT_NS)
                artifacts_found.append("reflections")
            
            return img_uint8.astype(np.float32) / 255.0, artifacts_found, True
            
        except Exception as e:
            logger.warning(f"Artifact removal failed: {e}")
            return image, [], False


class FundusDetector:
    """Detect if image is a retinal fundus photograph"""
    
    @staticmethod
    def detect(image: np.ndarray) -> Tuple[bool, float, List[str]]:
        """
        Verify image is a retinal fundus photograph.
        
        Checks:
        1. Red channel dominance (fundus has strong red component)
        2. Circular field of view
        3. Vessel-like edge patterns
        
        Args:
            image: RGB image (H, W, 3) float32 [0, 1]
            
        Returns:
            (is_fundus, confidence, issues)
        """
        issues = []
        confidence_factors = []
        
        try:
            if not CV2_AVAILABLE:
                return True, 0.5, ["CV2 not available - fundus detection skipped"]
            
            # Convert to uint8
            img_uint8 = (image * 255).astype(np.uint8)
            h, w = img_uint8.shape[:2]
            
            # 1. Check red channel dominance
            r, g, b = img_uint8[:, :, 0], img_uint8[:, :, 1], img_uint8[:, :, 2]
            r_mean, g_mean, b_mean = r.mean(), g.mean(), b.mean()
            
            # Fundus typically has higher red values
            red_dominance = r_mean / (g_mean + 1e-6)
            if red_dominance > CONFIG.RED_CHANNEL_DOMINANCE:
                confidence_factors.append(0.9)
            elif red_dominance > 0.6:
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.3)
                issues.append("Low red channel - may not be fundus")
            
            # 2. Check for circular FOV (fundus cameras have circular aperture)
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
            
            # Find dark border (outside FOV)
            _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                coverage = area / (h * w)
                
                if coverage > CONFIG.MIN_FUNDUS_COVERAGE:
                    confidence_factors.append(0.8)
                else:
                    confidence_factors.append(0.4)
                    issues.append("Limited fundus field detected")
            else:
                confidence_factors.append(0.5)
            
            # 3. Check for vessel-like structures (edges)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (h * w)
            
            if edge_density > CONFIG.MIN_VESSEL_DENSITY:
                confidence_factors.append(0.85)
            else:
                confidence_factors.append(0.4)
                issues.append("Low vessel pattern density")
            
            # Calculate overall confidence
            confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
            is_fundus = confidence > 0.5
            
            return is_fundus, confidence, issues
            
        except Exception as e:
            logger.warning(f"Fundus detection failed: {e}")
            return True, 0.5, [f"Detection error: {e}"]


class QualityScorer:
    """Calculate composite image quality score"""
    
    @staticmethod
    def score(image: np.ndarray) -> Dict[str, float]:
        """
        Calculate quality metrics and composite score.
        
        Args:
            image: RGB image (H, W, 3) float32 [0, 1]
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            if not CV2_AVAILABLE:
                return {
                    "composite": 0.5,
                    "sharpness": 0.5,
                    "illumination": 0.5,
                    "contrast": 0.5,
                    "snr_db": 20.0,
                    "grade": "fair"
                }
            
            img_uint8 = (image * 255).astype(np.uint8)
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
            
            # Sharpness (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness_raw = laplacian.var()
            sharpness = min(1.0, sharpness_raw / 500)
            
            # SNR
            mean_intensity = gray.mean()
            std_intensity = gray.std()
            snr_db = 20 * np.log10(mean_intensity / (std_intensity + 1e-6))
            snr_db = np.clip(snr_db, 0, 50)
            snr_normalized = snr_db / 40  # Normalize to ~1.0
            
            # Illumination uniformity
            h, w = gray.shape
            quadrants = [
                gray[:h//2, :w//2].mean(),
                gray[:h//2, w//2:].mean(),
                gray[h//2:, :w//2].mean(),
                gray[h//2:, w//2:].mean(),
            ]
            uniformity = 1 - (np.std(quadrants) / (np.mean(quadrants) + 1e-6))
            illumination = np.clip(uniformity, 0, 1)
            
            # Contrast
            contrast = min(1.0, std_intensity / 80)
            
            # Composite score (weighted)
            composite = (
                sharpness * 0.40 +
                snr_normalized * 0.30 +
                illumination * 0.15 +
                contrast * 0.15
            )
            composite = np.clip(composite, 0, 1)
            
            # Grade
            if composite >= 0.8:
                grade = "excellent"
            elif composite >= 0.6:
                grade = "good"
            elif composite >= 0.4:
                grade = "fair"
            elif composite >= 0.2:
                grade = "poor"
            else:
                grade = "ungradable"
            
            return {
                "composite": round(composite, 3),
                "sharpness": round(sharpness, 3),
                "illumination": round(illumination, 3),
                "contrast": round(contrast, 3),
                "snr_db": round(snr_db, 1),
                "grade": grade
            }
            
        except Exception as e:
            logger.warning(f"Quality scoring failed: {e}")
            return {
                "composite": 0.5,
                "sharpness": 0.5,
                "illumination": 0.5,
                "contrast": 0.5,
                "snr_db": 20.0,
                "grade": "fair"
            }


# =============================================================================
# MAIN PREPROCESSOR
# =============================================================================

class ImagePreprocessor:
    """
    Complete image preprocessing pipeline.
    
    Pipeline:
    1. Load and validate image
    2. Fundus detection
    3. Color normalization
    4. Illumination correction (optional)
    5. Contrast enhancement (CLAHE)
    6. Artifact removal
    7. Quality scoring
    8. Resize for model
    """
    
    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or CONFIG
        self.color_normalizer = ColorNormalizer()
        self.illumination_corrector = IlluminationCorrector()
        self.contrast_enhancer = ContrastEnhancer()
        self.artifact_remover = ArtifactRemover()
        self.fundus_detector = FundusDetector()
        self.quality_scorer = QualityScorer()
    
    def preprocess(self, image_bytes: bytes, skip_fundus_check: bool = False) -> PreprocessingResult:
        """
        Execute full preprocessing pipeline.
        
        Args:
            image_bytes: Raw image bytes
            skip_fundus_check: Skip fundus detection (for known-good images)
            
        Returns:
            PreprocessingResult with processed image and quality info
        """
        # Load image
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img_array = np.array(img, dtype=np.float32) / 255.0
        except Exception as e:
            raise PipelineException("VAL_051", {"error": str(e)})
        
        original_image = img_array.copy()
        warnings = []
        
        # Fundus detection
        if not skip_fundus_check:
            is_fundus, fundus_confidence, fundus_issues = self.fundus_detector.detect(img_array)
            
            if not is_fundus:
                raise PipelineException(
                    "VAL_040", 
                    {"confidence": fundus_confidence, "issues": fundus_issues}
                )
            
            if fundus_issues:
                warnings.extend(fundus_issues)
        else:
            is_fundus, fundus_confidence = True, 1.0
        
        # Color normalization
        img_array, color_normalized = self.color_normalizer.normalize(img_array)
        if not color_normalized:
            warnings.append("Color normalization skipped")
        
        # Contrast enhancement (CLAHE)
        img_array, clahe_applied = self.contrast_enhancer.enhance(img_array)
        if not clahe_applied:
            warnings.append("CLAHE enhancement skipped")
        
        # Artifact removal
        img_array, artifacts_found, artifacts_cleaned = self.artifact_remover.remove(img_array)
        
        # Quality scoring
        quality = self.quality_scorer.score(img_array)
        
        # Check quality gate
        if quality["composite"] < self.config.MIN_QUALITY_SCORE:
            raise PipelineException(
                "PRE_010",
                {"quality_score": quality["composite"], "minimum": self.config.MIN_QUALITY_SCORE}
            )
        
        # Resize for model
        h, w = img_array.shape[:2]
        target = self.config.MODEL_INPUT_SIZE
        if h != target or w != target:
            img_uint8 = (img_array * 255).astype(np.uint8)
            if CV2_AVAILABLE:
                img_resized = cv2.resize(img_uint8, (target, target), interpolation=cv2.INTER_LINEAR)
            else:
                img_pil = Image.fromarray(img_uint8)
                img_pil = img_pil.resize((target, target), Image.BILINEAR)
                img_resized = np.array(img_pil)
            img_array = img_resized.astype(np.float32) / 255.0
        
        return PreprocessingResult(
            image=img_array,
            original_image=original_image,
            quality_score=quality["composite"],
            quality_grade=quality["grade"],
            is_fundus=is_fundus,
            fundus_confidence=fundus_confidence,
            artifacts_removed=artifacts_found,
            warnings=warnings,
            sharpness_score=quality["sharpness"],
            illumination_score=quality["illumination"],
            contrast_score=quality["contrast"],
            snr_db=quality["snr_db"],
            color_normalized=color_normalized,
            clahe_applied=clahe_applied,
            artifacts_cleaned=artifacts_cleaned,
        )


# Singleton instance
image_preprocessor = ImagePreprocessor()

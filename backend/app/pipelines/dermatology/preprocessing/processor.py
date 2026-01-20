"""
Dermatology Pipeline Image Preprocessor

Multi-stage image preprocessing for skin lesion analysis.
"""

import logging
import time
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass

import numpy as np
import cv2

from ..config import PREPROCESSING_CONFIG
from ..schemas import PreprocessingStageResult, PreprocessingResult

logger = logging.getLogger(__name__)


class ColorConstancyNormalizer:
    """
    Color constancy using Shades of Gray algorithm.
    Removes illuminant color bias.
    """
    
    def __init__(self, norm_order: int = 6):
        self.norm_order = norm_order
    
    def normalize(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Apply color constancy normalization."""
        image_float = image.astype(np.float32) / 255.0
        
        # Estimate illuminant
        if self.norm_order == np.inf:
            illuminant = np.max(image_float, axis=(0, 1))
        else:
            illuminant = np.power(
                np.mean(np.power(image_float + 1e-6, self.norm_order), axis=(0, 1)),
                1.0 / self.norm_order
            )
        
        # Normalize illuminant
        illuminant = illuminant / (np.linalg.norm(illuminant) + 1e-6)
        
        # Apply correction
        target = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
        correction = target / (illuminant + 1e-6)
        
        corrected = image_float * correction
        corrected = np.clip(corrected, 0, 1)
        
        output = (corrected * 255).astype(np.uint8)
        
        # Confidence based on deviation from neutral
        neutral = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
        deviation = np.linalg.norm(illuminant - neutral)
        confidence = float(np.exp(-deviation * 2))
        
        return output, confidence


class IlluminationCorrector:
    """
    Corrects uneven illumination using CLAHE.
    """
    
    def __init__(self, clip_limit: float = 2.0, tile_size: Tuple[int, int] = (8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    
    def correct(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Apply illumination correction."""
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Measure uniformity before
        uniformity_before = self._measure_uniformity(l_channel)
        
        # Apply CLAHE to L channel
        l_corrected = self.clahe.apply(l_channel)
        
        # Measure uniformity after
        uniformity_after = self._measure_uniformity(l_corrected)
        
        # Merge and convert back
        corrected_lab = cv2.merge([l_corrected, a_channel, b_channel])
        corrected_rgb = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2RGB)
        
        improvement = uniformity_after - uniformity_before
        confidence = min(uniformity_after, 1.0)
        
        return corrected_rgb, confidence
    
    def _measure_uniformity(self, image: np.ndarray) -> float:
        """Measure illumination uniformity (0-1)."""
        # Divide into blocks
        h, w = image.shape
        block_h, block_w = h // 4, w // 4
        
        means = []
        for i in range(4):
            for j in range(4):
                block = image[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                means.append(np.mean(block))
        
        std = np.std(means)
        uniformity = 1.0 - min(1.0, std / 50.0)
        return uniformity


class HairArtifactRemover:
    """
    Removes hair and artifacts using morphological operations and inpainting.
    Based on DullRazor algorithm.
    """
    
    def __init__(self, kernel_size: int = 17, inpaint_radius: int = 5):
        self.kernel_size = kernel_size
        self.inpaint_radius = inpaint_radius
    
    def remove(self, image: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """Remove hair and artifacts."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect hair using morphological black-hat
        hair_mask = self._detect_hair(gray)
        
        # Detect other artifacts
        artifact_mask = self._detect_artifacts(image)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(hair_mask, artifact_mask)
        
        # Inpaint
        inpainted = cv2.inpaint(
            image, combined_mask, 
            self.inpaint_radius, 
            cv2.INPAINT_TELEA
        )
        
        # Calculate coverage
        hair_coverage = np.sum(hair_mask > 0) / hair_mask.size
        confidence = 1.0 - min(hair_coverage * 2, 0.5)
        
        return inpainted, confidence, combined_mask
    
    def _detect_hair(self, gray: np.ndarray) -> np.ndarray:
        """Detect hair using black-hat transform."""
        masks = []
        
        # Multiple orientations
        for angle in range(0, 180, 30):
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (self.kernel_size, 1)
            )
            # Rotate kernel
            M = cv2.getRotationMatrix2D(
                (self.kernel_size // 2, 0), angle, 1
            )
            kernel = cv2.warpAffine(
                kernel, M, (self.kernel_size, self.kernel_size)
            )
            
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            masks.append(blackhat)
        
        # Combine
        combined = np.max(masks, axis=0)
        
        # Threshold
        _, hair_mask = cv2.threshold(combined, 10, 255, cv2.THRESH_BINARY)
        
        # Cleanup
        kernel = np.ones((3, 3), np.uint8)
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)
        
        return hair_mask.astype(np.uint8)
    
    def _detect_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Detect non-hair artifacts."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Detect very bright spots (reflections)
        _, bright_mask = cv2.threshold(hsv[:, :, 2], 250, 255, cv2.THRESH_BINARY)
        
        return bright_mask.astype(np.uint8)


class ContrastEnhancer:
    """
    Enhances lesion-skin contrast.
    """
    
    def enhance(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Enhance contrast."""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Contrast before
        contrast_before = np.std(l)
        
        # Sigmoid-based enhancement on L channel
        l_float = l.astype(np.float32) / 255.0
        mean_l = np.mean(l_float)
        enhanced = 1 / (1 + np.exp(-8 * (l_float - mean_l)))
        l_enhanced = (enhanced * 255).astype(np.uint8)
        
        # Stretch A and B channels
        a_enhanced = self._stretch_channel(a)
        b_enhanced = self._stretch_channel(b)
        
        enhanced_lab = cv2.merge([l_enhanced, a_enhanced, b_enhanced])
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # Contrast after
        contrast_after = np.std(l_enhanced)
        
        improvement = contrast_after / (contrast_before + 1e-6)
        confidence = min(improvement / 1.5, 1.0)
        
        return enhanced_rgb, confidence
    
    def _stretch_channel(self, channel: np.ndarray) -> np.ndarray:
        """Linear stretch a color channel."""
        p2, p98 = np.percentile(channel, (2, 98))
        stretched = np.clip((channel - p2) * 255.0 / (p98 - p2 + 1e-6), 0, 255)
        return stretched.astype(np.uint8)


class ImageResizer:
    """
    Resizes images to model input dimensions.
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        preserve_aspect: bool = True,
        padding_color: Tuple[int, int, int] = (0, 0, 0)
    ):
        self.target_size = target_size
        self.preserve_aspect = preserve_aspect
        self.padding_color = padding_color
    
    def resize(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resize image to target dimensions."""
        h, w = image.shape[:2]
        target_w, target_h = self.target_size
        
        if self.preserve_aspect:
            # Calculate scale
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Pad
            pad_left = (target_w - new_w) // 2
            pad_top = (target_h - new_h) // 2
            
            output = np.full((target_h, target_w, 3), self.padding_color, dtype=np.uint8)
            output[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
            
            transform = {
                "scale": scale,
                "offset": (pad_left, pad_top),
                "original_size": (w, h),
                "resized_size": (new_w, new_h),
                "target_size": self.target_size
            }
        else:
            output = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            transform = {
                "scale_x": target_w / w,
                "scale_y": target_h / h,
                "original_size": (w, h),
                "target_size": self.target_size
            }
        
        return output, transform


class DermatologyPreprocessor:
    """
    Complete preprocessing pipeline for dermatology images.
    """
    
    def __init__(self, config=None):
        self.config = config or PREPROCESSING_CONFIG
        
        self.color_normalizer = ColorConstancyNormalizer(
            norm_order=self.config.color_constancy_norm_order
        )
        self.illumination_corrector = IlluminationCorrector(
            clip_limit=self.config.clahe_clip_limit,
            tile_size=self.config.clahe_tile_size
        )
        self.artifact_remover = HairArtifactRemover(
            kernel_size=self.config.hair_kernel_size,
            inpaint_radius=self.config.inpaint_radius
        )
        self.contrast_enhancer = ContrastEnhancer()
        self.resizer = ImageResizer(
            target_size=self.config.target_size,
            preserve_aspect=self.config.preserve_aspect,
            padding_color=self.config.padding_color
        )
    
    def preprocess(self, image: np.ndarray) -> PreprocessingResult:
        """Run complete preprocessing pipeline."""
        start_time = time.time()
        stages = []
        current_image = image.copy()
        warnings = []
        
        # Stage 1: Color constancy
        try:
            current_image, cc_confidence = self.color_normalizer.normalize(current_image)
            stages.append(PreprocessingStageResult(
                stage="color_constancy",
                success=True,
                confidence=cc_confidence,
                metrics={"confidence": cc_confidence}
            ))
            if cc_confidence < self.config.min_color_constancy_confidence:
                warnings.append("Low color constancy confidence")
        except Exception as e:
            logger.warning(f"Color constancy failed: {e}")
            stages.append(PreprocessingStageResult(
                stage="color_constancy",
                success=False,
                confidence=0.0,
                warning=str(e)
            ))
        
        # Stage 2: Illumination correction
        try:
            current_image, illum_confidence = self.illumination_corrector.correct(current_image)
            stages.append(PreprocessingStageResult(
                stage="illumination",
                success=True,
                confidence=illum_confidence,
                metrics={"uniformity": illum_confidence}
            ))
        except Exception as e:
            logger.warning(f"Illumination correction failed: {e}")
            stages.append(PreprocessingStageResult(
                stage="illumination",
                success=False,
                confidence=0.0,
                warning=str(e)
            ))
        
        # Stage 3: Hair/artifact removal
        try:
            current_image, hair_confidence, _ = self.artifact_remover.remove(current_image)
            stages.append(PreprocessingStageResult(
                stage="artifact_removal",
                success=True,
                confidence=hair_confidence,
                metrics={"coverage_removed": 1.0 - hair_confidence}
            ))
            if hair_confidence < 0.7:
                warnings.append("Significant hair/artifact coverage detected")
        except Exception as e:
            logger.warning(f"Artifact removal failed: {e}")
            stages.append(PreprocessingStageResult(
                stage="artifact_removal",
                success=False,
                confidence=0.0,
                warning=str(e)
            ))
        
        # Stage 4: Contrast enhancement
        try:
            current_image, contrast_confidence = self.contrast_enhancer.enhance(current_image)
            stages.append(PreprocessingStageResult(
                stage="contrast_enhancement",
                success=True,
                confidence=contrast_confidence,
                metrics={"improvement": contrast_confidence}
            ))
        except Exception as e:
            logger.warning(f"Contrast enhancement failed: {e}")
            stages.append(PreprocessingStageResult(
                stage="contrast_enhancement",
                success=False,
                confidence=0.0,
                warning=str(e)
            ))
        
        # Stage 5: Resize
        try:
            current_image, transform_info = self.resizer.resize(current_image)
            stages.append(PreprocessingStageResult(
                stage="resize",
                success=True,
                confidence=1.0,
                metrics=transform_info
            ))
        except Exception as e:
            logger.error(f"Resize failed: {e}")
            stages.append(PreprocessingStageResult(
                stage="resize",
                success=False,
                confidence=0.0,
                warning=str(e)
            ))
            transform_info = {}
        
        # Calculate overall confidence
        successful_stages = [s for s in stages if s.success]
        overall_confidence = (
            sum(s.confidence for s in successful_stages) / len(successful_stages)
            if successful_stages else 0.0
        )
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return PreprocessingResult(
            image=current_image,
            stages=stages,
            overall_confidence=overall_confidence,
            warnings=warnings,
            transform_info=transform_info,
            processing_time_ms=processing_time_ms
        )

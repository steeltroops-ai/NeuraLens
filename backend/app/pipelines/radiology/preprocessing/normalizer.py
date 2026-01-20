"""
Radiology Image Normalizer

Normalizes and enhances images for model input.
"""

import numpy as np
from PIL import Image
from io import BytesIO
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..config import RadiologyConfig


@dataclass
class NormalizationResult:
    """Result of image normalization."""
    normalized_array: np.ndarray
    original_shape: Tuple[int, int]
    output_shape: Tuple[int, int]
    normalization_method: str
    quality_metrics: Dict[str, float]


class ImageNormalizer:
    """
    Normalize images for TorchXRayVision input.
    
    Performs:
    - Grayscale conversion
    - Resizing to model input size
    - Intensity normalization
    - Optional enhancement
    """
    
    def __init__(self, target_size: int = 224):
        self.target_size = target_size
    
    def normalize(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Normalize image for model input.
        
        Args:
            image_bytes: Raw image data
        
        Returns:
            Dict with normalized image and metadata
        """
        # Load image
        img = Image.open(BytesIO(image_bytes))
        original_size = img.size
        
        # Convert to grayscale
        img_gray = img.convert('L')
        img_array = np.array(img_gray, dtype=np.float32)
        
        # Store original for quality metrics
        original_array = img_array.copy()
        
        # Resize to target size
        img_resized = img_gray.resize(
            (self.target_size, self.target_size),
            Image.Resampling.LANCZOS
        )
        img_array = np.array(img_resized, dtype=np.float32)
        
        # Normalize intensity (0-1 range)
        img_min = img_array.min()
        img_max = img_array.max()
        
        if img_max > img_min:
            img_normalized = (img_array - img_min) / (img_max - img_min)
        else:
            img_normalized = np.zeros_like(img_array)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(original_array)
        
        return {
            "normalized_array": img_normalized,
            "original_shape": original_size,
            "output_shape": (self.target_size, self.target_size),
            "normalization_method": "min_max",
            "quality_metrics": quality_metrics,
            "original_image": img
        }
    
    def _calculate_quality_metrics(self, img_array: np.ndarray) -> Dict[str, float]:
        """Calculate image quality metrics."""
        # Contrast (dynamic range)
        contrast = (np.max(img_array) - np.min(img_array)) / 255
        
        # Variance (detail content)
        variance = np.var(img_array)
        
        # Mean brightness
        brightness = np.mean(img_array) / 255
        
        # Sharpness estimate (gradient magnitude)
        grad_y = np.diff(img_array, axis=0)
        grad_x = np.diff(img_array, axis=1)
        sharpness = (np.mean(np.abs(grad_y)) + np.mean(np.abs(grad_x))) / 2
        sharpness = min(1.0, sharpness / 50)  # Normalize
        
        return {
            "contrast": round(contrast, 3),
            "variance": round(variance, 2),
            "brightness": round(brightness, 3),
            "sharpness": round(sharpness, 3),
            "dynamic_range": round(np.ptp(img_array), 1)
        }
    
    def prepare_for_model(
        self,
        image_bytes: bytes,
        model_type: str = "torchxrayvision"
    ) -> np.ndarray:
        """
        Prepare image for specific model input.
        
        Args:
            image_bytes: Raw image data
            model_type: Target model type
        
        Returns:
            Normalized array ready for model
        """
        result = self.normalize(image_bytes)
        img_array = result["normalized_array"]
        
        if model_type == "torchxrayvision":
            # TorchXRayVision expects [C, H, W] format
            # Normalize using TorchXRayVision convention
            img_array = img_array * 255  # Back to 0-255 for xrv normalization
            
        return img_array

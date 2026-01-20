"""
Dermatology Pipeline Utilities
"""

import numpy as np
import cv2
import base64
from io import BytesIO
from typing import Tuple


def image_to_base64(image: np.ndarray, format: str = 'png') -> str:
    """Convert numpy image to base64 string."""
    # Ensure RGB -> BGR for OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image
    
    _, buffer = cv2.imencode(f'.{format}', image_bgr)
    return base64.b64encode(buffer).decode('utf-8')


def base64_to_image(b64_string: str) -> np.ndarray:
    """Convert base64 string to numpy image."""
    img_data = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # BGR -> RGB
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def calculate_pixels_per_mm(
    image_width: int,
    image_height: int,
    known_distance_mm: float = None,
    known_distance_pixels: float = None
) -> float:
    """
    Calculate pixels per millimeter for calibration.
    
    If known_distance is provided, use it for calibration.
    Otherwise, estimate based on typical smartphone focal lengths.
    """
    if known_distance_mm and known_distance_pixels:
        return known_distance_pixels / known_distance_mm
    
    # Default estimation for smartphone at ~10cm distance
    # Typical smartphone: 4032x3024 sensor, ~15mm focal length
    # At 10cm distance, 1mm ~ 10 pixels
    return 10.0


def create_heatmap(
    attention_map: np.ndarray,
    original_image: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """Create a heatmap overlay on the original image."""
    # Normalize attention map
    attention_norm = (attention_map - attention_map.min())
    if attention_norm.max() > 0:
        attention_norm = attention_norm / attention_norm.max()
    
    # Convert to uint8
    attention_uint8 = (attention_norm * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(attention_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Resize to match original
    if heatmap.shape[:2] != original_image.shape[:2]:
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
    # Blend
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
    
    return overlay


def draw_segmentation_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    line_width: int = 2
) -> np.ndarray:
    """Draw segmentation contour on image."""
    overlay = image.copy()
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(overlay, contours, -1, color, line_width)
    
    return overlay

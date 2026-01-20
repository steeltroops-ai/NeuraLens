"""
Dermatology Pipeline Lesion Segmenter

Lesion detection, segmentation, and geometry extraction.
"""

import logging
import time
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass

import numpy as np
import cv2

from ..config import SEGMENTATION_CONFIG
from ..schemas import (
    BoundingBox,
    LesionGeometry,
    SegmentationResult
)

logger = logging.getLogger(__name__)


class LesionDetector:
    """
    Detects lesion location using traditional CV methods.
    Falls back when DL model is unavailable.
    """
    
    def __init__(self, config=None):
        self.config = config or SEGMENTATION_CONFIG
    
    def detect(self, image: np.ndarray) -> Tuple[Optional[BoundingBox], float]:
        """Detect primary lesion using color-based segmentation."""
        # Convert to different color spaces
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Get initial segmentation using Otsu on L channel
        l_channel = lab[:, :, 0]
        _, binary = cv2.threshold(
            l_channel, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return None, 0.0
        
        # Filter and select primary lesion
        h, w = image.shape[:2]
        img_area = h * w
        center = (w // 2, h // 2)
        
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Size filter
            if area < self.config.min_lesion_ratio * img_area:
                continue
            if area > self.config.max_lesion_ratio * img_area:
                continue
            
            # Compute centroid
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Score based on size and centrality
            size_score = min(area / (0.1 * img_area), 1.0)
            dist_from_center = np.sqrt((cx - center[0])**2 + (cy - center[1])**2)
            max_dist = np.sqrt(center[0]**2 + center[1]**2)
            centrality_score = 1 - (dist_from_center / max_dist)
            
            score = size_score * 0.4 + centrality_score * 0.6
            
            if score > best_score:
                best_score = score
                best_contour = contour
        
        if best_contour is None:
            return None, 0.0
        
        # Get bounding box
        x, y, bw, bh = cv2.boundingRect(best_contour)
        
        bbox = BoundingBox(
            x=x, y=y, width=bw, height=bh,
            confidence=best_score
        )
        
        return bbox, best_score


class SemanticSegmenter:
    """
    Performs semantic segmentation using traditional methods.
    Can be extended with DL models.
    """
    
    def __init__(self, config=None):
        self.config = config or SEGMENTATION_CONFIG
    
    def segment(
        self, 
        image: np.ndarray, 
        bbox: Optional[BoundingBox] = None
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Segment lesion from image.
        
        Returns:
            - Binary mask
            - Probability map
            - Confidence score
        """
        h, w = image.shape[:2]
        
        # Create initial segmentation
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # Multi-threshold approach
        masks = []
        
        # Otsu thresholding on L
        _, otsu_mask = cv2.threshold(
            l_channel, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        masks.append(otsu_mask)
        
        # Adaptive thresholding
        adaptive_mask = cv2.adaptiveThreshold(
            l_channel, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 5
        )
        masks.append(adaptive_mask)
        
        # K-means segmentation on color
        kmeans_mask = self._kmeans_segment(image)
        masks.append(kmeans_mask)
        
        # Combine masks with voting
        combined = np.zeros((h, w), dtype=np.float32)
        for mask in masks:
            combined += mask.astype(np.float32) / 255.0
        
        probability_map = combined / len(masks)
        
        # Generate binary mask
        binary_mask = (probability_map > self.config.segmentation_threshold).astype(np.uint8) * 255
        
        # Post-processing
        binary_mask = self._postprocess_mask(binary_mask, bbox)
        
        # Calculate confidence
        confidence = self._calculate_confidence(masks, binary_mask)
        
        return binary_mask, probability_map, confidence
    
    def _kmeans_segment(self, image: np.ndarray, k: int = 3) -> np.ndarray:
        """Segment using K-means clustering."""
        # Reshape for K-means
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        # Find darkest cluster (likely lesion)
        center_brightness = np.mean(centers, axis=1)
        darkest_cluster = np.argmin(center_brightness)
        
        # Create mask
        labels = labels.reshape(image.shape[:2])
        mask = (labels == darkest_cluster).astype(np.uint8) * 255
        
        return mask
    
    def _postprocess_mask(
        self, 
        mask: np.ndarray, 
        bbox: Optional[BoundingBox]
    ) -> np.ndarray:
        """Clean up segmentation mask."""
        # Fill holes
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return mask
        
        # Fill all contours
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
        
        # Keep only largest component
        largest = max(contours, key=cv2.contourArea)
        final = np.zeros_like(mask)
        cv2.drawContours(final, [largest], -1, 255, cv2.FILLED)
        
        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel)
        final = cv2.morphologyEx(final, cv2.MORPH_OPEN, kernel)
        
        return final
    
    def _calculate_confidence(
        self, 
        masks: List[np.ndarray], 
        final_mask: np.ndarray
    ) -> float:
        """Calculate segmentation confidence from mask agreement."""
        if len(masks) == 0:
            return 0.0
        
        # IoU between each mask and final
        ious = []
        for mask in masks:
            mask_binary = mask > 127
            final_binary = final_mask > 127
            
            intersection = np.sum(mask_binary & final_binary)
            union = np.sum(mask_binary | final_binary)
            
            if union > 0:
                ious.append(intersection / union)
        
        return float(np.mean(ious)) if ious else 0.0


class GeometryExtractor:
    """
    Extracts geometric properties from lesion mask.
    """
    
    def __init__(self, pixels_per_mm: float = 10.0):
        self.pixels_per_mm = pixels_per_mm
    
    def extract(
        self, 
        mask: np.ndarray,
        calibration: Optional[float] = None
    ) -> Optional[LesionGeometry]:
        """Extract geometry from mask."""
        ppm = calibration or self.pixels_per_mm
        
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        
        if len(contours) == 0:
            return None
        
        contour = max(contours, key=cv2.contourArea)
        
        # Basic measurements
        area_pixels = cv2.contourArea(contour)
        perimeter_pixels = cv2.arcLength(contour, True)
        
        # Bounding box
        x, y, bw, bh = cv2.boundingRect(contour)
        bbox = BoundingBox(x=x, y=y, width=bw, height=bh)
        
        # Equivalent diameter
        diameter_pixels = np.sqrt(4 * area_pixels / np.pi)
        
        # Fitted ellipse for axes
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (major, minor), angle = ellipse
            major_axis_pixels = max(major, minor)
            minor_axis_pixels = min(major, minor)
        else:
            cx, cy = x + bw // 2, y + bh // 2
            major_axis_pixels = max(bw, bh)
            minor_axis_pixels = min(bw, bh)
            angle = 0
        
        # Convert to mm
        area_mm2 = area_pixels / (ppm ** 2)
        diameter_mm = diameter_pixels / ppm
        major_axis_mm = major_axis_pixels / ppm
        minor_axis_mm = minor_axis_pixels / ppm
        
        # Shape features
        circularity = 4 * np.pi * area_pixels / (perimeter_pixels ** 2) if perimeter_pixels > 0 else 0
        
        # Asymmetry using moment-based analysis
        asymmetry_index = self._compute_asymmetry(mask, contour)
        
        # Border irregularity
        border_irregularity = self._compute_border_irregularity(contour)
        
        # Solidity (convex hull ratio)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area_pixels / hull_area if hull_area > 0 else 0
        
        # Aspect ratio
        aspect_ratio = major_axis_pixels / minor_axis_pixels if minor_axis_pixels > 0 else 1
        
        return LesionGeometry(
            area_pixels=area_pixels,
            perimeter_pixels=perimeter_pixels,
            diameter_pixels=diameter_pixels,
            area_mm2=area_mm2,
            diameter_mm=diameter_mm,
            major_axis_mm=major_axis_mm,
            minor_axis_mm=minor_axis_mm,
            center=(int(cx), int(cy)),
            bounding_box=bbox,
            circularity=circularity,
            asymmetry_index=asymmetry_index,
            border_irregularity=border_irregularity,
            solidity=solidity,
            aspect_ratio=aspect_ratio,
            orientation=angle
        )
    
    def _compute_asymmetry(
        self, 
        mask: np.ndarray, 
        contour: np.ndarray
    ) -> float:
        """Compute asymmetry index."""
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return 0.0
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Horizontal asymmetry
        left = mask[:, :cx]
        right = cv2.flip(mask[:, cx:], 1)
        
        # Align sizes
        min_width = min(left.shape[1], right.shape[1])
        left = left[:, -min_width:]
        right = right[:, :min_width]
        
        h_intersection = np.sum((left > 0) & (right > 0))
        h_union = np.sum((left > 0) | (right > 0))
        h_asym = 1 - (h_intersection / h_union) if h_union > 0 else 0
        
        # Vertical asymmetry
        top = mask[:cy, :]
        bottom = cv2.flip(mask[cy:, :], 0)
        
        min_height = min(top.shape[0], bottom.shape[0])
        top = top[-min_height:, :]
        bottom = bottom[:min_height, :]
        
        v_intersection = np.sum((top > 0) & (bottom > 0))
        v_union = np.sum((top > 0) | (bottom > 0))
        v_asym = 1 - (v_intersection / v_union) if v_union > 0 else 0
        
        return (h_asym + v_asym) / 2
    
    def _compute_border_irregularity(self, contour: np.ndarray) -> float:
        """Compute border irregularity."""
        if len(contour) < 10:
            return 0.0
        
        # Fit polygon and compare
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Compare perimeter of approximation vs original
        orig_perim = cv2.arcLength(contour, True)
        approx_perim = cv2.arcLength(approx, True)
        
        irregularity = 1 - (approx_perim / orig_perim) if orig_perim > 0 else 0
        
        # Also consider convexity defects
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)
        
        if defects is not None:
            significant_defects = sum(1 for d in defects if d[0][3] > 500)
            # Normalize by contour length
            defect_score = significant_defects / (len(contour) / 10)
            irregularity = (irregularity + min(defect_score, 1.0)) / 2
        
        return min(irregularity, 1.0)


class DermatologySegmenter:
    """
    Complete segmentation pipeline for dermatology images.
    """
    
    def __init__(self, config=None):
        self.config = config or SEGMENTATION_CONFIG
        self.detector = LesionDetector(config)
        self.segmenter = SemanticSegmenter(config)
        self.geometry_extractor = GeometryExtractor(
            self.config.default_pixels_per_mm
        )
    
    def segment(
        self, 
        image: np.ndarray,
        calibration: Optional[float] = None
    ) -> SegmentationResult:
        """Run complete segmentation pipeline."""
        start_time = time.time()
        warnings = []
        
        # Step 1: Detection
        bbox, detection_confidence = self.detector.detect(image)
        
        if bbox is None:
            return SegmentationResult(
                detected=False,
                mask=None,
                probability_map=None,
                confidence=0.0,
                geometry=None,
                validation_passed=False,
                warnings=["No lesion detected"]
            )
        
        # Step 2: Segmentation
        mask, prob_map, seg_confidence = self.segmenter.segment(image, bbox)
        
        if np.sum(mask > 0) == 0:
            return SegmentationResult(
                detected=True,
                mask=None,
                probability_map=prob_map,
                confidence=detection_confidence,
                geometry=None,
                validation_passed=False,
                warnings=["Segmentation produced empty mask"]
            )
        
        # Step 3: Geometry extraction
        geometry = self.geometry_extractor.extract(mask, calibration)
        
        if geometry is None:
            return SegmentationResult(
                detected=True,
                mask=mask,
                probability_map=prob_map,
                confidence=seg_confidence,
                geometry=None,
                validation_passed=False,
                warnings=["Could not extract geometry"]
            )
        
        # Validation
        validation_passed, val_warnings = self._validate_segmentation(geometry, image.shape)
        warnings.extend(val_warnings)
        
        # Overall confidence
        confidence = (detection_confidence + seg_confidence) / 2
        
        if seg_confidence < self.config.segmentation_threshold:
            warnings.append("Low segmentation confidence")
        
        return SegmentationResult(
            detected=True,
            mask=mask,
            probability_map=prob_map,
            confidence=confidence,
            geometry=geometry,
            validation_passed=validation_passed,
            warnings=warnings
        )
    
    def _validate_segmentation(
        self, 
        geometry: LesionGeometry, 
        image_shape: Tuple[int, ...]
    ) -> Tuple[bool, List[str]]:
        """Validate segmentation results."""
        h, w = image_shape[:2]
        img_area = h * w
        warnings = []
        valid = True
        
        # Size validation
        lesion_ratio = geometry.area_pixels / img_area
        if lesion_ratio < self.config.min_lesion_ratio:
            warnings.append("Lesion too small")
            valid = False
        if lesion_ratio > self.config.max_lesion_ratio:
            warnings.append("Lesion too large")
            valid = False
        
        # Diameter validation
        if geometry.diameter_mm < self.config.min_diameter_mm:
            warnings.append(f"Lesion diameter {geometry.diameter_mm:.1f}mm too small")
        if geometry.diameter_mm > self.config.max_diameter_mm:
            warnings.append(f"Lesion diameter {geometry.diameter_mm:.1f}mm unusually large")
        
        # Circularity validation
        if geometry.circularity < self.config.min_circularity:
            warnings.append("Unusual lesion shape")
        
        return valid, warnings

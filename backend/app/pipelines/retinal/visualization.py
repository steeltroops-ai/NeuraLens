"""
Visualization Service for Retinal Analysis

Generates visual outputs for retinal analysis:
- Vessel segmentation overlays (Requirement 6.1)
- Attention heatmaps (Requirements 6.2, 6.3)
- Artery/vein color-coding (Requirement 6.4)
- Optic disc measurement overlays (Requirement 6.5)
- Risk score gauge visualization (Requirement 6.6)
- Biomarker comparison charts (Requirement 6.7)
- Trend analysis graphs (Requirement 6.8)
- Image zoom and pan support (Requirement 6.9)
- Scale bar overlays (Requirement 6.10)

Author: NeuraLens Team
"""

import io
import logging
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import base64

logger = logging.getLogger(__name__)


class ColorPalette:
    """Color palette for visualizations"""
    # Risk category colors
    RISK_MINIMAL = (34, 197, 94)      # Green #22c55e
    RISK_LOW = (132, 204, 22)         # Lime #84cc16
    RISK_MODERATE = (234, 179, 8)     # Yellow #eab308
    RISK_ELEVATED = (249, 115, 22)    # Orange #f97316
    RISK_HIGH = (239, 68, 68)         # Red #ef4444
    RISK_CRITICAL = (153, 27, 27)     # Dark red #991b1b
    
    # Vessel colors
    ARTERY = (220, 50, 50)            # Bright red
    VEIN = (50, 50, 180)              # Blue
    VESSEL_GENERAL = (100, 200, 100)  # Green for general vessels
    
    # Anatomy colors
    OPTIC_DISC = (255, 200, 0)        # Gold
    MACULA = (200, 100, 255)          # Purple
    
    # Heatmap gradient (blue to red)
    HEATMAP_COLD = (0, 0, 255)        # Blue
    HEATMAP_WARM = (255, 255, 0)      # Yellow
    HEATMAP_HOT = (255, 0, 0)         # Red
    
    @staticmethod
    def get_risk_color(category: str) -> Tuple[int, int, int]:
        """Get color for risk category"""
        colors = {
            "minimal": ColorPalette.RISK_MINIMAL,
            "low": ColorPalette.RISK_LOW,
            "moderate": ColorPalette.RISK_MODERATE,
            "elevated": ColorPalette.RISK_ELEVATED,
            "high": ColorPalette.RISK_HIGH,
            "critical": ColorPalette.RISK_CRITICAL,
        }
        return colors.get(category.lower(), (128, 128, 128))


@dataclass
class VisualizationConfig:
    """Configuration for visualization generation"""
    output_size: Tuple[int, int] = (800, 800)
    overlay_opacity: float = 0.5
    vessel_line_width: int = 2
    annotation_font_size: int = 14
    scale_bar_length_mm: float = 1.0
    heatmap_opacity: float = 0.6


class RetinalVisualizationService:
    """
    Service for generating retinal analysis visualizations.
    
    Generates:
    - Vessel segmentation overlays
    - Attention heatmaps
    - Measurement annotations
    - Risk gauge visualizations
    - Trend charts
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
    
    def generate_vessel_overlay(
        self,
        original_image: np.ndarray,
        vessel_mask: np.ndarray,
        artery_mask: Optional[np.ndarray] = None,
        vein_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate vessel segmentation overlay on original image.
        
        Requirement 6.1: Display vessel segmentation overlay
        Requirement 6.4: Color-code arteries (red) and veins (blue)
        
        Args:
            original_image: Original fundus image (BGR)
            vessel_mask: Binary vessel segmentation mask
            artery_mask: Optional mask for arteries
            vein_mask: Optional mask for veins
            
        Returns:
            Image with vessel overlay (BGR)
        """
        # Resize masks if needed
        if vessel_mask.shape[:2] != original_image.shape[:2]:
            vessel_mask = cv2.resize(
                vessel_mask, 
                (original_image.shape[1], original_image.shape[0])
            )
        
        # Create overlay
        overlay = original_image.copy()
        
        if artery_mask is not None and vein_mask is not None:
            # Separate artery and vein coloring
            if artery_mask.shape[:2] != original_image.shape[:2]:
                artery_mask = cv2.resize(
                    artery_mask, 
                    (original_image.shape[1], original_image.shape[0])
                )
                vein_mask = cv2.resize(
                    vein_mask, 
                    (original_image.shape[1], original_image.shape[0])
                )
            
            # Apply artery color (red)
            overlay[artery_mask > 0] = ColorPalette.ARTERY
            # Apply vein color (blue)
            overlay[vein_mask > 0] = ColorPalette.VEIN
        else:
            # General vessel coloring
            overlay[vessel_mask > 0] = ColorPalette.VESSEL_GENERAL
        
        # Blend with original
        result = cv2.addWeighted(
            original_image, 
            1 - self.config.overlay_opacity,
            overlay, 
            self.config.overlay_opacity, 
            0
        )
        
        return result
    
    def generate_heatmap(
        self,
        original_image: np.ndarray,
        attention_map: np.ndarray,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Generate attention heatmap overlay.
        
        Requirement 6.2: Show attention heatmaps
        Requirement 6.3: Use blue-to-red gradient
        
        Args:
            original_image: Original fundus image (BGR)
            attention_map: Attention/activation map (0-1 float or 0-255 uint8)
            colormap: OpenCV colormap to use
            
        Returns:
            Image with heatmap overlay (BGR)
        """
        # Normalize attention map
        if attention_map.dtype != np.uint8:
            attention_normalized = (attention_map * 255).astype(np.uint8)
        else:
            attention_normalized = attention_map
        
        # Resize if needed
        if attention_normalized.shape[:2] != original_image.shape[:2]:
            attention_normalized = cv2.resize(
                attention_normalized,
                (original_image.shape[1], original_image.shape[0])
            )
        
        # Apply colormap (blue to red gradient)
        heatmap_colored = cv2.applyColorMap(attention_normalized, colormap)
        
        # Blend with original
        result = cv2.addWeighted(
            original_image,
            1 - self.config.heatmap_opacity,
            heatmap_colored,
            self.config.heatmap_opacity,
            0
        )
        
        return result
    
    def generate_measurement_overlay(
        self,
        original_image: np.ndarray,
        optic_disc_center: Optional[Tuple[int, int]] = None,
        optic_disc_radius: Optional[int] = None,
        cup_radius: Optional[int] = None,
        macula_center: Optional[Tuple[int, int]] = None,
        cup_to_disc_ratio: Optional[float] = None,
        scale_pixels_per_mm: float = 100.0
    ) -> np.ndarray:
        """
        Generate measurement overlay with anatomical annotations.
        
        Requirement 6.5: Overlay optic disc measurements
        Requirement 6.10: Include scale bars
        
        Args:
            original_image: Original fundus image (BGR)
            optic_disc_center: Center coordinates of optic disc
            optic_disc_radius: Radius of optic disc in pixels
            cup_radius: Radius of optic cup in pixels
            macula_center: Center coordinates of macula
            cup_to_disc_ratio: Cup-to-disc ratio value
            scale_pixels_per_mm: Pixels per millimeter for scale bar
            
        Returns:
            Image with measurement overlays (BGR)
        """
        result = original_image.copy()
        height, width = result.shape[:2]
        
        # Draw optic disc if detected
        if optic_disc_center and optic_disc_radius:
            # Outer disc circle
            cv2.circle(
                result,
                optic_disc_center,
                optic_disc_radius,
                ColorPalette.OPTIC_DISC,
                2
            )
            
            # Cup circle if provided
            if cup_radius:
                cv2.circle(
                    result,
                    optic_disc_center,
                    cup_radius,
                    (255, 100, 0),  # Orange for cup
                    2
                )
            
            # Add CDR label
            if cup_to_disc_ratio:
                label = f"CDR: {cup_to_disc_ratio:.2f}"
                label_pos = (optic_disc_center[0] - 30, optic_disc_center[1] + optic_disc_radius + 20)
                cv2.putText(
                    result,
                    label,
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    ColorPalette.OPTIC_DISC,
                    1
                )
        
        # Draw macula if detected
        if macula_center:
            cv2.circle(result, macula_center, 20, ColorPalette.MACULA, 2)
            cv2.putText(
                result,
                "Macula",
                (macula_center[0] - 25, macula_center[1] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                ColorPalette.MACULA,
                1
            )
        
        # Add scale bar (Requirement 6.10)
        scale_bar_pixels = int(self.config.scale_bar_length_mm * scale_pixels_per_mm)
        scale_bar_y = height - 30
        scale_bar_x_start = 20
        scale_bar_x_end = scale_bar_x_start + scale_bar_pixels
        
        # Draw scale bar
        cv2.line(
            result,
            (scale_bar_x_start, scale_bar_y),
            (scale_bar_x_end, scale_bar_y),
            (255, 255, 255),
            2
        )
        
        # Scale bar labels
        cv2.line(result, (scale_bar_x_start, scale_bar_y - 5), (scale_bar_x_start, scale_bar_y + 5), (255, 255, 255), 2)
        cv2.line(result, (scale_bar_x_end, scale_bar_y - 5), (scale_bar_x_end, scale_bar_y + 5), (255, 255, 255), 2)
        
        cv2.putText(
            result,
            f"{self.config.scale_bar_length_mm} mm",
            (scale_bar_x_start, scale_bar_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )
        
        return result
    
    def generate_risk_gauge(
        self,
        risk_score: float,
        risk_category: str,
        width: int = 400,
        height: int = 250
    ) -> np.ndarray:
        """
        Generate risk score gauge visualization.
        
        Requirement 6.6: Present risk score with color-coded gauge
        
        Args:
            risk_score: Risk score (0-100)
            risk_category: Risk category string
            width: Output image width
            height: Output image height
            
        Returns:
            Gauge visualization (BGR)
        """
        # Create blank canvas
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        center_x = width // 2
        center_y = height - 50
        radius = min(width // 2 - 20, height - 70)
        
        # Draw gauge background arc
        start_angle = 180
        end_angle = 360
        
        # Draw colored segments
        segment_angles = [
            (180, 210, ColorPalette.RISK_MINIMAL),    # 0-25
            (210, 228, ColorPalette.RISK_LOW),        # 25-40
            (228, 252, ColorPalette.RISK_MODERATE),   # 40-55
            (252, 282, ColorPalette.RISK_ELEVATED),   # 55-70
            (282, 318, ColorPalette.RISK_HIGH),       # 70-85
            (318, 360, ColorPalette.RISK_CRITICAL),   # 85-100
        ]
        
        for start, end, color in segment_angles:
            cv2.ellipse(
                img,
                (center_x, center_y),
                (radius, radius),
                0,
                start,
                end,
                color[::-1],  # BGR
                20
            )
        
        # Draw needle
        needle_angle = 180 + (risk_score / 100) * 180
        needle_rad = np.radians(needle_angle)
        needle_length = radius - 30
        needle_end_x = int(center_x + needle_length * np.cos(needle_rad))
        needle_end_y = int(center_y + needle_length * np.sin(needle_rad))
        
        cv2.line(img, (center_x, center_y), (needle_end_x, needle_end_y), (50, 50, 50), 3)
        cv2.circle(img, (center_x, center_y), 10, (50, 50, 50), -1)
        
        # Add score text
        score_text = f"{risk_score:.0f}"
        text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x = center_x - text_size[0] // 2
        text_y = center_y - 40
        cv2.putText(img, score_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (50, 50, 50), 3)
        
        # Add category text
        category_color = ColorPalette.get_risk_color(risk_category)
        cat_text = risk_category.upper()
        cat_size = cv2.getTextSize(cat_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cat_x = center_x - cat_size[0] // 2
        cv2.putText(
            img, 
            cat_text, 
            (cat_x, height - 15), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            category_color[::-1], 
            2
        )
        
        return img
    
    def generate_biomarker_chart(
        self,
        biomarkers: Dict[str, float],
        reference_ranges: Dict[str, Tuple[float, float]],
        width: int = 600,
        height: int = 400
    ) -> np.ndarray:
        """
        Generate biomarker comparison bar chart.
        
        Requirement 6.7: Display biomarker comparison charts
        
        Args:
            biomarkers: Dictionary of biomarker name -> value
            reference_ranges: Dictionary of biomarker name -> (min, max)
            width: Chart width
            height: Chart height
            
        Returns:
            Chart visualization (BGR)
        """
        # Create blank canvas
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        n_bars = len(biomarkers)
        if n_bars == 0:
            return img
        
        margin = 60
        bar_area_width = width - 2 * margin
        bar_area_height = height - 2 * margin
        bar_height = bar_area_height // (n_bars + 1)
        bar_spacing = bar_height // 4
        
        for i, (name, value) in enumerate(biomarkers.items()):
            y = margin + i * (bar_height + bar_spacing)
            
            # Get reference range
            ref_min, ref_max = reference_ranges.get(name, (0, 100))
            
            # Calculate bar width (normalized to 0-100%)
            max_val = ref_max * 1.5  # Allow some overflow
            bar_width = int((value / max_val) * bar_area_width)
            bar_width = min(bar_width, bar_area_width)
            
            # Determine color based on reference range
            if ref_min <= value <= ref_max:
                bar_color = (100, 200, 100)  # Green - normal
            elif value < ref_min:
                bar_color = (200, 200, 0)    # Yellow - low
            else:
                bar_color = (50, 50, 200)    # Red - high
            
            # Draw bar
            cv2.rectangle(
                img,
                (margin, y),
                (margin + bar_width, y + bar_height - bar_spacing),
                bar_color,
                -1
            )
            
            # Draw reference range indicator
            ref_start = int((ref_min / max_val) * bar_area_width)
            ref_end = int((ref_max / max_val) * bar_area_width)
            cv2.line(
                img,
                (margin + ref_start, y - 5),
                (margin + ref_start, y + bar_height - bar_spacing + 5),
                (0, 150, 0),
                2
            )
            cv2.line(
                img,
                (margin + ref_end, y - 5),
                (margin + ref_end, y + bar_height - bar_spacing + 5),
                (0, 150, 0),
                2
            )
            
            # Add label
            label = f"{name}: {value:.2f}"
            cv2.putText(
                img,
                label,
                (margin + 5, y + bar_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1
            )
        
        return img
    
    def generate_trend_chart(
        self,
        dates: List[str],
        values: List[float],
        label: str = "Risk Score",
        width: int = 600,
        height: int = 300
    ) -> np.ndarray:
        """
        Generate trend analysis line chart.
        
        Requirement 6.8: Show trend analysis graphs
        
        Args:
            dates: List of date strings
            values: List of corresponding values
            label: Chart label
            width: Chart width
            height: Chart height
            
        Returns:
            Trend chart visualization (BGR)
        """
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        if len(values) < 2:
            cv2.putText(img, "Insufficient data for trend", (50, height // 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            return img
        
        margin = 50
        chart_width = width - 2 * margin
        chart_height = height - 2 * margin
        
        # Normalize values to chart area
        min_val = min(values)
        max_val = max(values)
        val_range = max_val - min_val if max_val != min_val else 1
        
        # Generate points
        points = []
        for i, val in enumerate(values):
            x = margin + int((i / (len(values) - 1)) * chart_width)
            y = height - margin - int(((val - min_val) / val_range) * chart_height)
            points.append((x, y))
        
        # Draw grid
        for i in range(5):
            y = margin + int(i * chart_height / 4)
            cv2.line(img, (margin, y), (width - margin, y), (220, 220, 220), 1)
        
        # Draw line
        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i + 1], (66, 133, 244), 2)
        
        # Draw points
        for point in points:
            cv2.circle(img, point, 5, (66, 133, 244), -1)
        
        # Add label
        cv2.putText(img, label, (margin, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
        
        return img
    
    def image_to_base64(self, image: np.ndarray, format: str = "PNG") -> str:
        """Convert OpenCV image to base64 string for embedding in responses."""
        # Convert BGR to RGB for PIL
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        pil_image = Image.fromarray(image_rgb)
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def image_to_bytes(self, image: np.ndarray, format: str = ".png") -> bytes:
        """Convert OpenCV image to bytes."""
        _, buffer = cv2.imencode(format, image)
        return buffer.tobytes()


# Singleton instance
visualization_service = RetinalVisualizationService()

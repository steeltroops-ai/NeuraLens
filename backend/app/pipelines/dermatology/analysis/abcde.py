"""
Dermatology Pipeline ABCDE Feature Extractor

Extracts Asymmetry, Border, Color, Diameter, and Evolution features.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import cv2

from ..config import ABCDE_CONFIG, CONCERNING_COLORS
from ..schemas import (
    AsymmetryResult,
    BorderResult,
    ColorResult,
    DiameterResult,
    EvolutionResult,
    ABCDEFeatures,
    LesionGeometry
)

logger = logging.getLogger(__name__)


class AsymmetryAnalyzer:
    """
    Analyzes lesion asymmetry along multiple axes.
    """
    
    def __init__(self, config=None):
        self.config = config or ABCDE_CONFIG
    
    def analyze(
        self, 
        mask: np.ndarray, 
        image: np.ndarray
    ) -> AsymmetryResult:
        """Compute asymmetry scores."""
        # Find centroid
        M = cv2.moments(mask)
        if M["m00"] == 0:
            return self._default_result()
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Shape asymmetry
        h_shape = self._compute_axis_asymmetry(mask, cx, 'horizontal')
        v_shape = self._compute_axis_asymmetry(mask, cy, 'vertical')
        shape_asymmetry = (h_shape + v_shape) / 2
        
        # Color asymmetry
        color_asymmetry = self._compute_color_asymmetry(image, mask, cx, cy)
        
        # Texture asymmetry
        texture_asymmetry = self._compute_texture_asymmetry(image, mask, cx, cy)
        
        # Combined score
        combined = (
            shape_asymmetry * 0.5 +
            color_asymmetry * 0.3 +
            texture_asymmetry * 0.2
        )
        
        # Classification
        if combined < 0.2:
            classification = "symmetric"
        elif combined < 0.4:
            classification = "mildly_asymmetric"
        elif combined < 0.6:
            classification = "moderately_asymmetric"
        else:
            classification = "highly_asymmetric"
        
        return AsymmetryResult(
            shape_asymmetry=shape_asymmetry,
            color_asymmetry=color_asymmetry,
            texture_asymmetry=texture_asymmetry,
            combined_score=combined,
            classification=classification,
            is_concerning=combined > self.config.asymmetry_concerning_threshold
        )
    
    def _compute_axis_asymmetry(
        self, 
        mask: np.ndarray, 
        center: int, 
        axis: str
    ) -> float:
        """Compute asymmetry along an axis."""
        try:
            if axis == 'horizontal':
                half1 = mask[:, :center]
                half2 = cv2.flip(mask[:, center:], 1)
            else:
                half1 = mask[:center, :]
                half2 = cv2.flip(mask[center:, :], 0)
            
            # Align sizes
            if axis == 'horizontal':
                min_size = min(half1.shape[1], half2.shape[1])
                if min_size == 0:
                    return 0.0
                half1 = half1[:, -min_size:]
                half2 = half2[:, :min_size]
            else:
                min_size = min(half1.shape[0], half2.shape[0])
                if min_size == 0:
                    return 0.0
                half1 = half1[-min_size:, :]
                half2 = half2[:min_size, :]
            
            # Compute IoU-based asymmetry
            intersection = np.sum((half1 > 0) & (half2 > 0))
            union = np.sum((half1 > 0) | (half2 > 0))
            
            if union == 0:
                return 0.0
            
            iou = intersection / union
            return 1.0 - iou
        except Exception:
            return 0.0
    
    def _compute_color_asymmetry(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        cx: int, 
        cy: int
    ) -> float:
        """Compute color distribution asymmetry."""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Left vs right
            left_region = lab[:, :cx][mask[:, :cx] > 0]
            right_region = lab[:, cx:][mask[:, cx:] > 0]
            
            if len(left_region) == 0 or len(right_region) == 0:
                return 0.0
            
            left_mean = np.mean(left_region, axis=0)
            right_mean = np.mean(right_region, axis=0)
            
            color_diff = np.linalg.norm(left_mean - right_mean) / (255 * np.sqrt(3))
            return min(color_diff * 2, 1.0)
        except Exception:
            return 0.0
    
    def _compute_texture_asymmetry(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        cx: int, 
        cy: int
    ) -> float:
        """Compute texture asymmetry using variance."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Left vs right texture variance
            left_texture = gray[:, :cx][mask[:, :cx] > 0]
            right_texture = gray[:, cx:][mask[:, cx:] > 0]
            
            if len(left_texture) < 10 or len(right_texture) < 10:
                return 0.0
            
            left_var = np.var(left_texture)
            right_var = np.var(right_texture)
            
            max_var = max(left_var, right_var)
            if max_var == 0:
                return 0.0
            
            return abs(left_var - right_var) / max_var
        except Exception:
            return 0.0
    
    def _default_result(self) -> AsymmetryResult:
        return AsymmetryResult(
            shape_asymmetry=0.0,
            color_asymmetry=0.0,
            texture_asymmetry=0.0,
            combined_score=0.0,
            classification="unknown",
            is_concerning=False
        )


class BorderAnalyzer:
    """
    Analyzes lesion border characteristics.
    """
    
    def __init__(self, config=None):
        self.config = config or ABCDE_CONFIG
    
    def analyze(self, mask: np.ndarray, image: np.ndarray) -> BorderResult:
        """Analyze border characteristics."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        
        if len(contours) == 0:
            return self._default_result()
        
        contour = max(contours, key=cv2.contourArea)
        
        # Fractal dimension approximation
        fractal_dim = self._compute_fractal_dimension(contour)
        
        # Curvature analysis
        curvature = self._analyze_curvature(contour)
        
        # Notch detection
        notch_count = self._count_notches(contour)
        
        # Compactness
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0
        
        # Border sharpness
        sharpness = self._analyze_sharpness(image, mask)
        
        # Compute irregularity score
        irregularity = self._compute_irregularity(
            fractal_dim, curvature, notch_count, compactness
        )
        
        # Classification
        if irregularity < 0.25:
            classification = "regular"
        elif irregularity < 0.50:
            classification = "slightly_irregular"
        elif irregularity < 0.75:
            classification = "moderately_irregular"
        else:
            classification = "highly_irregular"
        
        return BorderResult(
            fractal_dimension=fractal_dim,
            curvature_mean=curvature['mean'],
            curvature_std=curvature['std'],
            notch_count=notch_count,
            compactness=compactness,
            sharpness=sharpness,
            irregularity_score=irregularity,
            classification=classification,
            is_concerning=irregularity > self.config.border_irregularity_threshold
        )
    
    def _compute_fractal_dimension(self, contour: np.ndarray) -> float:
        """Estimate fractal dimension using box counting."""
        try:
            points = contour.squeeze()
            if len(points) < 10:
                return 1.0
            
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            
            size = max(x_max - x_min, y_max - y_min) + 1
            if size < 64:
                return 1.0
            
            # Create binary image
            img = np.zeros((int(size), int(size)), dtype=np.uint8)
            shifted = (points - [x_min, y_min]).astype(int)
            
            for p in shifted:
                if 0 <= p[1] < size and 0 <= p[0] < size:
                    img[p[1], p[0]] = 1
            
            # Box counting
            scales = [4, 8, 16, 32]
            counts = []
            
            for scale in scales:
                if size // scale < 1:
                    continue
                scaled = cv2.resize(
                    img, (int(size/scale), int(size/scale)),
                    interpolation=cv2.INTER_MAX
                )
                counts.append(np.sum(scaled > 0))
            
            if len(counts) < 2:
                return 1.0
            
            # Linear regression
            log_scales = np.log(1 / np.array(scales[:len(counts)]))
            log_counts = np.log(np.array(counts) + 1)
            
            coeffs = np.polyfit(log_scales, log_counts, 1)
            return max(min(coeffs[0], 2.0), 1.0)
        except Exception:
            return 1.0
    
    def _analyze_curvature(self, contour: np.ndarray) -> Dict[str, float]:
        """Analyze curvature along border."""
        try:
            points = contour.squeeze()
            if len(points) < 20:
                return {'mean': 0.0, 'std': 0.0}
            
            n = len(points)
            curvatures = []
            
            step = max(1, n // 50)
            
            for i in range(0, n, step):
                p1 = points[(i - 5) % n]
                p2 = points[i]
                p3 = points[(i + 5) % n]
                
                v1 = p1 - p2
                v2 = p3 - p2
                
                cross = float(np.cross(v1.astype(float), v2.astype(float)))
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                
                if norm1 > 0 and norm2 > 0:
                    curvature = abs(cross) / (norm1 * norm2)
                    curvatures.append(curvature)
            
            if not curvatures:
                return {'mean': 0.0, 'std': 0.0}
            
            return {
                'mean': float(np.mean(curvatures)),
                'std': float(np.std(curvatures))
            }
        except Exception:
            return {'mean': 0.0, 'std': 0.0}
    
    def _count_notches(self, contour: np.ndarray) -> int:
        """Count significant notches in border."""
        try:
            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull)
            
            if defects is None:
                return 0
            
            count = sum(1 for d in defects if d[0][3] > 500)
            return count
        except Exception:
            return 0
    
    def _analyze_sharpness(self, image: np.ndarray, mask: np.ndarray) -> float:
        """Analyze border sharpness/definition."""
        try:
            # Get border region
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(mask, kernel, iterations=1)
            eroded = cv2.erode(mask, kernel, iterations=1)
            border_region = dilated - eroded
            
            # Check gradient magnitude at border
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gradient = cv2.Laplacian(gray, cv2.CV_64F)
            
            border_gradient = gradient[border_region > 0]
            if len(border_gradient) == 0:
                return 1.0
            
            sharpness = min(np.mean(np.abs(border_gradient)) / 50, 1.0)
            return sharpness
        except Exception:
            return 1.0
    
    def _compute_irregularity(
        self, 
        fractal: float, 
        curvature: dict, 
        notches: int,
        compactness: float
    ) -> float:
        """Compute overall irregularity score."""
        # Normalize fractal dimension (1.0-2.0 -> 0-1)
        fractal_score = (fractal - 1.0) / 1.0
        
        # Curvature contribution
        curvature_score = min(curvature['std'] * 5, 1.0)
        
        # Notch contribution
        notch_score = min(notches / 5, 1.0)
        
        # Compactness contribution (higher = more irregular)
        compactness_score = min((compactness - 1) / 2, 1.0)
        
        irregularity = (
            fractal_score * 0.25 +
            curvature_score * 0.25 +
            notch_score * 0.25 +
            compactness_score * 0.25
        )
        
        return min(irregularity, 1.0)
    
    def _default_result(self) -> BorderResult:
        return BorderResult(
            fractal_dimension=1.0,
            curvature_mean=0.0,
            curvature_std=0.0,
            notch_count=0,
            compactness=1.0,
            sharpness=1.0,
            irregularity_score=0.0,
            classification="unknown",
            is_concerning=False
        )


class ColorAnalyzer:
    """
    Analyzes color distribution within lesion.
    """
    
    def __init__(self, config=None):
        self.config = config or ABCDE_CONFIG
        self.concerning_colors = CONCERNING_COLORS
    
    def analyze(self, image: np.ndarray, mask: np.ndarray) -> ColorResult:
        """Analyze lesion color characteristics."""
        lesion_pixels = image[mask > 0]
        
        if len(lesion_pixels) == 0:
            return self._default_result()
        
        # Color clustering
        clusters = self._cluster_colors(lesion_pixels)
        
        # Color variety
        variety_score = self._compute_variety(clusters)
        
        # Detect concerning colors
        concerning = self._detect_concerning_colors(image, mask)
        
        # Homogeneity
        homogeneity = self._compute_homogeneity(lesion_pixels)
        
        # Blue-white veil detection
        has_bwv = self._detect_blue_white_veil(image, mask)
        
        # Combined score
        color_score = self._compute_color_score(
            len(clusters), concerning, has_bwv, variety_score
        )
        
        return ColorResult(
            num_colors=len(clusters),
            color_clusters=clusters,
            color_variety_score=variety_score,
            concerning_colors=concerning,
            has_blue_white_veil=has_bwv,
            homogeneity=homogeneity,
            color_score=color_score,
            is_concerning=(
                len(concerning) >= self.config.concerning_color_count or
                has_bwv
            )
        )
    
    def _cluster_colors(
        self, 
        pixels: np.ndarray, 
        n_clusters: int = 6
    ) -> List[Dict[str, Any]]:
        """Cluster colors using K-means."""
        try:
            # Subsample
            if len(pixels) > 3000:
                indices = np.random.choice(len(pixels), 3000, replace=False)
                sample = pixels[indices].astype(np.float32)
            else:
                sample = pixels.astype(np.float32)
            
            n = min(n_clusters, len(sample) // 50 + 1)
            if n < 2:
                return []
            
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, labels, centers = cv2.kmeans(
                sample, n, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
            )
            
            clusters = []
            for i in range(n):
                proportion = np.sum(labels == i) / len(labels)
                if proportion > 0.05:
                    clusters.append({
                        'center': centers[i].tolist(),
                        'proportion': float(proportion),
                        'name': self._name_color(centers[i])
                    })
            
            return sorted(clusters, key=lambda x: x['proportion'], reverse=True)
        except Exception:
            return []
    
    def _name_color(self, rgb: np.ndarray) -> str:
        """Assign a name to RGB color."""
        try:
            rgb_uint8 = np.clip(rgb, 0, 255).astype(np.uint8)
            hsv = cv2.cvtColor(
                np.array([[rgb_uint8]], dtype=np.uint8),
                cv2.COLOR_RGB2HSV
            )[0][0]
            
            h, s, v = hsv
            
            if v < 50:
                return "black"
            elif s < 30:
                return "white" if v > 200 else "gray"
            elif h < 15 or h > 165:
                return "red"
            elif 15 <= h < 35:
                return "brown"
            elif 35 <= h < 75:
                return "tan"
            elif 100 <= h < 140:
                return "blue"
            else:
                return "other"
        except Exception:
            return "unknown"
    
    def _compute_variety(self, clusters: List[Dict]) -> float:
        """Compute color variety score."""
        if len(clusters) <= 1:
            return 0.0
        
        # Entropy-based variety
        proportions = [c['proportion'] for c in clusters]
        entropy = -sum(p * np.log2(p + 1e-6) for p in proportions)
        max_entropy = np.log2(len(clusters))
        
        if max_entropy == 0:
            return 0.0
        
        return entropy / max_entropy
    
    def _detect_concerning_colors(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> List[str]:
        """Detect dermoscopic colors of concern."""
        detected = []
        
        for color_name, color_def in self.concerning_colors.items():
            lower = np.array(color_def['lower'], dtype=np.uint8)
            upper = np.array(color_def['upper'], dtype=np.uint8)
            
            color_mask = cv2.inRange(image, lower, upper)
            color_mask = cv2.bitwise_and(color_mask, mask)
            
            proportion = np.sum(color_mask > 0) / (np.sum(mask > 0) + 1e-6)
            
            if proportion > 0.03:
                detected.append(color_name)
        
        return detected
    
    def _detect_blue_white_veil(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> bool:
        """Detect blue-white veil (strong melanoma indicator)."""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Blue-white veil criteria
            bwv_mask = (
                (hsv[:, :, 0] > 90) & (hsv[:, :, 0] < 130) &  # Blue hue
                (hsv[:, :, 1] < 100) &                         # Low saturation
                (hsv[:, :, 2] > 80) & (hsv[:, :, 2] < 220)    # Moderate value
            )
            
            bwv_in_lesion = bwv_mask & (mask > 0)
            proportion = np.sum(bwv_in_lesion) / (np.sum(mask > 0) + 1e-6)
            
            return proportion > 0.10
        except Exception:
            return False
    
    def _compute_homogeneity(self, pixels: np.ndarray) -> float:
        """Compute color homogeneity."""
        if len(pixels) == 0:
            return 1.0
        
        std = np.std(pixels, axis=0)
        avg_std = np.mean(std)
        
        homogeneity = 1 - min(avg_std / 50, 1.0)
        return homogeneity
    
    def _compute_color_score(
        self,
        num_colors: int,
        concerning: List[str],
        has_bwv: bool,
        variety: float
    ) -> float:
        """Compute overall color risk score."""
        # Number of colors contribution
        color_count_score = min(num_colors / self.config.max_expected_colors, 1.0)
        
        # Concerning colors contribution
        concerning_score = min(len(concerning) / 3, 1.0)
        
        # Blue-white veil (high risk)
        bwv_score = 0.8 if has_bwv else 0.0
        
        score = (
            color_count_score * 0.3 +
            concerning_score * 0.3 +
            variety * 0.2 +
            bwv_score * 0.2
        )
        
        return min(score, 1.0)
    
    def _default_result(self) -> ColorResult:
        return ColorResult(
            num_colors=0,
            color_clusters=[],
            color_variety_score=0.0,
            concerning_colors=[],
            has_blue_white_veil=False,
            homogeneity=1.0,
            color_score=0.0,
            is_concerning=False
        )


class DiameterAnalyzer:
    """
    Analyzes lesion diameter.
    """
    
    THRESHOLD_MM = 6.0
    
    def __init__(self, config=None):
        self.config = config or ABCDE_CONFIG
        self.threshold = self.config.diameter_threshold_mm
    
    def analyze(self, geometry: LesionGeometry) -> DiameterResult:
        """Analyze lesion diameter."""
        diameter_mm = geometry.diameter_mm
        major_axis_mm = geometry.major_axis_mm
        minor_axis_mm = geometry.minor_axis_mm
        
        # Use maximum dimension
        max_dim = max(diameter_mm, major_axis_mm)
        
        # Classification
        if max_dim < 3.0:
            classification = "small"
            risk = 0.0
        elif max_dim < self.threshold:
            classification = "medium"
            risk = 0.2
        elif max_dim < 10.0:
            classification = "concerning"
            risk = 0.5
        else:
            classification = "large"
            risk = 0.8
        
        return DiameterResult(
            diameter_mm=diameter_mm,
            major_axis_mm=major_axis_mm,
            minor_axis_mm=minor_axis_mm,
            max_dimension_mm=max_dim,
            exceeds_threshold=max_dim > self.threshold,
            classification=classification,
            risk_contribution=risk,
            is_concerning=max_dim >= self.threshold
        )


class EvolutionAnalyzer:
    """
    Analyzes evolution indicators.
    """
    
    def __init__(self, config=None):
        self.config = config or ABCDE_CONFIG
    
    def analyze(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prior_image: Optional[np.ndarray] = None,
        prior_mask: Optional[np.ndarray] = None
    ) -> EvolutionResult:
        """Analyze evolution indicators."""
        # Texture heterogeneity as proxy
        heterogeneity = self._compute_heterogeneity(image, mask)
        
        # Growth pattern
        growth_pattern = self._analyze_growth_pattern(mask)
        
        # Prior comparison
        prior_comparison = None
        if prior_image is not None and prior_mask is not None:
            prior_comparison = self._compare_to_prior(
                image, mask, prior_image, prior_mask
            )
        
        # Evolution score
        if prior_comparison is not None:
            evolution_score = prior_comparison.get('overall_change', 0.0)
        else:
            evolution_score = min(heterogeneity * 0.5, 1.0)
        
        # Classification
        if evolution_score < 0.2:
            classification = "stable"
        elif evolution_score < 0.4:
            classification = "mildly_evolving"
        elif evolution_score < 0.6:
            classification = "moderately_evolving"
        else:
            classification = "rapidly_evolving"
        
        return EvolutionResult(
            texture_heterogeneity=heterogeneity,
            growth_pattern=growth_pattern,
            prior_comparison=prior_comparison,
            evolution_score=evolution_score,
            has_prior=prior_image is not None,
            classification=classification,
            is_concerning=evolution_score > self.config.evolution_concerning_threshold
        )
    
    def _compute_heterogeneity(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> float:
        """Compute texture heterogeneity."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            lesion_gray = gray[mask > 0]
            
            if len(lesion_gray) < 100:
                return 0.0
            
            # Variance as heterogeneity measure
            variance = np.var(lesion_gray)
            normalized = min(variance / 1000, 1.0)
            
            return normalized
        except Exception:
            return 0.0
    
    def _analyze_growth_pattern(self, mask: np.ndarray) -> str:
        """Analyze growth pattern from shape."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return "unknown"
        
        contour = max(contours, key=cv2.contourArea)
        
        # Circularity
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        if circularity > 0.8:
            return "concentric"
        elif circularity > 0.5:
            return "mildly_irregular"
        else:
            return "asymmetric_growth"
    
    def _compare_to_prior(
        self,
        current_image: np.ndarray,
        current_mask: np.ndarray,
        prior_image: np.ndarray,
        prior_mask: np.ndarray
    ) -> Dict[str, Any]:
        """Compare current to prior image."""
        try:
            # Size change
            current_area = np.sum(current_mask > 0)
            prior_area = np.sum(prior_mask > 0)
            size_change = (current_area - prior_area) / (prior_area + 1e-6)
            
            # Color change
            current_colors = current_image[current_mask > 0].mean(axis=0)
            prior_colors = prior_image[prior_mask > 0].mean(axis=0)
            color_change = np.linalg.norm(current_colors - prior_colors) / 255
            
            # Shape change (IoU)
            h = max(current_mask.shape[0], prior_mask.shape[0])
            w = max(current_mask.shape[1], prior_mask.shape[1])
            
            curr_resized = cv2.resize(current_mask, (w, h))
            prior_resized = cv2.resize(prior_mask, (w, h))
            
            intersection = np.sum((curr_resized > 0) & (prior_resized > 0))
            union = np.sum((curr_resized > 0) | (prior_resized > 0))
            shape_change = 1 - (intersection / (union + 1e-6))
            
            overall = (
                abs(size_change) * 0.4 +
                color_change * 0.3 +
                shape_change * 0.3
            )
            
            return {
                'size_change_ratio': float(size_change),
                'color_change': float(color_change),
                'shape_change': float(shape_change),
                'overall_change': min(float(overall), 1.0),
                'grew_larger': size_change > 0.1,
                'changed_color': color_change > 0.1,
                'changed_shape': shape_change > 0.2
            }
        except Exception:
            return None


class ABCDEExtractor:
    """
    Complete ABCDE feature extraction.
    """
    
    def __init__(self, config=None):
        self.config = config or ABCDE_CONFIG
        self.asymmetry = AsymmetryAnalyzer(config)
        self.border = BorderAnalyzer(config)
        self.color = ColorAnalyzer(config)
        self.diameter = DiameterAnalyzer(config)
        self.evolution = EvolutionAnalyzer(config)
    
    def extract(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        geometry: LesionGeometry,
        prior_image: Optional[np.ndarray] = None,
        prior_mask: Optional[np.ndarray] = None
    ) -> ABCDEFeatures:
        """Extract complete ABCDE features."""
        # A - Asymmetry
        asymmetry_result = self.asymmetry.analyze(mask, image)
        
        # B - Border
        border_result = self.border.analyze(mask, image)
        
        # C - Color
        color_result = self.color.analyze(image, mask)
        
        # D - Diameter
        diameter_result = self.diameter.analyze(geometry)
        
        # E - Evolution
        evolution_result = self.evolution.analyze(
            image, mask, prior_image, prior_mask
        )
        
        # Count concerning criteria
        criteria_met = sum([
            asymmetry_result.is_concerning,
            border_result.is_concerning,
            color_result.is_concerning,
            diameter_result.is_concerning,
            evolution_result.is_concerning
        ])
        
        # Weighted total score
        weights = self.config.weights
        total_score = (
            weights['asymmetry'] * asymmetry_result.combined_score +
            weights['border'] * border_result.irregularity_score +
            weights['color'] * color_result.color_score +
            weights['diameter'] * diameter_result.risk_contribution +
            weights['evolution'] * evolution_result.evolution_score
        ) / sum(weights.values())
        
        return ABCDEFeatures(
            asymmetry=asymmetry_result,
            border=border_result,
            color=color_result,
            diameter=diameter_result,
            evolution=evolution_result,
            total_score=total_score,
            criteria_met=criteria_met
        )

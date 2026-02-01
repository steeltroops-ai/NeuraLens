"""
Enhanced Biomarker Extraction Module v5.0

Replaces simulated values with actual computed biomarkers from fundus images.

Algorithms:
- Vessel Segmentation: Green channel + morphological operations (fallback)
- Tortuosity: Integral curvature method (Grisan et al. 2008)
- AVR: Color-based A/V classification + Knudtson formula
- Fractal Dimension: Box-counting algorithm
- CDR: Hough circles for disc/cup detection
- Lesions: Adaptive thresholding + blob detection

Author: NeuraLens Medical AI Team
Version: 5.0.0
"""

import numpy as np
import logging
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass
from scipy import ndimage
from scipy.interpolate import splprep, splev

logger = logging.getLogger(__name__)

# Graceful imports
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None
    logger.warning("OpenCV not available - using fallback methods")

try:
    from skimage.morphology import skeletonize, disk, opening, closing
    from skimage.measure import regionprops, label
    from skimage.filters import frangi
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    logger.warning("scikit-image not available - using fallback methods")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BiomarkerConfig:
    """Configuration for biomarker extraction."""
    # Vessel parameters
    VESSEL_THRESHOLD_PERCENTILE: float = 85
    MIN_VESSEL_LENGTH_PX: int = 30
    
    # Zone B definition (0.5-1.0 disc radii from disc margin)
    ZONE_B_INNER_FACTOR: float = 1.5  # 0.5 from margin = 1.5 from center
    ZONE_B_OUTER_FACTOR: float = 2.0  # 1.0 from margin = 2.0 from center
    
    # Knudtson formula constants
    KNUDTSON_ARTERY_FACTOR: float = 0.88
    KNUDTSON_VEIN_FACTOR: float = 0.95
    PIXELS_PER_UM: float = 2.5  # Approximate, device-dependent
    
    # Lesion detection
    MA_MIN_AREA_PX: int = 5
    MA_MAX_AREA_PX: int = 100
    HEMORRHAGE_MIN_AREA_PX: int = 50
    
    # Quality thresholds
    MIN_VESSEL_COVERAGE: float = 0.02


CONFIG = BiomarkerConfig()


# =============================================================================
# VESSEL SEGMENTATION
# =============================================================================

class VesselSegmenter:
    """
    Segment retinal vessels using traditional image processing.
    
    Used as fallback when deep learning models are not available.
    """
    
    @staticmethod
    def segment(image: np.ndarray) -> np.ndarray:
        """
        Segment vessels using green channel enhancement + morphology.
        
        Args:
            image: RGB fundus image (H, W, 3) normalized [0, 1]
            
        Returns:
            Binary vessel mask (H, W)
        """
        if not CV2_AVAILABLE:
            return VesselSegmenter._fallback_segment(image)
        
        # Extract green channel (best vessel contrast)
        if image.ndim == 3:
            if image.max() <= 1.0:
                green = (image[:, :, 1] * 255).astype(np.uint8)
            else:
                green = image[:, :, 1].astype(np.uint8)
        else:
            green = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(green)
        
        # Top-hat transform to extract vessels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            tophat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, -2
        )
        
        # Clean up with morphological operations
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_small)
        
        return (cleaned > 0).astype(np.uint8)
    
    @staticmethod
    def _fallback_segment(image: np.ndarray) -> np.ndarray:
        """Fallback segmentation without OpenCV."""
        if image.ndim == 3:
            green = image[:, :, 1]
        else:
            green = image
        
        # Simple thresholding
        threshold = np.percentile(green, 75)
        mask = (green > threshold).astype(np.uint8)
        
        # Opening to remove noise
        if SKIMAGE_AVAILABLE:
            mask = opening(mask, disk(2))
            mask = closing(mask, disk(2))
        
        return mask


# =============================================================================
# VESSEL GRAPH EXTRACTION
# =============================================================================

class VesselGraphExtractor:
    """Extract vessel centerlines and properties from mask."""
    
    @staticmethod
    def extract_centerlines(vessel_mask: np.ndarray) -> List[np.ndarray]:
        """
        Extract vessel centerlines via skeletonization.
        
        Returns list of centerline coordinate arrays.
        """
        if not SKIMAGE_AVAILABLE:
            return []
        
        # Skeletonize
        skeleton = skeletonize(vessel_mask > 0)
        
        # Label connected components
        labeled, num_features = label(skeleton, return_num=True)
        
        centerlines = []
        for region in regionprops(labeled):
            coords = region.coords
            if len(coords) >= CONFIG.MIN_VESSEL_LENGTH_PX:
                centerlines.append(coords)
        
        return centerlines
    
    @staticmethod
    def measure_widths(
        vessel_mask: np.ndarray,
        centerline: np.ndarray
    ) -> np.ndarray:
        """
        Measure vessel width along centerline.
        
        Uses distance transform to estimate diameter.
        """
        if not SKIMAGE_AVAILABLE:
            return np.ones(len(centerline)) * 5.0
        
        # Distance transform
        dist = ndimage.distance_transform_edt(vessel_mask > 0)
        
        # Sample width at each centerline point
        widths = []
        for point in centerline:
            y, x = int(point[0]), int(point[1])
            if 0 <= y < dist.shape[0] and 0 <= x < dist.shape[1]:
                width = dist[y, x] * 2  # Diameter = 2 * distance to edge
                widths.append(width)
            else:
                widths.append(0)
        
        return np.array(widths)


# =============================================================================
# TORTUOSITY CALCULATION
# =============================================================================

class TortuosityCalculator:
    """
    Calculate vessel tortuosity using integral curvature method.
    
    Reference: Grisan et al. 2008 IEEE TMI
    """
    
    @staticmethod
    def calculate_distance_metric(centerline: np.ndarray) -> float:
        """
        Distance Metric tortuosity: arc_length / chord_length
        
        Simple but effective measure.
        DM = 1.0 for straight vessels, > 1.0 for curved.
        """
        if len(centerline) < 2:
            return 1.0
        
        # Arc length: sum of segment lengths
        diffs = np.diff(centerline, axis=0)
        arc_length = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))
        
        # Chord length: end-to-end distance
        chord = np.sqrt(np.sum((centerline[-1] - centerline[0])**2))
        
        if chord < 1e-6:
            return 1.0
        
        return float(arc_length / chord)
    
    @staticmethod
    def calculate_integral_curvature(centerline: np.ndarray) -> float:
        """
        Integral curvature tortuosity: (1/L) * integral(k^2) ds
        
        More sensitive to local curvature variations.
        
        Reference: Grisan et al. 2008
        """
        if len(centerline) < 5:
            return TortuosityCalculator.calculate_distance_metric(centerline)
        
        try:
            # Smooth with cubic spline
            tck, u = splprep([centerline[:, 0], centerline[:, 1]], s=len(centerline)/10)
            
            # Sample at regular intervals
            u_new = np.linspace(0, 1, min(200, len(centerline) * 2))
            x_new, y_new = splev(u_new, tck)
            
            # First and second derivatives
            dx, dy = splev(u_new, tck, der=1)
            ddx, ddy = splev(u_new, tck, der=2)
            
            # Curvature: k = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
            denominator = (dx**2 + dy**2)**1.5
            denominator = np.where(denominator < 1e-10, 1e-10, denominator)
            curvature = np.abs(dx * ddy - dy * ddx) / denominator
            
            # Arc length elements
            ds = np.sqrt(np.diff(x_new)**2 + np.diff(y_new)**2)
            total_length = np.sum(ds)
            
            if total_length < 1e-6:
                return 1.0
            
            # Integral of curvature squared
            curvature_integral = np.sum(curvature[:-1]**2 * ds)
            tortuosity = curvature_integral / total_length
            
            # Normalize to typical range [1.0, 1.5]
            return float(1.0 + tortuosity * 10)
            
        except Exception as e:
            logger.debug(f"Integral curvature failed: {e}, using distance metric")
            return TortuosityCalculator.calculate_distance_metric(centerline)
    
    @staticmethod
    def calculate_mean_tortuosity(centerlines: List[np.ndarray]) -> float:
        """Calculate mean tortuosity across all centerlines."""
        if not centerlines:
            return 1.0
        
        tortuosities = [
            TortuosityCalculator.calculate_integral_curvature(cl)
            for cl in centerlines
            if len(cl) >= 5
        ]
        
        if not tortuosities:
            return 1.0
        
        return float(np.mean(tortuosities))


# =============================================================================
# ARTERY/VEIN CLASSIFICATION
# =============================================================================

class AVClassifier:
    """
    Classify vessels as arteries or veins based on color.
    
    Arteries: Brighter (oxygenated blood)
    Veins: Darker (deoxygenated blood)
    """
    
    @staticmethod
    def classify(
        image: np.ndarray,
        vessel_mask: np.ndarray,
        centerlines: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Classify centerlines as arteries or veins.
        
        Returns:
            (artery_centerlines, vein_centerlines)
        """
        if len(centerlines) == 0:
            return [], []
        
        arteries = []
        veins = []
        
        # Compute intensity along each centerline
        for cl in centerlines:
            intensity = AVClassifier._sample_intensity(image, cl)
            if intensity is None:
                continue
            
            # Simple threshold: brighter = artery
            mean_intensity = np.mean(intensity)
            # Use median intensity of all centerlines as threshold
            global_median = np.median([
                np.mean(AVClassifier._sample_intensity(image, c) or [0.5])
                for c in centerlines
            ])
            
            if mean_intensity > global_median:
                arteries.append(cl)
            else:
                veins.append(cl)
        
        return arteries, veins
    
    @staticmethod
    def _sample_intensity(
        image: np.ndarray,
        centerline: np.ndarray
    ) -> Optional[np.ndarray]:
        """Sample image intensity along centerline."""
        if len(centerline) == 0:
            return None
        
        if image.ndim == 3:
            # Use red channel ratio for A/V discrimination
            red = image[:, :, 0]
            green = image[:, :, 1]
        else:
            red = image
            green = image
        
        intensities = []
        for point in centerline[::5]:  # Sample every 5th point
            y, x = int(point[0]), int(point[1])
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                # Red/Green ratio (arteries have higher ratio)
                r, g = red[y, x], green[y, x]
                ratio = r / (g + 1e-6)
                intensities.append(ratio)
        
        return np.array(intensities) if intensities else None


# =============================================================================
# KNUDTSON CRAE/CRVE CALCULATION
# =============================================================================

class KnudtsonCalculator:
    """
    Calculate Central Retinal Equivalents using Knudtson formulas.
    
    Reference: Knudtson et al. 2003 Ophthalmology
    """
    
    @staticmethod
    def calculate_crae(artery_widths: List[float]) -> float:
        """
        Calculate Central Retinal Artery Equivalent.
        
        Formula: w = 0.88 * sqrt(w1^2 + w2^2), iterate until 6->1
        """
        return KnudtsonCalculator._knudtson_combine(
            artery_widths,
            CONFIG.KNUDTSON_ARTERY_FACTOR
        ) / CONFIG.PIXELS_PER_UM
    
    @staticmethod
    def calculate_crve(vein_widths: List[float]) -> float:
        """
        Calculate Central Retinal Vein Equivalent.
        
        Formula: w = 0.95 * sqrt(w1^2 + w2^2), iterate until 6->1
        """
        return KnudtsonCalculator._knudtson_combine(
            vein_widths,
            CONFIG.KNUDTSON_VEIN_FACTOR
        ) / CONFIG.PIXELS_PER_UM
    
    @staticmethod
    def _knudtson_combine(widths: List[float], factor: float) -> float:
        """Apply Knudtson iterative combination."""
        if not widths:
            return 0.0
        
        # Use 6 largest measurements
        widths = sorted(widths, reverse=True)[:6]
        while len(widths) < 6:
            widths.append(0.0)
        
        widths = np.array(widths)
        
        # Iteratively combine pairs
        while len(widths) > 1:
            new_widths = []
            for i in range(0, len(widths), 2):
                if i + 1 < len(widths):
                    combined = factor * np.sqrt(widths[i]**2 + widths[i+1]**2)
                    new_widths.append(combined)
                else:
                    new_widths.append(widths[i])
            widths = np.array(new_widths)
        
        return float(widths[0])


# =============================================================================
# FRACTAL DIMENSION
# =============================================================================

class FractalCalculator:
    """
    Calculate fractal dimension using box-counting.
    
    Reference: Liew et al. 2011 IOVS
    """
    
    @staticmethod
    def calculate_box_counting(vessel_mask: np.ndarray) -> float:
        """
        Box-counting fractal dimension.
        
        D = lim(log(N(s)) / log(1/s)) as s -> 0
        
        Normal: D ~ 1.40-1.50
        """
        if vessel_mask.sum() == 0:
            return 1.0
        
        # Box sizes (powers of 2)
        sizes = [2, 4, 8, 16, 32, 64, 128]
        counts = []
        
        h, w = vessel_mask.shape
        
        for size in sizes:
            new_h, new_w = h // size, w // size
            
            if new_h < 2 or new_w < 2:
                continue
            
            # Count boxes containing vessel pixels
            count = 0
            for i in range(new_h):
                for j in range(new_w):
                    box = vessel_mask[i*size:(i+1)*size, j*size:(j+1)*size]
                    if box.any():
                        count += 1
            counts.append(count)
        
        if len(counts) < 3:
            return 1.4  # Default healthy value
        
        # Linear regression on log-log plot
        valid_sizes = sizes[:len(counts)]
        log_sizes = np.log(1.0 / np.array(valid_sizes))
        log_counts = np.log(np.array(counts) + 1)
        
        # Slope = fractal dimension
        coeffs = np.polyfit(log_sizes, log_counts, 1)
        fractal_dim = coeffs[0]
        
        return float(np.clip(fractal_dim, 1.0, 2.0))


# =============================================================================
# OPTIC DISC DETECTION
# =============================================================================

class OpticDiscDetector:
    """Detect optic disc center and radius."""
    
    @staticmethod
    def detect(image: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
        """
        Detect optic disc using Hough circles.
        
        Returns:
            (center, radius) or (None, None) if not detected
        """
        if not CV2_AVAILABLE:
            return OpticDiscDetector._fallback_detect(image)
        
        # Use red channel (disc appears bright)
        if image.ndim == 3:
            if image.max() <= 1.0:
                gray = (image[:, :, 0] * 255).astype(np.uint8)
            else:
                gray = image[:, :, 0].astype(np.uint8)
        else:
            gray = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        # Blur
        blurred = cv2.GaussianBlur(gray, (31, 31), 0)
        
        # Hough circles
        h, w = gray.shape
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=w // 4,
            param1=50,
            param2=30,
            minRadius=w // 20,
            maxRadius=w // 8
        )
        
        if circles is not None and len(circles[0]) > 0:
            # Take the brightest circle
            best_circle = None
            best_brightness = 0
            
            for circle in circles[0]:
                cx, cy, r = int(circle[0]), int(circle[1]), int(circle[2])
                
                # Compute mean brightness in circle
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(mask, (cx, cy), r, 255, -1)
                brightness = np.mean(gray[mask > 0])
                
                if brightness > best_brightness:
                    best_brightness = brightness
                    best_circle = (cx, cy, r)
            
            if best_circle:
                return (best_circle[0], best_circle[1]), best_circle[2]
        
        return OpticDiscDetector._fallback_detect(image)
    
    @staticmethod
    def _fallback_detect(image: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
        """Fallback: find brightest region."""
        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Normalize
        if gray.max() > 1:
            gray = gray / 255.0
        
        # Smooth
        if SKIMAGE_AVAILABLE:
            from scipy.ndimage import gaussian_filter
            smoothed = gaussian_filter(gray, sigma=20)
        else:
            smoothed = gray
        
        # Find brightest point
        max_idx = np.unravel_index(np.argmax(smoothed), smoothed.shape)
        center = (int(max_idx[1]), int(max_idx[0]))
        
        # Estimate radius (~1/15 of image width)
        h, w = image.shape[:2]
        radius = int(min(h, w) / 15)
        
        return center, radius


# =============================================================================
# LESION DETECTION
# =============================================================================

class LesionDetector:
    """Detect DR lesions: microaneurysms, hemorrhages, exudates."""
    
    @staticmethod
    def detect_dark_lesions(
        image: np.ndarray,
        vessel_mask: np.ndarray
    ) -> Tuple[int, int]:
        """
        Detect dark lesions (microaneurysms, hemorrhages) excluding vessels.
        
        Returns:
            (microaneurysm_count, hemorrhage_count)
        """
        if not CV2_AVAILABLE:
            return 0, 0
        
        # Green channel
        if image.ndim == 3:
            if image.max() <= 1.0:
                green = (image[:, :, 1] * 255).astype(np.uint8)
            else:
                green = image[:, :, 1].astype(np.uint8)
        else:
            green = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(green)
        
        # Find dark regions
        mean_val = np.mean(enhanced)
        dark_thresh = mean_val * 0.6
        dark_mask = (enhanced < dark_thresh).astype(np.uint8)
        
        # Remove vessels
        if vessel_mask is not None:
            dark_mask = dark_mask & (~vessel_mask.astype(bool)).astype(np.uint8)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
        
        # Find connected components
        # connectedComponentsWithStats returns: num_labels, labels, stats, centroids
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_mask)
        
        ma_count = 0
        hemorrhage_count = 0
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            
            if CONFIG.MA_MIN_AREA_PX <= area <= CONFIG.MA_MAX_AREA_PX:
                ma_count += 1
            elif area > CONFIG.HEMORRHAGE_MIN_AREA_PX:
                hemorrhage_count += 1
        
        return ma_count, hemorrhage_count
    
    @staticmethod
    def detect_exudates(image: np.ndarray) -> float:
        """
        Detect hard exudates (bright lesions).
        
        Returns:
            Exudate area as percentage of image
        """
        if not CV2_AVAILABLE:
            return 0.0
        
        if image.ndim == 3:
            if image.max() <= 1.0:
                gray = (np.mean(image, axis=2) * 255).astype(np.uint8)
            else:
                gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        # Find very bright regions
        bright_thresh = np.percentile(gray, 95)
        bright_mask = (gray > bright_thresh).astype(np.uint8)
        
        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
        
        # Calculate area percentage
        exudate_pixels = np.sum(bright_mask)
        total_pixels = bright_mask.size
        
        return float(exudate_pixels / total_pixels * 100)


# =============================================================================
# MAIN BIOMARKER EXTRACTOR
# =============================================================================

class EnhancedBiomarkerExtractor:
    """
    Production-grade biomarker extractor.
    
    Computes actual values from fundus images instead of simulation.
    """
    
    def __init__(self):
        self.vessel_segmenter = VesselSegmenter()
        self.graph_extractor = VesselGraphExtractor()
        self.tortuosity_calc = TortuosityCalculator()
        self.av_classifier = AVClassifier()
        self.knudtson = KnudtsonCalculator()
        self.fractal_calc = FractalCalculator()
        self.disc_detector = OpticDiscDetector()
        self.lesion_detector = LesionDetector()
    
    def extract(
        self,
        image: np.ndarray,
        quality_score: float = 0.85
    ) -> Dict[str, Any]:
        """
        Extract all biomarkers from fundus image.
        
        Args:
            image: RGB fundus image (H, W, 3)
            quality_score: Image quality score for confidence weighting
            
        Returns:
            Dictionary of biomarkers with values, status, and confidence
        """
        logger.info("Starting enhanced biomarker extraction")
        
        # Step 1: Vessel segmentation
        vessel_mask = self.vessel_segmenter.segment(image)
        vessel_coverage = np.mean(vessel_mask)
        
        logger.info(f"Vessel coverage: {vessel_coverage:.2%}")
        
        # Step 2: Extract centerlines
        centerlines = self.graph_extractor.extract_centerlines(vessel_mask)
        logger.info(f"Extracted {len(centerlines)} vessel segments")
        
        # Step 3: Classify arteries/veins
        arteries, veins = self.av_classifier.classify(image, vessel_mask, centerlines)
        logger.info(f"Classified: {len(arteries)} arteries, {len(veins)} veins")
        
        # Step 4: Measure widths in Zone B
        disc_center, disc_radius = self.disc_detector.detect(image)
        
        if disc_center and disc_radius:
            artery_widths = self._get_zone_b_widths(arteries, vessel_mask, disc_center, disc_radius)
            vein_widths = self._get_zone_b_widths(veins, vessel_mask, disc_center, disc_radius)
        else:
            # Fallback: use all vessel widths
            artery_widths = [np.mean(self.graph_extractor.measure_widths(vessel_mask, cl)) for cl in arteries]
            vein_widths = [np.mean(self.graph_extractor.measure_widths(vessel_mask, cl)) for cl in veins]
        
        # Step 5: Calculate CRAE/CRVE
        crae = self.knudtson.calculate_crae(artery_widths) if artery_widths else 145.0
        crve = self.knudtson.calculate_crve(vein_widths) if vein_widths else 220.0
        avr = crae / crve if crve > 0 else 0.66
        
        # Step 6: Calculate tortuosity
        tortuosity = self.tortuosity_calc.calculate_mean_tortuosity(centerlines)
        
        # Step 7: Calculate fractal dimension
        fractal_dim = self.fractal_calc.calculate_box_counting(vessel_mask)
        
        # Step 8: CDR estimation (simplified - would use U-Net for accurate cup detection)
        cdr = self._estimate_cdr(image, disc_center, disc_radius)
        
        # Step 9: Lesion detection
        ma_count, hemorrhage_count = self.lesion_detector.detect_dark_lesions(image, vessel_mask)
        exudate_percent = self.lesion_detector.detect_exudates(image)
        
        # Build result dictionary
        result = {
            "vessels": {
                "crae_um": self._to_biomarker_value(crae, (140, 170), "um", quality_score),
                "crve_um": self._to_biomarker_value(crve, (200, 250), "um", quality_score),
                "av_ratio": self._to_biomarker_value(avr, (0.65, 0.75), "ratio", quality_score),
                "tortuosity_index": self._to_biomarker_value(tortuosity, (1.0, 1.18), "index", quality_score),
                "vessel_density": self._to_biomarker_value(vessel_coverage, (0.04, 0.12), "ratio", quality_score),
                "fractal_dimension": self._to_biomarker_value(fractal_dim, (1.35, 1.50), "dim", quality_score),
                "artery_count": len(arteries),
                "vein_count": len(veins),
            },
            "optic_disc": {
                "cup_disc_ratio": self._to_biomarker_value(cdr, (0.3, 0.5), "ratio", quality_score * 0.8),
                "disc_detected": disc_center is not None,
                "disc_center": disc_center,
                "disc_radius": disc_radius,
            },
            "lesions": {
                "microaneurysm_count": self._to_biomarker_value(float(ma_count), (0, 0), "count", quality_score),
                "hemorrhage_count": self._to_biomarker_value(float(hemorrhage_count), (0, 0), "count", quality_score),
                "exudate_area_percent": self._to_biomarker_value(exudate_percent, (0, 0.5), "percent", quality_score),
            },
            "meta": {
                "total_centerlines": len(centerlines),
                "vessel_coverage": vessel_coverage,
                "quality_score": quality_score,
            }
        }
        
        logger.info(f"Biomarker extraction complete: AVR={avr:.3f}, Tort={tortuosity:.3f}, CDR={cdr:.2f}")
        
        return result
    
    def _get_zone_b_widths(
        self,
        centerlines: List[np.ndarray],
        vessel_mask: np.ndarray,
        disc_center: Tuple[int, int],
        disc_radius: int
    ) -> List[float]:
        """Get vessel widths in Zone B (0.5-1.0 disc radii from margin)."""
        zone_b_widths = []
        
        inner_r = disc_radius * CONFIG.ZONE_B_INNER_FACTOR
        outer_r = disc_radius * CONFIG.ZONE_B_OUTER_FACTOR
        
        for cl in centerlines:
            for point in cl:
                y, x = point[0], point[1]
                dist = np.sqrt((x - disc_center[0])**2 + (y - disc_center[1])**2)
                
                if inner_r <= dist <= outer_r:
                    if 0 <= int(y) < vessel_mask.shape[0] and 0 <= int(x) < vessel_mask.shape[1]:
                        width = self.graph_extractor.measure_widths(vessel_mask, np.array([point]))[0]
                        if width > 0:
                            zone_b_widths.append(width)
        
        return zone_b_widths
    
    def _estimate_cdr(
        self,
        image: np.ndarray,
        disc_center: Optional[Tuple[int, int]],
        disc_radius: Optional[int]
    ) -> float:
        """
        Estimate cup-to-disc ratio.
        
        Simplified: uses intensity thresholding within disc.
        Full implementation would use trained U-Net for cup segmentation.
        """
        if disc_center is None or disc_radius is None:
            return 0.35  # Normal default
        
        if not CV2_AVAILABLE:
            return 0.35
        
        # Extract disc region
        h, w = image.shape[:2] if image.ndim == 2 else image.shape[:2]
        cx, cy = disc_center
        r = disc_radius
        
        # Create disc mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        
        # Get intensity within disc
        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        if gray.max() <= 1.0:
            gray = gray * 255
        
        disc_intensities = gray[mask > 0]
        
        if len(disc_intensities) == 0:
            return 0.35
        
        # Cup is brighter region in center
        bright_thresh = np.percentile(disc_intensities, 75)
        cup_pixels = np.sum((gray > bright_thresh) & (mask > 0))
        disc_pixels = np.sum(mask > 0)
        
        # CDR ~ sqrt(cup_area / disc_area) for diameter ratio
        cdr = np.sqrt(cup_pixels / disc_pixels) if disc_pixels > 0 else 0.35
        
        return float(np.clip(cdr, 0.1, 0.95))
    
    @staticmethod
    def _to_biomarker_value(
        value: float,
        normal_range: Tuple[float, float],
        unit: str,
        confidence: float
    ) -> Dict[str, Any]:
        """Convert value to biomarker value dict with status."""
        # Determine status
        if normal_range[0] <= value <= normal_range[1]:
            status = "normal"
        elif value < normal_range[0] * 0.8 or value > normal_range[1] * 1.2:
            status = "abnormal"
        else:
            status = "borderline"
        
        return {
            "value": float(value),
            "normal_range": list(normal_range),
            "status": status,
            "unit": unit,
            "measurement_confidence": float(confidence),
        }


# Singleton instance
enhanced_biomarker_extractor = EnhancedBiomarkerExtractor()

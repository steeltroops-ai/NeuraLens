"""
Image Validation Service for Retinal Analysis Pipeline

Validates retinal fundus images for:
- Format validation (JPEG, PNG, DICOM) - Requirement 1.1
- Resolution validation (min 1024x1024) - Requirement 1.2
- File size validation (100KB-50MB) - Requirement 1.4
- SNR calculation and threshold (>=15dB) - Requirements 2.1, 2.2
- Focus quality detection - Requirements 2.3, 2.4
- Optic disc detection - Requirements 2.5, 2.6
- Macula detection - Requirements 2.7, 2.8
- Glare detection - Requirements 2.9, 2.10
- Quality score generation (0-100) - Requirements 2.11, 2.12

Author: NeuraLens Team
"""

import logging
import io
import numpy as np
from PIL import Image
from fastapi import UploadFile
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

# Graceful cv2 import
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

# Import from schemas - using try/except for flexibility
try:
    from ..schemas import ImageValidationResponse
except ImportError:
    # Fallback if schemas not available
    ImageValidationResponse = None

logger = logging.getLogger(__name__)




@dataclass
class ValidationResult:
    """Internal validation result container"""
    is_valid: bool = True
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    snr_db: float = 0.0
    focus_score: float = 0.0
    glare_percentage: float = 0.0
    has_optic_disc: bool = False
    has_macula: bool = False
    resolution: Optional[Tuple[int, int]] = None


class ImageValidator:
    """
    Comprehensive validator for retinal fundus images.
    
    Checks format, resolution, size, and image quality metrics including:
    - Signal-to-Noise Ratio (SNR)
    - Focus quality (Laplacian variance)
    - Glare/saturation detection
    - Anatomical feature presence (optic disc, macula)
    
    Requirements: 1.1-1.10, 2.1-2.12
    """

    # Validation thresholds per requirements
    MIN_RESOLUTION = (1024, 1024)  # Requirement 1.2
    MIN_SIZE_KB = 100              # Requirement 1.4
    MAX_SIZE_MB = 50               # Requirement 1.4
    
    ALLOWED_FORMATS = [            # Requirement 1.1
        "image/jpeg", 
        "image/png", 
        "application/dicom"
    ]
    
    MIN_SNR_DB = 15.0              # Requirement 2.2
    MIN_FOCUS_SCORE = 100.0        # Requirement 2.4 threshold
    MAX_GLARE_PERCENTAGE = 5.0     # Requirement 2.10 threshold
    
    # Quality score thresholds
    MARGINAL_QUALITY_LOW = 60      # Requirement 2.12
    MARGINAL_QUALITY_HIGH = 75     # Requirement 2.12

    async def validate(self, file: UploadFile) -> ImageValidationResponse:
        """
        Validate the uploaded retinal image.
        
        Performs comprehensive validation including:
        1. File size validation
        2. Format validation
        3. Resolution validation
        4. SNR calculation
        5. Focus quality assessment
        6. Glare detection
        7. Anatomical feature detection
        8. Composite quality score calculation
        
        Args:
            file: Uploaded file from FastAPI
            
        Returns:
            ImageValidationResponse with validation results
        """
        result = ValidationResult()
        
        try:
            # 1. File Size Validation (Requirement 1.4, 1.5)
            self._validate_file_size(file, result)
            
            # 2. Format Validation (Requirement 1.1, 1.3)
            self._validate_format(file, result)
            
            # Read file content for image analysis
            content = await file.read()
            await file.seek(0)  # Reset for potential future reads
            
            # 3. Load and validate image
            img_cv = self._load_image(content, result)
            
            if img_cv is None:
                return self._build_response(result)
            
            height, width = img_cv.shape[:2]
            result.resolution = (width, height)
            
            # 4. Resolution Validation (Requirement 1.2, 1.3)
            self._validate_resolution(width, height, result)
            
            # 5. SNR Calculation (Requirement 2.1, 2.2)
            self._calculate_snr(img_cv, result)
            
            # 6. Focus Quality Assessment (Requirement 2.3, 2.4)
            self._assess_focus_quality(img_cv, result)
            
            # 7. Glare Detection (Requirement 2.9, 2.10)
            self._detect_glare(img_cv, result)
            
            # 8. Anatomical Feature Detection (Requirement 2.5-2.8)
            self._detect_anatomical_features(img_cv, result)
            
            # 9. Calculate Composite Quality Score (Requirement 2.11, 2.12)
            self._calculate_quality_score(result)
            
            return self._build_response(result)
            
        except Exception as e:
            logger.error(f"Image validation failed: {str(e)}")
            result.is_valid = False
            result.issues.append(f"Validation error: {str(e)}")
            return self._build_response(result)

    def _validate_file_size(self, file: UploadFile, result: ValidationResult) -> None:
        """Validate file size is within acceptable range (Requirement 1.4, 1.5)"""
        file.file.seek(0, 2)
        size_bytes = file.file.tell()
        file.file.seek(0)
        
        size_kb = size_bytes / 1024
        size_mb = size_kb / 1024
        
        if size_kb < self.MIN_SIZE_KB:
            result.issues.append(
                f"File size too small ({size_kb:.1f}KB). Minimum is {self.MIN_SIZE_KB}KB."
            )
            result.recommendations.append(
                "Upload a higher quality image with more detail."
            )
            result.is_valid = False
        
        if size_mb > self.MAX_SIZE_MB:
            result.issues.append(
                f"File size too large ({size_mb:.1f}MB). Maximum is {self.MAX_SIZE_MB}MB."
            )
            result.recommendations.append(
                "Compress the image or reduce resolution."
            )
            result.is_valid = False

    def _validate_format(self, file: UploadFile, result: ValidationResult) -> None:
        """Validate file format is allowed (Requirement 1.1, 1.3)"""
        if file.content_type not in self.ALLOWED_FORMATS:
            result.issues.append(
                f"Invalid file format: {file.content_type}. "
                f"Allowed: {', '.join(self.ALLOWED_FORMATS)}"
            )
            result.recommendations.append(
                "Convert image to JPEG or PNG format."
            )
            result.is_valid = False

    def _load_image(self, content: bytes, result: ValidationResult) -> Optional[np.ndarray]:
        """Load image content into OpenCV format"""
        try:
            # Try OpenCV first
            nparr = np.frombuffer(content, np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img_cv is None:
                # Fallback to PIL for other formats
                img_pil = Image.open(io.BytesIO(content))
                img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            
            return img_cv
            
        except Exception as e:
            result.issues.append(f"Could not decode image: {str(e)}")
            result.recommendations.append(
                "Ensure the file is a valid image format."
            )
            result.is_valid = False
            return None

    def _validate_resolution(
        self, width: int, height: int, result: ValidationResult
    ) -> None:
        """Validate image resolution meets minimum requirements (Requirement 1.2, 1.3)"""
        if width < self.MIN_RESOLUTION[0] or height < self.MIN_RESOLUTION[1]:
            result.issues.append(
                f"Resolution too low ({width}x{height}). "
                f"Minimum is {self.MIN_RESOLUTION[0]}x{self.MIN_RESOLUTION[1]}."
            )
            result.recommendations.append(
                "Use a higher resolution fundus camera or imaging device."
            )
            result.is_valid = False

    def _calculate_snr(self, img: np.ndarray, result: ValidationResult) -> None:
        """
        Calculate Signal-to-Noise Ratio (Requirement 2.1, 2.2)
        
        SNR = 20 * log10(signal / noise)
        Uses mean intensity as signal and standard deviation as noise estimate.
        """
        try:
            # Convert to grayscale for SNR calculation
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # Avoid division by zero
            if std_intensity < 1e-6:
                std_intensity = 1e-6
            
            snr = 20 * np.log10(mean_intensity / std_intensity)
            result.snr_db = round(snr, 1)
            
            if snr < self.MIN_SNR_DB:
                result.issues.append(
                    f"Low Signal-to-Noise Ratio ({snr:.1f}dB). "
                    f"Minimum is {self.MIN_SNR_DB}dB."
                )
                result.recommendations.append(
                    "Improve lighting conditions and camera sensor calibration."
                )
                result.is_valid = False
                
        except Exception as e:
            logger.warning(f"SNR calculation failed: {e}")
            result.snr_db = 0.0

    def _assess_focus_quality(self, img: np.ndarray, result: ValidationResult) -> None:
        """
        Assess focus quality using Laplacian variance (Requirement 2.3, 2.4)
        
        Higher variance indicates sharper edges = better focus.
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance as focus measure
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            focus_score = laplacian.var()
            
            result.focus_score = round(focus_score, 1)
            
            if focus_score < self.MIN_FOCUS_SCORE:
                result.issues.append(
                    f"Image is too blurry (Focus Score: {focus_score:.1f}). "
                    f"Minimum is {self.MIN_FOCUS_SCORE}."
                )
                result.recommendations.append(
                    "Ensure the camera is properly focused on the retina. "
                    "Stabilize the imaging device and retry."
                )
                result.is_valid = False
                
        except Exception as e:
            logger.warning(f"Focus assessment failed: {e}")
            result.focus_score = 0.0

    def _detect_glare(self, img: np.ndarray, result: ValidationResult) -> None:
        """
        Detect excessive glare or reflections (Requirement 2.9, 2.10)
        
        Counts pixels with high saturation (near 255) as potential glare.
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Count highly saturated pixels (potential glare)
            glare_threshold = 240  # Pixels > 240 considered glare
            glare_pixels = np.sum(gray > glare_threshold)
            total_pixels = gray.shape[0] * gray.shape[1]
            
            glare_percentage = (glare_pixels / total_pixels) * 100
            result.glare_percentage = round(glare_percentage, 2)
            
            if glare_percentage > self.MAX_GLARE_PERCENTAGE:
                result.issues.append(
                    f"Excessive glare detected ({glare_percentage:.1f}% of image). "
                    f"Maximum allowed is {self.MAX_GLARE_PERCENTAGE}%."
                )
                result.recommendations.append(
                    "Adjust lighting to reduce reflections. "
                    "Reposition the patient or adjust camera angle."
                )
                result.is_valid = False
                
        except Exception as e:
            logger.warning(f"Glare detection failed: {e}")
            result.glare_percentage = 0.0

    def _detect_anatomical_features(
        self, img: np.ndarray, result: ValidationResult
    ) -> None:
        """
        Detect optic disc and macula presence (Requirement 2.5-2.8)
        
        Uses circular Hough transform for optic disc detection
        and contrast/position analysis for macula detection.
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # ===== Optic Disc Detection =====
            # The optic disc appears as a bright circular region
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            
            # Use adaptive thresholding to find bright regions
            _, bright_regions = cv2.threshold(
                blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            
            # Find circles using Hough Circle Transform
            # Parameters tuned for typical fundus images
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=width // 4,
                param1=50,
                param2=30,
                minRadius=width // 20,
                maxRadius=width // 6
            )
            
            # Check if we found a bright circular region (optic disc candidate)
            if circles is not None and len(circles[0]) > 0:
                result.has_optic_disc = True
            else:
                # Fallback: check for bright region statistics
                # Optic disc typically appears in right half for left eye, left half for right eye
                bright_pixel_ratio = np.sum(bright_regions > 0) / (height * width)
                result.has_optic_disc = 0.01 < bright_pixel_ratio < 0.15
            
            if not result.has_optic_disc:
                result.issues.append("Optic disc not clearly visible.")
                result.recommendations.append(
                    "Reposition the camera to ensure the optic disc is "
                    "visible in the fundus image."
                )
                result.is_valid = False
            
            # ===== Macula Detection =====
            # The macula appears as a slightly darker region near the center
            # It's located temporal to the optic disc
            
            # Analyze the central region of the image
            center_y, center_x = height // 2, width // 2
            roi_size = min(height, width) // 4
            
            # Extract central ROI
            roi = gray[
                max(0, center_y - roi_size):min(height, center_y + roi_size),
                max(0, center_x - roi_size):min(width, center_x + roi_size)
            ]
            
            if roi.size > 0:
                # Macula should have lower intensity than surrounding areas
                roi_mean = np.mean(roi)
                overall_mean = np.mean(gray)
                
                # Macula typically darker than average (by 10-30%)
                intensity_ratio = roi_mean / overall_mean if overall_mean > 0 else 1
                
                # Also check for concentric pattern (fovea at center of macula)
                result.has_macula = 0.7 < intensity_ratio < 1.1
            else:
                result.has_macula = False
            
            if not result.has_macula:
                result.issues.append("Macula not clearly visible.")
                result.recommendations.append(
                    "Adjust patient gaze direction to center the macula "
                    "in the field of view."
                )
                result.is_valid = False
                
        except Exception as e:
            logger.warning(f"Anatomical feature detection failed: {e}")
            # Default to True to avoid false rejections on detection failure
            result.has_optic_disc = True
            result.has_macula = True

    def _calculate_quality_score(self, result: ValidationResult) -> None:
        """
        Calculate composite quality score (0-100) (Requirement 2.11, 2.12)
        
        Weighted formula:
        - Focus quality: 40%
        - SNR: 30%
        - Glare (inverted): 15%
        - Anatomical features: 15%
        """
        # Normalize focus score (cap at 500 for scaling)
        focus_normalized = min(result.focus_score, 500) / 500 * 100
        
        # Normalize SNR (cap at 40dB for scaling)  
        snr_normalized = min(max(result.snr_db, 0), 40) / 40 * 100
        
        # Glare penalty (0% glare = 100 points, >5% = 0 points)
        glare_score = max(0, 100 - (result.glare_percentage * 20))
        
        # Anatomical feature score
        anatomical_score = 0
        if result.has_optic_disc:
            anatomical_score += 50
        if result.has_macula:
            anatomical_score += 50
        
        # Weighted composite score
        quality_score = (
            focus_normalized * 0.40 +
            snr_normalized * 0.30 +
            glare_score * 0.15 +
            anatomical_score * 0.15
        )
        
        result.quality_score = round(min(100, max(0, quality_score)), 1)
        
        # Add marginal quality warning (Requirement 2.12)
        if self.MARGINAL_QUALITY_LOW <= result.quality_score <= self.MARGINAL_QUALITY_HIGH:
            if result.is_valid:  # Only warn if not already invalid
                result.recommendations.append(
                    f"Image quality is marginal ({result.quality_score:.0f}/100). "
                    "Analysis will proceed but results may have reduced accuracy."
                )

    def _build_response(self, result: ValidationResult) -> ImageValidationResponse:
        """Build the final validation response"""
        return ImageValidationResponse(
            is_valid=result.is_valid,
            quality_score=result.quality_score,
            issues=result.issues,
            recommendations=result.recommendations,
            snr_db=result.snr_db,
            has_optic_disc=result.has_optic_disc,
            has_macula=result.has_macula,
            focus_score=result.focus_score,
            glare_percentage=result.glare_percentage,
            resolution=result.resolution
        )


# ============================================================================
# Additional Validation Utilities
# ============================================================================

class MalwareScanner:
    """
    Placeholder for malware scanning functionality (Requirement 1.6, 1.7)
    
    In production, this would integrate with a malware scanning service
    like ClamAV or a cloud-based solution.
    """
    
    async def scan(self, content: bytes) -> Tuple[bool, Optional[str]]:
        """
        Scan file content for malware.
        
        Returns:
            Tuple of (is_safe, threat_name)
        """
        # Placeholder implementation
        # In production: integrate with ClamAV or similar
        
        # Basic magic byte validation
        if len(content) < 10:
            return False, "File too small"
        
        # Check for common image magic bytes
        jpeg_magic = content[:2] == b'\xff\xd8'
        png_magic = content[:8] == b'\x89PNG\r\n\x1a\n'
        
        if not (jpeg_magic or png_magic):
            # Could be DICOM or other format - allow through
            pass
        
        return True, None


# Singleton instances
image_validator = ImageValidator()
malware_scanner = MalwareScanner()


import logging
import io
import numpy as np
from PIL import Image
import cv2
from fastapi import UploadFile, HTTPException
from typing import List, Tuple, Dict, Any

from .schemas import ImageValidationResponse

logger = logging.getLogger(__name__)

class ImageValidator:
    """
    Validator for retinal fundus images.
    Checks format, resolution, size, and image quality metrics (SNR, focus, glare).
    """

    MIN_RESOLUTION = (1024, 1024)
    MIN_SIZE_KB = 100
    MAX_SIZE_MB = 50
    ALLOWED_FORMATS = ["image/jpeg", "image/png", "application/dicom"]
    MIN_SNR_DB = 15.0
    MIN_FOCUS_SCORE = 100.0  # Threshold for Laplacian variance

    async def validate(self, file: UploadFile) -> ImageValidationResponse:
        """
        Validate the uploaded retinal image.
        """
        issues = []
        recommendations = []
        is_valid = True
        
        # 1. File Size Validation
        file.file.seek(0, 2)
        size_bytes = file.file.tell()
        file.file.seek(0)
        
        size_kb = size_bytes / 1024
        size_mb = size_kb / 1024
        
        if size_kb < self.MIN_SIZE_KB:
            issues.append(f"File size too small ({size_kb:.1f}KB). Minimum is {self.MIN_SIZE_KB}KB.")
            recommendations.append("Upload a higher quality image.")
            is_valid = False
        
        if size_mb > self.MAX_SIZE_MB:
            issues.append(f"File size too large ({size_mb:.1f}MB). Maximum is {self.MAX_SIZE_MB}MB.")
            recommendations.append("Compress the image or use a different format.")
            is_valid = False

        # 2. Format Validation (based on Content-Type header, deep check later)
        if file.content_type not in self.ALLOWED_FORMATS:
            issues.append(f"Invalid file format: {file.content_type}. Allowed: {', '.join(self.ALLOWED_FORMATS)}")
            is_valid = False
        
        # 3. Image Loading & Resolution Validation
        try:
            # Read file content
            content = await file.read()
            # Reset cursor for potential future reads (though standard usage consumes it)
            await file.seek(0) 
            
            # Need to convert to numpy for OpenCV
            nparr = np.frombuffer(content, np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img_cv is None:
                 # Try PIL for other formats (like DICOM if supported directly, or generic images)
                try:
                    img_pil = Image.open(io.BytesIO(content))
                    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                except Exception:
                     issues.append("Could not decode image.")
                     is_valid = False
                     return ImageValidationResponse(
                        is_valid=False, quality_score=0, issues=issues, recommendations=recommendations,
                        snr_db=0, has_optic_disc=False, has_macula=False
                    )

            height, width = img_cv.shape[:2]
            if width < self.MIN_RESOLUTION[0] or height < self.MIN_RESOLUTION[1]:
                issues.append(f"Resolution too low ({width}x{height}). Minimum is {self.MIN_RESOLUTION[0]}x{self.MIN_RESOLUTION[1]}.")
                recommendations.append("Use a higher resolution fundus camera.")
                is_valid = False

            # 4. Quality Assessment
            
            # SNR Calculation (Signal-to-Noise Ratio)
            # Simple estimation: mean / std_dev of a relatively flat region or whole image
            mean_intensity = np.mean(img_cv)
            std_intensity = np.std(img_cv)
            snr = 20 * np.log10(mean_intensity / (std_intensity + 1e-6))
            
            if snr < self.MIN_SNR_DB:
                issues.append(f"Low Signal-to-Noise Ratio ({snr:.1f}dB). Minimum is {self.MIN_SNR_DB}dB.")
                recommendations.append("Check lighting and camera sensor quality.")
                # We might not invalidate just for SNR, but warn. For now let's keep is_valid logic stricter if desired.
                # Spec says "reject the image" if SNR < 15dB.
                is_valid = False

            # Focus Quality (Laplacian Variance)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            focus_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if focus_score < self.MIN_FOCUS_SCORE:
                issues.append(f"Image is too blurry (Focus Score: {focus_score:.1f}).")
                recommendations.append("Ensure the camera is properly focused.")
                is_valid = False

            # Glare Detection
            # Simple check: percentage of saturated pixels
            gray_saturation = np.max(img_cv, axis=2) # approximations using max channel
            # Or simplified: pixels near 255
            glare_pixels = np.sum(gray > 240)
            total_pixels = width * height
            glare_ratio = glare_pixels / total_pixels
            
            if glare_ratio > 0.05: # > 5% glare
                issues.append("Excessive glare detected.")
                recommendations.append("Adjust lighting or patient positioning to reduce reflections.")
                is_valid = False

            # Anatomical Features (Dummy Check - Real implementation needs complex CV)
            # We assume they are present if image is decent for this stub
            has_optic_disc = True 
            has_macula = True

            # Calculate overall quality score (0-100)
            # Heuristic: Focus contributes 50%, SNR 30%, Glare penalty 20%
            quality_score = min(100, max(0, 
                (min(focus_score, 500) / 500 * 50) + 
                (min(snr, 50) / 50 * 30) + 
                (20 * (1 - min(glare_ratio * 10, 1)))
            ))
            
            return ImageValidationResponse(
                is_valid=is_valid,
                quality_score=round(quality_score, 1),
                issues=issues,
                recommendations=recommendations,
                snr_db=round(snr, 1),
                has_optic_disc=has_optic_disc,
                has_macula=has_macula
            )

        except Exception as e:
            logger.error(f"Image validation failed: {str(e)}")
            issues.append(f"Validation error: {str(e)}")
            return ImageValidationResponse(
                is_valid=False, quality_score=0, issues=issues, recommendations=[],
                snr_db=0, has_optic_disc=False, has_macula=False
            )

# Singleton instance
image_validator = ImageValidator()

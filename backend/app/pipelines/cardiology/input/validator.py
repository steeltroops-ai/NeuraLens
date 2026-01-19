"""
Cardiology Pipeline - Input Validator
Validates ECG signals, echo images/videos, and clinical metadata.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import numpy as np
import logging

from ..config import INPUT_CONSTRAINTS, QUALITY_THRESHOLDS
from ..errors import ValidationError

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation result severity."""
    ERROR = "error"      # Reject input
    WARNING = "warning"  # Accept with caution
    INFO = "info"        # Informational


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    check_id: str
    passed: bool
    severity: ValidationSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report for all inputs."""
    is_valid: bool
    errors: List[ValidationResult] = field(default_factory=list)
    warnings: List[ValidationResult] = field(default_factory=list)
    info: List[ValidationResult] = field(default_factory=list)
    modalities_detected: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": [{"check": e.check_id, "message": e.message} for e in self.errors],
            "warnings": [{"check": w.check_id, "message": w.message} for w in self.warnings],
            "modalities_detected": self.modalities_detected,
        }


class ECGValidator:
    """Validate ECG signal inputs."""
    
    def __init__(self):
        self.constraints = INPUT_CONSTRAINTS["ecg"]
        self.quality = QUALITY_THRESHOLDS["ecg"]
    
    def validate(
        self,
        signal: np.ndarray,
        sample_rate: int,
        file_size_bytes: int = 0
    ) -> List[ValidationResult]:
        """Validate ECG signal data."""
        results = []
        
        # File size check
        if file_size_bytes > 0:
            size_mb = file_size_bytes / (1024 * 1024)
            if size_mb > self.constraints["max_file_size_mb"]:
                results.append(ValidationResult(
                    check_id="E_VAL_003",
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"ECG file size {size_mb:.1f}MB exceeds limit"
                ))
        
        # Sample rate validation
        if sample_rate < self.constraints["min_sample_rate_hz"]:
            results.append(ValidationResult(
                check_id="E_VAL_004",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Sample rate {sample_rate}Hz below minimum"
            ))
        elif sample_rate > self.constraints["max_sample_rate_hz"]:
            results.append(ValidationResult(
                check_id="E_VAL_004",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Sample rate {sample_rate}Hz exceeds maximum"
            ))
        
        # Empty signal check
        if len(signal) == 0:
            results.append(ValidationResult(
                check_id="E_VAL_010",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message="ECG signal array is empty"
            ))
            return results
        
        # Duration check
        duration = len(signal) / sample_rate
        if duration < self.constraints["min_duration_sec"]:
            results.append(ValidationResult(
                check_id="E_VAL_005",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Duration {duration:.1f}s below minimum {self.constraints['min_duration_sec']}s"
            ))
        elif duration > self.constraints["max_duration_sec"]:
            results.append(ValidationResult(
                check_id="E_VAL_009",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Duration {duration:.1f}s exceeds maximum"
            ))
        
        # Amplitude range check
        amp_range = self.constraints["amplitude_range_mv"]
        min_val, max_val = np.nanmin(signal), np.nanmax(signal)
        if min_val < amp_range[0] or max_val > amp_range[1]:
            results.append(ValidationResult(
                check_id="E_VAL_011",
                passed=False,
                severity=ValidationSeverity.WARNING,
                message=f"Signal range [{min_val:.2f}, {max_val:.2f}] outside typical range"
            ))
        
        # Flatline detection
        if self._check_flatline(signal, sample_rate):
            results.append(ValidationResult(
                check_id="E_PREP_004",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message="Flatline detected in signal"
            ))
        
        # Missing data check
        missing_ratio = np.sum(np.isnan(signal)) / len(signal)
        if missing_ratio > self.quality["max_missing_ratio"]:
            results.append(ValidationResult(
                check_id="E_VAL_012",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Missing data ratio {missing_ratio*100:.1f}% too high"
            ))
        
        return results
    
    def _check_flatline(self, signal: np.ndarray, sample_rate: int) -> bool:
        """Check for extended flatline segments."""
        flatline_samples = int(self.quality["max_flatline_sec"] * sample_rate)
        window = min(100, len(signal) // 10)
        
        for i in range(0, len(signal) - flatline_samples, window):
            segment = signal[i:i + flatline_samples]
            if np.std(segment) < 0.001:
                return True
        
        return False


class EchoValidator:
    """Validate echocardiography inputs."""
    
    def __init__(self):
        self.image_constraints = INPUT_CONSTRAINTS["echo_image"]
        self.video_constraints = INPUT_CONSTRAINTS["echo_video"]
    
    def validate_image(
        self,
        file_path: str,
        file_bytes: bytes
    ) -> List[ValidationResult]:
        """Validate echo image."""
        results = []
        
        # Check file extension
        ext = "." + file_path.lower().split(".")[-1]
        if ext not in self.image_constraints["allowed_formats"]:
            results.append(ValidationResult(
                check_id="E_VAL_002",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid image format: {ext}"
            ))
            return results
        
        # Check file size
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > self.image_constraints["max_file_size_mb"]:
            results.append(ValidationResult(
                check_id="E_VAL_003",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Image size {size_mb:.1f}MB exceeds limit"
            ))
        
        # Decode and check resolution
        try:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(file_bytes))
            width, height = img.size
            
            min_res = self.image_constraints["min_resolution"]
            max_res = self.image_constraints["max_resolution"]
            
            if width < min_res or height < min_res:
                results.append(ValidationResult(
                    check_id="E_VAL_006",
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Resolution {width}x{height} below minimum"
                ))
            elif width > max_res or height > max_res:
                results.append(ValidationResult(
                    check_id="E_VAL_006",
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Resolution {width}x{height} exceeds maximum"
                ))
                
        except Exception as e:
            results.append(ValidationResult(
                check_id="E_VAL_007",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Failed to decode image: {str(e)}"
            ))
        
        return results
    
    def validate_video(
        self,
        file_path: str,
        file_bytes: bytes
    ) -> List[ValidationResult]:
        """Validate echo video."""
        results = []
        
        # Check file extension
        ext = "." + file_path.lower().split(".")[-1]
        if ext not in self.video_constraints["allowed_formats"]:
            results.append(ValidationResult(
                check_id="E_VAL_002",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid video format: {ext}"
            ))
            return results
        
        # Check file size
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > self.video_constraints["max_file_size_mb"]:
            results.append(ValidationResult(
                check_id="E_VAL_003",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Video size {size_mb:.1f}MB exceeds limit"
            ))
        
        return results


class MetadataValidator:
    """Validate clinical metadata."""
    
    def __init__(self):
        self.constraints = INPUT_CONSTRAINTS["metadata"]
    
    def validate(self, metadata: Dict[str, Any]) -> List[ValidationResult]:
        """Validate clinical metadata JSON."""
        results = []
        
        if not metadata:
            return results  # Empty metadata is valid
        
        # Patient info validation
        patient = metadata.get("patient", {})
        
        # Age validation
        if "age_years" in patient:
            age = patient["age_years"]
            age_range = self.constraints["age_range"]
            if not age_range[0] <= age <= age_range[1]:
                results.append(ValidationResult(
                    check_id="E_VAL_008",
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Age {age} outside valid range"
                ))
        
        # Sex validation
        if "sex" in patient:
            if patient["sex"] not in self.constraints["valid_sex"]:
                results.append(ValidationResult(
                    check_id="E_VAL_008",
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid sex value: {patient['sex']}"
                ))
        
        return results


def validate_cardiology_input(
    ecg_signal: Optional[np.ndarray] = None,
    ecg_sample_rate: int = 500,
    echo_images: Optional[List[Tuple[str, bytes]]] = None,
    echo_video: Optional[Tuple[str, bytes]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> ValidationReport:
    """
    Validate all cardiology inputs.
    
    Args:
        ecg_signal: ECG signal array
        ecg_sample_rate: ECG sample rate in Hz
        echo_images: List of (filename, bytes) tuples
        echo_video: (filename, bytes) tuple
        metadata: Clinical metadata dictionary
    
    Returns:
        ValidationReport with all validation results
    """
    all_results = []
    modalities = []
    
    # Validate ECG
    if ecg_signal is not None and len(ecg_signal) > 0:
        modalities.append("ecg_signal")
        validator = ECGValidator()
        results = validator.validate(ecg_signal, ecg_sample_rate)
        all_results.extend(results)
    
    # Validate echo images
    if echo_images:
        modalities.append("echo_images")
        validator = EchoValidator()
        for filename, data in echo_images:
            results = validator.validate_image(filename, data)
            all_results.extend(results)
    
    # Validate echo video
    if echo_video:
        modalities.append("echo_video")
        validator = EchoValidator()
        filename, data = echo_video
        results = validator.validate_video(filename, data)
        all_results.extend(results)
    
    # Validate metadata
    if metadata:
        modalities.append("clinical_metadata")
        validator = MetadataValidator()
        results = validator.validate(metadata)
        all_results.extend(results)
    
    # Check minimum modality requirement
    if not modalities or (
        "ecg_signal" not in modalities and
        "echo_images" not in modalities and
        "echo_video" not in modalities
    ):
        all_results.append(ValidationResult(
            check_id="E_VAL_001",
            passed=False,
            severity=ValidationSeverity.ERROR,
            message="No valid modality provided (need ECG or Echo)"
        ))
    
    # Separate by severity
    errors = [r for r in all_results if r.severity == ValidationSeverity.ERROR and not r.passed]
    warnings = [r for r in all_results if r.severity == ValidationSeverity.WARNING and not r.passed]
    info = [r for r in all_results if r.severity == ValidationSeverity.INFO]
    
    return ValidationReport(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        info=info,
        modalities_detected=modalities,
    )

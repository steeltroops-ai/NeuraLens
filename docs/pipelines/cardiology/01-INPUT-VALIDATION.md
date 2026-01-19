# 01 - Input Interface and Validation

## Document Info
| Field | Value |
|-------|-------|
| Stage | 1 - Input Validation |
| Owner | Biomedical Signal Processing Engineer |
| Reviewer | Cardiologist |

---

## 1. Input Reception Interface

### 1.1 API Endpoint Specification

```
POST /api/cardiology/analyze
Content-Type: multipart/form-data
```

### 1.2 Request Structure

```python
class CardiologyAnalysisRequest:
    """Multi-modal cardiology analysis request."""
    
    # Echocardiography inputs (optional if ECG provided)
    echo_images: Optional[List[UploadFile]]    # JPEG/PNG images
    echo_video: Optional[UploadFile]           # MP4/AVI/DICOM video
    
    # ECG signal input (optional if Echo provided)
    ecg_signal: Optional[UploadFile]           # CSV/JSON waveform
    ecg_sample_rate: int = 500                 # Hz (100-1000)
    
    # Clinical metadata (always optional)
    clinical_metadata: Optional[str]           # JSON string
    
    # Processing options
    include_waveform: bool = False             # Return raw data in response
    generate_overlays: bool = True             # Generate visual overlays
    priority: str = "normal"                   # normal | urgent
```

---

## 2. Accepted Modalities

### 2.1 Echocardiography Images

| Parameter | Specification |
|-----------|---------------|
| **Formats** | JPEG (.jpg, .jpeg), PNG (.png) |
| **Min Resolution** | 256 x 256 pixels |
| **Max Resolution** | 4096 x 4096 pixels |
| **Color Space** | RGB or Grayscale |
| **Max File Size** | 10 MB per image |
| **Max Images** | 20 per request |

**Supported Views:**
| View ID | Name | Detection Priority |
|---------|------|-------------------|
| PLAX | Parasternal Long Axis | P1 - Primary |
| PSAX | Parasternal Short Axis | P2 - Secondary |
| A4C | Apical 4-Chamber | P1 - Primary |
| A2C | Apical 2-Chamber | P2 - Secondary |
| A3C | Apical 3-Chamber | P3 - Optional |
| SUBCOSTAL | Subcostal View | P3 - Optional |
| SUPRASTERNAL | Suprasternal View | P3 - Optional |

### 2.2 Echocardiography Videos

| Parameter | Specification |
|-----------|---------------|
| **Formats** | MP4 (.mp4), AVI (.avi), DICOM (.dcm) |
| **Codecs** | H.264, H.265, MPEG-4 |
| **Frame Rate** | 15-120 FPS |
| **Duration** | 1-60 seconds |
| **Min Resolution** | 256 x 256 pixels |
| **Max Resolution** | 1920 x 1080 pixels |
| **Max File Size** | 100 MB |

### 2.3 ECG Signals

| Parameter | Specification |
|-----------|---------------|
| **Formats** | CSV (.csv), JSON (.json) |
| **Sample Rate** | 100-1000 Hz (optimal: 500 Hz) |
| **Duration** | 5-300 seconds |
| **Channels** | 1-12 leads |
| **Amplitude** | -5 to +5 mV typical |
| **Max File Size** | 5 MB |

**CSV Format:**
```csv
# Single lead
time_s,lead_I
0.000,0.15
0.002,0.18
...

# Multi-lead
time_s,lead_I,lead_II,lead_V1,lead_V2,lead_V3,lead_V4,lead_V5,lead_V6
0.000,0.15,0.12,0.08,-0.05,0.02,0.10,0.14,0.11
...
```

**JSON Format:**
```json
{
  "sample_rate_hz": 500,
  "duration_seconds": 10,
  "leads": {
    "I": [0.15, 0.18, 0.22, ...],
    "II": [0.12, 0.14, 0.17, ...],
    "V1": [0.08, 0.06, 0.04, ...]
  },
  "unit": "mV",
  "timestamp_iso": "2026-01-19T10:30:00Z"
}
```

### 2.4 Clinical Metadata

```json
{
  "patient": {
    "age_years": 65,
    "sex": "male",                    // male | female | other
    "height_cm": 175,
    "weight_kg": 80
  },
  "vitals": {
    "systolic_bp_mmhg": 130,
    "diastolic_bp_mmhg": 85,
    "resting_hr_bpm": 72
  },
  "symptoms": [
    "chest_pain",
    "dyspnea",
    "palpitations",
    "fatigue",
    "syncope",
    "edema"
  ],
  "history": {
    "hypertension": true,
    "diabetes": false,
    "prior_mi": false,
    "prior_cabg": false,
    "afib_history": false
  },
  "medications": [
    {"name": "metoprolol", "dose_mg": 50},
    {"name": "lisinopril", "dose_mg": 10}
  ]
}
```

---

## 3. Validation Rules

### 3.1 Input Presence Validation

```python
INPUT_REQUIREMENTS = {
    "minimum_modalities": 1,  # At least echo OR ecg required
    "acceptable_combinations": [
        ["echo_images"],
        ["echo_video"],
        ["ecg_signal"],
        ["echo_images", "ecg_signal"],
        ["echo_video", "ecg_signal"],
        ["echo_images", "clinical_metadata"],
        ["ecg_signal", "clinical_metadata"],
        ["echo_images", "ecg_signal", "clinical_metadata"],
        ["echo_video", "ecg_signal", "clinical_metadata"]
    ]
}
```

### 3.2 Imaging Validation Checklist

| Check | Criterion | Error Code |
|-------|-----------|------------|
| File Type | Extension in [.jpg, .jpeg, .png, .mp4, .avi, .dcm] | E_IMG_001 |
| File Size | Under max limit per format | E_IMG_002 |
| Resolution | Within min-max bounds | E_IMG_003 |
| File Integrity | Can be decoded without error | E_IMG_004 |
| Frame Count (video) | >= 15 frames | E_IMG_005 |
| Frame Rate (video) | 15-120 FPS | E_IMG_006 |
| View Detection | At least one recognizable cardiac view | E_IMG_007 |
| Motion Check (video) | Non-static, contains motion | E_IMG_008 |
| Empty Frame Check | < 10% empty/black frames | E_IMG_009 |
| Corruption Check | No corrupted/truncated data | E_IMG_010 |

### 3.3 ECG Validation Checklist

| Check | Criterion | Error Code |
|-------|-----------|------------|
| File Type | Extension in [.csv, .json] | E_ECG_001 |
| File Size | Under 5 MB | E_ECG_002 |
| Sample Rate | 100-1000 Hz | E_ECG_003 |
| Duration | 5-300 seconds | E_ECG_004 |
| Signal Presence | Signal array not empty | E_ECG_005 |
| Amplitude Range | Values within -5 to +5 mV | E_ECG_006 |
| Flatline Check | No >2 second flatline segments | E_ECG_007 |
| Noise Check | SNR > 5 dB | E_ECG_008 |
| Missing Data | < 5% NaN/null values | E_ECG_009 |
| Lead Config | Valid lead names if specified | E_ECG_010 |
| Clipping Check | No signal saturation | E_ECG_011 |

### 3.4 Metadata Validation Checklist

| Check | Criterion | Error Code |
|-------|-----------|------------|
| Schema Valid | Matches expected JSON schema | E_META_001 |
| Age Range | 0-120 years | E_META_002 |
| Sex Value | One of: male, female, other | E_META_003 |
| BP Range | Systolic 60-300, Diastolic 30-200 | E_META_004 |
| Weight Range | 1-500 kg | E_META_005 |
| Height Range | 30-250 cm | E_META_006 |
| Symptom Codes | All symptoms in allowed list | E_META_007 |

---

## 4. Validation Implementation

```python
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
import numpy as np

class ValidationSeverity(Enum):
    ERROR = "error"       # Reject input
    WARNING = "warning"   # Accept with caution
    INFO = "info"         # Informational note

@dataclass
class ValidationResult:
    """Result of a single validation check."""
    check_id: str
    passed: bool
    severity: ValidationSeverity
    message: str
    details: Optional[dict] = None

@dataclass
class ValidationReport:
    """Complete validation report for all inputs."""
    is_valid: bool
    errors: List[ValidationResult]
    warnings: List[ValidationResult]
    info: List[ValidationResult]
    modalities_detected: List[str]
    
    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "errors": [{"check": e.check_id, "message": e.message} for e in self.errors],
            "warnings": [{"check": w.check_id, "message": w.message} for w in self.warnings],
            "modalities_detected": self.modalities_detected
        }


class EchoValidator:
    """Validate echocardiography inputs."""
    
    ALLOWED_IMAGE_TYPES = {".jpg", ".jpeg", ".png"}
    ALLOWED_VIDEO_TYPES = {".mp4", ".avi", ".dcm"}
    
    MIN_RESOLUTION = 256
    MAX_RESOLUTION = 4096
    MIN_FRAME_RATE = 15
    MAX_FRAME_RATE = 120
    MAX_IMAGE_SIZE_MB = 10
    MAX_VIDEO_SIZE_MB = 100
    
    def validate_image(self, file_path: str, file_bytes: bytes) -> List[ValidationResult]:
        results = []
        
        # Check file type
        ext = file_path.lower().split('.')[-1]
        if f".{ext}" not in self.ALLOWED_IMAGE_TYPES:
            results.append(ValidationResult(
                check_id="E_IMG_001",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid file type: .{ext}. Allowed: {self.ALLOWED_IMAGE_TYPES}"
            ))
            return results  # Can't proceed if file type is wrong
        
        # Check file size
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > self.MAX_IMAGE_SIZE_MB:
            results.append(ValidationResult(
                check_id="E_IMG_002",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"File size {size_mb:.1f}MB exceeds limit of {self.MAX_IMAGE_SIZE_MB}MB"
            ))
        
        # Decode and check resolution
        try:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(file_bytes))
            width, height = img.size
            
            if width < self.MIN_RESOLUTION or height < self.MIN_RESOLUTION:
                results.append(ValidationResult(
                    check_id="E_IMG_003",
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Resolution {width}x{height} below minimum {self.MIN_RESOLUTION}x{self.MIN_RESOLUTION}"
                ))
            elif width > self.MAX_RESOLUTION or height > self.MAX_RESOLUTION:
                results.append(ValidationResult(
                    check_id="E_IMG_003",
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Resolution {width}x{height} exceeds maximum {self.MAX_RESOLUTION}x{self.MAX_RESOLUTION}"
                ))
            else:
                results.append(ValidationResult(
                    check_id="E_IMG_003",
                    passed=True,
                    severity=ValidationSeverity.INFO,
                    message=f"Resolution {width}x{height} validated"
                ))
                
        except Exception as e:
            results.append(ValidationResult(
                check_id="E_IMG_004",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Failed to decode image: {str(e)}"
            ))
        
        return results
    
    def validate_video(self, file_path: str, file_bytes: bytes) -> List[ValidationResult]:
        """Validate echocardiography video."""
        results = []
        
        # Check file type
        ext = file_path.lower().split('.')[-1]
        if f".{ext}" not in self.ALLOWED_VIDEO_TYPES:
            results.append(ValidationResult(
                check_id="E_IMG_001",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid video type: .{ext}. Allowed: {self.ALLOWED_VIDEO_TYPES}"
            ))
            return results
        
        # Check file size
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > self.MAX_VIDEO_SIZE_MB:
            results.append(ValidationResult(
                check_id="E_IMG_002",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Video size {size_mb:.1f}MB exceeds limit of {self.MAX_VIDEO_SIZE_MB}MB"
            ))
        
        # Validate video properties using cv2
        try:
            import cv2
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            
            cap = cv2.VideoCapture(tmp_path)
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            
            # Frame count check
            if frame_count < 15:
                results.append(ValidationResult(
                    check_id="E_IMG_005",
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Video has only {frame_count} frames, minimum 15 required"
                ))
            
            # FPS check
            if fps < self.MIN_FRAME_RATE or fps > self.MAX_FRAME_RATE:
                results.append(ValidationResult(
                    check_id="E_IMG_006",
                    passed=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"Frame rate {fps:.1f} outside optimal range {self.MIN_FRAME_RATE}-{self.MAX_FRAME_RATE}"
                ))
            
            # Resolution check
            if width < self.MIN_RESOLUTION or height < self.MIN_RESOLUTION:
                results.append(ValidationResult(
                    check_id="E_IMG_003",
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Video resolution {width}x{height} below minimum"
                ))
                
        except Exception as e:
            results.append(ValidationResult(
                check_id="E_IMG_004",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Failed to decode video: {str(e)}"
            ))
        
        return results
    
    def detect_view(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Detect echocardiographic view from image.
        
        Returns:
            Tuple of (view_id, confidence)
        """
        # View classification model would be called here
        # For now, return placeholder
        return ("UNKNOWN", 0.0)


class ECGValidator:
    """Validate ECG signal inputs."""
    
    MIN_SAMPLE_RATE = 100
    MAX_SAMPLE_RATE = 1000
    OPTIMAL_SAMPLE_RATE = 500
    
    MIN_DURATION_SEC = 5
    MAX_DURATION_SEC = 300
    
    MAX_FILE_SIZE_MB = 5
    
    AMPLITUDE_RANGE = (-5.0, 5.0)  # mV
    MAX_FLATLINE_SEC = 2.0
    MIN_SNR_DB = 5.0
    MAX_MISSING_RATIO = 0.05
    
    def validate(
        self, 
        signal: np.ndarray, 
        sample_rate: int,
        file_size_bytes: int
    ) -> List[ValidationResult]:
        """Validate ECG signal data."""
        results = []
        
        # File size check
        size_mb = file_size_bytes / (1024 * 1024)
        if size_mb > self.MAX_FILE_SIZE_MB:
            results.append(ValidationResult(
                check_id="E_ECG_002",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"ECG file size {size_mb:.1f}MB exceeds limit of {self.MAX_FILE_SIZE_MB}MB"
            ))
        
        # Sample rate validation
        if sample_rate < self.MIN_SAMPLE_RATE:
            results.append(ValidationResult(
                check_id="E_ECG_003",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Sample rate {sample_rate}Hz below minimum {self.MIN_SAMPLE_RATE}Hz"
            ))
        elif sample_rate > self.MAX_SAMPLE_RATE:
            results.append(ValidationResult(
                check_id="E_ECG_003",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Sample rate {sample_rate}Hz exceeds maximum {self.MAX_SAMPLE_RATE}Hz"
            ))
        
        # Duration check
        duration = len(signal) / sample_rate
        if duration < self.MIN_DURATION_SEC:
            results.append(ValidationResult(
                check_id="E_ECG_004",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Duration {duration:.1f}s below minimum {self.MIN_DURATION_SEC}s"
            ))
        elif duration > self.MAX_DURATION_SEC:
            results.append(ValidationResult(
                check_id="E_ECG_004",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Duration {duration:.1f}s exceeds maximum {self.MAX_DURATION_SEC}s"
            ))
        
        # Empty signal check
        if len(signal) == 0:
            results.append(ValidationResult(
                check_id="E_ECG_005",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message="ECG signal array is empty"
            ))
            return results  # Can't continue validation
        
        # Amplitude range check
        min_val, max_val = np.nanmin(signal), np.nanmax(signal)
        if min_val < self.AMPLITUDE_RANGE[0] or max_val > self.AMPLITUDE_RANGE[1]:
            results.append(ValidationResult(
                check_id="E_ECG_006",
                passed=False,
                severity=ValidationSeverity.WARNING,
                message=f"Signal range [{min_val:.2f}, {max_val:.2f}]mV outside typical range"
            ))
        
        # Flatline detection
        flatline_check = self._check_flatline(signal, sample_rate)
        if not flatline_check["passed"]:
            results.append(ValidationResult(
                check_id="E_ECG_007",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=flatline_check["message"]
            ))
        
        # SNR estimation
        snr = self._estimate_snr(signal, sample_rate)
        if snr < self.MIN_SNR_DB:
            results.append(ValidationResult(
                check_id="E_ECG_008",
                passed=False,
                severity=ValidationSeverity.WARNING,
                message=f"Signal quality low: SNR {snr:.1f}dB below threshold {self.MIN_SNR_DB}dB"
            ))
        
        # Missing data check
        missing_ratio = np.sum(np.isnan(signal)) / len(signal)
        if missing_ratio > self.MAX_MISSING_RATIO:
            results.append(ValidationResult(
                check_id="E_ECG_009",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Missing data ratio {missing_ratio*100:.1f}% exceeds limit of {self.MAX_MISSING_RATIO*100}%"
            ))
        
        # Clipping check
        if self._check_clipping(signal):
            results.append(ValidationResult(
                check_id="E_ECG_011",
                passed=False,
                severity=ValidationSeverity.WARNING,
                message="Signal clipping detected - amplitude may be saturated"
            ))
        
        return results
    
    def _check_flatline(self, signal: np.ndarray, sample_rate: int) -> dict:
        """Check for extended flatline segments."""
        flatline_samples = int(self.MAX_FLATLINE_SEC * sample_rate)
        
        # Rolling standard deviation check
        window = min(100, len(signal) // 10)
        for i in range(0, len(signal) - flatline_samples, window):
            segment = signal[i:i + flatline_samples]
            if np.std(segment) < 0.001:
                return {
                    "passed": False,
                    "message": f"Flatline detected at {i/sample_rate:.1f}s for >{self.MAX_FLATLINE_SEC}s"
                }
        
        return {"passed": True, "message": "No flatline detected"}
    
    def _estimate_snr(self, signal: np.ndarray, sample_rate: int) -> float:
        """Estimate signal-to-noise ratio in dB."""
        try:
            from scipy.signal import butter, filtfilt
            
            # High-pass filter to estimate noise
            nyquist = sample_rate / 2
            b, a = butter(2, min(40 / nyquist, 0.99), btype='high')
            noise = filtfilt(b, a, signal)
            
            signal_power = np.var(signal)
            noise_power = np.var(noise)
            
            if noise_power > 0:
                return 10 * np.log10(signal_power / noise_power)
            return 20.0
        except:
            return 10.0  # Default if scipy unavailable
    
    def _check_clipping(self, signal: np.ndarray) -> bool:
        """Check for signal clipping/saturation."""
        max_val = np.max(np.abs(signal))
        
        # Check if many samples are at the max value (indicating clipping)
        at_max = np.sum(np.abs(signal) > max_val * 0.99) / len(signal)
        return at_max > 0.01  # More than 1% clipped


class MetadataValidator:
    """Validate clinical metadata."""
    
    REQUIRED_FIELDS = []  # All fields optional
    
    VALID_SEX = {"male", "female", "other"}
    
    VALID_SYMPTOMS = {
        "chest_pain", "dyspnea", "palpitations", "fatigue",
        "syncope", "edema", "dizziness", "orthopnea"
    }
    
    RANGES = {
        "age_years": (0, 120),
        "height_cm": (30, 250),
        "weight_kg": (1, 500),
        "systolic_bp_mmhg": (60, 300),
        "diastolic_bp_mmhg": (30, 200),
        "resting_hr_bpm": (20, 300)
    }
    
    def validate(self, metadata: dict) -> List[ValidationResult]:
        """Validate clinical metadata JSON."""
        results = []
        
        if not metadata:
            return results  # Empty metadata is valid
        
        # Patient info validation
        patient = metadata.get("patient", {})
        
        # Age validation
        if "age_years" in patient:
            age = patient["age_years"]
            if not self.RANGES["age_years"][0] <= age <= self.RANGES["age_years"][1]:
                results.append(ValidationResult(
                    check_id="E_META_002",
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Age {age} outside valid range"
                ))
        
        # Sex validation
        if "sex" in patient:
            if patient["sex"] not in self.VALID_SEX:
                results.append(ValidationResult(
                    check_id="E_META_003",
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid sex value: {patient['sex']}"
                ))
        
        # Vitals validation
        vitals = metadata.get("vitals", {})
        for field, value in vitals.items():
            if field in self.RANGES:
                min_val, max_val = self.RANGES[field]
                if not min_val <= value <= max_val:
                    results.append(ValidationResult(
                        check_id="E_META_004",
                        passed=False,
                        severity=ValidationSeverity.WARNING,
                        message=f"{field} value {value} outside expected range"
                    ))
        
        # Symptoms validation
        symptoms = metadata.get("symptoms", [])
        for symptom in symptoms:
            if symptom not in self.VALID_SYMPTOMS:
                results.append(ValidationResult(
                    check_id="E_META_007",
                    passed=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"Unknown symptom code: {symptom}"
                ))
        
        return results
```

---

## 5. Error Response Schema

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_FAILED",
    "message": "Input validation failed with 2 errors",
    "stage": "VALIDATION",
    "timestamp": "2026-01-19T10:30:00.000Z",
    "validation_report": {
      "is_valid": false,
      "errors": [
        {
          "check": "E_ECG_004",
          "message": "Duration 3.5s below minimum 5s"
        },
        {
          "check": "E_IMG_003",
          "message": "Resolution 128x128 below minimum 256x256"
        }
      ],
      "warnings": [
        {
          "check": "E_ECG_008",
          "message": "Signal quality low: SNR 4.2dB"
        }
      ],
      "modalities_detected": ["ecg_signal", "echo_images"]
    },
    "recoverable": true,
    "resubmission_hint": "Increase ECG duration to at least 5 seconds. Use higher resolution echo images."
  }
}
```

---

## 6. Validation Stage Confirmation

When validation passes, the pipeline returns a stage confirmation:

```json
{
  "stage_complete": "VALIDATION",
  "stage_id": 1,
  "status": "success",
  "timestamp": "2026-01-19T10:30:00.000Z",
  "summary": {
    "modalities_validated": ["ecg_signal", "echo_video", "clinical_metadata"],
    "ecg_duration_seconds": 30,
    "ecg_sample_rate_hz": 500,
    "video_frames": 450,
    "video_fps": 30,
    "metadata_fields_present": ["age", "sex", "bp", "symptoms"]
  },
  "next_stage": "PREPROCESSING"
}
```

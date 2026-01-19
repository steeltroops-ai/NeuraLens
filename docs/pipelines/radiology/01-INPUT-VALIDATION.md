# 01 - Input Interface and Validation

## Document Info
| Field | Value |
|-------|-------|
| Stage | 1 - Input Validation |
| Owner | Computer Vision Engineer |
| Reviewer | Radiologist |

---

## 1. Input Reception Interface

### 1.1 API Endpoint Specification

```
POST /api/radiology/analyze
Content-Type: multipart/form-data
```

### 1.2 Request Structure

```python
class RadiologyAnalysisRequest:
    """Multi-modal radiology analysis request."""
    
    # Single image inputs
    image: Optional[UploadFile]              # JPEG/PNG single image
    
    # DICOM inputs
    dicom_file: Optional[UploadFile]         # Single DICOM file
    dicom_series: Optional[List[UploadFile]] # Multiple DICOM files (volume)
    
    # Volumetric inputs
    volume_archive: Optional[UploadFile]     # ZIP/TAR containing DICOM series
    nifti_file: Optional[UploadFile]         # NIfTI volume (.nii, .nii.gz)
    
    # Modality hint (optional, auto-detected if not provided)
    modality_hint: Optional[str]             # "chest_xray" | "ct" | "mri"
    body_region_hint: Optional[str]          # "chest" | "abdomen" | "brain" | etc
    
    # Processing options
    generate_heatmap: bool = True            # Generate Grad-CAM overlay
    generate_segmentation: bool = True       # Generate anatomical masks
    return_dicom_metadata: bool = False      # Include DICOM tags in response
    priority: str = "normal"                 # normal | urgent
```

---

## 2. Accepted Input Formats

### 2.1 Standard Images (PNG/JPEG)

| Parameter | Specification |
|-----------|---------------|
| **Formats** | JPEG (.jpg, .jpeg), PNG (.png) |
| **Min Resolution** | 224 x 224 pixels |
| **Max Resolution** | 4096 x 4096 pixels |
| **Recommended Resolution** | >= 512 x 512 pixels |
| **Color Space** | Grayscale (preferred) or RGB |
| **Bit Depth** | 8-bit or 16-bit |
| **Max File Size** | 10 MB per image |

### 2.2 DICOM Files

| Parameter | Specification |
|-----------|---------------|
| **Format** | DICOM (.dcm, .dicom) |
| **DICOM Version** | 3.0 compliant |
| **Transfer Syntaxes** | Explicit VR Little Endian, Implicit VR Little Endian, JPEG Lossless |
| **Max File Size** | 50 MB per file |
| **Max Series Size** | 500 MB total |
| **Max Slices** | 1000 per series |

**Required DICOM Tags:**
| Tag | Name | Purpose |
|-----|------|---------|
| (0008,0060) | Modality | CR/CT/MR detection |
| (0028,0010) | Rows | Image dimensions |
| (0028,0011) | Columns | Image dimensions |
| (0028,0100) | Bits Allocated | Pixel depth |
| (0028,1050) | Window Center | Display settings (CT) |
| (0028,1051) | Window Width | Display settings (CT) |
| (0018,0050) | Slice Thickness | Volume reconstruction |
| (0020,0013) | Instance Number | Slice ordering |
| (0020,0032) | Image Position | Spatial coordinates |

**Optional but Recommended Tags:**
| Tag | Name | Purpose |
|-----|------|---------|
| (0018,0015) | Body Part Examined | Body region routing |
| (0008,0018) | SOP Instance UID | Unique identification |
| (0010,0040) | Patient Sex | Demographic context |
| (0010,1010) | Patient Age | Demographic context |
| (0018,0088) | Spacing Between Slices | 3D reconstruction |
| (0028,0030) | Pixel Spacing | Physical size |

### 2.3 Volumetric Data

| Parameter | Specification |
|-----------|---------------|
| **Formats** | DICOM series, NIfTI (.nii, .nii.gz) |
| **Archive Formats** | ZIP (.zip), TAR (.tar, .tar.gz) |
| **Min Slices** | 10 for volumetric analysis |
| **Max Slices** | 1000 |
| **Slice Thickness** | 0.5 - 10 mm |
| **Max Volume Size** | 500 MB |

---

## 3. Modality-Specific Requirements

### 3.1 Chest X-Ray
| Requirement | Value | Error Code |
|-------------|-------|------------|
| Orientation | PA, AP, or Lateral | E_CXR_001 |
| Min Resolution | 512 x 512 | E_CXR_002 |
| Lung Visibility | Both lungs visible | E_CXR_003 |
| Rotation | < 5 degrees | W_CXR_001 |
| Inspiration | Adequate (>= 9 ribs) | W_CXR_002 |

### 3.2 CT Scan
| Requirement | Value | Error Code |
|-------------|-------|------------|
| Slice Thickness | 0.5 - 5 mm | E_CT_001 |
| Slice Consistency | Uniform spacing +/- 10% | E_CT_002 |
| Window/Level | Valid HU range | W_CT_001 |
| Coverage | Complete anatomical region | W_CT_002 |
| Contrast Phase | Consistent across slices | W_CT_003 |

### 3.3 MRI Scan
| Requirement | Value | Error Code |
|-------------|-------|------------|
| Sequence Type | Identifiable (T1/T2/FLAIR/DWI) | E_MRI_001 |
| Slice Consistency | Uniform spacing | E_MRI_002 |
| Signal Intensity | No saturation | W_MRI_001 |
| Motion Artifacts | Minimal | W_MRI_002 |
| Bias Field | Acceptable uniformity | W_MRI_003 |

---

## 4. Validation Rules

### 4.1 Universal Validation Checklist

| Check | Criterion | Error Code | Severity |
|-------|-----------|------------|----------|
| File Exists | Non-empty file received | E_GEN_001 | Error |
| File Type | Extension in allowed list | E_GEN_002 | Error |
| File Size | Under max limit | E_GEN_003 | Error |
| File Integrity | Decodable without error | E_GEN_004 | Error |
| Image Dimensions | Within min-max bounds | E_GEN_005 | Error |
| Empty Check | Non-black/non-white image | E_GEN_006 | Error |
| Medical Content | Appears to be medical image | E_GEN_007 | Error |

### 4.2 DICOM-Specific Validation

| Check | Criterion | Error Code | Severity |
|-------|-----------|------------|----------|
| DICOM Parse | Valid DICOM structure | E_DCM_001 | Error |
| Required Tags | All required tags present | E_DCM_002 | Error |
| Modality Tag | Valid modality code | E_DCM_003 | Error |
| Pixel Data | Transfer syntax supported | E_DCM_004 | Error |
| Slice Order | Consistent instance numbers | E_DCM_005 | Error |
| Spatial Consistency | Matching orientation | E_DCM_006 | Warning |
| Window/Level | Valid range for modality | W_DCM_001 | Warning |

### 4.3 Volume-Specific Validation

| Check | Criterion | Error Code | Severity |
|-------|-----------|------------|----------|
| Min Slices | >= 10 slices | E_VOL_001 | Error |
| Max Slices | <= 1000 slices | E_VOL_002 | Error |
| Slice Gaps | No missing slices | E_VOL_003 | Error |
| Orientation Consistency | Same orientation | E_VOL_004 | Error |
| Dimension Consistency | Same XY dimensions | E_VOL_005 | Error |
| Spacing Uniformity | Uniform +/- 10% | W_VOL_001 | Warning |

### 4.4 Image Quality Validation

| Check | Criterion | Error Code | Severity |
|-------|-----------|------------|----------|
| Contrast | Sufficient dynamic range | W_QA_001 | Warning |
| Noise Level | Acceptable SNR | W_QA_002 | Warning |
| Blur | Acceptable sharpness | W_QA_003 | Warning |
| Artifacts | No severe artifacts | W_QA_004 | Warning |
| Cropping | No critical area cropped | E_QA_001 | Error |

---

## 5. Validation Implementation

```python
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from enum import Enum
import numpy as np
import pydicom
from PIL import Image
from io import BytesIO

class ValidationSeverity(Enum):
    ERROR = "error"       # Reject input
    WARNING = "warning"   # Accept with caution
    INFO = "info"         # Informational note

class ModalityType(Enum):
    CHEST_XRAY = "chest_xray"
    CT = "ct"
    MRI = "mri"
    UNKNOWN = "unknown"

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
    modality_detected: ModalityType
    body_region_detected: Optional[str]
    is_volumetric: bool
    slice_count: int
    dicom_metadata: Optional[dict]
    
    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "errors": [{"check": e.check_id, "message": e.message} for e in self.errors],
            "warnings": [{"check": w.check_id, "message": w.message} for w in self.warnings],
            "modality_detected": self.modality_detected.value,
            "body_region_detected": self.body_region_detected,
            "is_volumetric": self.is_volumetric,
            "slice_count": self.slice_count
        }


class ImageValidator:
    """Validate standard image inputs (PNG/JPEG)."""
    
    ALLOWED_TYPES = {".jpg", ".jpeg", ".png"}
    MIN_RESOLUTION = 224
    MAX_RESOLUTION = 4096
    MAX_FILE_SIZE_MB = 10
    
    # Minimum intensity variance to detect non-medical/blank images
    MIN_INTENSITY_VARIANCE = 100
    
    def validate(self, file_path: str, file_bytes: bytes) -> List[ValidationResult]:
        results = []
        
        # Check file type
        ext = "." + file_path.lower().split('.')[-1]
        if ext not in self.ALLOWED_TYPES:
            results.append(ValidationResult(
                check_id="E_GEN_002",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid file type: {ext}. Allowed: {self.ALLOWED_TYPES}"
            ))
            return results
        
        # Check file size
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > self.MAX_FILE_SIZE_MB:
            results.append(ValidationResult(
                check_id="E_GEN_003",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"File size {size_mb:.1f}MB exceeds limit of {self.MAX_FILE_SIZE_MB}MB"
            ))
        
        # Decode and validate image
        try:
            img = Image.open(BytesIO(file_bytes))
            width, height = img.size
            
            # Resolution check
            if width < self.MIN_RESOLUTION or height < self.MIN_RESOLUTION:
                results.append(ValidationResult(
                    check_id="E_GEN_005",
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Resolution {width}x{height} below minimum {self.MIN_RESOLUTION}x{self.MIN_RESOLUTION}"
                ))
            elif width > self.MAX_RESOLUTION or height > self.MAX_RESOLUTION:
                results.append(ValidationResult(
                    check_id="E_GEN_005",
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Resolution {width}x{height} exceeds maximum {self.MAX_RESOLUTION}x{self.MAX_RESOLUTION}"
                ))
            
            # Convert to grayscale for analysis
            img_gray = img.convert('L')
            img_array = np.array(img_gray)
            
            # Empty/blank image check
            variance = np.var(img_array)
            if variance < self.MIN_INTENSITY_VARIANCE:
                results.append(ValidationResult(
                    check_id="E_GEN_006",
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message="Image appears to be blank or nearly uniform"
                ))
            
            # Medical content detection (basic heuristic)
            if not self._appears_medical(img_array):
                results.append(ValidationResult(
                    check_id="E_GEN_007",
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message="Image does not appear to be a medical radiological image"
                ))
            
        except Exception as e:
            results.append(ValidationResult(
                check_id="E_GEN_004",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Failed to decode image: {str(e)}"
            ))
        
        return results
    
    def _appears_medical(self, img_array: np.ndarray) -> bool:
        """Basic heuristic to check if image looks like medical imaging."""
        # Check intensity distribution (medical images typically have specific patterns)
        hist, _ = np.histogram(img_array.flatten(), bins=256, range=(0, 255))
        
        # Medical X-rays typically have strong peaks at low and high intensities
        low_intensity = np.sum(hist[:50])
        high_intensity = np.sum(hist[200:])
        mid_intensity = np.sum(hist[50:200])
        
        total = np.sum(hist)
        
        # Heuristic: X-rays should have significant content in dark and light regions
        if (low_intensity / total > 0.1 or high_intensity / total > 0.1):
            return True
        
        # Accept if there's good dynamic range
        if np.ptp(img_array) > 100:
            return True
        
        return False


class DicomValidator:
    """Validate DICOM file inputs."""
    
    MAX_FILE_SIZE_MB = 50
    
    REQUIRED_TAGS = [
        (0x0008, 0x0060),  # Modality
        (0x0028, 0x0010),  # Rows
        (0x0028, 0x0011),  # Columns
        (0x0028, 0x0100),  # Bits Allocated
    ]
    
    SUPPORTED_MODALITIES = {"CR", "DX", "CT", "MR", "PT", "NM", "XA", "RF", "MG"}
    
    def validate(self, file_bytes: bytes) -> Tuple[List[ValidationResult], Optional[dict]]:
        results = []
        metadata = None
        
        # Check file size
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > self.MAX_FILE_SIZE_MB:
            results.append(ValidationResult(
                check_id="E_GEN_003",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"DICOM file size {size_mb:.1f}MB exceeds limit of {self.MAX_FILE_SIZE_MB}MB"
            ))
        
        # Parse DICOM
        try:
            from io import BytesIO
            ds = pydicom.dcmread(BytesIO(file_bytes))
            
            # Check required tags
            missing_tags = []
            for tag in self.REQUIRED_TAGS:
                if tag not in ds:
                    missing_tags.append(f"({tag[0]:04x},{tag[1]:04x})")
            
            if missing_tags:
                results.append(ValidationResult(
                    check_id="E_DCM_002",
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Missing required DICOM tags: {', '.join(missing_tags)}"
                ))
            
            # Check modality
            modality = getattr(ds, 'Modality', None)
            if modality and modality not in self.SUPPORTED_MODALITIES:
                results.append(ValidationResult(
                    check_id="E_DCM_003",
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Unsupported modality: {modality}. Supported: {self.SUPPORTED_MODALITIES}"
                ))
            
            # Check pixel data
            if not hasattr(ds, 'PixelData'):
                results.append(ValidationResult(
                    check_id="E_DCM_004",
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message="DICOM file does not contain pixel data"
                ))
            else:
                # Try to extract pixel array
                try:
                    pixel_array = ds.pixel_array
                except Exception as e:
                    results.append(ValidationResult(
                        check_id="E_DCM_004",
                        passed=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Cannot decode pixel data: {str(e)}"
                    ))
            
            # Extract metadata
            metadata = {
                "modality": getattr(ds, 'Modality', None),
                "rows": getattr(ds, 'Rows', None),
                "columns": getattr(ds, 'Columns', None),
                "bits_allocated": getattr(ds, 'BitsAllocated', None),
                "window_center": getattr(ds, 'WindowCenter', None),
                "window_width": getattr(ds, 'WindowWidth', None),
                "slice_thickness": getattr(ds, 'SliceThickness', None),
                "pixel_spacing": getattr(ds, 'PixelSpacing', None),
                "body_part": getattr(ds, 'BodyPartExamined', None),
                "patient_sex": getattr(ds, 'PatientSex', None),
                "patient_age": getattr(ds, 'PatientAge', None),
                "instance_number": getattr(ds, 'InstanceNumber', None),
                "image_position": getattr(ds, 'ImagePositionPatient', None),
            }
            
        except pydicom.errors.InvalidDicomError as e:
            results.append(ValidationResult(
                check_id="E_DCM_001",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid DICOM file structure: {str(e)}"
            ))
        except Exception as e:
            results.append(ValidationResult(
                check_id="E_DCM_001",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Failed to parse DICOM: {str(e)}"
            ))
        
        return results, metadata


class VolumeValidator:
    """Validate volumetric data (DICOM series, NIfTI)."""
    
    MIN_SLICES = 10
    MAX_SLICES = 1000
    MAX_SIZE_MB = 500
    SPACING_TOLERANCE = 0.1  # 10% tolerance for slice spacing
    
    def validate_series(
        self, 
        dicom_files: List[bytes],
        metadata_list: List[dict]
    ) -> List[ValidationResult]:
        results = []
        
        # Check slice count
        slice_count = len(dicom_files)
        if slice_count < self.MIN_SLICES:
            results.append(ValidationResult(
                check_id="E_VOL_001",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Volume has only {slice_count} slices, minimum {self.MIN_SLICES} required"
            ))
        elif slice_count > self.MAX_SLICES:
            results.append(ValidationResult(
                check_id="E_VOL_002",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Volume has {slice_count} slices, maximum {self.MAX_SLICES} allowed"
            ))
        
        # Check total size
        total_size_mb = sum(len(f) for f in dicom_files) / (1024 * 1024)
        if total_size_mb > self.MAX_SIZE_MB:
            results.append(ValidationResult(
                check_id="E_GEN_003",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Volume size {total_size_mb:.1f}MB exceeds limit of {self.MAX_SIZE_MB}MB"
            ))
        
        if slice_count < 2:
            return results
        
        # Check dimension consistency
        rows = set(m.get('rows') for m in metadata_list if m.get('rows'))
        cols = set(m.get('columns') for m in metadata_list if m.get('columns'))
        
        if len(rows) > 1 or len(cols) > 1:
            results.append(ValidationResult(
                check_id="E_VOL_005",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message="Inconsistent image dimensions across slices"
            ))
        
        # Check slice ordering and gaps
        instance_numbers = [m.get('instance_number') for m in metadata_list if m.get('instance_number') is not None]
        if instance_numbers:
            sorted_nums = sorted(instance_numbers)
            expected = list(range(sorted_nums[0], sorted_nums[-1] + 1))
            if sorted_nums != expected:
                missing = set(expected) - set(sorted_nums)
                if missing:
                    results.append(ValidationResult(
                        check_id="E_VOL_003",
                        passed=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Missing slices detected: {len(missing)} gaps in instance numbers"
                    ))
        
        # Check slice spacing uniformity
        thicknesses = [m.get('slice_thickness') for m in metadata_list if m.get('slice_thickness')]
        if thicknesses and len(set(thicknesses)) > 1:
            results.append(ValidationResult(
                check_id="W_VOL_001",
                passed=True,
                severity=ValidationSeverity.WARNING,
                message="Non-uniform slice thickness detected"
            ))
        
        return results


class ChestXrayValidator:
    """Chest X-ray specific validation."""
    
    MIN_RESOLUTION = 512
    
    def validate(self, img_array: np.ndarray) -> List[ValidationResult]:
        results = []
        
        height, width = img_array.shape[:2]
        
        # Resolution check
        if width < self.MIN_RESOLUTION or height < self.MIN_RESOLUTION:
            results.append(ValidationResult(
                check_id="E_CXR_002",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Chest X-ray resolution {width}x{height} below minimum {self.MIN_RESOLUTION}x{self.MIN_RESOLUTION}"
            ))
        
        # Check for bilateral lung visibility (basic heuristic)
        # Divide image into left and right halves and check for similar structure
        left_half = img_array[:, :width//2]
        right_half = img_array[:, width//2:]
        
        left_var = np.var(left_half)
        right_var = np.var(right_half)
        
        # Both halves should have significant structure
        if left_var < 100 or right_var < 100:
            results.append(ValidationResult(
                check_id="E_CXR_003",
                passed=False,
                severity=ValidationSeverity.WARNING,
                message="One or both lung fields may not be fully visible"
            ))
        
        return results
```

---

## 6. Input Schema Definition

### 6.1 Accepted Input Schema

```json
{
  "type": "object",
  "properties": {
    "image": {
      "type": "file",
      "contentType": ["image/jpeg", "image/png"],
      "maxSize": "10MB"
    },
    "dicom_file": {
      "type": "file",
      "contentType": ["application/dicom"],
      "maxSize": "50MB"
    },
    "dicom_series": {
      "type": "array",
      "items": {
        "type": "file",
        "contentType": ["application/dicom"]
      },
      "maxItems": 1000,
      "maxTotalSize": "500MB"
    },
    "volume_archive": {
      "type": "file",
      "contentType": ["application/zip", "application/x-tar"],
      "maxSize": "500MB"
    },
    "nifti_file": {
      "type": "file",
      "contentType": ["application/octet-stream"],
      "extensions": [".nii", ".nii.gz"],
      "maxSize": "500MB"
    },
    "modality_hint": {
      "type": "string",
      "enum": ["chest_xray", "ct", "mri"]
    },
    "body_region_hint": {
      "type": "string",
      "enum": ["chest", "abdomen", "pelvis", "brain", "spine", "extremity"]
    },
    "generate_heatmap": {
      "type": "boolean",
      "default": true
    },
    "generate_segmentation": {
      "type": "boolean",
      "default": true
    },
    "return_dicom_metadata": {
      "type": "boolean",
      "default": false
    },
    "priority": {
      "type": "string",
      "enum": ["normal", "urgent"],
      "default": "normal"
    }
  },
  "oneOf": [
    {"required": ["image"]},
    {"required": ["dicom_file"]},
    {"required": ["dicom_series"]},
    {"required": ["volume_archive"]},
    {"required": ["nifti_file"]}
  ]
}
```

---

## 7. Error Response Schema

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
          "check": "E_GEN_005",
          "message": "Resolution 128x128 below minimum 224x224"
        },
        {
          "check": "E_GEN_007",
          "message": "Image does not appear to be a medical radiological image"
        }
      ],
      "warnings": [
        {
          "check": "W_QA_002",
          "message": "Image noise level higher than optimal"
        }
      ],
      "modality_detected": "unknown",
      "is_volumetric": false,
      "slice_count": 1
    },
    "recoverable": true,
    "resubmission_hint": "Upload a higher resolution medical image in JPEG, PNG, or DICOM format."
  }
}
```

---

## 8. Validation Stage Confirmation

When validation passes, the pipeline returns a stage confirmation:

```json
{
  "stage_complete": "VALIDATION",
  "stage_id": 1,
  "status": "success",
  "timestamp": "2026-01-19T10:30:00.000Z",
  "summary": {
    "modality_detected": "chest_xray",
    "body_region": "chest",
    "is_volumetric": false,
    "slice_count": 1,
    "image_dimensions": [1024, 1024],
    "dicom_tags_extracted": true,
    "warnings_count": 0
  },
  "next_stage": "PREPROCESSING"
}
```

---

## 9. Complete Error Code Reference

### 9.1 General Errors (E_GEN_xxx)
| Code | Message | Recoverable | User Action |
|------|---------|-------------|-------------|
| E_GEN_001 | No file received | No | Upload a file |
| E_GEN_002 | Invalid file format | No | Use supported format |
| E_GEN_003 | File size exceeds limit | No | Reduce file size |
| E_GEN_004 | File decode failed | No | Upload valid file |
| E_GEN_005 | Resolution out of range | No | Use appropriate resolution |
| E_GEN_006 | Image appears blank | No | Upload valid image |
| E_GEN_007 | Non-medical image | No | Upload medical image |

### 9.2 DICOM Errors (E_DCM_xxx)
| Code | Message | Recoverable | User Action |
|------|---------|-------------|-------------|
| E_DCM_001 | Invalid DICOM structure | No | Check DICOM file |
| E_DCM_002 | Missing required tags | No | Complete DICOM file |
| E_DCM_003 | Unsupported modality | No | Use supported modality |
| E_DCM_004 | Pixel data error | No | Check transfer syntax |
| E_DCM_005 | Slice order error | Partial | Verify series |
| E_DCM_006 | Spatial inconsistency | Warning | Review orientation |

### 9.3 Modality Errors (E_CXR/CT/MRI_xxx)
| Code | Message | Recoverable | User Action |
|------|---------|-------------|-------------|
| E_CXR_001 | Unknown orientation | Partial | Use PA/AP/Lateral |
| E_CXR_002 | Resolution too low | No | Use higher resolution |
| E_CXR_003 | Lung fields not visible | Partial | Full chest image |
| E_CT_001 | Invalid slice thickness | No | Check scan parameters |
| E_CT_002 | Inconsistent spacing | Partial | Verify acquisition |
| E_MRI_001 | Unknown sequence | Partial | Specify sequence |
| E_MRI_002 | Inconsistent slices | Partial | Verify acquisition |

### 9.4 Volume Errors (E_VOL_xxx)
| Code | Message | Recoverable | User Action |
|------|---------|-------------|-------------|
| E_VOL_001 | Too few slices | No | Upload complete volume |
| E_VOL_002 | Too many slices | No | Split volume |
| E_VOL_003 | Missing slices | Partial | Upload complete series |
| E_VOL_004 | Orientation mismatch | No | Consistent orientation |
| E_VOL_005 | Dimension mismatch | No | Consistent dimensions |

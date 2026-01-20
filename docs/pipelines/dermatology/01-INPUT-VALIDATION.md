# 01 - Input Interface and Validation

## Purpose
Define how the dermatology pipeline receives, validates, and acknowledges skin lesion images from the frontend API.

---

## 1. Input Reception Interface

### API Endpoint
```
POST /api/dermatology/analyze
Content-Type: multipart/form-data
```

### Request Schema
```typescript
interface DermatologyAnalysisRequest {
  // Required
  image: File;                    // Skin lesion image
  
  // Optional metadata
  image_source?: "smartphone" | "dermatoscope" | "clinical";
  body_location?: BodyLocation;   // Anatomical location
  patient_age?: number;           // Age in years
  patient_sex?: "male" | "female" | "other";
  skin_type?: FitzpatrickType;    // I-VI
  lesion_duration?: string;       // "new" | "<3months" | "3-12months" | ">1year"
  has_changed?: boolean;          // Recent changes observed
  prior_image_id?: string;        // For longitudinal comparison
  session_id?: string;            // Tracking ID
}

type BodyLocation = 
  | "head_face" | "head_scalp" | "head_neck"
  | "trunk_anterior" | "trunk_posterior" | "trunk_lateral"
  | "arm_upper" | "arm_lower" | "hand_palm" | "hand_dorsal"
  | "leg_upper" | "leg_lower" | "foot_sole" | "foot_dorsal"
  | "nail" | "genital" | "other";

type FitzpatrickType = 1 | 2 | 3 | 4 | 5 | 6;
```

### Receipt Acknowledgment
```json
{
  "receipt": {
    "acknowledged": true,
    "request_id": "derm_20260120_abc123",
    "timestamp": "2026-01-20T10:30:00Z",
    "image_received": true,
    "file_hash": "sha256:abc123...",
    "file_size_bytes": 2456789,
    "estimated_processing_time_ms": 4500
  }
}
```

---

## 2. Accepted Input Specifications

### 2.1 Smartphone Images if using smartphone

| Parameter | Requirement | Optimal |
|-----------|-------------|---------|
| Resolution | >= 1920x1080 | 3024x4032 (12MP) |
| Format | JPEG, PNG, HEIC | JPEG |
| Color Depth | 8-bit RGB | 8-bit RGB |
| File Size | 500KB - 20MB | 1-5MB |
| Aspect Ratio | 3:4 to 16:9 | 3:4 |
| Focus | Lesion in focus | Macro mode |
| Distance | 5-20cm from lesion | 10cm |
| Lighting | Even, natural/clinical | Diffuse daylight |

### 2.2 Dermatoscope Images using dermatoscope

| Parameter | Requirement | Optimal |
|-----------|-------------|---------|
| Resolution | >= 640x480 | 1920x1080+ |
| Format | JPEG, PNG, DICOM | DICOM |
| Magnification | 10x-20x | 10x |
| Polarization | Cross-polarized preferred | Cross-polarized |
| Immersion | Dry or oil-immersion | Noted in metadata |
| Color Calibration | Device-specific | Calibrated |

---

## 3. Validation Pipeline

### Stage 1: File Validation
```python
class FileValidator:
    ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/heic", "application/dicom"]
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    MIN_FILE_SIZE = 10 * 1024         # 10KB
    
    def validate(self, file: UploadFile) -> ValidationResult:
        checks = [
            self._check_mime_type(file),
            self._check_file_size(file),
            self._check_file_integrity(file),
            self._check_not_empty(file),
        ]
        return ValidationResult(passed=all(checks))
```

### Stage 2: Image Quality Validation
```python
class ImageQualityValidator:
    """Validates image quality for dermatological analysis."""
    
    def validate(self, image: np.ndarray) -> QualityReport:
        return QualityReport(
            resolution=self._check_resolution(image),
            focus=self._check_focus_score(image),
            illumination=self._check_illumination(image),
            color_balance=self._check_color_balance(image),
            overall_quality=self._compute_overall_quality(),
        )
```

#### 3.2.1 Resolution Validation
```python
def _check_resolution(self, image: np.ndarray) -> ResolutionCheck:
    height, width = image.shape[:2]
    megapixels = (height * width) / 1_000_000
    
    if megapixels < 0.3:  # < 0.3MP
        return ResolutionCheck(passed=False, issue="Resolution too low for reliable analysis")
    elif megapixels < 1.0:  # 0.3-1MP
        return ResolutionCheck(passed=True, warning="Low resolution may affect accuracy")
    else:
        return ResolutionCheck(passed=True)
```

#### 3.2.2 Focus/Sharpness Validation
```python
def _check_focus_score(self, image: np.ndarray) -> FocusCheck:
    """
    Uses Laplacian variance to detect blur.
    Higher variance = sharper image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    SHARP_THRESHOLD = 100
    BLURRY_THRESHOLD = 30
    
    if laplacian_var < BLURRY_THRESHOLD:
        return FocusCheck(
            passed=False,
            score=laplacian_var,
            issue="Image is too blurry. Please retake with better focus."
        )
    elif laplacian_var < SHARP_THRESHOLD:
        return FocusCheck(
            passed=True,
            score=laplacian_var,
            warning="Moderate blur detected. Analysis may be affected."
        )
    return FocusCheck(passed=True, score=laplacian_var)
```

#### 3.2.3 Illumination Validation
```python
def _check_illumination(self, image: np.ndarray) -> IlluminationCheck:
    """
    Checks for proper exposure and even lighting.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    value_channel = hsv[:, :, 2]
    
    mean_brightness = np.mean(value_channel)
    std_brightness = np.std(value_channel)
    
    # Check for overexposure
    overexposed_ratio = np.sum(value_channel > 250) / value_channel.size
    underexposed_ratio = np.sum(value_channel < 10) / value_channel.size
    
    issues = []
    if overexposed_ratio > 0.15:
        issues.append("Significant overexposure detected")
    if underexposed_ratio > 0.20:
        issues.append("Image is too dark")
    if std_brightness > 80:
        issues.append("Uneven lighting detected")
        
    return IlluminationCheck(
        passed=len(issues) == 0,
        mean_brightness=mean_brightness,
        uniformity=1.0 - (std_brightness / 128),
        issues=issues
    )
```

#### 3.2.4 Color Balance Validation
```python
def _check_color_balance(self, image: np.ndarray) -> ColorBalanceCheck:
    """
    Validates color cast and white balance.
    Important for accurate lesion color assessment.
    """
    # Calculate channel means
    r_mean = np.mean(image[:, :, 0])
    g_mean = np.mean(image[:, :, 1])
    b_mean = np.mean(image[:, :, 2])
    
    # Check for severe color cast
    max_diff = max(abs(r_mean - g_mean), abs(g_mean - b_mean), abs(r_mean - b_mean))
    
    if max_diff > 50:
        return ColorBalanceCheck(
            passed=False,
            color_cast="severe",
            issue="Strong color cast may affect diagnosis accuracy"
        )
    elif max_diff > 30:
        return ColorBalanceCheck(
            passed=True,
            color_cast="moderate",
            warning="Moderate color cast detected. Will attempt correction."
        )
    return ColorBalanceCheck(passed=True, color_cast="none")
```

### Stage 3: Content Validation
```python
class ContentValidator:
    """Validates image content is appropriate for dermatological analysis."""
    
    def validate(self, image: np.ndarray) -> ContentReport:
        return ContentReport(
            is_skin_image=self._detect_skin_presence(image),
            lesion_detected=self._detect_lesion_presence(image),
            lesion_centered=self._check_lesion_centering(image),
            occlusions=self._detect_occlusions(image),
            body_part=self._detect_body_part(image),
        )
```

#### 3.3.1 Skin Detection
```python
def _detect_skin_presence(self, image: np.ndarray) -> SkinDetectionResult:
    """
    Uses color-based skin detection in YCrCb color space.
    Supports Fitzpatrick skin types I-VI.
    """
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    
    # Broad skin color range (all Fitzpatrick types)
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
    
    if skin_ratio < 0.10:
        return SkinDetectionResult(
            is_skin=False,
            skin_ratio=skin_ratio,
            error="Image does not appear to contain skin"
        )
    elif skin_ratio < 0.30:
        return SkinDetectionResult(
            is_skin=True,
            skin_ratio=skin_ratio,
            warning="Low skin visibility. Ensure lesion is clearly visible."
        )
    return SkinDetectionResult(is_skin=True, skin_ratio=skin_ratio)
```

#### 3.3.2 Lesion Visibility Check
```python
def _detect_lesion_presence(self, image: np.ndarray) -> LesionDetectionResult:
    """
    Quick check for presence of a distinct lesion region.
    Uses edge detection and color contrast analysis.
    """
    # Convert to LAB for perceptual color difference
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Detect high-contrast regions
    edges = cv2.Canny(lab[:, :, 0], 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter for lesion-like contours
    candidate_contours = [c for c in contours if 100 < cv2.contourArea(c) < image.size * 0.5]
    
    if len(candidate_contours) == 0:
        return LesionDetectionResult(
            detected=False,
            error="No distinct lesion detected. Please ensure lesion is visible."
        )
    
    return LesionDetectionResult(
        detected=True,
        candidate_count=len(candidate_contours)
    )
```

#### 3.3.3 Centering Validation
```python
def _check_lesion_centering(self, image: np.ndarray) -> CenteringCheck:
    """
    Verifies lesion is reasonably centered in frame.
    """
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    
    # Define central region (inner 60%)
    margin_x = width * 0.2
    margin_y = height * 0.2
    
    # Get primary lesion centroid
    lesion_centroid = self._get_lesion_centroid(image)
    
    if lesion_centroid is None:
        return CenteringCheck(passed=True, warning="Could not determine lesion center")
    
    lx, ly = lesion_centroid
    
    # Check if lesion center is within central region
    in_center = (
        margin_x < lx < (width - margin_x) and
        margin_y < ly < (height - margin_y)
    )
    
    if not in_center:
        return CenteringCheck(
            passed=True,  # Warning only, not a hard fail
            warning="Lesion is off-center. Best results when lesion is centered."
        )
    return CenteringCheck(passed=True)
```

#### 3.3.4 Occlusion Detection
```python
def _detect_occlusions(self, image: np.ndarray) -> OcclusionReport:
    """
    Detects common occlusions: hair, shadows, rulers, ink markings.
    """
    occlusions = []
    
    # Hair detection (dark thin lines)
    hair_score = self._detect_hair(image)
    if hair_score > 0.15:
        occlusions.append(Occlusion(
            type="hair",
            severity=hair_score,
            message="Hair partially obscures lesion. Consider hair removal."
        ))
    
    # Shadow detection
    shadow_score = self._detect_shadows(image)
    if shadow_score > 0.20:
        occlusions.append(Occlusion(
            type="shadow",
            severity=shadow_score,
            message="Shadows detected. Diffuse lighting recommended."
        ))
    
    # Ruler/scale detection
    if self._detect_ruler(image):
        occlusions.append(Occlusion(
            type="ruler",
            severity=0.1,
            message="Ruler detected - will be masked during analysis."
        ))
    
    # Ink marking detection
    ink_score = self._detect_ink_markings(image)
    if ink_score > 0.05:
        occlusions.append(Occlusion(
            type="ink",
            severity=ink_score,
            message="Ink markings detected near lesion."
        ))
    
    return OcclusionReport(
        has_occlusions=len(occlusions) > 0,
        occlusions=occlusions,
        severe=any(o.severity > 0.3 for o in occlusions)
    )
```

---

## 4. Validation Error Codes

| Code | Category | Message | Recoverable |
|------|----------|---------|-------------|
| `E_VAL_001` | File | Invalid file type. Accepted: JPEG, PNG, HEIC, DICOM | Yes |
| `E_VAL_002` | File | File too large. Maximum: 50MB | Yes |
| `E_VAL_003` | File | File too small or empty | Yes |
| `E_VAL_004` | File | Corrupted or unreadable file | Yes |
| `E_VAL_010` | Resolution | Resolution too low (<0.3MP) | Yes |
| `E_VAL_011` | Focus | Image too blurry for analysis | Yes |
| `E_VAL_012` | Exposure | Image overexposed | Yes |
| `E_VAL_013` | Exposure | Image underexposed | Yes |
| `E_VAL_014` | Lighting | Uneven illumination | Yes |
| `E_VAL_015` | Color | Severe color cast detected | Yes |
| `E_VAL_020` | Content | No skin detected in image | Yes |
| `E_VAL_021` | Content | No lesion detected | Yes |
| `E_VAL_022` | Content | Multiple lesions - single lesion preferred | Warning |
| `E_VAL_023` | Content | Lesion too close to image edge | Yes |
| `E_VAL_024` | Content | Lesion occluded by hair/artifacts | Warning |
| `E_VAL_030` | Metadata | Invalid body location specified | Warning |
| `E_VAL_031` | Metadata | Invalid Fitzpatrick type | Warning |

---

## 5. Validation Response Schema

### Success Response
```json
{
  "validation": {
    "status": "passed",
    "request_id": "derm_20260120_abc123",
    "checks_performed": 12,
    "checks_passed": 12,
    "warnings": [
      {
        "code": "W_VAL_024",
        "message": "Minor hair occlusion detected. Analysis will proceed."
      }
    ],
    "quality_score": 0.87,
    "ready_for_analysis": true
  }
}
```

### Failure Response
```json
{
  "validation": {
    "status": "failed",
    "request_id": "derm_20260120_abc123",
    "checks_performed": 8,
    "checks_passed": 6,
    "errors": [
      {
        "code": "E_VAL_011",
        "message": "Image too blurry for reliable analysis",
        "user_message": {
          "title": "Image Out of Focus",
          "explanation": "The image appears blurry which would affect our ability to accurately assess the lesion.",
          "action": "Please retake the photo ensuring the lesion is in sharp focus. Try using your camera's macro mode if available."
        },
        "recoverable": true
      }
    ],
    "quality_score": 0.32,
    "ready_for_analysis": false,
    "retake_guidance": {
      "recommended": true,
      "tips": [
        "Hold camera steady or use a tripod",
        "Tap on the lesion to focus before taking photo",
        "Ensure good lighting - avoid direct sunlight",
        "Keep camera 10-15cm from the lesion"
      ]
    }
  }
}
```

---

## 6. Input Validation Checklist

```markdown
## Pre-Analysis Validation Checklist

### File Validation
- [ ] File type is JPEG, PNG, HEIC, or DICOM
- [ ] File size between 10KB and 50MB
- [ ] File is not corrupted
- [ ] File contains valid image data

### Image Quality
- [ ] Resolution >= 1920x1080 (smartphone) or >= 640x480 (dermatoscope)
- [ ] Focus score >= 30 (Laplacian variance)
- [ ] Mean brightness between 40-220
- [ ] No more than 15% overexposed pixels
- [ ] No more than 20% underexposed pixels
- [ ] Illumination uniformity >= 0.4
- [ ] Color cast differential < 50

### Content Validation
- [ ] Skin detected (>= 10% of image)
- [ ] Lesion presence detected
- [ ] Lesion center within central 60% of image
- [ ] No severe occlusions (hair < 30%, shadow < 30%)
- [ ] Single primary lesion preferred

### Metadata Validation (if provided)
- [ ] Body location is valid enum value
- [ ] Fitzpatrick type is 1-6
- [ ] Age is reasonable (0-120)
- [ ] Session ID format valid
```

---

## 7. Implementation Notes

### Thread Safety
All validators are stateless and thread-safe. Each request creates new validator instances.

### Performance
- File validation: < 50ms
- Image quality checks: < 200ms
- Content validation: < 500ms
- Total validation: < 1s typical

### Caching
- Validation results cached by file hash for 5 minutes
- Prevents duplicate processing of same image

### Logging
All validation steps are logged with:
- Request ID
- Check name
- Result (pass/fail/warning)
- Metrics captured
- Timestamp

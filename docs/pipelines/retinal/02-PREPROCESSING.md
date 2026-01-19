# Retinal Pipeline - Preprocessing Stages Specification

## Document Info
| Field | Value |
|-------|-------|
| Version | 4.0.0 |
| Pipeline Stage | 2 - Preprocessing |

---

## 1. Preprocessing Pipeline Overview

```
Raw Fundus Image
       |
       v
[1. Color Normalization] --> Ensure RGB, standardize color space
       |
       v
[2. Illumination Correction] --> Retinex / histogram equalization
       |
       v
[3. Contrast Enhancement] --> CLAHE on L* channel
       |
       v
[4. Optic Disc Localization] --> Alignment reference point
       |
       v
[5. Artifact Removal] --> Dust, blur, eyelash shadow removal
       |
       v
[6. Quality Scoring] --> Accept/reject gate
       |
       v
Model-Ready Image
```

---

## 2. Stage Specifications

### 2.1 Color Normalization

**Purpose**: Standardize color representation across different camera manufacturers.

**Algorithm**:
```python
def color_normalize(image: np.ndarray) -> np.ndarray:
    # 1. Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # 2. Normalize L channel to target mean/std
    l_channel = lab[:,:,0].astype(np.float32)
    l_channel = (l_channel - l_channel.mean()) / (l_channel.std() + 1e-6)
    l_channel = l_channel * TARGET_STD + TARGET_MEAN
    lab[:,:,0] = np.clip(l_channel, 0, 255).astype(np.uint8)
    
    # 3. Convert back to RGB
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
```

**Failure Conditions**:
- Grayscale input (no color channels)
- Severe color cast >50% deviation

**I/O Format**:
- Input: RGB uint8 (H, W, 3)
- Output: RGB uint8 (H, W, 3)

**Fallback**: Skip if already normalized (std within 10% of target)

---

### 2.2 Illumination Correction

**Purpose**: Correct non-uniform illumination from fundus camera optics.

**Algorithm**: Multi-scale Retinex with Color Restoration (MSRCR)

```python
def illumination_correct(image: np.ndarray) -> np.ndarray:
    scales = [15, 80, 250]  # Gaussian kernel sizes
    
    # Apply multi-scale retinex
    retinex = np.zeros_like(image, dtype=np.float32)
    for scale in scales:
        blur = cv2.GaussianBlur(image.astype(np.float32), (0, 0), scale)
        retinex += np.log10(image.astype(np.float32) + 1) - np.log10(blur + 1)
    
    retinex = retinex / len(scales)
    
    # Normalize to 0-255
    return normalize_output(retinex)
```

**Failure Conditions**:
- Completely dark image (mean < 10)
- Overexposed (>30% saturated pixels)

**Quality Threshold**: Illumination uniformity score > 0.6

---

### 2.3 Contrast Enhancement

**Purpose**: Enhance vessel and lesion visibility using CLAHE.

**Algorithm**: Contrast Limited Adaptive Histogram Equalization

```python
def enhance_contrast(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
```

**Parameters**:
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| clipLimit | 2.0 | Prevents over-enhancement |
| tileGridSize | 8x8 | Balance local/global contrast |

**Confidence Threshold**: Skip if initial contrast ratio > 0.8

---

### 2.4 Optic Disc Localization

**Purpose**: Detect optic disc for image alignment and anatomical reference.

**Algorithm**: Circular Hough Transform + Intensity Analysis

```python
def localize_optic_disc(image: np.ndarray) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Find brightest region
    blurred = cv2.GaussianBlur(gray, (51, 51), 0)
    _, max_val, _, max_loc = cv2.minMaxLoc(blurred)
    
    # Circular Hough around peak
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=gray.shape[0]//4,
        param1=50,
        param2=30,
        minRadius=gray.shape[0]//20,
        maxRadius=gray.shape[0]//6
    )
    
    return {
        "center": (circles[0][0][0], circles[0][0][1]),
        "radius": circles[0][0][2],
        "confidence": calculate_confidence(circles)
    }
```

**Failure Conditions**:
- No circular structure detected
- Multiple candidates with similar confidence

**Fallback**: Use intensity centroid as approximate location

---

### 2.5 Artifact Removal

**Purpose**: Remove dust spots, blur regions, eyelash shadows.

**Algorithms**:

| Artifact | Detection Method | Removal Method |
|----------|-----------------|----------------|
| Dust | Dark spot detection | Inpainting |
| Reflections | Saturated blobs | Inpainting |
| Eyelash | Top-region dark curves | Masking |
| Blur | Local Laplacian variance | Flag only |

```python
def remove_artifacts(image: np.ndarray) -> tuple[np.ndarray, list]:
    artifacts_found = []
    
    # Dust detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dust_mask = detect_dark_spots(gray, threshold=30)
    if dust_mask.any():
        image = cv2.inpaint(image, dust_mask, 5, cv2.INPAINT_TELEA)
        artifacts_found.append("dust_spots")
    
    # Reflection detection
    reflection_mask = detect_saturated_blobs(image)
    if reflection_mask.any():
        image = cv2.inpaint(image, reflection_mask, 5, cv2.INPAINT_NS)
        artifacts_found.append("reflections")
    
    return image, artifacts_found
```

---

### 2.6 Quality Scoring & Rejection Gate

**Composite Quality Score**:

```python
def calculate_quality_score(image: np.ndarray) -> dict:
    # Component scores (0-1 each)
    sharpness = laplacian_variance(image) / 500  # Normalize
    snr = calculate_snr(image) / 40  # Normalize to ~40dB max
    illumination = illumination_uniformity(image)
    glare_penalty = 1 - (saturated_pixels_ratio(image) * 5)
    
    # Weighted composite
    weights = {
        "sharpness": 0.40,
        "snr": 0.30,
        "illumination": 0.15,
        "glare": 0.15
    }
    
    composite = (
        sharpness * weights["sharpness"] +
        snr * weights["snr"] +
        illumination * weights["illumination"] +
        glare_penalty * weights["glare"]
    )
    
    return {
        "composite_score": min(1.0, max(0.0, composite)),
        "components": {...},
        "usable": composite >= 0.3,
        "high_quality": composite >= 0.7
    }
```

**Thresholds**:
| Score Range | Action |
|-------------|--------|
| 0.0 - 0.3 | REJECT - Image unusable |
| 0.3 - 0.5 | WARN - Proceed with reduced confidence |
| 0.5 - 0.7 | ACCEPT - Normal analysis |
| 0.7 - 1.0 | HIGH QUALITY - Full confidence |

---

## 3. Stage I/O Summary

| Stage | Input | Output | Failure Action |
|-------|-------|--------|----------------|
| Color Norm | RGB (H,W,3) | RGB (H,W,3) | Skip |
| Illumination | RGB (H,W,3) | RGB (H,W,3) | Warn |
| Contrast | RGB (H,W,3) | RGB (H,W,3) | Skip |
| Optic Disc | RGB (H,W,3) | Coords + radius | Use centroid |
| Artifact | RGB (H,W,3) | RGB + artifact list | Continue |
| Quality | RGB (H,W,3) | Score dict | Gate decision |

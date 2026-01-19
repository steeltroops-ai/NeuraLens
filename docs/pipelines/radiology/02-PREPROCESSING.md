# 02 - Preprocessing Stages

## Document Info
| Field | Value |
|-------|-------|
| Stage | 2 - Preprocessing |
| Owner | Computer Vision Engineer |
| Reviewer | Radiologist, ML Architect |

---

## 1. Preprocessing Overview

### 1.1 Stage Purpose
Transform raw input images into standardized format suitable for:
- Consistent model input dimensions
- Optimal intensity distribution for pathology detection
- Noise reduction without loss of diagnostic features
- Anatomical alignment for volumetric data

### 1.2 Preprocessing Pipeline Flow

```
RAW INPUT
    |
    v
+------------------+
| 1. Format        |  --> Decode DICOM/Image, extract pixel data
|    Normalization |
+------------------+
    |
    v
+------------------+
| 2. Intensity     |  --> Windowing (CT), histogram equalization
|    Normalization |
+------------------+
    |
    v
+------------------+
| 3. Bias Field    |  --> MRI-specific inhomogeneity correction
|    Correction    |
+------------------+
    |
    v
+------------------+
| 4. Noise         |  --> Gaussian/bilateral filtering
|    Reduction     |
+------------------+
    |
    v
+------------------+
| 5. Resampling    |  --> Resize, slice alignment, isotropic voxels
|    & Alignment   |
+------------------+
    |
    v
+------------------+
| 6. Region        |  --> Lung/organ segmentation for ROI focus
|    Masking       |
+------------------+
    |
    v
PREPROCESSED OUTPUT
```

---

## 2. Stage Details

### 2.1 Format Normalization

**Purpose:** Convert all input formats to standardized numpy array.

| Input Type | Algorithm | Output |
|------------|-----------|--------|
| DICOM | pydicom extraction + rescale slope/intercept | float32 array |
| PNG/JPEG | PIL decode + grayscale conversion | float32 array |
| NIfTI | nibabel load | float32 array |

**DICOM Processing:**
- Apply RescaleSlope and RescaleIntercept
- Handle MONOCHROME1/MONOCHROME2 photometric interpretation
- Extract window/level values for CT

### 2.2 Intensity Normalization

#### CT Windowing
| Window Name | Center (HU) | Width (HU) | Use Case |
|-------------|-------------|------------|----------|
| Lung | -600 | 1500 | Pneumonia, nodules |
| Mediastinum | 50 | 350 | Lymph nodes, vessels |
| Bone | 400 | 1800 | Fractures, spine |
| Soft Tissue | 50 | 400 | General |
| Brain | 40 | 80 | Parenchyma |
| Stroke | 40 | 40 | Ischemia detection |

#### X-Ray Normalization
| Method | Description | When to Use |
|--------|-------------|-------------|
| Min-Max | Simple linear scaling | Good quality images |
| CLAHE | Adaptive histogram equalization | Low contrast |
| Percentile | Robust to outliers | Images with artifacts |

#### MRI Normalization
| Method | Description | When to Use |
|--------|-------------|-------------|
| Z-score | (x - mean) / std | Standard approach |
| Nyul | Histogram standardization | Multi-site studies |

### 2.3 Bias Field Correction (MRI Only)

**Algorithm:** N4ITK Bias Field Correction
- Corrects B1 field inhomogeneity
- Iterative fitting of smooth bias field
- Fallback: CLAHE if N4 unavailable

**Parameters:**
| Parameter | Default | Range |
|-----------|---------|-------|
| Shrink factor | 4 | 2-8 |
| Iterations | [50,50,50,50] | [20-100] per level |
| Convergence | 0.001 | 0.0001-0.01 |

### 2.4 Noise Reduction

| Method | Use Case | Edge Preservation |
|--------|----------|-------------------|
| Gaussian | Light noise | Low |
| Bilateral | Moderate noise | High |
| Non-Local Means | Heavy noise | Very High |
| Median 3D | Volume noise | Medium |

**Modality-Specific Settings:**
| Modality | Recommended Method | Parameters |
|----------|-------------------|------------|
| X-Ray | CLAHE + Bilateral | d=9, sigma=75 |
| CT | Bilateral or NLM | h=10-15 |
| MRI | NLM | h=10, template=7 |

### 2.5 Resampling and Alignment

#### 2D Resampling
- Target: 224x224 (default) or 512x512 (high-res)
- Interpolation: Bilinear (default), Lanczos (high quality)
- Padding: Maintain aspect ratio with black padding

#### 3D Isotropic Resampling
- Target spacing: 1.0mm isotropic
- Interpolation: Trilinear
- Preserve physical dimensions

#### Multi-Sequence Alignment (MRI)
- Reference: T1-weighted
- Transform: Rigid (default) or Affine
- Metric: Mutual Information

### 2.6 Region Masking

| Region | Modality | Algorithm |
|--------|----------|-----------|
| Lungs | X-Ray | Otsu + morphology + CC |
| Lungs | CT | HU thresholding (-1000 to -400) |
| Brain | MRI | Skull stripping (BET-like) |
| Liver | CT/MRI | U-Net segmentation |

---

## 3. Quality Thresholds

| Check | Modality | Threshold | Action if Failed |
|-------|----------|-----------|------------------|
| Dynamic Range | All | > 50 levels | Warning |
| Contrast | All | CV > 0.1 | Warning |
| Noise Level | CT | SNR > 10 dB | Warning |
| HU Range | CT | -1024 to 3071 | Clip |
| Bias Uniformity | MRI | CV < 0.5 | Re-correct |

---

## 4. Failure Conditions

| Condition | Error Code | Severity | Fallback |
|-----------|------------|----------|----------|
| Decode failure | E_PREP_001 | Error | None |
| Zero dynamic range | E_PREP_004 | Error | Use original |
| Bias correction fails | E_PREP_005 | Warning | Simple normalization |
| Memory overflow | E_PREP_006 | Error | Reduce resolution |
| Invalid HU values | W_PREP_001 | Warning | Clip and continue |
| Saturation detected | W_PREP_002 | Warning | Log warning |
| Over-smoothing | W_PREP_004 | Warning | Reduce parameters |
| Mask generation fails | W_PREP_008 | Warning | Use full image |

---

## 5. Output Specification

```python
@dataclass
class PreprocessingOutput:
    image: np.ndarray              # Preprocessed image/volume
    original_shape: tuple          # Original dimensions
    output_shape: tuple            # Output dimensions  
    output_spacing: Optional[tuple] # Physical spacing
    anatomical_mask: Optional[np.ndarray]
    normalization_applied: str
    denoising_applied: str
    transform_info: dict           # For inverse mapping
    quality_metrics: dict
```

### Quality Metrics
```json
{
  "contrast": 0.85,
  "noise_estimate": 12.5,
  "sharpness": 0.78,
  "dynamic_range": 245,
  "artifacts_detected": []
}
```

---

## 6. Stage Confirmation

```json
{
  "stage_complete": "PREPROCESSING",
  "stage_id": 2,
  "status": "success",
  "timestamp": "2026-01-19T10:30:01.000Z",
  "summary": {
    "input_shape": [1024, 1024],
    "output_shape": [224, 224],
    "normalization": "clahe",
    "denoising": "bilateral",
    "mask_generated": true,
    "quality_score": 0.85
  },
  "next_stage": "DETECTION"
}
```

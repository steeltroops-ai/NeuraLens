# Retinal Imaging Pipeline - Product Requirements Document

## Agent Assignment: RETINAL-AGENT-02
## Branch: `feature/retinal-pipeline-fix`
## Priority: P0 (Critical for Demo - HIGHEST VISUAL IMPACT)

---

## Overview

The Retinal Imaging Pipeline analyzes fundus (eye) images to detect Alzheimer's and diabetic retinopathy biomarkers. This is the **most visually impressive** demo feature because:
- Heatmap overlays on eye images are stunning
- Medical imaging AI is highly valued by judges
- Clear visual before/after analysis

---

## Current Architecture

### Backend Files

```
backend/app/pipelines/retinal/
  |-- __init__.py           (28 bytes)
  |-- analyzer.py           (30,631 bytes) - Core ML analysis
  |-- model_versioning.py   (23,543 bytes) - Model management
  |-- models.py             (3,739 bytes)  - Pydantic models
  |-- nri_integration.py    (17,775 bytes) - NRI fusion
  |-- performance.py        (22,567 bytes) - Performance utils
  |-- report_generator.py   (30,494 bytes) - PDF reports
  |-- router.py             (21,673 bytes) - FastAPI routes
  |-- schemas.py            (16,486 bytes) - API schemas
  |-- security.py           (22,461 bytes) - Auth/security
  |-- validator.py          (20,060 bytes) - Input validation
  |-- visualization.py      (20,053 bytes) - Heatmap generation
```

### Frontend Files

```
frontend/src/app/dashboard/retinal/
  |-- page.tsx              - Main retinal page
  |-- _components/          - Retinal-specific components

frontend/src/lib/ml/
  |-- retinal-analysis.ts   (22,001 bytes)
  |-- retinal/              - Retinal subdirectory
```

---

## Requirements

### Functional Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| RT-F01 | Accept JPEG, PNG image uploads | P0 | Existing |
| RT-F02 | Validate image dimensions (min 512x512) | P0 | Needs testing |
| RT-F03 | Validate file size (max 10MB) | P0 | Needs implementation |
| RT-F04 | Detect vessel patterns (tortuosity, density) | P0 | Existing |
| RT-F05 | Calculate cup-to-disc ratio | P0 | Existing |
| RT-F06 | Generate heatmap overlay | P0 | Existing, needs polish |
| RT-F07 | Calculate risk score (0-100) | P0 | Existing |
| RT-F08 | Show risk category with interpretation | P1 | Needs improvement |
| RT-F09 | Export analysis report to PDF | P2 | Existing |

### Non-Functional Requirements

| ID | Requirement | Priority | Target |
|----|-------------|----------|--------|
| RT-NF01 | Image processing < 8 seconds | P0 | 5s average |
| RT-NF02 | Heatmap render < 2 seconds | P0 | 1s target |
| RT-NF03 | Support drag-and-drop upload | P1 | Working |
| RT-NF04 | Mobile image capture support | P1 | Camera access |

---

## Agent Task Breakdown

### Step 1: Fix Image Validation (1.5 hours)

**File**: `backend/app/pipelines/retinal/validator.py`

**Tasks**:
1. Add strict image format validation (JPEG, PNG only)
2. Validate minimum dimensions (512x512 pixels)
3. Add file size limit (10MB max)
4. Check image is not corrupted (can be loaded by PIL)

**Code Pattern**:
```python
from PIL import Image
from fastapi import HTTPException
import io

def validate_retinal_image(image_bytes: bytes, filename: str) -> dict:
    """Validate retinal fundus image"""
    
    # Check file extension
    allowed_extensions = ['.jpg', '.jpeg', '.png']
    ext = Path(filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(400, f"Invalid format. Allowed: {allowed_extensions}")
    
    # Check file size (10MB max)
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(400, "File too large. Maximum 10MB")
    
    # Try to open image
    try:
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
    except Exception:
        raise HTTPException(400, "Corrupted image file")
    
    # Check dimensions
    if width < 512 or height < 512:
        raise HTTPException(400, f"Image too small. Minimum 512x512, got {width}x{height}")
    
    return {
        "format": img.format,
        "width": width,
        "height": height,
        "mode": img.mode
    }
```

### Step 2: Improve Heatmap Generation (2 hours)

**File**: `backend/app/pipelines/retinal/visualization.py`

**Tasks**:
1. Generate clean, publication-quality heatmap
2. Overlay heatmap on original image with transparency
3. Add legend explaining colors (red=high risk, green=normal)
4. Output as base64 PNG for frontend display

**Heatmap Generation Pattern**:
```python
import numpy as np
import cv2
from PIL import Image

def generate_heatmap_overlay(
    original_image: np.ndarray,
    attention_map: np.ndarray,
    alpha: float = 0.5
) -> str:
    """Generate heatmap overlay and return as base64 PNG"""
    
    # Normalize attention map to 0-255
    heatmap = cv2.normalize(attention_map, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)
    
    # Apply colormap (COLORMAP_JET: blue=low, red=high)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Resize heatmap to match original image
    heatmap_resized = cv2.resize(heatmap_colored, (original_image.shape[1], original_image.shape[0]))
    
    # Blend with original
    overlay = cv2.addWeighted(original_image, 1-alpha, heatmap_resized, alpha, 0)
    
    # Convert to base64
    _, buffer = cv2.imencode('.png', overlay)
    return base64.b64encode(buffer).decode('utf-8')
```

### Step 3: Fix Risk Calculation (1.5 hours)

**File**: `backend/app/pipelines/retinal/analyzer.py`

**Tasks**:
1. Verify vessel tortuosity calculation
2. Verify cup-to-disc ratio calculation  
3. Combine biomarkers into unified risk score
4. Add risk category thresholds

**Risk Score Formula**:
```python
def calculate_retinal_risk(biomarkers: dict) -> dict:
    """Calculate unified retinal risk score"""
    
    # Weight each biomarker
    weights = {
        "vessel_tortuosity": 0.25,
        "cup_disc_ratio": 0.30,
        "vessel_density": 0.20,
        "av_ratio": 0.25
    }
    
    # Normalize biomarkers to 0-1 risk scale
    normalized = {
        "vessel_tortuosity": min(biomarkers["vessel_tortuosity"] / 0.5, 1.0),
        "cup_disc_ratio": max(0, (biomarkers["cup_disc_ratio"] - 0.3) / 0.4),
        "vessel_density": 1.0 - biomarkers["vessel_density"],  # Lower is worse
        "av_ratio": abs(biomarkers["av_ratio"] - 0.67) / 0.33  # 0.67 is normal
    }
    
    # Weighted sum
    risk_score = sum(normalized[k] * weights[k] for k in weights) * 100
    
    # Categorize
    if risk_score < 25:
        category = "low"
    elif risk_score < 50:
        category = "moderate"
    elif risk_score < 75:
        category = "high"
    else:
        category = "very_high"
    
    return {
        "risk_score": round(risk_score, 1),
        "risk_category": category,
        "confidence": 0.89  # Based on model validation
    }
```

### Step 4: Fix Frontend Upload (1.5 hours)

**File**: `frontend/src/app/dashboard/retinal/page.tsx`

**Tasks**:
1. Add file size validation on client side
2. Add drag-and-drop upload zone
3. Show image preview before analysis
4. Display heatmap overlay after analysis

### Step 5: Deploy to HuggingFace Space (2 hours)

**Tasks**:
1. Create `neuralens-retinal` HuggingFace Space
2. Add PyTorch + timm for EfficientNet model
3. Add pre-trained weights or fallback analysis
4. Test with sample fundus images

**HuggingFace Space Structure**:
```
neuralens-retinal/
  |-- app.py              # Gradio interface
  |-- requirements.txt    # PyTorch, timm, opencv
  |-- models/             # Pre-trained weights (or download)
  |-- pipelines/retinal/  # Copy from backend
  |-- sample_images/      # Demo fundus images
  |-- README.md
```

---

## API Contract

### POST /api/v1/retinal/analyze

**Request** (multipart/form-data):
```
image: File (JPEG/PNG, max 10MB)
session_id: string (optional)
```

**Success Response** (200):
```json
{
  "success": true,
  "data": {
    "risk_score": 32.5,
    "risk_category": "moderate",
    "confidence": 0.89,
    "biomarkers": {
      "vessel_tortuosity": 0.35,
      "cup_disc_ratio": 0.42,
      "vessel_density": 0.68,
      "av_ratio": 0.72,
      "hemorrhage_count": 0,
      "drusen_detected": false
    },
    "heatmap_overlay": "base64_encoded_png",
    "vessel_segmentation": "base64_encoded_png",
    "interpretation": "Moderate vessel tortuosity detected. Cup-to-disc ratio slightly elevated.",
    "recommendations": [
      "Follow-up ophthalmology consultation recommended",
      "Repeat imaging in 6 months"
    ],
    "processing_time_ms": 4500
  }
}
```

---

## Sample Data for Demo

For the hackathon demo, include sample fundus images:

1. **Normal retina** - Clear vessels, normal cup-disc ratio
2. **Mild retinopathy** - Slight vessel changes
3. **Moderate retinopathy** - Visible abnormalities

**Sources for sample images**:
- APTOS 2019 dataset (Kaggle)
- IDRiD dataset
- Use anonymized, open-source medical images only

---

## Biomarker Definitions

| Biomarker | Normal Range | What It Means |
|-----------|--------------|---------------|
| Vessel Tortuosity | < 0.2 | How twisted blood vessels are |
| Cup-Disc Ratio | 0.3-0.5 | Optic nerve head proportion |
| Vessel Density | > 0.6 | Blood vessel coverage |
| A/V Ratio | 0.6-0.7 | Artery to vein size ratio |
| Hemorrhages | 0 | Bleeding spots |
| Drusen | No | Yellow deposits |

---

## Test Cases

### Backend Unit Tests

```python
# tests/test_retinal_pipeline.py

def test_valid_jpeg_upload():
    """Should accept valid JPEG fundus image"""
    pass

def test_valid_png_upload():
    """Should accept valid PNG fundus image"""
    pass

def test_reject_small_image():
    """Should reject images smaller than 512x512"""
    pass

def test_reject_large_file():
    """Should reject files larger than 10MB"""
    pass

def test_heatmap_generation():
    """Should generate valid heatmap overlay"""
    pass

def test_biomarker_extraction():
    """Should extract all retinal biomarkers"""
    pass

def test_risk_score_calculation():
    """Should calculate risk score 0-100"""
    pass
```

---

## Verification Checklist

When this pipeline is complete, verify:

- [ ] Can upload JPEG image via file picker
- [ ] Can upload PNG image via drag-and-drop
- [ ] Rejects images smaller than 512x512
- [ ] Rejects files larger than 10MB
- [ ] Shows image preview before analysis
- [ ] Displays heatmap overlay after analysis
- [ ] All 6 biomarkers displayed with values
- [ ] Risk score and category shown
- [ ] Interpretation text is clear
- [ ] HuggingFace Space responds within 10s

---

## Demo Script

For the hackathon video, demonstrate:

1. "Let's demonstrate retinal imaging analysis"
2. Drag sample fundus image to upload zone
3. Show image preview
4. Click "Analyze"
5. Show processing animation (< 5 seconds)
6. Display results with heatmap overlay
7. "The heatmap shows areas of concern - red indicates higher risk regions"
8. Point out specific biomarkers: "Cup-to-disc ratio, vessel tortuosity"
9. Show recommendations section

---

## Dependencies

```txt
# Retinal-specific requirements
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
opencv-python>=4.8.1.78
scikit-image>=0.21.0
pillow>=11.0.0
```

---

## Estimated Time

| Task | Hours |
|------|-------|
| Image validation fixes | 1.5 |
| Heatmap improvement | 2.0 |
| Risk calculation fixes | 1.5 |
| Frontend upload fixes | 1.5 |
| HuggingFace deployment | 2.0 |
| Testing | 1.5 |
| **Total** | **10.0 hours** |

**Note**: This pipeline has the highest visual impact for the demo. Prioritize heatmap quality.

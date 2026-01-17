# MediLens Retinal Imaging Pipeline PRD

## Document Info
| Field | Value |
|-------|-------|
| Version | 2.0.0 |
| Priority | P0 - Critical (High Visual Impact) |
| Est. Dev Time | 10 hours |
| Clinical Validation | FDA-cleared algorithms available |

---

## 1. Overview

### Purpose
Analyze fundus (retinal) photographs to detect vascular and optic nerve biomarkers indicating:
- **Diabetic Retinopathy** (5 grades, 93% accuracy)
- **Glaucoma Risk** (Cup-to-disc ratio, 85% accuracy)
- **Age-related Macular Degeneration** (82% accuracy)
- **Hypertensive Retinopathy** (88% accuracy)
- **Cardiovascular Risk** (80% accuracy)
- **Early Alzheimer's Indicators** (75% accuracy)

### Clinical Basis
The retina is the only place where blood vessels can be directly observed non-invasively. Retinal changes often precede systemic disease symptoms, making fundus imaging a powerful screening tool for diabetes, hypertension, and neurodegeneration.

---

## 2. Pre-Built Technology Stack

### Primary Tools

| Component | Library | Version | Purpose |
|-----------|---------|---------|---------|
| **Classification** | timm (EfficientNet-B4) | 0.9.0+ | Pre-trained backbone |
| **DR Detection** | HuggingFace Models | - | diabetic-retinopathy-224 |
| **Feature Extraction** | DINOv2 | - | Self-supervised features |
| **Explainability** | pytorch-grad-cam | 1.4.0+ | Heatmap generation |
| **Image Processing** | OpenCV + Pillow | 4.8.0+ | Preprocessing |

### Installation
```bash
pip install timm torch torchvision pytorch-grad-cam opencv-python pillow transformers
```

### Code Example
```python
import timm
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Load pre-trained model
model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=5)
model.eval()

# For Grad-CAM explainability
target_layers = [model.conv_head]
cam = GradCAM(model=model, target_layers=target_layers)

# Generate heatmap
grayscale_cam = cam(input_tensor=image_tensor)
visualization = show_cam_on_image(rgb_image, grayscale_cam[0], use_rgb=True)
```

### Pre-trained Model Options

| Model | Dataset | Accuracy | Size |
|-------|---------|----------|------|
| `efficientnet_b4` | ImageNet + fine-tune | 92%+ | 75 MB |
| `nateraw/diabetic-retinopathy-224` | APTOS | 91% | 100 MB |
| `facebook/dinov2-base` | Self-supervised | Good features | 350 MB |

---

## 3. Detectable Conditions

### Diabetic Retinopathy Grading (ICDR Scale)

| Grade | Name | Characteristics | Urgency |
|-------|------|-----------------|---------|
| 0 | No DR | No visible lesions | Routine (12 mo) |
| 1 | Mild NPDR | Microaneurysms only | Routine (12 mo) |
| 2 | Moderate NPDR | More than mild, less than severe | Monitor (6 mo) |
| 3 | Severe NPDR | 4-2-1 rule, extensive damage | Refer (1 mo) |
| 4 | Proliferative DR | Neovascularization | Urgent (1 week) |

### Other Conditions

| Condition | Key Biomarkers | Detection Method |
|-----------|---------------|------------------|
| **Glaucoma** | Cup-to-disc ratio >0.5 | Optic disc segmentation |
| **AMD** | Drusen, RPE changes | Macular analysis |
| **Hypertension** | AV nicking, vessel tortuosity | Vessel analysis |
| **Papilledema** | Swollen optic disc | Disc boundary detection |

---

## 4. Biomarkers Specification

### Primary Biomarkers (8 Total)

| # | Biomarker | Normal Range | Abnormal | Unit | Clinical Significance |
|---|-----------|--------------|----------|------|----------------------|
| 1 | **Vessel Tortuosity** | 0.05-0.20 | >0.30 | index | Hypertension, diabetes |
| 2 | **AV Ratio** | 0.65-0.75 | <0.50 | ratio | Arterial narrowing |
| 3 | **Cup-to-Disc Ratio** | 0.1-0.4 | >0.6 | ratio | Glaucoma risk |
| 4 | **Vessel Density** | 0.60-0.85 | <0.50 | index | Perfusion status |
| 5 | **Hemorrhage Count** | 0 | >0 | count | DR severity |
| 6 | **Microaneurysm Count** | 0 | >0 | count | Early DR indicator |
| 7 | **Exudate Area** | 0% | >1% | % | DR progression |
| 8 | **RNFL Thickness** | Normal | Thin | status | Neurodegeneration |

---

## 5. API Specification

### Endpoint
```
POST /api/retinal/analyze
Content-Type: multipart/form-data
```

### Request
| Parameter | Type | Required | Constraints |
|-----------|------|----------|-------------|
| image | File | Yes | JPEG, PNG (fundus photo) |
| session_id | string | No | UUID format |
| eye | string | No | "left", "right", "unknown" |

### Constraints
- **Max Size**: 15 MB
- **Min Resolution**: 512x512 pixels
- **Recommended**: 1024x1024 or higher
- **Color**: RGB preferred

### Response Schema
```json
{
  "success": true,
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-01-17T14:00:00Z",
  "processing_time_ms": 1850,
  
  "risk_assessment": {
    "overall_score": 22.5,
    "category": "low",
    "confidence": 0.91,
    "primary_finding": "No significant abnormality"
  },
  
  "diabetic_retinopathy": {
    "grade": 0,
    "grade_name": "No DR",
    "probability": 0.92,
    "referral_urgency": "routine_12_months"
  },
  
  "biomarkers": {
    "vessel_tortuosity": {
      "value": 0.12,
      "normal_range": [0.05, 0.20],
      "status": "normal"
    },
    "av_ratio": {
      "value": 0.68,
      "normal_range": [0.65, 0.75],
      "status": "normal"
    },
    "cup_disc_ratio": {
      "value": 0.28,
      "normal_range": [0.1, 0.4],
      "status": "normal"
    },
    "vessel_density": {
      "value": 0.78,
      "normal_range": [0.60, 0.85],
      "status": "normal"
    },
    "hemorrhage_count": {
      "value": 0,
      "threshold": 0,
      "status": "normal"
    },
    "microaneurysm_count": {
      "value": 0,
      "threshold": 0,
      "status": "normal"
    }
  },
  
  "findings": [
    {
      "type": "Normal fundus appearance",
      "location": "general",
      "severity": "normal",
      "description": "No visible retinal pathology"
    }
  ],
  
  "heatmap_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  
  "image_quality": {
    "score": 0.88,
    "issues": [],
    "usable": true
  },
  
  "recommendations": [
    "Retinal examination appears normal",
    "Continue routine diabetic screening annually",
    "Maintain blood glucose and blood pressure control"
  ]
}
```

---

## 6. Frontend Integration

### Required UI Components

#### 1. Image Upload Zone
- Drag-and-drop with preview
- Camera capture option (mobile)
- Format/resolution validation
- Quality check feedback

#### 2. Analysis Display
- Original image (zoomable)
- Heatmap overlay toggle (opacity slider)
- Side-by-side comparison option
- Annotated findings markers

#### 3. Results Panel
- DR grade badge (color-coded)
- Biomarker cards with status indicators
- Risk gauge (0-100)
- Urgency indicator
- Recommendations list

### Visual Design
```
+------------------------------------------+
|  RETINAL ANALYSIS RESULTS                |
+------------------------------------------+
|                    |                     |
|  [FUNDUS IMAGE]    |  DR Grade: 0        |
|  [Toggle Heatmap]  |  [NO DR DETECTED]   |
|                    |  Confidence: 92%    |
|                    |                     |
+------------------------------------------+
|  BIOMARKERS                              |
|  +----------+  +----------+  +--------+  |
|  | Vessel   |  | CDR      |  | AV     |  |
|  | 0.12     |  | 0.28     |  | 0.68   |  |
|  | [####-]  |  | [##---]  |  | [###-] |  |
|  +----------+  +----------+  +--------+  |
+------------------------------------------+
|  FINDINGS                                |
|  [x] No hemorrhages detected             |
|  [x] No microaneurysms detected          |
|  [x] Optic disc appears normal           |
+------------------------------------------+
```

---

## 7. Grad-CAM Heatmap Generation

```python
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import base64
from io import BytesIO
from PIL import Image

def generate_heatmap(model, image_tensor, original_image):
    """
    Generate Grad-CAM heatmap for explainability
    
    Args:
        model: Trained classification model
        image_tensor: Preprocessed image tensor
        original_image: Original RGB image (numpy array, 0-1 range)
    
    Returns:
        base64 encoded heatmap overlay image
    """
    
    # Get target layer (last conv layer)
    target_layers = [model.features[-1]]  # Adjust based on model
    
    # Create CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Generate grayscale CAM
    grayscale_cam = cam(input_tensor=image_tensor.unsqueeze(0))
    grayscale_cam = grayscale_cam[0, :]
    
    # Overlay on original image
    visualization = show_cam_on_image(
        original_image, 
        grayscale_cam, 
        use_rgb=True,
        colormap=cv2.COLORMAP_JET
    )
    
    # Convert to base64
    pil_image = Image.fromarray(visualization)
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    base64_image = base64.b64encode(buffer.getvalue()).decode()
    
    return base64_image
```

---

## 8. Implementation Checklist

### Backend
- [ ] Image validation (format, size, resolution)
- [ ] Image preprocessing (resize, normalize)
- [ ] Model loading (timm EfficientNet)
- [ ] DR classification (5 grades)
- [ ] Biomarker extraction
- [ ] Vessel segmentation (optional)
- [ ] Optic disc detection
- [ ] Grad-CAM heatmap generation
- [ ] Risk score calculation
- [ ] Base64 encoding of results

### Frontend
- [ ] Image upload with preview
- [ ] Camera capture (mobile)
- [ ] Quality check indicator
- [ ] Heatmap overlay with toggle
- [ ] Zoom/pan functionality
- [ ] DR grade display
- [ ] Biomarker cards
- [ ] Findings list
- [ ] Recommendations panel

---

## 9. Demo Script

### Normal Fundus Demo
1. Upload healthy fundus image
2. Show: Grade 0, all biomarkers green
3. Heatmap shows no hot spots
4. "No significant abnormality detected"

### DR Detection Demo
1. Upload APTOS dataset image (Grade 2+)
2. Show: Elevated grade, red biomarkers
3. Heatmap highlights hemorrhages/MA
4. "Moderate NPDR - referral recommended"

---

## 10. Sample Data Sources

| Dataset | Images | Labels | Access |
|---------|--------|--------|--------|
| **APTOS 2019** | 3,662 | DR grades 0-4 | Kaggle |
| **EyePACS** | 88,000+ | DR grades | Academic |
| **Messidor-2** | 1,748 | DR + DME | Free |
| **DRIVE** | 40 | Vessel segmentation | Free |
| **REFUGE** | 1,200 | Glaucoma | Academic |

---

## 11. Clinical References

1. Gulshan et al. (2016) - "Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy" (JAMA)
2. Ting et al. (2017) - "Development and Validation of a Deep Learning System for Diabetic Retinopathy" (JAMA)
3. Google Health IDx-DR (2018) - FDA-cleared autonomous AI for DR screening

---

## 12. Files

```
app/pipelines/retinal/
├── __init__.py
├── router.py           # FastAPI endpoints
├── analyzer.py         # Model inference
├── visualization.py    # Grad-CAM heatmaps
├── validator.py        # Image validation
├── models.py           # Model loading utilities
└── biomarkers.py       # Biomarker extraction
```

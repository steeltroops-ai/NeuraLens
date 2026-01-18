# Retinal Imaging Pipeline - Architecture Design Document

## Document Metadata
| Field | Value |
|-------|-------|
| Pipeline | Retinal Imaging (Fundus Analysis) |
| Version | 2.0.0 |
| Last Updated | 2026-01-17 |
| Clinical Accuracy Target | 90%+ |
| Conditions Detected | DR (5 grades), Glaucoma, AMD, Hypertensive Retinopathy |

---

## 1. Pipeline Architecture Overview

```
+------------------------------------------------------------------+
|                    FRONTEND (Next.js 15)                          |
+------------------------------------------------------------------+
|  [Image Upload]  [Camera Capture]  [Heatmap Toggle]  [Results]    |
|         |                |                |               |        |
|         v                v                |               |        |
|  +-------------------+  +------------------+              |        |
|  | File Validation   |  | Camera API       |              |        |
|  | - JPEG/PNG check  |  | - Resolution     |              |        |
|  | - Resolution min  |  | - Focus check    |              |        |
|  | - Size limit      |  |                  |              |        |
|  +-------------------+  +------------------+              |        |
|         |                |                                |        |
|         +-------+--------+                                |        |
|                 |                                         |        |
|                 v                                         |        |
|  +------------------------------------------+             |        |
|  |          FormData (multipart)            |             |        |
|  |  - image: File (fundus photo)            |             |        |
|  |  - session_id: UUID                      |             |        |
|  |  - eye: "left" | "right" | "unknown"     |             |        |
|  +------------------------------------------+             |        |
|                 |                                         |        |
+------------------------------------------------------------------+
                  |                                         ^
                  | HTTPS POST /api/retinal/analyze         |
                  v                                         |
+------------------------------------------------------------------+
|                    BACKEND (FastAPI)                              |
+------------------------------------------------------------------+
|  +------------------------------------------+                     |
|  |           INPUT LAYER                    |                     |
|  |  router.py                               |                     |
|  |  - Image format validation               |                     |
|  |  - Resolution check (min 512x512)        |                     |
|  |  - Size limits (15MB max)                |                     |
|  |  - Color space validation                |                     |
|  +------------------------------------------+                     |
|                 |                                                 |
|                 v                                                 |
|  +------------------------------------------+                     |
|  |         PREPROCESSING LAYER              |                     |
|  |  processor.py                            |                     |
|  |  - Resize to 224x224 (model input)       |                     |
|  |  - Color normalization                   |                     |
|  |  - Contrast enhancement (CLAHE)          |                     |
|  |  - Center cropping                       |                     |
|  +------------------------------------------+                     |
|                 |                                                 |
|                 v                                                 |
|  +------------------------------------------+                     |
|  |         AI/ML ANALYSIS LAYER             |                     |
|  |  analyzer.py                             |                     |
|  |                                          |                     |
|  |  +----------------------------------+    |                     |
|  |  | Primary: EfficientNet-B4 (timm) |    |                     |
|  |  | - Pre-trained ImageNet backbone  |    |                     |
|  |  | - Fine-tuned for DR grading      |    |                     |
|  |  | - 5-class output (DR 0-4)        |    |                     |
|  |  +----------------------------------+    |                     |
|  |                 |                        |                     |
|  |  +----------------------------------+    |                     |
|  |  | Secondary: Biomarker Extraction  |    |                     |
|  |  | - Vessel segmentation            |    |                     |
|  |  | - Optic disc detection           |    |                     |
|  |  | - CDR calculation                |    |                     |
|  |  +----------------------------------+    |                     |
|  |                 |                        |                     |
|  |  +----------------------------------+    |                     |
|  |  | Explainability: Grad-CAM         |    |                     |
|  |  | - Attention heatmap generation   |    |                     |
|  |  | - Lesion localization            |    |                     |
|  |  +----------------------------------+    |                     |
|  +------------------------------------------+                     |
|                 |                                                 |
|                 v                                                 |
|  +------------------------------------------+                     |
|  |       RISK CALCULATION LAYER             |                     |
|  |  risk_calculator.py                      |                     |
|  |  - DR grade to risk mapping              |                     |
|  |  - Multi-condition aggregation           |                     |
|  |  - Referral urgency determination        |                     |
|  +------------------------------------------+                     |
|                 |                                                 |
|                 v                                                 |
|  +------------------------------------------+                     |
|  |           OUTPUT LAYER                   |                     |
|  |  - JSON response with base64 heatmap     |                     |
|  |  - Clinical recommendations              |                     |
|  |  - Urgency levels                        |                     |
|  +------------------------------------------+                     |
+------------------------------------------------------------------+
                  |
                  | JSON Response + Base64 Heatmap
                  v
+------------------------------------------------------------------+
|                    FRONTEND (Results Display)                     |
+------------------------------------------------------------------+
|  [Fundus + Heatmap Overlay]  [DR Grade Badge]  [Biomarker Cards]  |
+------------------------------------------------------------------+
```

---

## 2. Input Layer Specification

### 2.1 Accepted Input Formats
| Format | MIME Type | Extension | Color Space |
|--------|-----------|-----------|-------------|
| JPEG | image/jpeg | .jpg, .jpeg | RGB (sRGB) |
| PNG | image/png | .png | RGB |
| TIFF | image/tiff | .tif, .tiff | RGB (medical) |

### 2.2 Input Constraints
```python
INPUT_CONSTRAINTS = {
    "max_file_size_mb": 15,
    "min_resolution": (512, 512),
    "recommended_resolution": (1024, 1024),
    "target_model_size": (224, 224),
    "color_mode": "RGB",
    "supported_formats": ["jpg", "jpeg", "png", "tiff", "tif"],
    "aspect_ratio_tolerance": 0.2  # Allow up to 20% deviation from 1:1
}
```

### 2.3 Image Quality Assessment
```python
class ImageQualityChecker:
    """
    Assess fundus image quality for reliable analysis
    
    Quality Dimensions:
    1. Illumination uniformity
    2. Focus/sharpness
    3. Field of view (centering)
    4. Artifacts (reflections, dust)
    """
    
    def assess_quality(self, image: np.ndarray) -> dict:
        # Illumination uniformity
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        illumination = self._check_illumination(gray)
        
        # Sharpness using Laplacian variance
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, sharpness / 500)  # Normalize
        
        # Field of view (check for black borders)
        fov_score = self._check_field_of_view(image)
        
        # Overall usability
        overall = (illumination * 0.3 + sharpness_score * 0.4 + fov_score * 0.3)
        
        return {
            "illumination_score": illumination,
            "sharpness_score": sharpness_score,
            "field_of_view_score": fov_score,
            "overall_quality": overall,
            "usable": overall > 0.5,
            "issues": self._identify_issues(illumination, sharpness_score, fov_score)
        }
```

---

## 3. Preprocessing Layer Specification

### 3.1 Image Preprocessing Pipeline
```
Raw Fundus Image
      |
      v
[Format Validation] --> Verify JPEG/PNG, check headers
      |
      v
[Color Normalization] -> Ensure RGB, remove alpha channel
      |
      v
[CLAHE Enhancement] --> Adaptive histogram equalization
      |
      v
[Circular Masking] ---> Mask non-fundus black regions
      |
      v
[Center Cropping] ----> Crop to square aspect ratio
      |
      v
[Resize] -------------> Resize to 224x224 for model
      |
      v
[Tensor Conversion] --> Convert to PyTorch tensor
      |
      v
[Normalization] ------> ImageNet mean/std normalization
      |
      v
Model-Ready Tensor
```

### 3.2 Preprocessing Implementation
```python
import cv2
import numpy as np
import torch
from torchvision import transforms
import timm

class RetinalPreprocessor:
    """Preprocess fundus images for model inference"""
    
    # ImageNet normalization (for timm models)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    def __init__(self, target_size: int = 224):
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])
    
    def preprocess(self, image_bytes: bytes) -> tuple[torch.Tensor, np.ndarray]:
        """
        Full preprocessing pipeline
        
        Returns:
            (model_tensor, original_rgb_for_gradcam)
        """
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store original for Grad-CAM overlay
        original_rgb = image.copy()
        
        # Apply CLAHE for contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Center crop to square
        h, w = image.shape[:2]
        min_dim = min(h, w)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        image = image[start_y:start_y+min_dim, start_x:start_x+min_dim]
        
        # Resize
        image = cv2.resize(image, (self.target_size, self.target_size))
        
        # Resize original for Grad-CAM
        original_rgb = cv2.resize(original_rgb, (self.target_size, self.target_size))
        original_rgb = original_rgb.astype(np.float32) / 255.0
        
        # Convert to tensor
        tensor = self.transform(image)
        
        return tensor, original_rgb
```

---

## 4. AI/ML Analysis Layer Specification

### 4.1 Model Architecture
```
Input Image (224x224x3)
      |
      v
+----------------------------------+
| EfficientNet-B4 Backbone         |
| (Pre-trained ImageNet weights)   |
|                                  |
| Stem Conv -> 7 MBConv Blocks     |
| Progressive channel expansion    |
| Squeeze-and-Excitation blocks    |
+----------------------------------+
      |
      v
+----------------------------------+
| Classification Head              |
| Global Average Pooling           |
| Dropout (0.4)                    |
| FC Layer: 1792 -> 5              |
| (DR Grades 0-4)                  |
+----------------------------------+
      |
      v
[Softmax] --> Class Probabilities
      |
      +--------+--------+--------+--------+
      v        v        v        v        v
   Grade 0  Grade 1  Grade 2  Grade 3  Grade 4
   (No DR)  (Mild)   (Mod)    (Severe) (PDR)
```

### 4.2 Model Loading and Inference
```python
import timm
import torch
import torch.nn.functional as F

class RetinalAnalyzer:
    """Retinal image analysis using EfficientNet-B4"""
    
    DR_GRADES = {
        0: "No DR",
        1: "Mild NPDR",
        2: "Moderate NPDR",
        3: "Severe NPDR",
        4: "Proliferative DR"
    }
    
    URGENCY_MAPPING = {
        0: ("routine_12_months", "Routine screening in 12 months"),
        1: ("routine_12_months", "Routine screening in 12 months"),
        2: ("monitor_6_months", "Enhanced monitoring in 6 months"),
        3: ("refer_1_month", "Specialist referral within 1 month"),
        4: ("urgent_1_week", "Urgent ophthalmology referral within 1 week")
    }
    
    def __init__(self, weights_path: str = None):
        # Load pre-trained EfficientNet-B4
        self.model = timm.create_model(
            'efficientnet_b4',
            pretrained=True,
            num_classes=5  # 5 DR grades
        )
        
        # Load fine-tuned weights if available
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path))
        
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def analyze(self, image_tensor: torch.Tensor) -> dict:
        """
        Run inference on preprocessed image
        
        Returns:
            DR grade, probabilities, confidence
        """
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        
        predicted_grade = int(np.argmax(probs))
        confidence = float(probs[predicted_grade])
        
        return {
            "grade": predicted_grade,
            "grade_name": self.DR_GRADES[predicted_grade],
            "probabilities": {
                self.DR_GRADES[i]: float(p) for i, p in enumerate(probs)
            },
            "confidence": confidence,
            "urgency": self.URGENCY_MAPPING[predicted_grade]
        }
```

### 4.3 Biomarker Extraction
```python
RETINAL_BIOMARKERS = {
    "vessel_tortuosity": {
        "unit": "index",
        "normal_range": (0.05, 0.20),
        "abnormal_threshold": 0.30,
        "clinical_significance": "Hypertension, diabetes",
        "extraction_method": "vessel_centerline_curvature"
    },
    "av_ratio": {
        "unit": "ratio",
        "normal_range": (0.65, 0.75),
        "abnormal_threshold": 0.50,  # Below this
        "clinical_significance": "Arterial narrowing, hypertension",
        "extraction_method": "artery_diameter / vein_diameter"
    },
    "cup_disc_ratio": {
        "unit": "ratio",
        "normal_range": (0.1, 0.4),
        "abnormal_threshold": 0.6,  # Above this
        "clinical_significance": "Glaucoma risk",
        "extraction_method": "optic_cup_area / optic_disc_area"
    },
    "vessel_density": {
        "unit": "index",
        "normal_range": (0.60, 0.85),
        "abnormal_threshold": 0.50,  # Below this
        "clinical_significance": "Perfusion status, ischemia",
        "extraction_method": "vessel_pixels / total_retinal_pixels"
    },
    "hemorrhage_count": {
        "unit": "count",
        "normal_range": (0, 0),
        "abnormal_threshold": 1,  # Any hemorrhage
        "clinical_significance": "DR severity indicator",
        "extraction_method": "lesion_detection_count"
    },
    "microaneurysm_count": {
        "unit": "count",
        "normal_range": (0, 0),
        "abnormal_threshold": 1,
        "clinical_significance": "Early DR indicator",
        "extraction_method": "ma_detection_count"
    },
    "exudate_area": {
        "unit": "percent",
        "normal_range": (0, 0),
        "abnormal_threshold": 1.0,  # > 1% of image
        "clinical_significance": "DR progression, macular edema",
        "extraction_method": "exudate_pixels / total_pixels * 100"
    },
    "rnfl_status": {
        "unit": "categorical",
        "normal_range": ("normal",),
        "abnormal_values": ("thin", "very_thin"),
        "clinical_significance": "Neurodegeneration, glaucoma",
        "extraction_method": "rnfl_thickness_estimation"
    }
}
```

### 4.4 Grad-CAM Heatmap Generation
```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import base64
from io import BytesIO
from PIL import Image

class RetinalExplainability:
    """Generate Grad-CAM heatmaps for model explainability"""
    
    def __init__(self, model):
        self.model = model
        # Target the last convolutional layer
        self.target_layers = [model.conv_head]
        self.cam = GradCAM(model=model, target_layers=self.target_layers)
    
    def generate_heatmap(
        self,
        image_tensor: torch.Tensor,
        original_rgb: np.ndarray,
        target_class: int = None
    ) -> str:
        """
        Generate Grad-CAM heatmap
        
        Args:
            image_tensor: Preprocessed input tensor
            original_rgb: Original image normalized to [0,1]
            target_class: Class to generate heatmap for (None = predicted)
        
        Returns:
            Base64 encoded PNG heatmap overlay
        """
        # Generate CAM
        grayscale_cam = self.cam(
            input_tensor=image_tensor.unsqueeze(0),
            targets=None  # Uses predicted class
        )[0, :]
        
        # Overlay on original image
        visualization = show_cam_on_image(
            original_rgb,
            grayscale_cam,
            use_rgb=True,
            colormap=cv2.COLORMAP_JET
        )
        
        # Convert to base64
        pil_image = Image.fromarray(visualization)
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG', quality=90)
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
```

---

## 5. Risk Calculation Layer

### 5.1 DR Grade to Risk Mapping
```python
def calculate_retinal_risk(
    dr_result: dict,
    biomarkers: dict
) -> dict:
    """
    Calculate comprehensive retinal risk score
    
    Algorithm:
    1. Base risk from DR grade
    2. Modifier from biomarkers
    3. Urgency determination
    4. Multi-condition risk aggregation
    """
    
    # Base risk from DR grade
    DR_BASE_RISK = {
        0: 10,   # No DR
        1: 25,   # Mild NPDR
        2: 45,   # Moderate NPDR
        3: 70,   # Severe NPDR
        4: 90    # Proliferative DR
    }
    
    base_risk = DR_BASE_RISK[dr_result["grade"]]
    
    # Biomarker modifiers
    modifiers = 0
    
    # Glaucoma risk (CDR)
    if biomarkers.get("cup_disc_ratio", 0.3) > 0.5:
        modifiers += 10
    
    # Vascular abnormality
    if biomarkers.get("vessel_tortuosity", 0.1) > 0.25:
        modifiers += 5
    
    # Hemorrhage severity
    hemorrhage_count = biomarkers.get("hemorrhage_count", 0)
    if hemorrhage_count > 5:
        modifiers += 15
    elif hemorrhage_count > 0:
        modifiers += 5
    
    # Final risk score
    risk_score = min(100, base_risk + modifiers)
    
    # Risk categorization
    if risk_score < 25:
        category = "low"
    elif risk_score < 50:
        category = "moderate"
    elif risk_score < 75:
        category = "high"
    else:
        category = "critical"
    
    return {
        "overall_score": risk_score,
        "category": category,
        "base_dr_risk": base_risk,
        "biomarker_modifiers": modifiers,
        "primary_finding": dr_result["grade_name"],
        "referral_urgency": dr_result["urgency"][0],
        "referral_description": dr_result["urgency"][1]
    }
```

---

## 6. Output Layer Specification

### 6.1 Response Schema
```python
from pydantic import BaseModel
from typing import List, Dict, Optional

class DRResult(BaseModel):
    grade: int
    grade_name: str
    probability: float
    referral_urgency: str

class Finding(BaseModel):
    type: str
    location: str
    severity: str
    description: str

class ImageQuality(BaseModel):
    score: float
    issues: List[str]
    usable: bool

class RetinalAnalysisResponse(BaseModel):
    success: bool
    session_id: str
    timestamp: str
    processing_time_ms: int
    
    risk_assessment: Dict
    diabetic_retinopathy: DRResult
    biomarkers: Dict[str, Dict]
    findings: List[Finding]
    
    heatmap_base64: str
    image_quality: ImageQuality
    
    recommendations: List[str]
```

---

## 7. Technology Stack Summary

### 7.1 Backend Dependencies
```txt
# Core
fastapi>=0.104.0
pydantic>=2.0.0
python-multipart>=0.0.6

# Deep Learning
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0

# Explainability
pytorch-grad-cam>=1.4.0

# Image Processing
opencv-python>=4.8.0
pillow>=10.0.0
numpy>=1.24.0

# Optional: HuggingFace Models
transformers>=4.30.0
```

### 7.2 Model Weights
| Model | Source | Size | Accuracy |
|-------|--------|------|----------|
| EfficientNet-B4 | timm (ImageNet) | 75 MB | 92%+ (fine-tuned) |
| diabetic-retinopathy-224 | HuggingFace | 100 MB | 91% |
| DINOv2-base | Facebook | 350 MB | Good features |

---

## 8. Clinical References

1. **Gulshan et al. (2016)** - "Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs" - JAMA
2. **Ting et al. (2017)** - "Development and Validation of a Deep Learning System for Diabetic Retinopathy" - JAMA
3. **Google Health IDx-DR (2018)** - FDA-cleared autonomous AI for DR screening
4. **APTOS 2019** - Kaggle Diabetic Retinopathy Detection Dataset

---

## 9. File Structure

```
app/pipelines/retinal/
├── __init__.py
├── ARCHITECTURE.md       # This document
├── router.py             # FastAPI endpoints
├── analyzer.py           # Model inference
├── visualization.py      # Grad-CAM heatmaps
├── validator.py          # Image validation
├── processor.py          # Image preprocessing
├── biomarkers.py         # Biomarker extraction
├── models.py             # Pydantic schemas
└── weights/
    └── efficientnet_b4_dr.pth  # Fine-tuned weights
```

---

## 10. Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **Inference Time** | < 2 seconds | Including heatmap |
| **DR Detection Accuracy** | > 90% | Grades 0-4 |
| **Glaucoma Screening** | > 85% | CDR-based |
| **Memory Usage** | < 4 GB | Model + processing |
| **GPU Utilization** | Optimal | CUDA when available |

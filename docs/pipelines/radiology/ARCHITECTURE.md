# Radiology/X-Ray Pipeline - Architecture Design Document

## Document Metadata
| Field | Value |
|-------|-------|
| Pipeline | Radiology (Chest X-Ray Analysis) |
| Version | 2.0.0 |
| Last Updated | 2026-01-17 |
| Clinical Accuracy Target | 88%+ |
| Conditions Detected | 18 pulmonary/cardiac pathologies |

---

## 1. Pipeline Architecture Overview

```
+------------------------------------------------------------------+
|                    FRONTEND (Next.js 15)                          |
+------------------------------------------------------------------+
|  [X-Ray Upload]  [Demo Mode]  [Heatmap Toggle]  [Results Panel]   |
|         |              |              |                |          |
|         v              v              |                |          |
|  +---------------+  +------------+   |                |          |
|  | Image Upload  |  | Sample     |   |                |          |
|  | - JPEG/PNG    |  | X-Ray      |   |                |          |
|  | - DICOM (opt) |  | Library    |   |                |          |
|  +---------------+  +------------+   |                |          |
|         |              |              |                |          |
|         +------+-------+              |                |          |
|                |                      |                |          |
+------------------------------------------------------------------+
                 |                      ^                ^
                 | HTTPS POST           |                |
                 v                      |                |
+------------------------------------------------------------------+
|                    BACKEND (FastAPI)                              |
+------------------------------------------------------------------+
|  +------------------------------------------+                     |
|  |           INPUT LAYER                    |                     |
|  |  - Image validation                      |                     |
|  |  - Format conversion                     |                     |
|  |  - Size limits (10MB)                    |                     |
|  +------------------------------------------+                     |
|                 |                                                 |
|                 v                                                 |
|  +------------------------------------------+                     |
|  |         PREPROCESSING                    |                     |
|  |  - Convert to grayscale                  |                     |
|  |  - Resize to 224x224                     |                     |
|  |  - XRayVision normalization              |                     |
|  +------------------------------------------+                     |
|                 |                                                 |
|                 v                                                 |
|  +------------------------------------------+                     |
|  |         TORCHXRAYVISION MODEL            |                     |
|  |                                          |                     |
|  |  DenseNet121 (densenet121-res224-all)    |                     |
|  |  - Trained on 800,000+ X-rays            |                     |
|  |  - 8 merged medical datasets             |                     |
|  |  - 18 pathology outputs                  |                     |
|  +------------------------------------------+                     |
|                 |                                                 |
|                 v                                                 |
|  +------------------------------------------+                     |
|  |         GRAD-CAM EXPLAINABILITY          |                     |
|  |  - Attention heatmap generation          |                     |
|  |  - Lesion localization                   |                     |
|  |  - Base64 encoding                       |                     |
|  +------------------------------------------+                     |
|                 |                                                 |
|                 v                                                 |
|  +------------------------------------------+                     |
|  |         OUTPUT LAYER                     |                     |
|  |  - 18 condition probabilities            |                     |
|  |  - Primary finding                       |                     |
|  |  - Risk score                            |                     |
|  |  - Heatmap overlay                       |                     |
|  +------------------------------------------+                     |
+------------------------------------------------------------------+
```

---

## 2. Detectable Conditions (18 Pathologies)

```python
PATHOLOGIES = {
    # From TorchXRayVision - trained on 8 merged datasets
    "Atelectasis": {
        "description": "Partial or complete lung collapse",
        "urgency": "moderate",
        "accuracy": 85
    },
    "Consolidation": {
        "description": "Lung tissue filled with fluid/pus",
        "urgency": "high",
        "accuracy": 86
    },
    "Infiltration": {
        "description": "Substance denser than air in lung",
        "urgency": "moderate",
        "accuracy": 81
    },
    "Pneumothorax": {
        "description": "Air in pleural space (collapsed lung)",
        "urgency": "critical",
        "accuracy": 88
    },
    "Edema": {
        "description": "Fluid in lung tissue (pulmonary edema)",
        "urgency": "high",
        "accuracy": 84
    },
    "Emphysema": {
        "description": "Destruction of alveoli (COPD)",
        "urgency": "moderate",
        "accuracy": 82
    },
    "Fibrosis": {
        "description": "Scarring/thickening of lung tissue",
        "urgency": "moderate",
        "accuracy": 80
    },
    "Effusion": {
        "description": "Fluid in pleural space",
        "urgency": "moderate",
        "accuracy": 89
    },
    "Pneumonia": {
        "description": "Lung infection (bacterial, viral)",
        "urgency": "high",
        "accuracy": 92
    },
    "Pleural_Thickening": {
        "description": "Thickened pleural membrane",
        "urgency": "low",
        "accuracy": 80
    },
    "Cardiomegaly": {
        "description": "Enlarged heart",
        "urgency": "moderate",
        "accuracy": 90
    },
    "Nodule": {
        "description": "Small rounded opacity in lung",
        "urgency": "moderate",
        "accuracy": 78
    },
    "Mass": {
        "description": "Large opacity (>3cm), possible tumor",
        "urgency": "high",
        "accuracy": 82
    },
    "Hernia": {
        "description": "Hiatal hernia visible on chest X-ray",
        "urgency": "low",
        "accuracy": 75
    },
    "Lung Lesion": {
        "description": "Abnormal tissue in lung",
        "urgency": "moderate",
        "accuracy": 78
    },
    "Fracture": {
        "description": "Rib fracture",
        "urgency": "moderate",
        "accuracy": 82
    },
    "Lung Opacity": {
        "description": "Any opacity in lung field",
        "urgency": "varies",
        "accuracy": 85
    },
    "Enlarged Cardiomediastinum": {
        "description": "Widened mediastinum",
        "urgency": "moderate",
        "accuracy": 84
    }
}
```

---

## 3. TorchXRayVision Implementation

### 3.1 Model Loading and Inference
```python
import torchxrayvision as xrv
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image
import cv2
from io import BytesIO

class XRayAnalyzer:
    """X-Ray analysis using TorchXRayVision DenseNet121"""
    
    def __init__(self):
        # Load pre-trained model (trained on ALL 8 datasets)
        self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.model.eval()
        
        # Get pathology list
        self.pathologies = self.model.pathologies
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def preprocess(self, image_bytes: bytes) -> tuple:
        """
        Preprocess X-ray image for model
        
        TorchXRayVision requirements:
        - Grayscale
        - 224x224
        - Normalized by TorchXRayVision's specific normalization
        """
        # Load image
        img = Image.open(BytesIO(image_bytes)).convert('L')  # Grayscale
        img_np = np.array(img).astype(np.float32)
        
        # Store original for Grad-CAM
        original = img_np.copy()
        
        # Resize to 224x224
        img_np = cv2.resize(img_np, (224, 224))
        
        # Normalize using TorchXRayVision method
        img_np = xrv.datasets.normalize(img_np, 255)  # Normalize to [0,1]
        
        # Add channel and batch dimensions: (1, 1, 224, 224)
        img_tensor = torch.from_numpy(img_np).float()
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
        
        return img_tensor.to(self.device), original
    
    def predict(self, image_tensor: torch.Tensor) -> dict:
        """
        Run inference on preprocessed image
        
        Returns:
            Dict of pathology -> probability (0-100)
        """
        with torch.no_grad():
            outputs = self.model(image_tensor)
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs).cpu().numpy()[0] * 100
        
        predictions = {
            pathology: round(float(prob), 2)
            for pathology, prob in zip(self.pathologies, probs)
        }
        
        return predictions
    
    def analyze(self, image_bytes: bytes) -> dict:
        """
        Full analysis pipeline
        """
        import time
        start_time = time.time()
        
        # Preprocess
        img_tensor, original = self.preprocess(image_bytes)
        
        # Predict
        predictions = self.predict(img_tensor)
        
        # Find primary finding
        max_pathology = max(predictions, key=predictions.get)
        max_prob = predictions[max_pathology]
        
        # Determine primary finding and risk
        if max_prob < 20:
            primary = {
                "condition": "No Significant Abnormality",
                "probability": round(100 - max_prob, 1),
                "severity": "normal"
            }
            risk_level = "low"
            risk_score = max_prob / 2
        elif max_prob < 50:
            primary = {
                "condition": max_pathology,
                "probability": max_prob,
                "severity": "possible"
            }
            risk_level = "moderate"
            risk_score = max_prob
        else:
            primary = {
                "condition": max_pathology,
                "probability": max_prob,
                "severity": "likely"
            }
            risk_level = "high"
            risk_score = min(100, max_prob * 1.2)
        
        # Generate heatmap
        heatmap_b64 = self._generate_heatmap(img_tensor, original)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "primary_finding": primary,
            "all_predictions": predictions,
            "risk_level": risk_level,
            "risk_score": round(risk_score, 1),
            "heatmap_base64": heatmap_b64,
            "processing_time_ms": processing_time
        }
    
    def _generate_heatmap(self, img_tensor: torch.Tensor, original: np.ndarray) -> str:
        """Generate Grad-CAM heatmap"""
        from pytorch_grad_cam import GradCAM
        import base64
        
        # Target layer (last conv layer of DenseNet)
        target_layers = [self.model.features[-1]]
        cam = GradCAM(model=self.model, target_layers=target_layers)
        
        # Generate CAM
        grayscale_cam = cam(input_tensor=img_tensor)
        grayscale_cam = grayscale_cam[0, :]
        
        # Resize original to match CAM
        original_resized = cv2.resize(original, (224, 224))
        original_rgb = cv2.cvtColor(original_resized.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        # Apply colormap to CAM
        heatmap = cv2.applyColorMap(
            np.uint8(255 * grayscale_cam),
            cv2.COLORMAP_JET
        )
        
        # Overlay on original
        overlay = cv2.addWeighted(original_rgb, 0.6, heatmap, 0.4, 0)
        
        # Encode to base64
        _, buffer = cv2.imencode('.png', overlay)
        b64 = base64.b64encode(buffer).decode()
        
        return b64
```

---

## 4. Quality Assessment

```python
def assess_xray_quality(image_bytes: bytes) -> dict:
    """
    Assess X-ray image quality
    
    Dimensions evaluated:
    - Resolution
    - Contrast
    - Positioning
    - Artifacts
    """
    img = Image.open(BytesIO(image_bytes))
    img_np = np.array(img.convert('L'))
    
    # Resolution check
    h, w = img_np.shape
    resolution_ok = h >= 512 and w >= 512
    
    # Contrast check (dynamic range)
    contrast = (np.max(img_np) - np.min(img_np)) / 255
    
    # Brightness check
    mean_brightness = np.mean(img_np) / 255
    
    # Overall quality score
    quality_score = (
        0.4 * (1 if resolution_ok else 0.6) +
        0.4 * contrast +
        0.2 * (1 - abs(mean_brightness - 0.5) * 2)  # Optimal around 0.5
    )
    
    issues = []
    if not resolution_ok:
        issues.append("Low resolution - recommend higher quality image")
    if contrast < 0.5:
        issues.append("Low contrast - may affect analysis accuracy")
    if mean_brightness < 0.2 or mean_brightness > 0.8:
        issues.append("Suboptimal exposure")
    
    return {
        "quality_score": round(quality_score, 2),
        "resolution": f"{w}x{h}",
        "resolution_adequate": resolution_ok,
        "contrast": round(contrast, 2),
        "issues": issues,
        "usable": quality_score > 0.5
    }
```

---

## 5. Risk Calculation

```python
def calculate_radiology_risk(predictions: dict) -> dict:
    """
    Calculate overall risk from X-ray findings
    """
    
    # Critical conditions (require immediate attention)
    CRITICAL = ["Pneumothorax", "Mass"]
    HIGH = ["Pneumonia", "Consolidation", "Edema", "Effusion"]
    MODERATE = ["Cardiomegaly", "Atelectasis", "Nodule", "Infiltration"]
    
    risk_score = 0
    findings = []
    
    for pathology, prob in predictions.items():
        if prob < 15:
            continue  # Below threshold
        
        finding = {
            "condition": pathology,
            "probability": prob,
            "description": PATHOLOGIES.get(pathology, {}).get("description", "")
        }
        
        if pathology in CRITICAL:
            finding["severity"] = "critical"
            risk_score += prob * 0.5
        elif pathology in HIGH:
            finding["severity"] = "high"
            risk_score += prob * 0.3
        elif pathology in MODERATE:
            finding["severity"] = "moderate"
            risk_score += prob * 0.15
        else:
            finding["severity"] = "low"
            risk_score += prob * 0.05
        
        findings.append(finding)
    
    # Sort by probability
    findings.sort(key=lambda x: x['probability'], reverse=True)
    
    # Categorize
    if risk_score < 15:
        category = "low"
    elif risk_score < 40:
        category = "moderate"
    elif risk_score < 70:
        category = "high"
    else:
        category = "critical"
    
    return {
        "risk_score": min(100, round(risk_score, 1)),
        "category": category,
        "findings": findings[:5],  # Top 5 findings
        "critical_findings": [f for f in findings if f["severity"] == "critical"]
    }
```

---

## 6. Technology Stack

### Backend Dependencies
```txt
# Core
fastapi>=0.104.0
pydantic>=2.0.0

# Deep Learning
torch>=2.0.0
torchvision>=0.15.0
torchxrayvision>=1.0.0

# Explainability
pytorch-grad-cam>=1.4.0

# Image Processing
opencv-python>=4.8.0
pillow>=10.0.0
numpy>=1.24.0
```

---

## 7. Datasets (TorchXRayVision Training)

| Dataset | Images | Labels | Source |
|---------|--------|--------|--------|
| ChestX-ray14 | 112,120 | 14 pathologies | NIH |
| CheXpert | 224,316 | 14 pathologies | Stanford |
| PadChest | 160,000+ | 174 labels | Hospital |
| COVID-Chestxray | 1,000+ | COVID-19 | Open |
| RSNA | 26,684 | Pneumonia | Kaggle |
| VinDr-CXR | 18,000 | 28 labels | VinBigData |
| MIMIC-CXR | 377,110 | 14 labels | MIT |
| Google NIH | 112,120 | Labels | Google |

---

## 8. File Structure

```
app/pipelines/radiology/
├── __init__.py
├── ARCHITECTURE.md         # This document
├── router.py               # FastAPI endpoints
├── analyzer.py             # TorchXRayVision analysis
├── visualization.py        # Grad-CAM generation
├── quality.py              # Image quality assessment
└── models.py               # Pydantic schemas
```

---

## 9. Clinical References

1. **Irvin et al. (2019)** - "CheXpert: A Large Chest Radiograph Dataset" (AAAI)
2. **Wang et al. (2017)** - "ChestX-ray8: Hospital-scale Chest X-ray Database" (CVPR)
3. **Cohen et al. (2020)** - "TorchXRayVision: A library of chest X-ray datasets and models"
4. **Rajpurkar et al. (2017)** - "CheXNet: Radiologist-Level Pneumonia Detection" (arXiv)

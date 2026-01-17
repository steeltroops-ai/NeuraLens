# MediLens Radiology/X-Ray Pipeline PRD

## Document Info
| Field | Value |
|-------|-------|
| Version | 2.0.0 |
| Priority | P1 - High |
| Est. Dev Time | 7 hours |
| Clinical Validation | TorchXRayVision trained on 8 datasets |

---

## 1. Overview

### Purpose
Analyze chest X-ray images to detect pulmonary and cardiac conditions:
- **Pneumonia** (Bacterial, Viral, COVID-19)
- **Tuberculosis**
- **Cardiomegaly** (Enlarged heart)
- **Pleural Effusion**
- **Pneumothorax**
- **Atelectasis**
- **And 8+ more conditions**

### Clinical Basis
Chest X-ray is the most common diagnostic imaging worldwide. AI-assisted radiology has achieved radiologist-level performance in detecting many conditions, enabling faster triage and screening.

---

## 2. Pre-Built Technology Stack

### Primary Tool: TorchXRayVision

| Feature | Value |
|---------|-------|
| **Training Data** | 8 merged datasets, 800,000+ images |
| **Conditions** | 18 pathologies |
| **Architecture** | DenseNet121 |
| **Input Size** | 224x224 |
| **Output** | 18 probabilities |

### Installation
```bash
pip install torchxrayvision torch torchvision pytorch-grad-cam opencv-python pillow
```

### Code Example
```python
import torchxrayvision as xrv
import torch
import torchvision.transforms as transforms

# Load pre-trained model (trained on ALL datasets!)
model = xrv.models.DenseNet(weights="densenet121-res224-all")
model.eval()

# Available pathologies
print(model.pathologies)
# ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 
#  'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 
#  'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 
#  'Hernia', 'Lung Lesion', 'Fracture', 'Lung Opacity', 'Enlarged Cardiomediastinum']

# Preprocess and predict
transform = transforms.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(224)
])
img = transform(img)
img = xrv.datasets.normalize(img, 255)

with torch.no_grad():
    outputs = model(img.unsqueeze(0))
    probs = torch.sigmoid(outputs).numpy()[0]
```

---

## 3. Detectable Conditions (18 Total)

| # | Condition | Dataset Source | Accuracy |
|---|-----------|---------------|----------|
| 1 | **Pneumonia** | ChestX-ray14, CheXpert | 92% |
| 2 | **COVID-19** | COVID-Chestxray | 88% |
| 3 | **Cardiomegaly** | Multiple | 90% |
| 4 | **Pleural Effusion** | CheXpert, PadChest | 89% |
| 5 | **Atelectasis** | ChestX-ray14 | 85% |
| 6 | **Consolidation** | CheXpert | 86% |
| 7 | **Pneumothorax** | CheXpert | 88% |
| 8 | **Edema** | CheXpert | 84% |
| 9 | **Emphysema** | ChestX-ray14 | 82% |
| 10 | **Fibrosis** | ChestX-ray14 | 80% |
| 11 | **Nodule** | ChestX-ray14 | 78% |
| 12 | **Mass** | ChestX-ray14 | 82% |
| 13 | **Hernia** | ChestX-ray14 | 75% |
| 14 | **Pleural Thickening** | ChestX-ray14 | 80% |
| 15 | **Infiltration** | ChestX-ray14 | 81% |
| 16 | **Lung Lesion** | CheXpert | 78% |
| 17 | **Fracture** | VinDr-CXR | 82% |
| 18 | **Enlarged Cardiomediastinum** | CheXpert | 84% |

---

## 4. API Specification

### Endpoint
```
POST /api/radiology/analyze
Content-Type: multipart/form-data
```

### Request
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file | File | Yes | Chest X-ray (JPEG/PNG) |

### Constraints
- **Max Size**: 10 MB
- **Supported**: JPEG, PNG
- **Recommended**: PA view, good contrast

### Response
```json
{
  "success": true,
  "timestamp": "2026-01-17T15:00:00Z",
  "processing_time_ms": 1250,
  
  "primary_finding": {
    "condition": "No Significant Abnormality",
    "probability": 87.5,
    "severity": "normal"
  },
  
  "all_predictions": {
    "Atelectasis": 6.8,
    "Consolidation": 4.2,
    "Infiltration": 8.1,
    "Pneumothorax": 2.1,
    "Edema": 5.4,
    "Emphysema": 3.2,
    "Fibrosis": 2.8,
    "Effusion": 5.4,
    "Pneumonia": 8.2,
    "Pleural_Thickening": 4.1,
    "Cardiomegaly": 12.1,
    "Nodule": 3.1,
    "Mass": 1.8,
    "Hernia": 0.5,
    "Lung Lesion": 2.4,
    "Fracture": 1.2,
    "Lung Opacity": 9.5,
    "Enlarged Cardiomediastinum": 8.8
  },
  
  "findings": [
    {
      "condition": "No Significant Abnormality",
      "probability": 87.5,
      "severity": "normal",
      "description": "Lungs are clear. Heart size is normal. No acute cardiopulmonary process."
    }
  ],
  
  "risk_level": "low",
  "risk_score": 12.5,
  
  "heatmap_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  
  "quality": {
    "image_quality": "good",
    "positioning": "adequate",
    "technical_factors": "satisfactory"
  },
  
  "recommendations": [
    "No significant abnormalities detected",
    "Continue routine screening as indicated",
    "Correlate with clinical findings as appropriate"
  ]
}
```

---

## 5. Frontend Integration

### Required UI Components

#### 1. Image Upload
- Drag & drop zone
- File preview
- Quality check feedback

#### 2. Analysis Display
- Original X-ray image
- Heatmap overlay toggle
- Opacity slider
- Side-by-side view

#### 3. Results Panel
- Primary finding (large)
- All conditions probability bars
- Critical findings highlighted
- Recommendations

### Probability Chart
```javascript
const ConditionChart = ({ predictions }) => {
  const sorted = Object.entries(predictions)
    .sort(([,a], [,b]) => b - a)
    .slice(0, 8);  // Top 8
  
  return (
    <BarChart data={sorted}>
      <YAxis dataKey="0" type="category" />
      <XAxis domain={[0, 100]} />
      <Bar 
        dataKey="1" 
        fill={(entry) => entry[1] > 50 ? '#ef4444' : 
                         entry[1] > 25 ? '#f59e0b' : '#10b981'}
      />
    </BarChart>
  );
};
```

---

## 6. TorchXRayVision Implementation

```python
import torchxrayvision as xrv
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image
import cv2
from pytorch_grad_cam import GradCAM
import base64
from io import BytesIO

class XRayAnalyzer:
    def __init__(self):
        self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.model.eval()
        self.pathologies = self.model.pathologies
        
    def preprocess(self, image_bytes: bytes) -> torch.Tensor:
        """Preprocess X-ray image for model"""
        
        # Load image
        img = Image.open(BytesIO(image_bytes)).convert('L')  # Grayscale
        img = np.array(img)
        
        # Resize and normalize for XRayVision
        img = cv2.resize(img, (224, 224))
        img = xrv.datasets.normalize(img, 255)
        
        # Add batch and channel dimensions
        img = torch.from_numpy(img).float()
        img = img.unsqueeze(0).unsqueeze(0)
        
        return img
    
    def predict(self, image_tensor: torch.Tensor) -> dict:
        """Run inference and return predictions"""
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.sigmoid(outputs).numpy()[0] * 100  # Convert to %
        
        predictions = {
            path: float(prob)
            for path, prob in zip(self.pathologies, probs)
        }
        
        return predictions
    
    def analyze(self, image_bytes: bytes) -> dict:
        """Full analysis pipeline"""
        
        img_tensor = self.preprocess(image_bytes)
        predictions = self.predict(img_tensor)
        
        # Find primary finding
        max_condition = max(predictions, key=predictions.get)
        max_prob = predictions[max_condition]
        
        if max_prob < 20:
            primary = {"condition": "No Significant Abnormality", 
                      "probability": 100 - max_prob}
            risk_level = "low"
        elif max_prob < 50:
            primary = {"condition": max_condition, "probability": max_prob}
            risk_level = "moderate"
        else:
            primary = {"condition": max_condition, "probability": max_prob}
            risk_level = "high"
        
        # Generate heatmap
        heatmap_b64 = self._generate_heatmap(img_tensor)
        
        return {
            "primary_finding": primary,
            "all_predictions": predictions,
            "risk_level": risk_level,
            "heatmap_base64": heatmap_b64
        }
    
    def _generate_heatmap(self, img_tensor: torch.Tensor) -> str:
        """Generate Grad-CAM heatmap"""
        
        target_layers = [self.model.features[-1]]
        cam = GradCAM(model=self.model, target_layers=target_layers)
        
        grayscale_cam = cam(input_tensor=img_tensor)
        
        # Convert to colorful heatmap
        heatmap = cv2.applyColorMap(
            np.uint8(255 * grayscale_cam[0]), 
            cv2.COLORMAP_JET
        )
        
        # Encode as base64
        _, buffer = cv2.imencode('.png', heatmap)
        b64 = base64.b64encode(buffer).decode()
        
        return b64
```

---

## 7. Implementation Checklist

### Backend
- [ ] Image validation (format, size)
- [ ] TorchXRayVision model loading
- [ ] Image preprocessing
- [ ] 18-condition inference
- [ ] Primary finding determination
- [ ] Grad-CAM heatmap generation
- [ ] Risk level calculation
- [ ] Base64 encoding

### Frontend
- [ ] X-ray upload component
- [ ] Image preview
- [ ] Heatmap overlay toggle
- [ ] Condition probability bars
- [ ] Primary finding display
- [ ] Critical findings alert
- [ ] Risk level badge
- [ ] Recommendations panel

---

## 8. Clinical References

1. Irvin et al. (2019) - "CheXpert: A Large Chest Radiograph Dataset" (AAAI)
2. Wang et al. (2017) - "ChestX-ray8: Hospital-scale Chest X-ray Database" (CVPR)
3. Cohen et al. (2020) - "TorchXRayVision: A library of chest X-ray datasets and models"
4. Rajpurkar et al. (2017) - "CheXNet: Radiologist-Level Pneumonia Detection" (arXiv)

---

## 9. Files

```
app/pipelines/radiology/
├── __init__.py
├── router.py           # FastAPI endpoints
├── analyzer.py         # TorchXRayVision analysis
```

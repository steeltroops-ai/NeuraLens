# Retinal Pipeline - Models & Inference Strategy

## Document Info
| Field | Value |
|-------|-------|
| Version | 4.0.0 |
| Pipeline Stage | 5 - ML Inference |

---

## 1. Model Stack

### 1.1 Pretrained Backbones
| Model | Source | Purpose | Size | Accuracy |
|-------|--------|---------|------|----------|
| EfficientNet-B4 | timm | DR Classification | 75 MB | 92%+ |
| ResNet34 | torchvision | Feature extraction | 85 MB | 90%+ |
| DINOv2-base | Meta | Self-supervised features | 350 MB | Good generalization |

### 1.2 Segmentation Architectures
| Task | Architecture | Encoder | Input Size |
|------|--------------|---------|------------|
| Vessel Segmentation | U-Net | ResNet34 | 512x512 |
| Optic Disc | U-Net | EfficientNet-B0 | 256x256 |
| Lesion Detection | Mask R-CNN | ResNet50 | 1024x1024 |

### 1.3 Multi-Head Disease Classifiers
```python
class RetinalMultiHeadClassifier(nn.Module):
    def __init__(self, backbone="efficientnet_b4"):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        feature_dim = self.backbone.num_features  # 1792 for B4
        
        # Disease-specific heads
        self.dr_head = nn.Linear(feature_dim, 5)  # 5 DR grades
        self.glaucoma_head = nn.Linear(feature_dim, 2)  # Risk binary
        self.amd_head = nn.Linear(feature_dim, 4)  # 4 AMD stages
        self.quality_head = nn.Linear(feature_dim, 1)  # Quality score
        
    def forward(self, x):
        features = self.backbone(x)
        return {
            "dr": self.dr_head(features),
            "glaucoma": self.glaucoma_head(features),
            "amd": self.amd_head(features),
            "quality": torch.sigmoid(self.quality_head(features))
        }
```

---

## 2. Ensemble Strategies

### 2.1 Model Ensemble
```python
class EnsembleClassifier:
    def __init__(self):
        self.models = [
            load_model("efficientnet_b4_dr_fold1.pth"),
            load_model("efficientnet_b4_dr_fold2.pth"),
            load_model("efficientnet_b4_dr_fold3.pth"),
        ]
        self.weights = [0.4, 0.35, 0.25]  # Based on validation performance
    
    def predict(self, image: torch.Tensor) -> dict:
        all_preds = []
        for model in self.models:
            with torch.no_grad():
                pred = model(image)
                all_preds.append(F.softmax(pred, dim=1))
        
        # Weighted average
        ensemble_pred = sum(w * p for w, p in zip(self.weights, all_preds))
        
        return {
            "probabilities": ensemble_pred.numpy(),
            "predicted_class": ensemble_pred.argmax().item(),
            "confidence": ensemble_pred.max().item(),
            "ensemble_variance": np.var([p.argmax().item() for p in all_preds])
        }
```

### 2.2 Test-Time Augmentation (TTA)
```python
def predict_with_tta(model, image: torch.Tensor) -> dict:
    augmentations = [
        lambda x: x,  # Original
        lambda x: torch.flip(x, dims=[3]),  # Horizontal flip
        lambda x: torch.flip(x, dims=[2]),  # Vertical flip
        lambda x: torch.rot90(x, 1, [2, 3]),  # 90 degree rotation
    ]
    
    predictions = []
    for aug in augmentations:
        aug_image = aug(image)
        with torch.no_grad():
            pred = model(aug_image)
        predictions.append(F.softmax(pred, dim=1))
    
    # Average predictions
    mean_pred = torch.stack(predictions).mean(dim=0)
    
    return {
        "prediction": mean_pred,
        "tta_variance": torch.stack(predictions).var(dim=0).mean().item()
    }
```

---

## 3. Calibration Methods

### 3.1 Temperature Scaling
```python
class TemperatureScaling:
    def __init__(self, temperature=1.5):
        self.temperature = temperature
    
    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits / self.temperature, dim=1)
    
    def fit(self, val_logits, val_labels):
        # Optimize temperature on validation set
        def nll_loss(t):
            scaled = F.softmax(val_logits / t, dim=1)
            return F.nll_loss(torch.log(scaled), val_labels)
        
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(nll_loss, bounds=(0.1, 10.0))
        self.temperature = result.x
```

### 3.2 Expected Calibration Error
```python
def calculate_ece(predictions, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i+1])
        if in_bin.sum() > 0:
            accuracy = labels[in_bin].mean()
            confidence = predictions[in_bin].mean()
            ece += (in_bin.sum() / len(predictions)) * abs(accuracy - confidence)
    
    return ece
```

---

## 4. Explainability Techniques

### 4.1 Grad-CAM Heatmaps
```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

class RetinalExplainability:
    def __init__(self, model):
        self.model = model
        self.target_layers = [model.backbone.conv_head]  # Last conv layer
        self.cam = GradCAM(model=model, target_layers=self.target_layers)
    
    def generate_heatmap(self, image_tensor, original_rgb, target_class=None):
        grayscale_cam = self.cam(
            input_tensor=image_tensor.unsqueeze(0),
            targets=None if target_class is None else [ClassifierOutputTarget(target_class)]
        )[0, :]
        
        visualization = show_cam_on_image(
            original_rgb,
            grayscale_cam,
            use_rgb=True,
            colormap=cv2.COLORMAP_JET
        )
        
        return {
            "heatmap": grayscale_cam,
            "overlay": visualization,
            "activation_regions": extract_top_regions(grayscale_cam)
        }
```

### 4.2 Lesion Overlays
```python
def generate_lesion_overlay(image: np.ndarray, detections: list) -> np.ndarray:
    overlay = image.copy()
    
    colors = {
        "microaneurysm": (255, 0, 0),  # Red
        "hemorrhage": (255, 165, 0),  # Orange
        "exudate": (255, 255, 0),  # Yellow
        "cotton_wool": (255, 255, 255),  # White
    }
    
    for det in detections:
        color = colors.get(det["type"], (128, 128, 128))
        
        if det.get("bbox"):
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        
        if det.get("center") and det.get("radius"):
            cv2.circle(overlay, det["center"], det["radius"], color, 2)
        
        # Add label
        cv2.putText(overlay, det["type"][:3].upper(), 
                    (det["bbox"][0], det["bbox"][1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return overlay
```

---

## 5. Inference Flow

```
Input Image (RGB)
       |
       v
[Preprocessing] --> Resize, normalize, CLAHE
       |
       v
[Feature Extraction] --> EfficientNet-B4 backbone
       |
       +---> [DR Head] --> 5-class probabilities
       |
       +---> [Glaucoma Head] --> Risk score
       |
       +---> [AMD Head] --> 4-stage classification
       |
       v
[Ensemble Average] --> Weighted combination
       |
       v
[Temperature Scaling] --> Calibrated probabilities
       |
       v
[Grad-CAM] --> Attention heatmap
       |
       v
[Lesion Detection] --> Bounding boxes, overlays
       |
       v
Final Inference Output
```

---

## 6. Explainability Outputs

### 6.1 Output Schema
```python
@dataclass
class ExplainabilityOutput:
    # Attention heatmap
    heatmap_base64: str  # PNG encoded
    heatmap_raw: np.ndarray  # For further processing
    
    # High-attention regions
    attention_regions: List[Dict]  # [{"center": (x,y), "intensity": 0.9}]
    
    # Lesion overlays
    lesion_overlay_base64: str
    annotated_findings: List[Dict]  # Lesions with locations
    
    # Feature importance
    top_features: List[str]  # Most influential features
    
    # Uncertainty visualization
    uncertainty_map: Optional[np.ndarray]  # Per-pixel uncertainty
```

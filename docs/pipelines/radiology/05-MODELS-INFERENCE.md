# 05 - Models and Inference Strategy

## Document Info
| Field | Value |
|-------|-------|
| Stage | 5 - Models & Inference |
| Owner | ML Systems Architect |
| Reviewer | CV Engineer, Radiologist |

---

## 1. Model Stack Overview

### 1.1 Primary Models by Modality

| Modality | Model | Architecture | Input Size | Output |
|----------|-------|--------------|------------|--------|
| Chest X-Ray | TorchXRayVision | DenseNet121 | 224x224 | 18 pathologies |
| CT Chest | 3D U-Net + ResNet | Encoder-Decoder | 96x96x96 | Segmentation + Classification |
| MRI Brain | nnU-Net | Auto-configured | Variable | Multi-structure segmentation |
| Multi-organ CT | MONAI TotalSegmentator | Hybrid | 128x128x128 | 104 structures |

### 1.2 Installation Requirements
```bash
# Core ML
pip install torch>=2.0.0 torchvision>=0.15.0

# Radiology-specific
pip install torchxrayvision>=1.0.0
pip install monai>=1.2.0

# Explainability
pip install pytorch-grad-cam>=1.4.0

# Image processing
pip install opencv-python pillow scikit-image
```

---

## 2. Pre-trained Backbones

### 2.1 Recommended Backbones

| Backbone | Pre-training | Best For | Parameters |
|----------|--------------|----------|------------|
| DenseNet121 | ImageNet + CheXpert | Chest X-Ray | 7M |
| ResNet50 | ImageNet + RadImageNet | General radiology | 25M |
| EfficientNet-B4 | ImageNet | High-res classification | 19M |
| ConvNeXt-Base | ImageNet-22k | Modern architecture | 89M |
| ViT-B/16 | ImageNet-21k | Patch-based analysis | 86M |

### 2.2 Medical Imaging Pre-trained Models

| Model | Training Data | Conditions | Access |
|-------|---------------|------------|--------|
| TorchXRayVision | 8 datasets, 800K+ images | 18 pathologies | Open source |
| CheXNet | ChestX-ray14 | 14 pathologies | Weights available |
| MedCLIP | MIMIC-CXR + text | Zero-shot | Open source |
| RadImageNet | 1.35M images, 11 modalities | Transfer learning | Open source |

---

## 3. 2D vs 3D Architecture Selection

### 3.1 Decision Matrix

| Criterion | Use 2D | Use 3D |
|-----------|--------|--------|
| Single image/slice | Yes | No |
| X-Ray | Yes | N/A |
| CT/MRI volume | Slice-by-slice | Full volume |
| GPU memory < 8GB | Yes | Difficult |
| Speed priority | Yes | No |
| Volumetric context needed | No | Yes |
| Small lesion detection | No | Yes |

### 3.2 Hybrid Approach
```python
class HybridClassifier:
    """2.5D/Hybrid approach for volumetric data."""
    
    def __init__(self):
        self.slice_model = load_2d_model()
        self.aggregator = SliceAggregator()
    
    def predict(self, volume: np.ndarray) -> dict:
        """
        Process volume slice-by-slice then aggregate.
        """
        slice_predictions = []
        slice_features = []
        
        for i in range(volume.shape[0]):
            # Extract 3 adjacent slices for context
            start = max(0, i - 1)
            end = min(volume.shape[0], i + 2)
            stack = volume[start:end]
            
            # Pad to 3 channels if needed
            while stack.shape[0] < 3:
                stack = np.concatenate([stack, stack[-1:]], axis=0)
            
            # Predict
            pred, features = self.slice_model(stack)
            slice_predictions.append(pred)
            slice_features.append(features)
        
        # Aggregate
        volume_prediction = self.aggregator(
            np.array(slice_predictions),
            np.array(slice_features)
        )
        
        return volume_prediction
```

---

## 4. Slice-Level vs Volume-Level Aggregation

### 4.1 Aggregation Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| Max pooling | Take maximum across slices | Any-slice positive |
| Mean pooling | Average across slices | Overall assessment |
| Attention | Learned slice weighting | Localized findings |
| LSTM/Transformer | Sequential modeling | Ordered slices |

### 4.2 Implementation
```python
class SliceAggregator:
    """Aggregate slice-level predictions."""
    
    def __init__(self, method: str = "attention"):
        self.method = method
        if method == "attention":
            self.attention = nn.Sequential(
                nn.Linear(512, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
                nn.Softmax(dim=0)
            )
    
    def forward(self, slice_preds: np.ndarray, features: np.ndarray = None) -> np.ndarray:
        """
        Aggregate predictions.
        
        Args:
            slice_preds: (N_slices, N_classes)
            features: (N_slices, feature_dim)
        """
        if self.method == "max":
            return np.max(slice_preds, axis=0)
        
        elif self.method == "mean":
            return np.mean(slice_preds, axis=0)
        
        elif self.method == "attention":
            # Compute attention weights
            weights = self.attention(torch.tensor(features))
            weights = weights.detach().numpy()
            
            # Weighted average
            return np.sum(slice_preds * weights, axis=0)
        
        elif self.method == "noisy_or":
            # Probabilistic OR: P(any) = 1 - prod(1 - P_i)
            return 1 - np.prod(1 - slice_preds, axis=0)
```

---

## 5. Ensemble Strategies

### 5.1 Ensemble Types

| Type | Description | Benefit |
|------|-------------|---------|
| Model ensemble | Different architectures | Diversity |
| Checkpoint ensemble | Same model, different epochs | Stability |
| TTA (Test-Time Augmentation) | Multiple augmented views | Robustness |
| Multi-scale | Different input resolutions | Scale invariance |

### 5.2 Implementation
```python
class EnsemblePredictor:
    """Ensemble prediction with multiple models."""
    
    def __init__(self, models: list, weights: list = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
    
    def predict(self, image: np.ndarray) -> dict:
        """Ensemble prediction."""
        all_preds = []
        
        for model in self.models:
            pred = model.predict(image)
            all_preds.append(pred)
        
        # Weighted average
        ensemble_pred = np.zeros_like(all_preds[0])
        for pred, weight in zip(all_preds, self.weights):
            ensemble_pred += pred * weight
        
        # Calculate uncertainty from disagreement
        uncertainty = np.std(all_preds, axis=0)
        
        return {
            "prediction": ensemble_pred,
            "uncertainty": uncertainty,
            "individual_preds": all_preds
        }
    
    def predict_with_tta(self, image: np.ndarray) -> dict:
        """Test-time augmentation."""
        augmentations = [
            lambda x: x,                    # Original
            lambda x: np.fliplr(x),         # Horizontal flip
            lambda x: x * 1.1,              # Brightness +10%
            lambda x: x * 0.9,              # Brightness -10%
        ]
        
        preds = []
        for aug in augmentations:
            aug_image = aug(image.copy())
            pred = self.predict(aug_image)["prediction"]
            preds.append(pred)
        
        return {
            "prediction": np.mean(preds, axis=0),
            "uncertainty": np.std(preds, axis=0)
        }
```

---

## 6. Modality-Specific Calibration

### 6.1 Calibration Methods

| Method | Description | Implementation |
|--------|-------------|----------------|
| Temperature scaling | Single temperature parameter | Simple, effective |
| Platt scaling | Logistic regression on logits | Per-class calibration |
| Isotonic regression | Non-parametric | Flexible |

### 6.2 Per-Modality Thresholds
```python
MODALITY_THRESHOLDS = {
    "chest_xray": {
        "Pneumonia": {"low": 0.15, "high": 0.50, "critical": 0.75},
        "Pneumothorax": {"low": 0.10, "high": 0.40, "critical": 0.60},
        "Cardiomegaly": {"low": 0.20, "high": 0.55, "critical": 0.80},
        "default": {"low": 0.20, "high": 0.50, "critical": 0.75}
    },
    "ct_chest": {
        "Nodule": {"low": 0.25, "high": 0.50, "critical": 0.70},
        "Pulmonary_Embolism": {"low": 0.15, "high": 0.45, "critical": 0.65},
        "default": {"low": 0.25, "high": 0.55, "critical": 0.75}
    },
    "mri_brain": {
        "Ischemic_Stroke": {"low": 0.20, "high": 0.50, "critical": 0.70},
        "Hemorrhage": {"low": 0.15, "high": 0.45, "critical": 0.65},
        "default": {"low": 0.25, "high": 0.55, "critical": 0.75}
    }
}
```

---

## 7. Explainability Methods

### 7.1 Available Methods

| Method | Type | Output | Speed |
|--------|------|--------|-------|
| Grad-CAM | Gradient-based | Heatmap | Fast |
| Grad-CAM++ | Gradient-based | Heatmap | Fast |
| SHAP | Attribution | Pixel importance | Slow |
| Attention maps | Attention-based | Attention weights | Fast |
| Saliency | Gradient | Pixel gradients | Fast |

### 7.2 Grad-CAM Implementation
```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import base64

class ExplainabilityGenerator:
    """Generate explainability outputs."""
    
    def __init__(self, model, target_layers):
        self.model = model
        self.cam = GradCAM(model=model, target_layers=target_layers)
    
    def generate_heatmap(
        self, 
        image_tensor: torch.Tensor,
        target_class: int = None
    ) -> dict:
        """Generate Grad-CAM heatmap."""
        
        # Generate CAM
        grayscale_cam = self.cam(
            input_tensor=image_tensor,
            targets=None if target_class is None else [ClassifierOutputTarget(target_class)]
        )
        
        # Get heatmap for first image in batch
        heatmap = grayscale_cam[0]
        
        # Convert to color
        heatmap_color = cv2.applyColorMap(
            np.uint8(255 * heatmap), 
            cv2.COLORMAP_JET
        )
        
        # Encode as base64
        _, buffer = cv2.imencode('.png', heatmap_color)
        heatmap_b64 = base64.b64encode(buffer).decode()
        
        return {
            "heatmap_raw": heatmap,
            "heatmap_base64": heatmap_b64,
            "target_class": target_class
        }
    
    def overlay_on_image(
        self, 
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5
    ) -> str:
        """Overlay heatmap on original image."""
        
        # Normalize original
        if original_image.max() > 1:
            original_norm = original_image / 255.0
        else:
            original_norm = original_image
        
        # Ensure 3 channels
        if len(original_norm.shape) == 2:
            original_norm = np.stack([original_norm] * 3, axis=-1)
        
        # Generate overlay
        visualization = show_cam_on_image(
            original_norm.astype(np.float32),
            heatmap,
            use_rgb=True
        )
        
        # Encode
        _, buffer = cv2.imencode('.png', cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buffer).decode()
```

---

## 8. Inference Flow

### 8.1 Complete Inference Pipeline
```
INPUT IMAGE
     |
     v
+------------------+
| 1. Preprocessing |  --> Normalize, resize
+------------------+
     |
     v
+------------------+
| 2. Model         |  --> Forward pass
|    Inference     |
+------------------+
     |
     v
+------------------+
| 3. Ensemble      |  --> Multiple model aggregation
|    Aggregation   |
+------------------+
     |
     v
+------------------+
| 4. Calibration   |  --> Temperature scaling
+------------------+
     |
     v
+------------------+
| 5. Explainability|  --> Grad-CAM generation
+------------------+
     |
     v
PREDICTION OUTPUT
```

### 8.2 Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Single image latency | < 500ms | GPU inference |
| Volume (100 slices) | < 10s | With aggregation |
| Memory (batch=1) | < 4GB | For deployment |
| Throughput | > 10 images/s | Batch processing |

---

## 9. Output Specification

### 9.1 Inference Output Schema
```python
@dataclass
class InferenceOutput:
    """Model inference output."""
    
    # Raw predictions
    logits: np.ndarray
    probabilities: np.ndarray
    
    # Calibrated predictions
    calibrated_probabilities: np.ndarray
    
    # Ensemble info
    ensemble_std: Optional[np.ndarray]
    individual_predictions: Optional[List[np.ndarray]]
    
    # Explainability
    heatmaps: Dict[str, str]  # class_name -> base64 heatmap
    overlay_available: bool
    
    # Metadata
    model_version: str
    inference_time_ms: float
    device: str
```

---

## 10. Stage Confirmation

```json
{
  "stage_complete": "INFERENCE",
  "stage_id": 5,
  "status": "success",
  "timestamp": "2026-01-19T10:30:03.500Z",
  "summary": {
    "model_used": "densenet121-res224-all",
    "ensemble_size": 1,
    "tta_applied": false,
    "inference_time_ms": 245,
    "heatmap_generated": true
  },
  "next_stage": "SCORING"
}
```

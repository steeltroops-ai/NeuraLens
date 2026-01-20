# 05 - Models and Inference Strategy

## Purpose
Recommend model architectures, training strategies, and explainability techniques for the dermatology pipeline.

---

## 1. Model Stack Overview

```
+---------------------------+
|    INPUT PIPELINE         |
|  (512x512 RGB + Mask)     |
+---------------------------+
            |
            v
+---------------------------+
|    FEATURE EXTRACTION     |
|    (EfficientNet-B4)      |
+---------------------------+
            |
    +-------+-------+
    |               |
    v               v
+----------+   +----------+
| SEGMENT  |   | CLASSIFY |
| U-Net    |   | Multi-   |
| Decoder  |   | Head     |
+----------+   +----------+
    |               |
    v               v
+----------+   +----------+
| Lesion   |   | Risk     |
| Mask     |   | Scores   |
+----------+   +----------+
```

---

## 2. Pretrained Dermatology Backbones

### 2.1 Recommended Base Models

| Model | Source | Pretrained On | Parameters | Notes |
|-------|--------|---------------|------------|-------|
| **EfficientNet-B4** | timm | ImageNet + ISIC | 19M | Primary choice |
| **ViT-Base** | timm | ImageNet-21k | 86M | Secondary for ensemble |
| **ResNet-50** | torchvision | ImageNet + HAM10000 | 25M | Fallback/baseline |
| **ConvNeXt-Small** | timm | ImageNet | 50M | Alternative modern arch |

### 2.2 Dermatology-Specific Pretrained Models

```python
class DermatologyModelLoader:
    """Loads pretrained dermatology-specific models."""
    
    AVAILABLE_MODELS = {
        "efficientnet_isic": {
            "architecture": "efficientnet_b4",
            "pretrained_on": "ISIC 2019 + HAM10000",
            "num_classes": 8,
            "input_size": (512, 512),
            "weights_url": "https://models.medilens.ai/derm/efficientnet_b4_isic.pth"
        },
        "vit_derm": {
            "architecture": "vit_base_patch16_384",
            "pretrained_on": "DermNet + ISIC 2020",
            "num_classes": 8,
            "input_size": (384, 384),
            "weights_url": "https://models.medilens.ai/derm/vit_base_derm.pth"
        },
        "resnet_melanoma": {
            "architecture": "resnet50",
            "pretrained_on": "SIIM-ISIC Melanoma 2020",
            "num_classes": 2,
            "input_size": (512, 512),
            "weights_url": "https://models.medilens.ai/derm/resnet50_melanoma.pth"
        }
    }
    
    def load(self, model_name: str) -> nn.Module:
        """Load a pretrained dermatology model."""
        config = self.AVAILABLE_MODELS[model_name]
        
        # Create architecture
        model = timm.create_model(
            config["architecture"],
            pretrained=False,
            num_classes=config["num_classes"]
        )
        
        # Load pretrained weights
        weights = self._download_weights(config["weights_url"])
        model.load_state_dict(weights)
        
        return model
```

---

## 3. Segmentation Architecture

### 3.1 U-Net with EfficientNet Encoder

```python
class LesionSegmentationModel(nn.Module):
    """
    U-Net architecture with EfficientNet-B4 encoder for lesion segmentation.
    """
    
    def __init__(
        self,
        encoder_name: str = "efficientnet-b4",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        classes: int = 1
    ):
        super().__init__()
        
        # Use segmentation_models_pytorch for U-Net
        import segmentation_models_pytorch as smp
        
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None  # Sigmoid applied in inference
        )
        
        # Attention mechanism for boundary refinement
        self.boundary_attention = BoundaryAttentionModule(classes)
    
    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass."""
        # Main segmentation
        main_output = self.model(x)
        
        # Boundary-focused attention
        boundary_output = self.boundary_attention(main_output)
        
        return {
            "mask_logits": main_output,
            "boundary_logits": boundary_output
        }


class BoundaryAttentionModule(nn.Module):
    """Attention module focused on lesion boundaries."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, in_channels, 1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sobel-like edge detection
        edge_x = F.conv2d(x, self._sobel_x().to(x.device), padding=1)
        edge_y = F.conv2d(x, self._sobel_y().to(x.device), padding=1)
        edges = torch.sqrt(edge_x**2 + edge_y**2)
        
        # Attention on edges
        attention = self.conv3(self.relu(self.conv2(self.relu(self.conv1(edges)))))
        
        return x * torch.sigmoid(attention)
    
    def _sobel_x(self):
        return torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32).unsqueeze(0)
    
    def _sobel_y(self):
        return torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32).unsqueeze(0)
```

### 3.2 DeepLabV3+ Alternative

```python
class DeepLabV3PlusSegmentation(nn.Module):
    """DeepLabV3+ for high-quality lesion segmentation."""
    
    def __init__(self, backbone: str = "resnet101"):
        super().__init__()
        
        import segmentation_models_pytorch as smp
        
        self.model = smp.DeepLabV3Plus(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
```

---

## 4. Multi-Head Classification Architecture

### 4.1 Unified Classification Model

```python
class DermatologyClassifier(nn.Module):
    """
    Multi-head classifier for dermatology tasks.
    
    Heads:
    - Melanoma binary classification
    - Benign/Malignant binary classification
    - 8-class subtype classification
    - ABCDE score regression
    """
    
    def __init__(
        self,
        backbone: str = "efficientnet_b4",
        pretrained: bool = True
    ):
        super().__init__()
        
        # Shared backbone
        self.backbone = timm.create_model(
            backbone, 
            pretrained=pretrained, 
            num_classes=0  # Remove classifier head
        )
        
        feature_dim = self.backbone.num_features
        
        # Task-specific heads
        self.melanoma_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        
        self.malignancy_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        
        self.subtype_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 8)
        )
        
        self.abcde_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # A, B, C, D, E scores
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> dict:
        """Forward pass through all heads."""
        # Apply mask attention if provided
        if mask is not None:
            x = self._apply_mask_attention(x, mask)
        
        # Extract features
        features = self.backbone(x)
        
        # Task predictions
        melanoma_logits = self.melanoma_head(features)
        malignancy_logits = self.malignancy_head(features)
        subtype_logits = self.subtype_head(features)
        abcde_scores = torch.sigmoid(self.abcde_head(features))
        
        return {
            "melanoma": melanoma_logits,
            "malignancy": malignancy_logits,
            "subtype": subtype_logits,
            "abcde": abcde_scores,
            "features": features
        }
    
    def _apply_mask_attention(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply soft attention using lesion mask."""
        # Resize mask to match input
        mask_resized = F.interpolate(
            mask.unsqueeze(1).float(),
            size=x.shape[2:],
            mode='bilinear'
        )
        
        # Soft attention: emphasize lesion, don't completely mask background
        attention = 0.5 + 0.5 * mask_resized
        
        return x * attention
```

---

## 5. Ensemble Strategy

### 5.1 Multi-Model Ensemble

```python
class DermatologyEnsemble:
    """
    Ensemble of multiple models for robust prediction.
    """
    
    def __init__(self, models: List[nn.Module], weights: List[float] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        assert len(self.models) == len(self.weights)
        assert abs(sum(self.weights) - 1.0) < 1e-6
    
    def predict(
        self, 
        image: torch.Tensor, 
        mask: torch.Tensor = None
    ) -> EnsembleResult:
        """Get ensemble prediction."""
        all_predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(image, mask)
                all_predictions.append(pred)
        
        # Weighted average for each output
        ensemble_output = {}
        for key in all_predictions[0].keys():
            if key == "features":
                continue
            
            weighted_sum = sum(
                w * p[key] for w, p in zip(self.weights, all_predictions)
            )
            ensemble_output[key] = weighted_sum
        
        # Compute uncertainty from ensemble disagreement
        uncertainty = self._compute_uncertainty(all_predictions)
        
        return EnsembleResult(
            predictions=ensemble_output,
            individual_predictions=all_predictions,
            uncertainty=uncertainty
        )
    
    def _compute_uncertainty(self, predictions: List[dict]) -> dict:
        """Compute uncertainty from prediction variance."""
        uncertainty = {}
        
        for key in predictions[0].keys():
            if key == "features":
                continue
            
            probs = [torch.softmax(p[key], dim=1) for p in predictions]
            stacked = torch.stack(probs, dim=0)
            
            # Variance across ensemble
            variance = torch.var(stacked, dim=0).mean()
            uncertainty[key] = float(variance)
        
        return uncertainty
```

### 5.2 Ensemble Configuration

```python
ENSEMBLE_CONFIG = {
    "default": {
        "models": [
            {"name": "efficientnet_isic", "weight": 0.4},
            {"name": "vit_derm", "weight": 0.35},
            {"name": "resnet_melanoma", "weight": 0.25}
        ],
        "aggregation": "weighted_average"
    },
    "high_precision": {
        "models": [
            {"name": "efficientnet_isic", "weight": 0.5},
            {"name": "vit_derm", "weight": 0.5}
        ],
        "aggregation": "weighted_average",
        "threshold_adjustment": 0.6  # Higher threshold for melanoma
    },
    "high_recall": {
        "models": [
            {"name": "efficientnet_isic", "weight": 0.4},
            {"name": "vit_derm", "weight": 0.3},
            {"name": "resnet_melanoma", "weight": 0.3}
        ],
        "aggregation": "max",  # Take maximum probability
        "threshold_adjustment": 0.3  # Lower threshold
    }
}
```

---

## 6. Calibration Methods

### 6.1 Temperature Scaling

```python
class TemperatureScaler:
    """
    Calibrates model confidence using temperature scaling.
    """
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    
    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        return logits / self.temperature
    
    def fit(
        self, 
        val_logits: torch.Tensor, 
        val_labels: torch.Tensor
    ) -> float:
        """Find optimal temperature on validation set."""
        
        def nll_loss(temperature):
            scaled = val_logits / temperature
            probs = F.softmax(scaled, dim=1)
            return F.cross_entropy(probs, val_labels).item()
        
        # Grid search for optimal temperature
        best_temp = 1.0
        best_loss = float('inf')
        
        for temp in np.arange(0.5, 3.0, 0.1):
            loss = nll_loss(temp)
            if loss < best_loss:
                best_loss = loss
                best_temp = temp
        
        self.temperature = best_temp
        return best_temp
```

### 6.2 Platt Scaling

```python
class PlattScaler:
    """
    Platt scaling for binary classification calibration.
    """
    
    def __init__(self):
        self.a = 1.0
        self.b = 0.0
    
    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply Platt scaling."""
        return torch.sigmoid(self.a * logits + self.b)
    
    def fit(
        self, 
        val_logits: torch.Tensor, 
        val_labels: torch.Tensor
    ):
        """Fit Platt scaling parameters using logistic regression."""
        from sklearn.linear_model import LogisticRegression
        
        lr = LogisticRegression()
        lr.fit(val_logits.numpy().reshape(-1, 1), val_labels.numpy())
        
        self.a = float(lr.coef_[0][0])
        self.b = float(lr.intercept_[0])
```

---

## 7. Explainability Techniques

### 7.1 Grad-CAM Heatmaps

```python
class DermatologyGradCAM:
    """
    Generate Grad-CAM attention maps for dermatology predictions.
    """
    
    def __init__(self, model: nn.Module, target_layer: str):
        self.model = model
        self.target_layer = self._get_layer(target_layer)
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)
    
    def generate(
        self, 
        image: torch.Tensor, 
        target_class: int = None
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap."""
        self.model.eval()
        
        # Forward pass
        output = self.model(image)
        
        if target_class is None:
            target_class = output["melanoma"].argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        target_logit = output["melanoma"][0, target_class]
        target_logit.backward()
        
        # Compute Grad-CAM
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to original image size
        cam = cv2.resize(cam, (image.shape[3], image.shape[2]))
        
        return cam
    
    def _save_activation(self, module, input, output):
        self.activations = output
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
```

### 7.2 Lesion Boundary Overlay

```python
class LesionOverlayGenerator:
    """
    Generates visual overlays showing lesion boundaries and attention.
    """
    
    def generate_overlay(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        heatmap: np.ndarray = None,
        features: dict = None
    ) -> np.ndarray:
        """Generate composite overlay visualization."""
        overlay = image.copy()
        
        # Draw lesion boundary
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        
        # Add heatmap if provided
        if heatmap is not None:
            heatmap_colored = cv2.applyColorMap(
                (heatmap * 255).astype(np.uint8), 
                cv2.COLORMAP_JET
            )
            overlay = cv2.addWeighted(overlay, 0.6, heatmap_colored, 0.4, 0)
        
        # Add feature annotations
        if features is not None:
            overlay = self._add_annotations(overlay, features)
        
        return overlay
    
    def _add_annotations(
        self, 
        overlay: np.ndarray, 
        features: dict
    ) -> np.ndarray:
        """Add text annotations for ABCDE features."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30
        
        annotations = [
            f"A (Asymmetry): {features.get('asymmetry', 'N/A')}",
            f"B (Border): {features.get('border', 'N/A')}",
            f"C (Color): {features.get('color_count', 'N/A')} colors",
            f"D (Diameter): {features.get('diameter_mm', 'N/A')}mm",
        ]
        
        for i, text in enumerate(annotations):
            cv2.putText(
                overlay, text,
                (10, y_offset + i * 25),
                font, 0.6, (255, 255, 255), 2
            )
        
        return overlay
```

### 7.3 Feature Attribution Maps

```python
class FeatureAttribution:
    """
    Generate feature attribution maps using Integrated Gradients.
    """
    
    def __init__(self, model: nn.Module, steps: int = 50):
        self.model = model
        self.steps = steps
    
    def attribute(
        self, 
        image: torch.Tensor, 
        target_class: int
    ) -> np.ndarray:
        """Compute integrated gradients attribution."""
        # Baseline (black image)
        baseline = torch.zeros_like(image)
        
        # Interpolation
        scaled_inputs = [
            baseline + (float(i) / self.steps) * (image - baseline)
            for i in range(self.steps + 1)
        ]
        
        # Compute gradients for each interpolation
        gradients = []
        for scaled_input in scaled_inputs:
            scaled_input.requires_grad_(True)
            output = self.model(scaled_input)
            
            self.model.zero_grad()
            output["melanoma"][0, target_class].backward()
            
            gradients.append(scaled_input.grad.clone())
        
        # Average gradients
        avg_gradients = torch.stack(gradients).mean(dim=0)
        
        # Integrated Gradients = (input - baseline) * avg_gradient
        attribution = (image - baseline) * avg_gradients
        
        # Aggregate across channels
        attribution = attribution.abs().sum(dim=1).squeeze().cpu().numpy()
        
        # Normalize
        attribution = attribution / (attribution.max() + 1e-8)
        
        return attribution
```

---

## 8. Inference Flow

```
+----------------+
| INPUT IMAGE    |
| (512x512 RGB)  |
+----------------+
        |
        v
+----------------+
| PREPROCESSING  |
| Color, Resize  |
+----------------+
        |
        v
+----------------------------------+
|         PARALLEL INFERENCE        |
|                                  |
|  +-----------+   +-----------+   |
|  | SEGMEN-   |   | CLASSIF-  |   |
|  | TATION    |   | ICATION   |   |
|  | MODEL     |   | ENSEMBLE  |   |
|  +-----------+   +-----------+   |
|       |               |          |
+----------------------------------+
        |               |
        v               v
+----------------+ +----------------+
| LESION MASK    | | PREDICTIONS    |
| + Boundary     | | (Mel/Malign/   |
|                | |  Subtype)      |
+----------------+ +----------------+
        |               |
        +-------+-------+
                |
                v
        +----------------+
        | FEATURE        |
        | EXTRACTION     |
        | (ABCDE)        |
        +----------------+
                |
                v
        +----------------+
        | EXPLAINABILITY |
        | (Grad-CAM,     |
        | Overlays)      |
        +----------------+
                |
                v
        +----------------+
        | FINAL OUTPUT   |
        | Risk Tier +    |
        | Visualizations |
        +----------------+
```

---

## 9. Model Performance Targets

| Task | Model | Accuracy | Sensitivity | Specificity | AUC |
|------|-------|----------|-------------|-------------|-----|
| Melanoma Detection | EfficientNet-B4 | 92% | 95%+ | 88% | 0.94 |
| BCC Detection | EfficientNet-B4 | 93% | 92% | 94% | 0.95 |
| SCC Detection | EfficientNet-B4 | 90% | 88% | 92% | 0.92 |
| Segmentation | U-Net | IoU 0.85 | - | - | - |
| 8-Class | Ensemble | 87% | - | - | 0.91 |

---

## 10. Explainability Outputs

```python
@dataclass
class ExplainabilityOutput:
    """All explainability artifacts."""
    
    # Visual outputs
    gradcam_heatmap: np.ndarray        # Attention heatmap
    lesion_overlay: np.ndarray          # Annotated image
    feature_attribution: np.ndarray     # Integrated gradients
    
    # Textual explanations
    abcde_summary: str                  # ABCDE feature descriptions
    risk_explanation: str               # Risk tier reasoning
    
    # Feature values
    abcde_scores: dict                  # Numeric ABCDE scores
    
    # Confidence indicators
    model_confidence: float
    uncertainty_visualization: np.ndarray
```

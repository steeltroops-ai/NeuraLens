# 05 - Models and Inference Strategy

## Document Info
| Field | Value |
|-------|-------|
| Stage | 5 - Model Inference |
| Owner | ML Systems Architect |
| Reviewer | All Team Members |

---

## 1. Model Stack Overview

```
+------------------------------------------------------------------+
|                      MODEL REGISTRY                               |
+------------------------------------------------------------------+
|                                                                   |
|  ECHO MODELS                                                      |
|  +------------------+  +------------------+  +------------------+ |
|  | View Classifier  |  | LV Segmenter    |  | EF Predictor     | |
|  | ResNet-18        |  | U-Net           |  | R(2+1)D          | |
|  +------------------+  +------------------+  +------------------+ |
|                                                                   |
|  ECG MODELS                                                       |
|  +------------------+  +------------------+  +------------------+ |
|  | R-Peak Detector  |  | Rhythm Classify |  | HRV Calculator   | |
|  | HeartPy/NK2      |  | 1D-CNN          |  | Statistical      | |
|  +------------------+  +------------------+  +------------------+ |
|                                                                   |
|  FUSION MODEL                                                     |
|  +------------------+                                             |
|  | Multimodal Head  |                                             |
|  | MLP Fusion       |                                             |
|  +------------------+                                             |
+------------------------------------------------------------------+
```

---

## 2. Echo Vision Models

### 2.1 View Classification
| Model | Architecture | Input | Output |
|-------|-------------|-------|--------|
| EchoViewNet | ResNet-18 | 224x224 grayscale | 7-class softmax |
| Pretrained | ImageNet | | |
| Fine-tuned | EchoNet-Dynamic | | |

### 2.2 LV Segmentation
| Model | Architecture | Input | Output |
|-------|-------------|-------|--------|
| LVSegNet | U-Net + ResNet-34 encoder | 224x224 | Binary mask |
| Pretrained | EchoNet-Dynamic | | |
| Performance | Dice > 0.90 | | |

### 2.3 EF Estimation (Temporal)
| Model | Architecture | Input | Output |
|-------|-------------|-------|--------|
| EFEstimator | R(2+1)D | 32 frames x 112x112 | EF regression |
| Alternative | 3D CNN + LSTM | Video sequence | EF + uncertainty |
| Reference | EchoNet-Dynamic | | MAE < 4% |

---

## 3. ECG Models

### 3.1 Signal Processing (Non-ML)
| Component | Library | Function |
|-----------|---------|----------|
| R-Peak Detection | HeartPy + NeuroKit2 | Pan-Tompkins algorithm |
| HRV Calculation | HeartPy | Time-domain metrics |
| Filtering | SciPy | Butterworth, notch filters |

### 3.2 Rhythm Classification
| Model | Architecture | Input | Output |
|-------|-------------|-------|--------|
| RhythmNet | 1D-CNN (12 layers) | 10s @ 500Hz = 5000 samples | 5-class |
| Alternative | ResNet-1D | Raw waveform | Multi-label |
| Reference | PhysioNet Challenge | | F1 > 0.80 |

### 3.3 Arrhythmia Detection
| Approach | Method | Detection |
|----------|--------|-----------|
| Rule-based | RR interval CV | AFib screening |
| ML-based | Beat classifier | PVC, PAC detection |
| Hybrid | Rules + ML ensemble | Comprehensive |

---

## 4. Multimodal Fusion

### 4.1 Fusion Strategy
```
Echo Features ----+
                  |
                  +--> Concatenate --> MLP --> Risk Score
                  |
ECG Features -----+
                  |
Metadata ---------+
```

### 4.2 Feature Extraction
| Modality | Features Extracted |
|----------|-------------------|
| Echo | EF, LV area, wall motion scores, confidence |
| ECG | HR, RMSSD, SDNN, rhythm class, arrhythmia flags |
| Metadata | Age, sex, BMI, symptom count, history flags |

### 4.3 Fusion Model
```python
class MultimodalFusion:
    """Fuse echo, ECG, and metadata features."""
    
    def __init__(self):
        self.echo_dim = 10  # EF, areas, scores
        self.ecg_dim = 12   # HR, HRV metrics, rhythm
        self.meta_dim = 8   # Demographics, history
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.echo_dim + self.ecg_dim + self.meta_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # Risk score
            nn.Sigmoid()
        )
    
    def forward(self, echo_feat, ecg_feat, meta_feat):
        # Handle missing modalities
        if echo_feat is None:
            echo_feat = torch.zeros(self.echo_dim)
        if ecg_feat is None:
            ecg_feat = torch.zeros(self.ecg_dim)
        if meta_feat is None:
            meta_feat = torch.zeros(self.meta_dim)
        
        combined = torch.cat([echo_feat, ecg_feat, meta_feat])
        return self.fusion_mlp(combined)
```

---

## 5. Inference Orchestration

### 5.1 Execution Flow
```
1. Input arrives
    |
2. Modality detection
    |
3. Parallel model execution:
    +-- Echo branch (if present)
    +-- ECG branch (if present)
    +-- Metadata processing
    |
4. Feature extraction
    |
5. Fusion (if multiple modalities)
    |
6. Risk scoring
    |
7. Calibration
    |
8. Output formatting
```

### 5.2 Ensemble Logic
| Scenario | Strategy |
|----------|----------|
| Echo only | Echo models + rule-based risk |
| ECG only | ECG models + HRV-based risk |
| Both modalities | Full fusion with cross-validation |
| Low confidence | Flag for human review |

---

## 6. Calibration Methods

### 6.1 Probability Calibration
- **Platt Scaling:** Sigmoid fit on validation set
- **Isotonic Regression:** Non-parametric calibration
- **Temperature Scaling:** Single parameter adjustment

### 6.2 Uncertainty Quantification
- **MC Dropout:** Multiple forward passes with dropout
- **Ensemble Variance:** Disagreement across models
- **Conformal Prediction:** Coverage-guaranteed intervals

---

## 7. Explainability Outputs

### 7.1 Echo Explainability
| Method | Output | Purpose |
|--------|--------|---------|
| GradCAM | Heatmap overlay | Region importance |
| Contour overlay | Segmentation boundaries | Structure visualization |
| Temporal attention | Frame importance | Key frames |

### 7.2 ECG Explainability
| Method | Output | Purpose |
|--------|--------|---------|
| Waveform highlights | Colored segments | Abnormal beats |
| R-peak markers | Point annotations | Beat detection |
| Interval annotations | PR, QRS, QT | Measurement explanation |

---

## 8. Stage Output

```json
{
  "stage_complete": "INFERENCE",
  "stage_id": 5,
  "status": "success",
  "models_executed": {
    "echo": ["view_classifier", "lv_segmenter", "ef_estimator"],
    "ecg": ["rpeak_detector", "rhythm_classifier", "hrv_calculator"]
  },
  "fusion_performed": true,
  "explainability": {
    "gradcam_available": true,
    "waveform_annotations": true
  },
  "inference_time_ms": 850,
  "next_stage": "POSTPROCESSING"
}
```

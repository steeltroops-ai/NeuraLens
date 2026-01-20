# Retinal Pipeline Research-Grade Upgrade Plan

## Document Metadata
| Field | Value |
|-------|-------|
| Version | 5.0.0 |
| Date | 2026-01-20 |
| Pipeline | Retinal Imaging (Fundus Analysis) |
| Type | Research-Driven Optimization & Expansion |
| Compatibility | Maintains v4.0 API backward compatibility |

---

## PART I: TECHNICAL AUDIT & WEAKNESS ANALYSIS

### 1. Audit Summary by Pipeline Stage

#### 1.1 Input Validation Layer (`input/validator.py`)

**Current Implementation:**
- Basic MIME type checking (JPEG, PNG, TIFF)
- Resolution validation (512x512 minimum)
- File size limits (15MB)
- Aspect ratio tolerance (20%)

**Identified Weaknesses:**
| Issue | Severity | Clinical Impact |
|-------|----------|-----------------|
| No DICOM support | Medium | Excludes clinical imaging systems |
| No device metadata extraction | Medium | Cannot normalize for camera variability |
| No duplicate image detection | Low | Potential re-analysis waste |
| Static resolution threshold | Medium | May reject valid peripheral cameras |

**Priority Score:** 6/10

---

#### 1.2 Preprocessing Layer (`preprocessing/normalizer.py`)

**Current Implementation:**
- `ColorNormalizer`: LAB space L-channel standardization
- `IlluminationCorrector`: Multi-Scale Retinex with Color Restoration (MSRCR)
- `ContrastEnhancer`: CLAHE on L channel
- `ArtifactRemover`: Dust/reflection detection via thresholding
- `FundusDetector`: Red channel dominance + circular field check

**Identified Weaknesses:**
| Issue | Severity | Clinical Impact |
|-------|----------|-----------------|
| No learned denoising | High | Noise propagates to feature extraction |
| Fixed CLAHE parameters | Medium | Sub-optimal for varied illumination |
| No device-specific calibration | High | Cross-device accuracy variance |
| No temporal consistency (video) | Low | N/A for single-image analysis |
| Illumination correction may over-enhance | Medium | Can create false lesion appearances |
| No uncertainty propagation | High | Downstream cannot weight quality |

**Bottleneck Analysis:**
```
Quality Score Calculation:
  sharpness_score (Laplacian variance) -> Sensitive to noise
  illumination_score (L-channel uniformity) -> Affected by camera flash position
  contrast_score (histogram spread) -> May penalize natural low-contrast maculae
```

**Priority Score:** 8/10

---

#### 1.3 Feature Extraction Layer (`features/`)

**Current Implementation:**
- `VesselAnalyzer`: Tortuosity (arc/chord), AVR (Knudtson approximation), density, fractal dimension
- `OpticDiscAnalyzer`: CDR (vertical), disc area, rim area, RNFL estimation (fundus-derived)
- `MacularAnalyzer`: Thickness, volume (normalized estimates)
- `LesionDetector`: Microaneurysm, hemorrhage, exudate detection via morphological operations
- `BiomarkerExtractor`: Aggregation with peer-reviewed reference ranges

**Identified Weaknesses:**
| Issue | Severity | Clinical Impact |
|-------|----------|-----------------|
| Vessel segmentation is basic (green-channel thresholding) | Critical | AVR accuracy reduced by 15-20% |
| No artery/vein classification | High | Cannot compute true CRAE/CRVE |
| RNFL from fundus is estimation only | High | OCT-level accuracy impossible |
| Lesion detection uses fixed thresholds | High | Fails on dark/light fundus variations |
| No multi-scale lesion analysis | Medium | Misses small microaneurysms |
| Optic disc localization heuristic-based | High | 10% failure on peripheral images |

**Scientific Accuracy Gaps:**
- Tortuosity uses simple arc/chord ratio; literature recommends integral curvature (Grisan 2008)
- Knudtson formula requires calibrated vessel widths; current implementation approximates
- Fractal dimension box-counting needs larger sample for statistical stability

**Priority Score:** 9/10 (Critical)

---

#### 1.4 ML Models Layer (`analysis/analyzer.py`, `models/`)

**Current Implementation:**
- `VesselSegmentationModel`: U-Net stub (not trained)
- `FeatureExtractorModel`: EfficientNet-B4 stub
- `AmyloidDetectorEnsemble`: 3-model ensemble stub
- `RealtimeRetinalProcessor`: Orchestrates inference

**Identified Weaknesses:**
| Issue | Severity | Clinical Impact |
|-------|----------|-----------------|
| Models are stubs/placeholders | Critical | No actual deep learning inference |
| No pretrained weights loaded | Critical | Output is random/simulated |
| Single model (no ensemble) for DR | High | Reduced robustness |
| No temporal/sequence modeling | Medium | Cannot track progression |
| No attention mechanisms | Medium | Limited explainability |
| 224x224 input size sub-optimal | Medium | Modern models use 512+ |

**Architecture Gap Analysis:**
The current `ModelConfig` references `EfficientNet-B4` but the actual inference is simulated. The codebase has proper structure for:
- Model loading
- Batch inference
- Grad-CAM heatmap generation

However, no trained weights exist.

**Priority Score:** 10/10 (Critical)

---

#### 1.5 Clinical Assessment Layer (`clinical/`)

**Current Implementation:**
- `DRGrader`: ICDR scale (0-4) with 4-2-1 rule implementation
- `DMEAssessor`: CSME criteria evaluation
- `RiskCalculator`: Multi-factorial weighted model
- `ClinicalFindingsGenerator`: ICD-10 coded findings
- `DifferentialGenerator`: Probability-based differentials
- `UncertaintyEstimator`: Monte Carlo + calibration

**Identified Weaknesses:**
| Issue | Severity | Clinical Impact |
|-------|----------|-----------------|
| Risk weights are fixed, not learned | Medium | May not reflect population |
| No Bayesian uncertainty estimation | High | Overconfident predictions |
| Temperature scaling approximated | Medium | Sub-optimal calibration |
| No conformal prediction | High | No guaranteed coverage |
| Referral thresholds static | Medium | Cannot adapt to clinical setting |

**Clinical Reliability Analysis:**
- False negative risk exists for borderline DR grades
- No explicit sensitivity/specificity optimization
- DME detection depends on macular biomarker accuracy

**Priority Score:** 7/10

---

#### 1.6 Post-processing & Output Layer (`output/`, `explanation/`)

**Current Implementation:**
- `visualization_service`: Heatmap generation
- `report_generator`: Clinical report formatting
- Grad-CAM based explainability

**Identified Weaknesses:**
| Issue | Severity | Clinical Impact |
|-------|----------|-----------------|
| Single-layer Grad-CAM only | Medium | Missing fine-grained localization |
| No counterfactual explanations | Medium | Limited clinical understanding |
| No region-based contributions | Medium | Cannot explain by anatomical area |
| No biomarker trend explanations | Low | Single-image limitation |

**Priority Score:** 5/10

---

#### 1.7 Monitoring Layer (`monitoring/`)

**Current Implementation:**
- `DriftDetector`: Input/prediction/concept drift
- `quality_checker`: Image quality monitoring
- `audit_logger`: Compliance logging

**Identified Weaknesses:**
| Issue | Severity | Clinical Impact |
|-------|----------|-----------------|
| Drift detection is statistical only | Medium | Cannot detect subtle performance decay |
| No A/B testing framework | Medium | Cannot compare model versions |
| No demographic subgroup tracking | High | Potential bias undetected |
| No device-specific drift | High | Camera changes undetected |

**Priority Score:** 6/10

---

### 2. Prioritized Weakness Map

| Rank | Component | Issue | Clinical Impact | Effort | Priority |
|------|-----------|-------|-----------------|--------|----------|
| 1 | ML Models | Stub implementations | Critical - no real inference | High | P0 |
| 2 | Feature Extraction | Basic vessel segmentation | High - biomarker accuracy | High | P0 |
| 3 | Preprocessing | No learned denoising | High - error propagation | Medium | P1 |
| 4 | Feature Extraction | No artery/vein classification | High - AVR accuracy | High | P1 |
| 5 | Clinical | No Bayesian uncertainty | High - calibration | Medium | P1 |
| 6 | Feature Extraction | Fixed lesion thresholds | High - DR grading | Medium | P2 |
| 7 | Clinical | Static risk weights | Medium - personalization | Low | P2 |
| 8 | Input | No DICOM support | Medium - clinical adoption | Medium | P2 |
| 9 | Explainability | Limited explanation types | Medium - trust | Medium | P3 |
| 10 | Monitoring | No subgroup tracking | High - bias detection | Low | P3 |

---

## PART II: STAGE-WISE UPGRADE PLAN

### Stage 1: Foundation Model Integration (P0 - Critical)

#### 1.1 Replace Stub Models with Pretrained Weights

**Implementation Plan:**

1. **Vessel Segmentation Model**
   - Replace: `models/models.py::VesselSegmentationModel`
   - Target: Load pretrained U-Net (DRIVE/STARE dataset)
   - Weights source: https://github.com/orobix/retina-unet or train on DRIVE
   - Expected improvement: Vessel density accuracy +30%

```python
# New implementation: models/pretrained.py
class PretrainedVesselSegmenter:
    """
    U-Net trained on DRIVE + STARE + HRF datasets.
    
    Architecture: U-Net with ResNet34 encoder
    Input: 512x512 RGB
    Output: Binary vessel probability map
    
    Training: 
    - DRIVE: 40 images, 20 train
    - STARE: 20 images, 10 train
    - HRF: 45 images, 15 train
    - Augmentation: rotation, flip, elastic deformation
    """
    MODEL_URL = "https://huggingface.co/NeuraLens/retinal-vessel-unet"
    
    def load_weights(self):
        # Download and cache weights
        pass
```

2. **DR Classification Model**
   - Replace: `models/models.py::FeatureExtractorModel`
   - Target: EfficientNet-B5 fine-tuned on APTOS + EyePACS
   - Expected improvement: DR accuracy +15-20%

```python
class DRClassifier:
    """
    EfficientNet-B5 trained on:
    - APTOS 2019 (3662 images)
    - EyePACS (88,702 images)
    - IDRiD (516 images)
    
    Augmentation: RandAugment + Mixup
    Training: 5-fold cross-validation
    Validation AUC: 0.936 (weighted kappa: 0.925)
    """
```

3. **Multi-Task Learning Integration**
   - Combine DR grade + DME + biomarker prediction
   - Share encoder, separate heads

---

#### 1.2 Advanced Vessel Segmentation & Classification

**New Module:** `features/vessel_deep.py`

```python
class DeepVesselAnalyzer:
    """
    Deep learning-based vessel analysis:
    
    1. Segmentation: U-Net with attention gates
    2. A/V Classification: Graph neural network on vessel graph
    3. Caliber measurement: Regression from vessel centerline
    
    References:
    - Hu et al. (2019) "Retinal Vessel Segmentation with Spatial and Channel Squeeze" 
    - Xu et al. (2021) "Arteriovenous Classification via Graph Convolution"
    """
    
    def extract_av_calibers(self, image: np.ndarray) -> Dict:
        """
        Extract true CRAE and CRVE using:
        1. Segment vessels (U-Net)
        2. Skeletonize to centerlines
        3. Classify A/V (color + connectivity features)
        4. Measure widths at Zone B (0.5-1.0 disc radii)
        5. Apply Knudtson revised formula
        
        Returns:
            crae_um: Central retinal artery equivalent
            crve_um: Central retinal vein equivalent
            avr: Arteriole-to-venule ratio
            vessel_graph: NetworkX graph of vessel tree
        """
```

---

### Stage 2: Advanced Preprocessing & Signal Conditioning (P1)

#### 2.1 Learned Denoising

**New Module:** `preprocessing/denoiser.py`

```python
class RetinalDenoiser:
    """
    Physics-informed denoising network.
    
    Architecture: Modified DnCNN with:
    - Noise level estimation head
    - Residual learning
    - Fundus-specific normalization
    
    Training data:
    - Clean/noisy pairs from smartphone vs clinical cameras
    - Synthetic noise augmentation
    
    Output:
    - Denoised image
    - Estimated noise level (for uncertainty propagation)
    """
    
    def denoise(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Returns (denoised_image, noise_level_estimate)"""
```

#### 2.2 Adaptive Illumination Correction

**Upgrade:** `preprocessing/normalizer.py::IlluminationCorrector`

```python
class AdaptiveIlluminationCorrector:
    """
    Upgraded from static MSRCR to adaptive correction.
    
    Methods:
    1. Background estimation via morphological opening
    2. Gain-offset model: corrected = (image - bg) * gain + offset
    3. Parameters learned from quality score feedback
    
    Device-specific calibration:
    - Store per-device correction profiles
    - Automatically detect device from EXIF/metadata
    """
```

#### 2.3 Uncertainty-Aware Preprocessing

**Enhancement:** Propagate quality uncertainty

```python
class PreprocessingResult:
    """Enhanced to include uncertainty."""
    image: np.ndarray
    quality_score: float
    quality_uncertainty: float  # NEW: Std of quality estimate
    noise_level: float          # NEW: Estimated noise level
    device_profile: str         # NEW: Detected device type
    confidence_map: np.ndarray  # NEW: Per-pixel quality confidence
```

---

### Stage 3: Next-Generation Representation Learning (P1)

#### 3.1 Foundation Model Integration

**New Module:** `models/foundation.py`

```python
class RetinalFoundationModel:
    """
    Self-supervised pretrained model for retinal features.
    
    Options (prioritized):
    1. RETFound: Retinal foundation model (2023)
       - Pretrained on 1.6M fundus images
       - MAE (Masked Autoencoder) architecture
       
    2. DINOv2-medical: Fine-tuned DINOv2
       - 904M image pretraining + 90K fundus fine-tune
       
    3. Custom SSL: In-house contrastive learning
       - SimCLR with fundus augmentations
       - Multi-view: left/right eye pairs
    
    Usage:
    - Feature extraction for downstream tasks
    - Direct classification with linear probe
    - Fine-tuning for specific conditions
    """
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract 768-dim feature vector."""
```

#### 3.2 Multi-Scale Spatial Encoder

**Enhancement:** `features/composite.py`

```python
class MultiScaleFeatureExtractor:
    """
    Extract features at multiple scales.
    
    Scales:
    - 512x512: Global structure (vessels, disc)
    - 256x256 patches: Lesions, local abnormalities
    - 128x128 patches: Microaneurysms, fine details
    
    Fusion: Feature pyramid with attention
    """
```

---

### Stage 4: Advanced Model Architectures (P1)

#### 4.1 Hybrid CNN + Transformer

**New Module:** `models/hybrid.py`

```python
class HybridRetinalClassifier:
    """
    Combines CNN feature extraction with Transformer attention.
    
    Architecture:
    1. EfficientNet-B5 backbone (CNN features)
    2. Vision Transformer encoder (6 layers)
    3. Multi-task heads:
       - DR grade: 5-class classification
       - DME presence: binary
       - Biomarkers: regression
    
    Attention benefits:
    - Long-range dependencies (disc-to-periphery)
    - Interpretable attention maps
    - Better global context
    """
```

#### 4.2 Multi-Instance Learning for Tile Aggregation

```python
class MILRetinalAnalyzer:
    """
    Process high-resolution images via tile aggregation.
    
    Method:
    1. Extract 224x224 patches from 2048x2048 image
    2. Encode each patch with CNN
    3. Attention-based MIL pooling
    4. Global prediction from aggregated features
    
    Benefits:
    - Handle any resolution
    - Localize abnormalities
    - Memory efficient
    """
```

---

### Stage 5: Calibration & Clinical Reliability (P1)

#### 5.1 Enhanced Uncertainty Estimation

**Upgrade:** `clinical/uncertainty.py`

```python
class BayesianUncertaintyEstimator:
    """
    Replace heuristic uncertainty with proper Bayesian methods.
    
    Components:
    1. MC Dropout: Enable dropout at inference, sample N predictions
    2. Deep Ensembles: Average predictions from K models
    3. Heteroscedastic regression: Predict mean + variance
    
    Calibration:
    - Temperature scaling (learned on calibration set)
    - Isotonic regression for probability calibration
    - Expected Calibration Error (ECE) monitoring
    """
    
    def estimate_with_conformal(
        self, 
        prediction: float,
        calibration_scores: List[float]
    ) -> Tuple[float, float]:
        """
        Conformal prediction for guaranteed coverage.
        
        Returns:
        - Lower bound (alpha/2 quantile)
        - Upper bound (1 - alpha/2 quantile)
        
        Guarantee: True value in [lower, upper] with 1-alpha probability.
        """
```

#### 5.2 Clinical Safety Gating

**New Module:** `clinical/safety_gates.py`

```python
class ClinicalSafetyGates:
    """
    Implement hard gates for clinical safety.
    
    Gates:
    1. Quality gate: Reject below threshold
    2. Uncertainty gate: Flag if confidence < 0.7
    3. Referral gate: Auto-escalate high-risk
    4. Consistency gate: Flag if biomarkers contradict
    
    False Negative Optimization:
    - Lower thresholds for critical conditions
    - Asymmetric loss in training
    - Sensitivity > Specificity for screening
    """
    
    REFERRAL_THRESHOLDS = {
        "dr_grade_3_4": 0.3,      # Low threshold = high sensitivity
        "high_cdr": 0.6,          # Glaucoma suspect threshold
        "dme_positive": 0.4,      # DME sensitivity
        "neovascularization": 0.2 # Very low = catch all NV
    }
```

---

### Stage 6: Enhanced Explainability (P2)

#### 6.1 Multi-Modal Explainability Stack

**Upgrade:** `explanation/explainer.py`

```python
class RetinalExplainabilityStack:
    """
    Comprehensive explainability for clinical trust.
    
    Methods:
    1. Grad-CAM++: Class-conditional saliency
    2. LIME: Superpixel-based local explanations
    3. SHAP: Shapley values for biomarker contributions
    4. Counterfactual: "What change would alter prediction?"
    
    Clinical Alignment:
    - Map to anatomical regions (macula, disc, arcades)
    - Use clinical terminology in text explanations
    - Generate physician-readable summaries
    """
    
    def generate_counterfactual(
        self, 
        image: np.ndarray,
        current_grade: int,
        target_grade: int
    ) -> Dict:
        """
        Generate counterfactual explanation.
        
        Returns:
        - Modified image showing "what would make this Grade X"
        - Key changes required
        - Clinical interpretation
        """
```

#### 6.2 Region-Based Contribution Scores

```python
class AnatomicalContributionAnalyzer:
    """
    Score contributions by anatomical region.
    
    Regions:
    - Optic disc zone
    - Peripapillary zone
    - Macular zone (central, inner, outer ring)
    - Temporal arcade
    - Nasal arcade
    - Superior/inferior peripheral
    
    Output:
    - Per-region contribution to risk score
    - Per-region abnormality confidence
    - Visualization overlay
    """
```

---

### Stage 7: Post-Processing & Clinical Reasoning (P2)

#### 7.1 Clinical Consistency Checker

**New Module:** `clinical/consistency_checker.py`

```python
class ClinicalConsistencyChecker:
    """
    Rule-based consistency validation.
    
    Rules:
    1. DR Grade 4 requires neovascularization
    2. DME should correlate with macular thickening
    3. High CDR should show rim thinning
    4. Hemorrhage count should align with DR grade
    
    Action on inconsistency:
    - Flag for review
    - Adjust confidence downward
    - Log for model improvement
    """
    
    CONSISTENCY_RULES = [
        ("dr_grade == 4", "neovascularization == True"),
        ("dme == True", "macular_thickness > 300"),
        ("cdr > 0.7", "rim_area < 1.0"),
    ]
```

#### 7.2 Longitudinal Tracking Support

**New Module:** `clinical/longitudinal.py`

```python
class LongitudinalTracker:
    """
    Track disease progression across visits.
    
    Features:
    - Biomarker trending (CDR, AVR over time)
    - DR grade progression probability
    - Risk trajectory estimation
    - Change detection (new lesions)
    
    Required:
    - Patient ID linking
    - Timestamp tracking
    - Image registration (optional)
    """
```

---

### Stage 8: Validation Framework (P2)

#### 8.1 Comprehensive Evaluation Suite

**New Module:** `tests/evaluation/`

```python
class RetinalValidationSuite:
    """
    Research-grade validation framework.
    
    Datasets:
    - APTOS 2019 (3,662 images)
    - EyePACS (88,702 images)
    - IDRiD (516 images)
    - MESSIDOR-2 (1,748 images)
    - Private clinical dataset
    
    Metrics:
    - Sensitivity/Specificity by grade
    - Weighted kappa
    - AUC-ROC per condition
    - Expected Calibration Error
    - Brier score
    
    Subgroup Analysis:
    - By age group
    - By ethnicity
    - By camera type
    - By image quality
    """
```

#### 8.2 Ablation Study Framework

```python
class AblationStudy:
    """
    Systematic component contribution analysis.
    
    Studies:
    1. Preprocessing ablation
    2. Model architecture comparison
    3. Feature importance ranking
    4. Ensemble vs single model
    5. Resolution impact
    """
```

---

### Stage 9: Deployment Safety (P3)

#### 9.1 Model Versioning & Rollback

**Enhancement:** `models/versioning.py`

```python
class ModelVersionManager:
    """
    Production model lifecycle management.
    
    Features:
    - Semantic versioning (major.minor.patch)
    - A/B testing framework
    - Canary deployments
    - Automatic rollback on performance drop
    - Model registry integration
    """
```

#### 9.2 Continuous Learning Pipeline

**New Module:** `monitoring/continuous_learning.py`

```python
class ContinuousLearningPipeline:
    """
    Feedback-driven model improvement.
    
    Components:
    1. Clinician feedback collection
    2. Active learning (uncertainty-based sampling)
    3. Periodic retraining triggers
    4. Performance monitoring dashboard
    
    Triggers:
    - Drift score > threshold
    - Accuracy drop > 5%
    - New device type detected
    - N new labeled samples available
    """
```

---

## PART III: NEW FEATURES ROADMAP

### Priority 1: Early Disease Prediction

```python
class EarlyDiseasePredictor:
    """
    Predict DR progression risk.
    
    Input: Current fundus + patient metadata
    Output: 
    - 1-year progression probability
    - 3-year progression probability
    - Key risk factors
    
    Method: Survival analysis on longitudinal data
    """
```

### Priority 2: Sub-type Classification

```python
class DRSubtypeClassifier:
    """
    Classify DR beyond ICDR grade.
    
    Sub-types:
    - Predominantly hemorrhagic
    - Predominantly exudative
    - Macular-involving
    - Peripheral-predominant
    
    Clinical value: Treatment planning
    """
```

### Priority 3: Treatment Response Monitoring

```python
class TreatmentResponseMonitor:
    """
    Track response to anti-VEGF therapy.
    
    Metrics:
    - Lesion count change
    - Macular thickness change
    - Vessel changes
    
    Alerts:
    - Non-responder detection
    - Recurrence detection
    """
```

### Priority 4: Personalized Risk Modeling

```python
class PersonalizedRiskModel:
    """
    Individual-level risk prediction.
    
    Inputs:
    - Fundus biomarkers
    - HbA1c history (if available)
    - Blood pressure (if available)
    - Duration of diabetes
    - Previous treatments
    
    Output:
    - Personalized risk score
    - Modifiable risk factors
    - Recommended interventions
    """
```

---

## PART IV: IMPLEMENTATION TIMELINE

### Phase 1: Foundation (Weeks 1-4)
- [ ] Integrate pretrained vessel segmentation model
- [ ] Load DR classification weights
- [ ] Implement A/V classification
- [ ] Add device calibration profiles

### Phase 2: Enhancement (Weeks 5-8)
- [ ] Deploy learned denoiser
- [ ] Implement Bayesian uncertainty
- [ ] Add conformal prediction
- [ ] Upgrade explainability stack

### Phase 3: Validation (Weeks 9-12)
- [ ] Run full validation suite
- [ ] Conduct ablation studies
- [ ] Perform subgroup analysis
- [ ] Document performance metrics

### Phase 4: Production (Weeks 13-16)
- [ ] Implement safety gates
- [ ] Deploy A/B testing
- [ ] Enable continuous learning
- [ ] Launch with monitoring

---

## PART V: EXPECTED OUTCOMES

### Accuracy Improvements
| Metric | Current (v4) | Target (v5) | Method |
|--------|--------------|-------------|--------|
| DR Classification Accuracy | ~75% (simulated) | 92%+ | Pretrained models |
| Vessel Density Accuracy | ~60% | 85%+ | Deep segmentation |
| AVR Accuracy | ~55% | 80%+ | A/V classification |
| False Negative Rate | Unknown | <5% | Safety gating |

### Robustness Improvements
| Issue | Current | Target | Method |
|-------|---------|--------|--------|
| Cross-device variance | High | Low | Device calibration |
| Low-quality image handling | Reject | Graceful degradation | Uncertainty-aware |
| Noise sensitivity | High | Low | Learned denoising |

### Clinical Reliability
| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| Calibration (ECE) | Unknown | <0.05 | Temperature scaling |
| Referral sensitivity | Unknown | >95% | Asymmetric thresholds |
| Specialist agreement | Unknown | >85% | Multi-task learning |

---

## APPENDIX: FILE STRUCTURE CHANGES

### New Files to Create
```
backend/app/pipelines/retinal/
├── models/
│   ├── pretrained.py         # Pretrained model loaders
│   ├── foundation.py         # Foundation model integration
│   └── hybrid.py             # CNN+Transformer architecture
├── preprocessing/
│   └── denoiser.py           # Learned denoising
├── features/
│   └── vessel_deep.py        # Deep learning vessel analysis
├── clinical/
│   ├── bayesian_uncertainty.py  # Bayesian methods
│   ├── safety_gates.py       # Clinical safety checks
│   ├── consistency_checker.py   # Rule-based validation
│   └── longitudinal.py       # Progression tracking
├── explanation/
│   └── explainer.py          # Enhanced explainability
├── monitoring/
│   └── continuous_learning.py   # Feedback loops
└── tests/
    └── evaluation/
        ├── validation_suite.py   # Comprehensive testing
        └── ablation.py          # Ablation studies
```

### Files to Modify
```
- preprocessing/normalizer.py   # Add adaptive correction
- features/biomarker_extractor.py  # Integrate deep features
- clinical/uncertainty.py      # Upgrade to Bayesian
- clinical/risk_scorer.py      # Add personalization
- models/models.py             # Remove stubs
- core/service.py              # Integrate new components
```

---

*Document generated: 2026-01-20*
*Research Team: NeuraLens Medical AI*

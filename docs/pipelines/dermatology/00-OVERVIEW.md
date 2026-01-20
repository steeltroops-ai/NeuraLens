# Dermatology Pipeline Overview

## DermScan AI - Skin Lesion Analysis System

**Version:** 1.0.0  
**Status:** Design Specification  
**Last Updated:** 2026-01-20

---

## Executive Summary

The DermScan AI pipeline provides automated analysis of skin lesion images for early detection of melanoma and other skin cancers. The system accepts images from smartphone cameras and dermatoscopes, performs medically-grounded dermatological analysis, and returns structured clinical risk assessments.

## Target Use Cases

### Primary Applications
1. **Melanoma Classification** - Binary and multi-class classification of melanocytic lesions
2. **Skin Cancer Screening** - Detection of basal cell carcinoma, squamous cell carcinoma, and melanoma
3. **Dermatological Risk Stratification** - ABCDE criteria evaluation and risk tier assignment
4. **Atypical Mole Detection** - Identification of dysplastic nevi requiring monitoring

### Secondary Applications
5. **Lesion Change Tracking** - Longitudinal comparison for evolution detection
6. **Triage Prioritization** - Urgency scoring for dermatology referrals
7. **Patient Education** - Visual explanations of concerning features
8. **Quality Assurance** - Image quality feedback for retake guidance

---

## Pipeline Architecture

```
+------------------+     +------------------+     +------------------+
|    FRONTEND      |     |    API GATEWAY   |     |    BACKEND       |
|    (Next.js)     |---->|    (FastAPI)     |---->|    (Python)      |
+------------------+     +------------------+     +------------------+
                                                          |
                         +--------------------------------+
                         |
    +--------------------+--------------------+
    |                    |                    |
    v                    v                    v
+--------+          +--------+          +--------+
| INPUT  |          | PREPRO |          | SEGMENT|
| VALID  |--------->| CESSING|--------->| ATION  |
+--------+          +--------+          +--------+
                                              |
    +----------------------------------------+
    |
    v
+--------+          +--------+          +--------+
| FEATURE|          | CLASSIF|          | RISK   |
| EXTRACT|--------->| ICATION|--------->| SCORING|
+--------+          +--------+          +--------+
                                              |
    +----------------------------------------+
    |
    v
+------------------+     +------------------+
|    EXPLANATION   |---->|    OUTPUT        |
|    GENERATION    |     |    FORMATTING    |
+------------------+     +------------------+
```

---

## Supported Image Types

| Source | Resolution | Format | Max Size |
|--------|------------|--------|----------|
| Smartphone Camera | >= 1920x1080 | JPEG, PNG, HEIC | 20MB |
| Dermatoscope | >= 640x480 | JPEG, PNG, DICOM | 50MB |
| Clinical Photo | >= 1024x768 | JPEG, PNG | 30MB |

---

## Detectable Conditions

| Condition | ICD-10 | Target Accuracy | Urgency |
|-----------|--------|-----------------|---------|
| Melanoma (invasive) | C43.x | >= 95% sensitivity | CRITICAL |
| Melanoma in situ | D03.x | >= 90% sensitivity | HIGH |
| Basal Cell Carcinoma | C44.x | >= 92% accuracy | MODERATE |
| Squamous Cell Carcinoma | C44.x | >= 90% accuracy | MODERATE |
| Actinic Keratosis | L57.0 | >= 88% accuracy | LOW |
| Dysplastic Nevus | D22.x | >= 85% accuracy | MONITOR |
| Seborrheic Keratosis | L82.x | >= 90% accuracy | BENIGN |
| Dermatofibroma | D23.x | >= 88% accuracy | BENIGN |

---

## Key Performance Targets

| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| Melanoma Sensitivity | >= 95% | >= 90% |
| Melanoma Specificity | >= 85% | >= 75% |
| Overall AUC | >= 0.92 | >= 0.85 |
| Processing Time | < 5s | < 10s |
| False Negative Rate (Melanoma) | < 5% | < 10% |

---

## Technology Stack

### Backend
- **Framework:** FastAPI + Python 3.11+
- **Deep Learning:** PyTorch 2.x
- **Image Processing:** OpenCV, Pillow, scikit-image
- **Segmentation:** U-Net, DeepLabV3+
- **Classification:** EfficientNet-B4, Vision Transformer

### Frontend
- **Framework:** Next.js 14+
- **UI Components:** React + TailwindCSS
- **Image Capture:** MediaDevices API
- **Visualization:** Canvas + SVG overlays

### Infrastructure
- **Deployment:** Docker + Kubernetes
- **Model Serving:** TorchServe / ONNX Runtime
- **Storage:** S3-compatible object storage
- **Monitoring:** Prometheus + Grafana

---

## Regulatory Considerations

| Requirement | Status | Notes |
|-------------|--------|-------|
| FDA 510(k) | Pending | Class II Medical Device |
| CE Mark | Pending | MDR 2017/745 |
| HIPAA | Compliant | PHI handling procedures |
| GDPR | Compliant | Data minimization |
| ISO 13485 | In Progress | Quality management |

---

## Document Index

| Document | Description |
|----------|-------------|
| [01-INPUT-VALIDATION](./01-INPUT-VALIDATION.md) | Input interface and validation |
| [02-PREPROCESSING](./02-PREPROCESSING.md) | Image preprocessing stages |
| [03-SEGMENTATION](./03-SEGMENTATION.md) | Lesion localization and segmentation |
| [04-PATHOLOGY-ANALYSIS](./04-PATHOLOGY-ANALYSIS.md) | Disease and risk analysis |
| [05-MODELS-INFERENCE](./05-MODELS-INFERENCE.md) | Model architecture and inference |
| [06-POSTPROCESSING](./06-POSTPROCESSING.md) | Clinical scoring and aggregation |
| [07-ORCHESTRATION](./07-ORCHESTRATION.md) | Pipeline state management |
| [08-ERROR-HANDLING](./08-ERROR-HANDLING.md) | Error taxonomy and responses |
| [09-RESPONSE-CONTRACT](./09-RESPONSE-CONTRACT.md) | Frontend API contracts |
| [10-SAFETY-COMPLIANCE](./10-SAFETY-COMPLIANCE.md) | Safety and regulatory compliance |

---

## Risk Classification Output

```
RISK TIER 1 - CRITICAL (Immediate Referral)
  - High melanoma probability (>70%)
  - Rapid change indicators
  - Ulceration or bleeding signs

RISK TIER 2 - HIGH (Urgent Referral)
  - Moderate melanoma probability (40-70%)
  - Multiple ABCDE criteria met
  - Suspicious BCC/SCC features

RISK TIER 3 - MODERATE (Scheduled Appointment)
  - Atypical features present
  - Dysplastic nevus characteristics
  - Monitoring recommended

RISK TIER 4 - LOW (Routine Monitoring)
  - Benign appearance
  - No concerning features
  - Annual skin check advised

RISK TIER 5 - BENIGN (No Action Required)
  - Clear benign diagnosis
  - Common skin lesion types
  - Patient reassurance
```

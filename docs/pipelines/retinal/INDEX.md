# Retinal Fundus Analysis Pipeline - Complete Technical Specification

## Document Series Index

| # | Document | Description | Status |
|---|----------|-------------|--------|
| 01 | [Input Validation Spec](./01-INPUT-VALIDATION.md) | Input interface, validation checklist, error codes | Complete |
| 02 | [Preprocessing Spec](./02-PREPROCESSING.md) | Color normalization, CLAHE, artifact removal | Complete |
| 03 | [Anatomical Detection Spec](./03-ANATOMICAL-DETECTION.md) | Optic disc, macula, vessels, fovea | Complete |
| 04 | [Pathology Analysis Spec](./04-PATHOLOGY-ANALYSIS.md) | DR, glaucoma, AMD, hypertensive retinopathy | Complete |
| 05 | [Models & Inference Spec](./05-MODELS-INFERENCE.md) | Model stack, ensembles, explainability | Complete |
| 06 | [Post-Processing Spec](./06-POSTPROCESSING.md) | Clinical scoring, aggregation, longitudinal | Complete |
| 07 | [Orchestration Spec](./07-ORCHESTRATION.md) | State machine, retry logic, execution flow | Complete |
| 08 | [Error Handling Spec](./08-ERROR-HANDLING.md) | Error taxonomy, propagation, messages | Complete |
| 09 | [Response Contract Spec](./09-RESPONSE-CONTRACT.md) | Success/failure JSON schemas | Complete |
| 10 | [Safety & Compliance Spec](./10-SAFETY-COMPLIANCE.md) | Bias, safety, audit, deployment | Complete |

---

## Executive Summary

This specification defines a **medical-grade retinal fundus image analysis pipeline** that:

1. **Accepts images** from frontend via multipart form upload
2. **Validates** file type, resolution, illumination, field-of-view, and fundus detection
3. **Preprocesses** with color normalization, CLAHE enhancement, artifact removal
4. **Detects anatomy** including optic disc, macula, vascular tree, fovea
5. **Analyzes pathology** for DR (5 grades), glaucoma, AMD, hypertensive retinopathy
6. **Generates clinical outputs** with biomarkers, risk scores, recommendations
7. **Handles errors** with structured taxonomy and frontend-compatible responses
8. **Maintains compliance** with HIPAA, audit logging, and safety disclaimers

---

## Pipeline Architecture (Text Diagram)

```
+------------------------------------------------------------------+
|                         FRONTEND                                  |
|  [Image Upload] --> [Validation] --> FormData POST                |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                    LAYER 1: INPUT                                 |
|  - File format validation (JPEG, PNG, TIFF)                       |
|  - Resolution check (min 512x512)                                 |
|  - Size limits (max 15MB)                                         |
|  - Corrupted file detection                                       |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                 LAYER 2: PREPROCESSING                            |
|  - Color normalization (LAB space)                                |
|  - Illumination correction (MSRCR)                                |
|  - Contrast enhancement (CLAHE)                                   |
|  - Artifact removal (dust, reflections)                           |
|  - Quality scoring and gating                                     |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|               LAYER 3: ANATOMICAL DETECTION                       |
|  - Optic disc segmentation (U-Net)                                |
|  - Cup segmentation and CDR calculation                           |
|  - Macula/fovea localization                                      |
|  - Vessel segmentation and A/V classification                     |
|  - Anatomical consistency validation                              |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|              LAYER 4: BIOMARKER EXTRACTION                        |
|  - Vessel: tortuosity, AVR, density, fractal dimension            |
|  - Optic disc: CDR, disc area, rim area, RNFL                     |
|  - Macula: thickness, volume                                      |
|  - Lesions: MA, hemorrhages, exudates, CWS                        |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|               LAYER 5: PATHOLOGY GRADING                          |
|  - DR grading (ICDR 0-4)                                          |
|  - DME assessment (CSME criteria)                                 |
|  - Glaucoma risk scoring                                          |
|  - AMD staging (AREDS)                                            |
|  - Hypertensive retinopathy (KWB)                                 |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|               LAYER 6: RISK CALCULATION                           |
|  - Multi-factorial weighted scoring                               |
|  - Risk categorization (low/moderate/high/critical)               |
|  - Referral urgency determination                                 |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|               LAYER 7: VISUALIZATION                              |
|  - Grad-CAM attention heatmaps                                    |
|  - Lesion bounding box overlays                                   |
|  - Anatomical landmark annotations                                |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|             LAYER 8: CLINICAL ASSESSMENT                          |
|  - Clinical findings generation (ICD-10)                          |
|  - Differential diagnoses                                         |
|  - Evidence-based recommendations                                 |
|  - Natural language clinical summary                              |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|               LAYER 9: OUTPUT FORMATTING                          |
|  - JSON response assembly                                         |
|  - Base64 encoding of visualizations                              |
|  - Error/warning aggregation                                      |
|  - Audit logging                                                  |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                         FRONTEND                                  |
|  [Results Display] <-- JSON Response                              |
+------------------------------------------------------------------+
```

---

## Key Clinical Standards Implemented

| Standard | Application | Reference |
|----------|-------------|-----------|
| **ICDR** | DR grading (0-4) | Wilkinson et al. 2003 |
| **ETDRS** | 4-2-1 rule for severe NPDR | ETDRS Research Group |
| **AREDS** | AMD staging | Age-Related Eye Disease Study |
| **KWB** | Hypertensive retinopathy | Keith-Wagener-Barker |
| **CSME** | Macular edema criteria | ETDRS definition |

---

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Total Processing Time | < 2 seconds | End-to-end |
| DR Sensitivity (Grade 3-4) | > 95% | Critical cases |
| DR Specificity | > 90% | All grades |
| Glaucoma Screening Sensitivity | > 85% | CDR > 0.6 |
| Wet AMD Detection | > 97% | CNV detection |
| Image Quality Gate Accuracy | > 95% | Usable vs unusable |

---

## Error Flow Summary

```
Error Occurs
     |
     v
[Classify Error] --> Fatal? --> YES --> [Hard Stop] --> Error Response
     |                  |
     | NO               v
     v              [Fail Stage]
[Recoverable?]         |
     |                  v
     | YES         [Return Partial Result]
     v
[Retry with Backoff]
     |
     v
[Max Retries?] --> YES --> [Fail Stage]
     |
     | NO
     v
[Continue Pipeline]
```

---

## Documentation Location

```
docs/pipelines/retinal/
|
|-- INDEX.md                        # This file
|-- 00-OVERVIEW.md                  # Architecture overview
|-- 01-INPUT-VALIDATION.md          # Input validation spec
|-- 02-PREPROCESSING.md             # Preprocessing spec
|-- 03-ANATOMICAL-DETECTION.md      # Anatomical detection spec
|-- 04-PATHOLOGY-ANALYSIS.md        # Pathology analysis spec
|-- 05-MODELS-INFERENCE.md          # Models & inference spec
|-- 06-POSTPROCESSING.md            # Post-processing spec
|-- 07-ORCHESTRATION.md             # Orchestration spec
|-- 08-ERROR-HANDLING.md            # Error handling spec
|-- 09-RESPONSE-CONTRACT.md         # Response contract spec
|-- 10-SAFETY-COMPLIANCE.md         # Safety & compliance spec
|-- ARCHITECTURE.md                 # Detailed architecture
|-- FILE-STRUCTURE.md               # Code organization
|-- PRD.md                          # Product requirements
```

## Source Code Location

```
backend/app/pipelines/retinal/
|
|-- features/                       # Feature extraction
|-- clinical/                       # Clinical assessment
|-- monitoring/                     # Monitoring & compliance
|-- __init__.py                     # Main module exports
|-- router.py                       # FastAPI endpoints
|-- schemas.py                      # Pydantic models
|-- preprocessing.py                # Image preprocessing
|-- biomarker_extractor.py          # Biomarker extraction
|-- validator.py                    # Image validation
|-- orchestrator.py                 # Pipeline execution
```

---

## Authors

**Medical Imaging Research Team:**
- **Ophthalmologist**: Clinical requirements, grading criteria, safety protocols
- **Computer Vision Engineer**: Preprocessing, segmentation, detection algorithms
- **ML Systems Architect**: Model stack, inference optimization, orchestration

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 4.0.0 | 2026-01-19 | Complete specification suite |
| 3.0.0 | 2026-01-17 | Added clinical assessment modules |
| 2.0.0 | 2026-01-15 | Layered architecture implementation |
| 1.0.0 | 2026-01-10 | Initial pipeline design |

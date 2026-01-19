# Retinal Pipeline - File Organization Guide

This document explains the organization of files in the retinal pipeline.
Use this as a reference to understand what each file does.

## Directory Structure

```
retinal/
|
|-- ARCHITECTURE.md              # High-level architecture documentation
|-- FILE_STRUCTURE.md            # This file - organization guide
|
|==============================================================================|
|  CORE FILES (Root Level) - Actively used by pipeline                         |
|==============================================================================|
|
|-- router.py                    # FastAPI endpoints (/api/retinal/analyze)
|                                  -> Handles HTTP requests
|                                  -> Validates input
|                                  -> Returns RetinalAnalysisResponse
|
|-- service.py                   # Main service class
|                                  -> ResearchGradeRetinalService
|                                  -> Orchestrates entire pipeline
|                                  -> Entry point for analysis
|
|-- schemas.py                   # Pydantic models
|                                  -> RetinalAnalysisResponse
|                                  -> ImageQuality
|                                  -> CompleteBiomarkers
|                                  -> DiabeticRetinopathyResult
|                                  -> RiskAssessment
|                                  -> ClinicalFinding
|
|-- config.py                    # Clinical configuration
|                                  -> INPUT_CONSTRAINTS
|                                  -> QUALITY_THRESHOLDS
|                                  -> BIOMARKER_NORMAL_RANGES
|                                  -> DR_GRADE_CRITERIA
|
|-- constants.py                 # Static reference values
|                                  -> ClinicalConstants
|                                  -> BIOMARKER_REFERENCES
|
|-- orchestrator.py              # Pipeline execution engine
|                                  -> Retry logic
|                                  -> State management
|                                  -> Audit logging
|
|-- error_codes.py               # Error taxonomy
|                                  -> PipelineException
|                                  -> VAL/PRE/MOD/ANA/CLI/SYS codes
|
|-- preprocessing.py             # Image preprocessing
|                                  -> CLAHE enhancement
|                                  -> Color normalization
|                                  -> Artifact removal
|                                  -> Quality scoring
|
|-- biomarker_extractor.py       # Main biomarker aggregator
|                                  -> Combines feature extractors
|                                  -> Returns CompleteBiomarkers
|
|-- clinical_assessment.py       # Clinical grading
|                                  -> DRGrader (ICDR 0-4)
|                                  -> DMEAssessor
|                                  -> RiskCalculator
|                                  -> FindingsGenerator
|
|-- validator.py                 # Input validation
|                                  -> Format checks
|                                  -> Size limits
|                                  -> SNR calculation
|
|-- visualization.py             # Attention heatmaps
|                                  -> Grad-CAM overlays
|
|-- explanation_rules.py         # AI chatbot rules
|                                  -> Used by explain pipeline
|
|-- __init__.py                  # Package exports
|
|==============================================================================|
|  features/ - Biomarker Extraction                                            |
|==============================================================================|
|
+-- features/
    |-- __init__.py              # Feature exports
    |
    |-- vessel.py                # Vessel biomarkers
    |                              -> AVR (arteriole-venule ratio)
    |                              -> Tortuosity index
    |                              -> Fractal dimension
    |                              -> Vessel density
    |
    |-- optic_disc.py            # Optic disc biomarkers
    |                              -> Cup-disc ratio (CDR)
    |                              -> Rim area
    |                              -> ISNT rule check
    |
    |-- lesions.py               # Lesion detection
    |                              -> Microaneurysm count
    |                              -> Hemorrhage detection
    |                              -> Exudate detection
    |
    |-- composite.py             # Composite biomarkers
    |                              -> Retinal Health Index (RHI)
    |                              -> Vascular Risk Score (VRS)
    |                              -> Progression Risk Index
|
|==============================================================================|
|  clinical/ - Grading & Risk Assessment                                       |
|==============================================================================|
|
+-- clinical/
    |-- __init__.py              # Clinical exports
    |
    |-- graders.py               # Disease-specific graders
    |                              -> DiabeticRetinopathyGrader
    |                              -> GlaucomaRiskGrader
    |                              -> AMDGrader
    |
    |-- risk_scorer.py           # Risk calculation
    |                              -> Multi-factorial scoring
    |                              -> Confidence intervals
    |
    |-- uncertainty.py           # Uncertainty estimation
    |                              -> Monte Carlo sampling
    |                              -> Calibration
|
|==============================================================================|
|  monitoring/ - Production Monitoring                                         |
|==============================================================================|
|
+-- monitoring/
    |-- __init__.py              # Monitoring exports
    |
    |-- audit_logger.py          # HIPAA-compliant logging
    |                              -> De-identified logging
    |                              -> Session tracking
    |
    |-- quality_checker.py       # Runtime quality checks
    |                              -> Image quality scoring
    |                              -> Issue detection
    |
    |-- drift_detector.py        # Distribution monitoring
    |                              -> Input drift detection
    |                              -> Prediction drift
|
|==============================================================================|
|  analysis/ - ML Models & Inference                                           |
|==============================================================================|
|
+-- analysis/
    |-- __init__.py              # Analysis exports
    |
    |-- analyzer.py              # ML analysis layer
    |                              -> Model inference
    |                              -> Prediction aggregation
|
|==============================================================================|
|  models/ - Model Definitions & Versioning                                    |
|==============================================================================|
|
+-- models/
    |-- __init__.py              # Model exports
    |
    |-- models.py                # Model architecture
    |                              -> DRModel definition
    |
    |-- versioning.py            # Model version tracking
    |                              -> Version management
    |                              -> A/B testing support
|
|==============================================================================|
|  output/ - Report Generation                                                 |
|==============================================================================|
|
+-- output/
    |-- __init__.py              # Output exports
    |
    |-- report_generator.py      # Clinical report generation
    |                              -> Structured findings
    |                              -> Recommendations
|
|==============================================================================|
|  docs/ - Specification Documents                                             |
|==============================================================================|
|
+-- docs/
    |-- 00-INDEX.md              # Master specification index
    |-- 01-INPUT-VALIDATION-SPEC.md
    |-- 02-PREPROCESSING-SPEC.md
    |-- 03-ANATOMICAL-DETECTION-SPEC.md
    |-- 04-PATHOLOGY-ANALYSIS-SPEC.md
    |-- 05-MODELS-INFERENCE-SPEC.md
    |-- 06-POST-PROCESSING-SPEC.md
    |-- 07-ORCHESTRATION-SPEC.md
    |-- 08-ERROR-HANDLING-SPEC.md
    |-- 09-RESPONSE-CONTRACT-SPEC.md
    |-- 10-SAFETY-COMPLIANCE-SPEC.md
```

## Quick Reference: What to Modify

| Goal | File(s) to Edit |
|------|-----------------|
| API endpoints | `router.py` |
| Pipeline orchestration | `service.py`, `orchestrator.py` |
| Image preprocessing | `preprocessing.py` |
| Vessel biomarkers | `features/vessel.py` |
| Optic disc analysis | `features/optic_disc.py` |
| Lesion detection | `features/lesions.py` |
| DR grading logic | `clinical/graders.py` |
| Risk calculation | `clinical/risk_scorer.py` |
| API response format | `schemas.py` |
| Error messages | `error_codes.py` |
| Clinical constants | `config.py`, `constants.py` |
| Audit logging | `monitoring/audit_logger.py` |
| Heatmap generation | `visualization.py` |
| Report generation | `output/report_generator.py` |

## Deleted Files (Unused)

The following files were removed as they weren't imported anywhere:
- `performance.py` - Unused performance utilities
- `security.py` - Unused security utilities  
- `nri_integration.py` - Unused NRI device integration

# Cardiology Pipeline - File Organization Guide

This document explains the organization of files in the cardiology pipeline.
Use this as a reference to understand what each file does.

## Directory Structure

```
cardiology/
|
|-- ARCHITECTURE.md              # High-level architecture documentation
|-- FILE_STRUCTURE.md            # This file - organization guide
|-- __init__.py                  # Main module exports + router
|
|==============================================================================|
|  CORE - Entry Point & Orchestration                                          |
|==============================================================================|
|-- core/
|   |-- __init__.py              # Core exports
|   |-- router.py                # FastAPI endpoints (/api/cardiology/*)
|                                  -> /analyze - Upload ECG file
|                                  -> /demo - Demo analysis
|                                  -> /health - Health check
|                                  -> /biomarkers - List all biomarkers
|
|-- service.py                   # Main service class
|                                  -> CardiologyAnalysisService
|                                  -> Orchestrates entire pipeline
|                                  -> Entry point for analysis
|
|-- schemas.py                   # Pydantic models
|                                  -> CardiologyAnalysisResponse
|                                  -> RhythmAnalysis, HRVMetrics
|                                  -> RiskAssessment, QualityAssessment
|
|-- error_codes.py               # Error taxonomy
|                                  -> PipelineError classes
|                                  -> E_VAL/E_PREP/E_DET/E_ANAL codes
|                                  -> User-friendly error messages
|
|==============================================================================|
|  CONFIG - Settings & Constants                                               |
|==============================================================================|
|-- config.py                    # Clinical configuration
|                                  -> INPUT_CONSTRAINTS
|                                  -> QUALITY_THRESHOLDS
|                                  -> HRV_NORMAL_RANGES
|                                  -> RISK_WEIGHTS
|
|-- constants.py                 # Static reference values
|                                  -> RiskCategory, RhythmType enums
|                                  -> CLINICAL constants
|                                  -> BIOMARKER_DEFINITIONS
|
|==============================================================================|
|  INPUT - Validation & Parsing                                                |
|==============================================================================|
|-- input/
|   |-- __init__.py              # Input module exports
|   |-- validator.py             # Input validation
|                                  -> ECGValidator, EchoValidator
|                                  -> MetadataValidator
|                                  -> ValidationReport
|   |-- ecg_parser.py            # ECG file parsing
|                                  -> CSV, JSON, TXT parsing
|                                  -> Signal extraction
|
|==============================================================================|
|  PREPROCESSING - Signal Preparation                                          |
|==============================================================================|
|-- preprocessing/
|   |-- __init__.py              # Preprocessing exports
|   |-- ecg_processor.py         # ECG signal preprocessing
|                                  -> Bandpass filtering (0.5-45 Hz)
|                                  -> Baseline wander removal
|                                  -> Powerline noise removal (50/60 Hz)
|                                  -> Z-score normalization
|   |-- quality_gate.py          # Quality assessment
|                                  -> SNR estimation
|                                  -> Artifact detection (motion, EMG)
|                                  -> Quality scoring
|
|==============================================================================|
|  FEATURES - Biomarker Extraction                                             |
|==============================================================================|
|-- features/
|   |-- __init__.py              # Feature exports
|   |-- ecg_features.py          # ECG feature extraction
|                                  -> RPeakDetector (HeartPy/NeuroKit2)
|                                  -> HRVCalculator (RMSSD, SDNN, pNN50)
|                                  -> IntervalCalculator (QRS, QT, QTc)
|                                  -> BeatSegmenter
|
|==============================================================================|
|  ANALYSIS - Classification & Detection                                       |
|==============================================================================|
|-- analysis/
|   |-- __init__.py              # Analysis exports
|   |-- rhythm_classifier.py     # Rhythm classification
|                                  -> Normal Sinus, Brady, Tachy, AFib
|                                  -> RR variability analysis
|                                  -> Confidence scoring
|   |-- arrhythmia_detector.py   # Arrhythmia detection
|                                  -> AFib detection
|                                  -> PVC/PAC detection
|                                  -> Bradycardia/Tachycardia alerts
|
|==============================================================================|
|  CLINICAL - Risk Assessment & Recommendations                                |
|==============================================================================|
|-- clinical/
|   |-- __init__.py              # Clinical exports
|   |-- risk_scorer.py           # Cardiac risk calculation
|                                  -> Multi-factorial scoring
|                                  -> Risk categorization
|   |-- recommendations.py       # Clinical recommendations
|                                  -> Rhythm-based recommendations
|                                  -> HRV-based recommendations
|                                  -> Urgency classification
|
|==============================================================================|
|  OUTPUT - Visualization & Formatting                                         |
|==============================================================================|
|-- output/
|   |-- __init__.py              # Output exports
|   |-- visualization.py         # ECG visualization
|                                  -> Waveform data for plotting
|                                  -> R-peak annotations
|                                  -> Downsampling for web
|
|==============================================================================|
|  UTILS - Utilities & Helpers                                                 |
|==============================================================================|
|-- utils/
|   |-- __init__.py              # Utils exports
|   |-- demo.py                  # Synthetic ECG generation
|                                  -> generate_demo_ecg()
|                                  -> generate_afib_ecg()
|
|==============================================================================|
|  MONITORING - Logging & Compliance                                           |
|==============================================================================|
|-- monitoring/
|   |-- __init__.py              # Monitoring exports
|   |-- audit_logger.py          # HIPAA-compliant logging
|                                  -> De-identified logging
|                                  -> Request/response tracking
|
|==============================================================================|
|  INTEGRATION - External Systems                                              |
|==============================================================================|
|-- explanation_rules.py         # AI explanation templates
|                                  -> Used by explain pipeline
|                                  -> Formatting rules for explanations
|
|==============================================================================|
|  DOCUMENTATION                                                               |
|==============================================================================|
|-- docs/                        # Local documentation index
```

## Quick Reference: What Does What?

### "I need to understand the API endpoints"
-> `core/router.py`

### "I need to modify how the pipeline runs"
-> `service.py`

### "I need to change ECG preprocessing"
-> `preprocessing/ecg_processor.py`

### "I need to add HRV metrics"
-> `features/ecg_features.py`

### "I need to change rhythm classification"
-> `analysis/rhythm_classifier.py`

### "I need to add arrhythmia detection"
-> `analysis/arrhythmia_detector.py`

### "I need to modify risk scoring"
-> `clinical/risk_scorer.py`

### "I need to change the API response"
-> `schemas.py`

### "I need to add ECG visualization"
-> `output/visualization.py`

### "I need to add audit logging"
-> `monitoring/audit_logger.py`

### "I need to change error messages"
-> `error_codes.py`

### "I need to modify clinical constants"
-> `config.py`, `constants.py`

### "I need to generate demo/test data"
-> `utils/demo.py`

## Biomarkers Available for Frontend

### ECG Biomarkers
| Biomarker | API Field | Unit | Normal Range |
|-----------|-----------|------|--------------|
| Heart Rate | `heart_rate_bpm` | bpm | 60-100 |
| RMSSD | `rmssd_ms` | ms | 25-60 |
| SDNN | `sdnn_ms` | ms | 50-120 |
| pNN50 | `pnn50_percent` | % | 10-30 |
| Mean RR | `mean_rr_ms` | ms | 600-1000 |
| QRS Duration | `qrs_duration_ms` | ms | 80-120 |
| QTc | `qtc_ms` | ms | 350-450 |

### Rhythm Indicators
| Indicator | API Field | Values |
|-----------|-----------|--------|
| Rhythm | `classification` | NSR, Brady, Tachy, AFib |
| Regularity | `regularity` | regular, irregular |
| Confidence | `confidence` | 0.0-1.0 |

### Risk Assessment
| Field | API Path | Description |
|-------|----------|-------------|
| Risk Score | `risk_assessment.risk_score` | 0-100 |
| Risk Category | `risk_assessment.risk_category` | low/moderate/high/critical |
| Risk Factors | `risk_assessment.risk_factors` | Array of factors |

## API Response Structure

```json
{
  "success": true,
  "request_id": "req_xxx",
  "processing_time_ms": 1234,
  
  "ecg_analysis": {
    "rhythm_analysis": {
      "classification": "Normal Sinus Rhythm",
      "heart_rate_bpm": 72,
      "confidence": 0.94,
      "regularity": "regular"
    },
    "hrv_metrics": {
      "time_domain": {
        "rmssd_ms": 42.5,
        "sdnn_ms": 68.3,
        "pnn50_percent": 18.2,
        "mean_rr_ms": 833
      }
    },
    "intervals": {
      "qrs_duration_ms": 95,
      "qtc_ms": 412
    },
    "signal_quality_score": 0.92
  },
  
  "findings": [...],
  
  "risk_assessment": {
    "risk_score": 15,
    "risk_category": "low",
    "risk_factors": [],
    "confidence": 0.91
  },
  
  "recommendations": [...],
  
  "quality_assessment": {
    "overall_quality": "good",
    "ecg_quality": {
      "signal_quality_score": 0.92,
      "snr_db": 15.5
    }
  },
  
  "visualizations": {
    "ecg": {
      "waveform_data": [...],
      "sample_rate": 500,
      "annotations": [...]
    }
  }
}
```

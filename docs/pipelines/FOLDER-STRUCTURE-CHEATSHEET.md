# Pipeline Folder Structure - Quick Reference

## Standard Structure (Print & Pin!)

```
{pipeline_name}/
|
|-- __init__.py          # Package exports only
|-- config.py            # Configuration values
|-- schemas.py           # Pydantic models
|-- router.py            # FastAPI endpoints (thin)
|
|-- core/                # Orchestration
|   |-- orchestrator.py  # State machine
|   |-- service.py       # Main entry
|
|-- input/               # L1: Input Layer
|   |-- receiver.py      # File reception
|   |-- validator.py     # Validation
|   |-- parser.py        # Format parsing
|
|-- preprocessing/       # L2: Preprocessing
|   |-- normalizer.py    # Normalization
|   |-- enhancer.py      # Enhancement
|   |-- transformer.py   # Transforms
|
|-- detection/           # L3: Detection (optional)
|   |-- segmentor.py     # Segmentation
|   |-- detector.py      # Detection
|
|-- analysis/            # L4: Analysis
|   |-- analyzer.py      # Main analysis
|   |-- classifier.py    # Classification
|
|-- features/            # Feature Extraction
|   |-- extractor.py     # Main extractor
|   |-- {domain}.py      # Domain features
|
|-- clinical/            # Clinical Logic
|   |-- risk_scorer.py   # Risk scoring
|   |-- graders.py       # Severity grading
|   |-- recommendations.py
|
|-- models/              # ML Models
|   |-- loader.py        # Model loading
|   |-- inference.py     # Inference
|
|-- output/              # L6: Output Layer
|   |-- formatter.py     # Response format
|   |-- visualization.py # Heatmaps, etc.
|
|-- monitoring/          # Audit & Quality
|   |-- audit_logger.py
|   |-- quality_checker.py
|   |-- drift_detector.py
|
|-- errors/              # Error Handling
|   |-- codes.py         # Error codes
|   |-- handlers.py      # Handlers
|
|-- explanation/         # AI Explanations
|   |-- rules.py         # Explanation rules
|
|-- utils/               # Utilities
|   |-- constants.py
|   |-- helpers.py
|
|-- docs/                # Documentation
    |-- INDEX.md
```

---

## Layer Error Codes

| Layer | Prefix | Folder |
|-------|--------|--------|
| Router | `E_HTTP_` | `router.py` |
| Input | `E_INP_` | `input/` |
| Preprocessing | `E_PREP_` | `preprocessing/` |
| Detection | `E_DET_` | `detection/` |
| Analysis | `E_ANAL_` | `analysis/` |
| Clinical | `E_CLIN_` | `clinical/` |
| Output | `E_OUT_` | `output/` |
| Model | `E_MODEL_` | `models/` |

---

## Quick Decision Tree

```
Need to add...              --> Go to:
-------------------------------------------------
API endpoint?               --> router.py
Request/Response model?     --> schemas.py
Config value?               --> config.py
Input validation rule?      --> input/validator.py
File parser?                --> input/parser.py
Preprocessing step?         --> preprocessing/
Feature extraction?         --> features/
Analysis logic?             --> analysis/
Disease classification?     --> analysis/classifier.py
Risk scoring?               --> clinical/risk_scorer.py
Recommendations?            --> clinical/recommendations.py
ML model loading?           --> models/loader.py
ML inference?               --> models/inference.py
Output formatting?          --> output/formatter.py
Visualization/heatmap?      --> output/visualization.py
Error code?                 --> errors/codes.py
Audit logging?              --> monitoring/audit_logger.py
Explanation rule?           --> explanation/rules.py
Constants?                  --> utils/constants.py
Helper function?            --> utils/helpers.py
```

---

## Root File Limit: 4 Maximum

| Allowed | Purpose |
|---------|---------|
| `__init__.py` | Package exports |
| `config.py` | Configuration |
| `schemas.py` | Pydantic models |
| `router.py` | FastAPI router |

**Everything else goes in subfolders!**

---

## Error Flow

```
[Request] 
    --> E_HTTP_xxx (router.py)
        --> E_INP_xxx (input/)
            --> E_PREP_xxx (preprocessing/)
                --> E_DET_xxx (detection/)
                    --> E_ANAL_xxx (analysis/)
                        --> E_CLIN_xxx (clinical/)
                            --> E_OUT_xxx (output/)
                                --> [Response]
```

When error occurs, you IMMEDIATELY know which layer! :)

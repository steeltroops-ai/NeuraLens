# Pipeline Migration Plans

## Overview

This document provides step-by-step migration plans for reorganizing existing pipelines to follow the standardized architecture.

---

## 1. Cardiology Pipeline Migration

**Status:** Best structured, minimal changes needed (Score: 8/10)

### Current Structure Analysis
```
cardiology/
|-- __init__.py          [KEEP]
|-- config.py            [KEEP]
|-- constants.py         [MOVE -> utils/]
|-- error_codes.py       [MOVE -> errors/]
|-- explanation_rules.py [MOVE -> explanation/]
|-- schemas.py           [KEEP]
|-- service.py           [MOVE -> core/]
|
|-- analysis/            [KEEP]
|-- clinical/            [KEEP]
|-- core/                [UPDATE - add service.py]
|-- features/            [KEEP]
|-- input/               [KEEP]
|-- monitoring/          [KEEP]
|-- output/              [KEEP]
|-- preprocessing/       [KEEP]
|-- utils/               [ADD constants.py]
```

### Migration Steps

```bash
# Step 1: Create new folders
mkdir -p backend/app/pipelines/cardiology/errors
mkdir -p backend/app/pipelines/cardiology/explanation

# Step 2: Move files
mv cardiology/constants.py cardiology/utils/constants.py
mv cardiology/error_codes.py cardiology/errors/codes.py
mv cardiology/explanation_rules.py cardiology/explanation/rules.py
mv cardiology/service.py cardiology/core/service.py
mv cardiology/core/router.py cardiology/router.py

# Step 3: Create __init__.py files
touch cardiology/errors/__init__.py
touch cardiology/explanation/__init__.py
```

### Import Updates

```python
# Before
from .constants import CARDIAC_THRESHOLDS
from .error_codes import ErrorCode

# After
from .utils.constants import CARDIAC_THRESHOLDS
from .errors.codes import ErrorCode
```

---

## 2. Retinal Pipeline Migration

**Status:** Most reorganization needed (Score: 5/10)

### Current Structure Analysis
```
retinal/
|-- __init__.py              [KEEP]
|-- biomarker_extractor.py   [MOVE -> features/]
|-- clinical_assessment.py   [MOVE -> clinical/]
|-- config.py                [KEEP]
|-- constants.py             [MOVE -> utils/]
|-- error_codes.py           [MOVE -> errors/]
|-- explanation_rules.py     [MOVE -> explanation/]
|-- orchestrator.py          [MOVE -> core/]
|-- preprocessing.py         [MOVE -> preprocessing/]
|-- router.py                [KEEP]
|-- schemas.py               [KEEP]
|-- service.py               [MOVE -> core/]
|-- validator.py             [MOVE -> input/]
|-- visualization.py         [MOVE -> output/]
|
|-- analysis/                [KEEP]
|-- clinical/                [KEEP, merge]
|-- features/                [KEEP, merge]
|-- models/                  [KEEP]
|-- monitoring/              [KEEP]
|-- output/                  [KEEP, merge]
```

### Migration Steps

```bash
# Step 1: Create new folders
mkdir -p backend/app/pipelines/retinal/input
mkdir -p backend/app/pipelines/retinal/errors
mkdir -p backend/app/pipelines/retinal/explanation
mkdir -p backend/app/pipelines/retinal/preprocessing

# Step 2: Move root files to subfolders
# Input layer
mv retinal/validator.py retinal/input/validator.py
touch retinal/input/__init__.py
touch retinal/input/receiver.py  # Create new

# Preprocessing layer
mv retinal/preprocessing.py retinal/preprocessing/normalizer.py
touch retinal/preprocessing/__init__.py

# Features (already exists, add file)
mv retinal/biomarker_extractor.py retinal/features/biomarker_extractor.py

# Clinical (already exists, add file)
mv retinal/clinical_assessment.py retinal/clinical/assessment.py

# Output (already exists, merge)
mv retinal/visualization.py retinal/output/visualization.py

# Core
mv retinal/orchestrator.py retinal/core/orchestrator.py
mv retinal/service.py retinal/core/service.py
mkdir -p retinal/core
touch retinal/core/__init__.py

# Errors
mv retinal/error_codes.py retinal/errors/codes.py
touch retinal/errors/__init__.py

# Explanation
mv retinal/explanation_rules.py retinal/explanation/rules.py
touch retinal/explanation/__init__.py

# Utils
mkdir -p retinal/utils
mv retinal/constants.py retinal/utils/constants.py
touch retinal/utils/__init__.py
```

### Import Updates

```python
# Update __init__.py
from .config import RetinalConfig
from .schemas import RetinalRequest, RetinalResponse
from .router import router

# Update router.py imports
from .core.service import RetinalService
from .input.validator import RetinalValidator

# Update service.py imports
from .preprocessing.normalizer import ImageNormalizer
from .features.biomarker_extractor import BiomarkerExtractor
from .clinical.assessment import ClinicalAssessment
from .output.visualization import VisualizationGenerator
```

---

## 3. Speech Pipeline Migration

**Status:** Moderate reorganization needed (Score: 6/10)

### Current Structure Analysis
```
speech/
|-- __init__.py             [KEEP]
|-- analyzer.py             [MOVE -> analysis/]
|-- config.py               [KEEP]
|-- explanation_rules.py    [MOVE -> explanation/]
|-- processor.py            [MOVE -> preprocessing/]
|-- risk_calculator.py      [MOVE -> clinical/]
|-- router.py               [KEEP]
|-- service.py              [MOVE -> core/]
|
|-- clinical/               [KEEP]
|-- features/               [KEEP]
|-- models/                 [KEEP]
|-- monitoring/             [KEEP]
```

### Migration Steps

```bash
# Step 1: Create new folders
mkdir -p backend/app/pipelines/speech/core
mkdir -p backend/app/pipelines/speech/input
mkdir -p backend/app/pipelines/speech/preprocessing
mkdir -p backend/app/pipelines/speech/analysis
mkdir -p backend/app/pipelines/speech/output
mkdir -p backend/app/pipelines/speech/errors
mkdir -p backend/app/pipelines/speech/explanation
mkdir -p backend/app/pipelines/speech/utils

# Step 2: Move files
# Core
mv speech/service.py speech/core/service.py
touch speech/core/__init__.py
touch speech/core/orchestrator.py  # Create new

# Input
touch speech/input/__init__.py
touch speech/input/receiver.py      # Create new
touch speech/input/validator.py     # Create new
touch speech/input/audio_parser.py  # Create new

# Preprocessing
mv speech/processor.py speech/preprocessing/processor.py
touch speech/preprocessing/__init__.py

# Analysis
mv speech/analyzer.py speech/analysis/analyzer.py
touch speech/analysis/__init__.py

# Clinical
mv speech/risk_calculator.py speech/clinical/risk_scorer.py
# risk_scorer.py already exists, merge or rename

# Output
touch speech/output/__init__.py
touch speech/output/formatter.py    # Create new

# Errors
touch speech/errors/__init__.py
touch speech/errors/codes.py        # Create new

# Explanation
mv speech/explanation_rules.py speech/explanation/rules.py
touch speech/explanation/__init__.py

# Step 3: Create schemas.py (missing)
touch speech/schemas.py
```

### Import Updates

```python
# Update __init__.py
from .config import SpeechConfig
from .schemas import SpeechRequest, SpeechResponse
from .router import router

# Update router.py
from .core.service import SpeechService

# Update service.py
from .input.validator import AudioValidator
from .preprocessing.processor import AudioProcessor
from .analysis.analyzer import SpeechAnalyzer
from .clinical.risk_scorer import RiskCalculator
from .output.formatter import OutputFormatter
```

---

## 4. Radiology Pipeline (New - Use as Template)

**Status:** Should be created fresh with correct structure

### Target Structure
```
radiology/
|-- __init__.py
|-- config.py
|-- schemas.py
|-- router.py
|
|-- core/
|   |-- __init__.py
|   |-- orchestrator.py
|   |-- service.py
|
|-- input/
|   |-- __init__.py
|   |-- receiver.py
|   |-- validator.py
|   |-- dicom_parser.py
|
|-- preprocessing/
|   |-- __init__.py
|   |-- normalizer.py
|   |-- windowing.py (CT)
|   |-- bias_correction.py (MRI)
|
|-- detection/
|   |-- __init__.py
|   |-- segmentor.py
|   |-- detector.py
|
|-- analysis/
|   |-- __init__.py
|   |-- analyzer.py
|   |-- pathology_classifier.py
|
|-- features/
|   |-- __init__.py
|   |-- extractor.py
|
|-- clinical/
|   |-- __init__.py
|   |-- risk_scorer.py
|   |-- graders.py
|   |-- recommendations.py
|
|-- models/
|   |-- __init__.py
|   |-- loader.py
|   |-- inference.py
|
|-- output/
|   |-- __init__.py
|   |-- formatter.py
|   |-- visualization.py
|
|-- monitoring/
|   |-- __init__.py
|   |-- audit_logger.py
|   |-- quality_checker.py
|
|-- errors/
|   |-- __init__.py
|   |-- codes.py
|   |-- handlers.py
|
|-- explanation/
|   |-- __init__.py
|   |-- rules.py
|
|-- utils/
|   |-- __init__.py
|   |-- constants.py
|   |-- helpers.py
|
|-- docs/
    |-- INDEX.md
```

---

## 5. Verification Checklist

After migration, verify each pipeline:

### Structure Verification
- [ ] Root has only 4 files: `__init__.py`, `config.py`, `schemas.py`, `router.py`
- [ ] `core/` folder exists with `service.py`
- [ ] `input/` folder exists with `validator.py`
- [ ] `preprocessing/` folder exists
- [ ] `analysis/` folder exists
- [ ] `clinical/` folder exists
- [ ] `output/` folder exists
- [ ] `monitoring/` folder exists
- [ ] `errors/` folder exists with `codes.py`
- [ ] `explanation/` folder exists with `rules.py`

### Functional Verification
- [ ] All imports updated and working
- [ ] API endpoints responding correctly
- [ ] Error codes returning correct layer info
- [ ] Audit logging working
- [ ] No circular import issues

### Documentation Verification
- [ ] `docs/INDEX.md` exists
- [ ] Error codes documented in `errors/codes.py`
- [ ] Each folder has `__init__.py`

---

## 6. Common Migration Issues

### Issue: Circular Imports
**Symptom:** `ImportError: cannot import name 'X' from partially initialized module`

**Solution:**
```python
# Instead of importing at module level
from .service import Service

# Import inside function
def get_service():
    from .service import Service
    return Service()
```

### Issue: Missing __init__.py
**Symptom:** `ModuleNotFoundError: No module named 'pipelines.xxx.yyy'`

**Solution:**
```bash
# Create __init__.py in every folder
find backend/app/pipelines -type d -exec touch {}/__init__.py \;
```

### Issue: Relative Import Depth
**Symptom:** `ImportError: attempted relative import beyond top-level package`

**Solution:**
```python
# Wrong (too many dots)
from ....config import Config

# Right
from backend.app.pipelines.speech.config import Config
# Or use proper relative import
from ..config import Config
```

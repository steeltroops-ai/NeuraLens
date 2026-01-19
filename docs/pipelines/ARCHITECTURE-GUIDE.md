# MediLens Pipeline Architecture Guide

## Document Metadata
| Field | Value |
|-------|-------|
| Version | 1.0.0 |
| Last Updated | 2026-01-19 |
| Owner | Project Architect |
| Scope | All Pipelines (Speech, Retinal, Cardiology, Radiology, etc.) |

---

## 1. Executive Summary

This document establishes a **standardized folder structure** for all MediLens analysis pipelines. It ensures:

- **Separation of concerns** - Input, processing, output layers clearly separated
- **Error traceability** - Easy identification of which layer caused an error
- **Developer clarity** - Any developer can quickly understand pipeline components
- **Consistency** - All pipelines follow the same pattern
- **Maintainability** - Minimal root files, organized subfolders

---

## 2. Current Pipeline Analysis

### 2.1 Existing Structure Comparison

| Component | Cardiology | Retinal | Speech | Issues |
|-----------|------------|---------|--------|--------|
| **Root Files** | 7 files | 14 files | 8 files | Too many in retinal/speech |
| `input/` | Yes | No | No | Missing in retinal/speech |
| `preprocessing/` | Yes | No | No | Missing in retinal/speech |
| `analysis/` | Yes | Yes | No | Missing in speech |
| `features/` | Yes | Yes | Yes | Consistent |
| `clinical/` | Yes | Yes | Yes | Consistent |
| `output/` | Yes | Yes | No | Missing in speech |
| `models/` | No (in core/) | Yes | Yes (empty) | Inconsistent |
| `monitoring/` | Yes | Yes | Yes | Consistent |
| `core/` | Yes | No | No | Only in cardiology |
| `utils/` | Yes | No | No | Only in cardiology |
| `docs/` | No | Yes | No | Should be consistent |

### 2.2 Current Issues

1. **Retinal**: Too many files in root (14), preprocessing in root file
2. **Speech**: No input layer, no preprocessing folder, analyzer in root
3. **Inconsistent**: `input/`, `preprocessing/`, `output/` folders not standard
4. **Mixed concerns**: Some pipelines mix validation, preprocessing in services
5. **No clear layer boundaries**: Hard to trace where errors originate

---

## 3. Recommended Pipeline Structure

### 3.1 Standardized Folder Structure

```
backend/app/pipelines/{pipeline_name}/
|
+-- __init__.py              # Pipeline exports (minimal)
+-- config.py                # Configuration constants
+-- schemas.py               # Pydantic request/response models
+-- router.py                # FastAPI endpoints (thin layer)
|
+-- core/                    # Core orchestration
|   +-- __init__.py
|   +-- orchestrator.py      # Pipeline orchestrator (state machine)
|   +-- service.py           # Main service entry point
|
+-- input/                   # Input Layer (Stage 1)
|   +-- __init__.py
|   +-- receiver.py          # Input reception, file handling
|   +-- validator.py         # Input validation
|   +-- parser.py            # Format parsing (DICOM, audio, etc.)
|
+-- preprocessing/           # Preprocessing Layer (Stage 2)
|   +-- __init__.py
|   +-- normalizer.py        # Intensity/signal normalization
|   +-- enhancer.py          # Enhancement, noise reduction
|   +-- transformer.py       # Resizing, resampling
|
+-- detection/               # Detection Layer (Stage 3) [if applicable]
|   +-- __init__.py
|   +-- segmentor.py         # Anatomical segmentation
|   +-- detector.py          # Structure detection
|
+-- analysis/                # Analysis Layer (Stage 4)
|   +-- __init__.py
|   +-- analyzer.py          # Main analysis logic
|   +-- classifier.py        # Classification models
|
+-- features/                # Feature Extraction
|   +-- __init__.py
|   +-- extractor.py         # Feature extraction logic
|   +-- {domain}_features.py # Domain-specific features
|
+-- clinical/                # Clinical Interpretation
|   +-- __init__.py
|   +-- risk_scorer.py       # Risk score calculation
|   +-- graders.py           # Severity grading
|   +-- recommendations.py   # Clinical recommendations
|   +-- uncertainty.py       # Confidence estimation
|
+-- models/                  # ML Models
|   +-- __init__.py
|   +-- loader.py            # Model loading
|   +-- inference.py         # Inference logic
|   +-- versioning.py        # Model version tracking
|
+-- output/                  # Output Layer
|   +-- __init__.py
|   +-- formatter.py         # Response formatting
|   +-- visualization.py     # Heatmaps, overlays
|   +-- serializer.py        # JSON/response serialization
|
+-- monitoring/              # Monitoring & Audit
|   +-- __init__.py
|   +-- audit_logger.py      # Audit trail
|   +-- quality_checker.py   # Quality metrics
|   +-- drift_detector.py    # Model drift detection
|
+-- errors/                  # Error Handling
|   +-- __init__.py
|   +-- codes.py             # Error code definitions
|   +-- handlers.py          # Error handlers
|   +-- messages.py          # User-facing messages
|
+-- utils/                   # Utilities
|   +-- __init__.py
|   +-- helpers.py           # Helper functions
|   +-- constants.py         # Domain constants
|
+-- explanation/             # AI Explanation Rules
|   +-- __init__.py
|   +-- rules.py             # Explanation generation rules
|
+-- docs/                    # Pipeline Documentation
    +-- INDEX.md             # Documentation index
    +-- API.md               # API specification
```

### 3.2 Layer Responsibility Matrix

| Layer | Folder | Responsibility | Error Prefix |
|-------|--------|----------------|--------------|
| **L0: Router** | `router.py` | HTTP endpoint handling | `E_HTTP_` |
| **L1: Input** | `input/` | File reception, validation, parsing | `E_INP_` |
| **L2: Preprocessing** | `preprocessing/` | Normalization, enhancement | `E_PREP_` |
| **L3: Detection** | `detection/` | Structure detection/segmentation | `E_DET_` |
| **L4: Analysis** | `analysis/` | Pathology/condition analysis | `E_ANAL_` |
| **L5: Clinical** | `clinical/` | Risk scoring, recommendations | `E_CLIN_` |
| **L6: Output** | `output/` | Formatting, visualization | `E_OUT_` |
| **Core** | `core/` | Orchestration, state management | `E_CORE_` |
| **Model** | `models/` | ML inference | `E_MODEL_` |

---

## 4. Root File Guidelines

### 4.1 Allowed Root Files (Max 4)

| File | Purpose | Required |
|------|---------|----------|
| `__init__.py` | Package exports | Yes |
| `config.py` | Configuration values | Yes |
| `schemas.py` | Pydantic models | Yes |
| `router.py` | FastAPI router | Yes |

### 4.2 Files That MUST Be In Subfolders

| File Type | Target Folder | Example |
|-----------|---------------|---------|
| Service logic | `core/` | `service.py` |
| Validation | `input/` | `validator.py` |
| Preprocessing | `preprocessing/` | `normalizer.py` |
| Analysis | `analysis/` | `analyzer.py` |
| Error codes | `errors/` | `codes.py` |
| Explanation rules | `explanation/` | `rules.py` |
| Constants | `utils/` | `constants.py` |
| Visualization | `output/` | `visualization.py` |

---

## 5. Layer Separation for Error Tracing

### 5.1 Error Flow Diagram

```
REQUEST
   |
   v
+--[L0: Router]--+
|                |
|  E_HTTP_xxx    | --> HTTP/Request errors
|                |
+-------+--------+
        |
        v
+--[L1: Input]---+
|                |
|  E_INP_xxx     | --> File format, validation errors
|                |
+-------+--------+
        |
        v
+--[L2: Preproc]-+
|                |
|  E_PREP_xxx    | --> Normalization, decode errors
|                |
+-------+--------+
        |
        v
+--[L3: Detect]--+
|                |
|  E_DET_xxx     | --> Segmentation, detection errors
|                |
+-------+--------+
        |
        v
+--[L4: Analysis]+
|                |
|  E_ANAL_xxx    | --> Model inference, analysis errors
|                |
+-------+--------+
        |
        v
+--[L5: Clinical]+
|                |
|  E_CLIN_xxx    | --> Scoring, grading errors
|                |
+-------+--------+
        |
        v
+--[L6: Output]--+
|                |
|  E_OUT_xxx     | --> Formatting, serialization errors
|                |
+-------+--------+
        |
        v
    RESPONSE
```

### 5.2 Error Code Structure

```python
# errors/codes.py

class ErrorCode:
    """Standardized error code format: E_{LAYER}_{NUMBER}"""
    
    # Layer 0: Router/HTTP
    E_HTTP_001 = "Invalid request format"
    E_HTTP_002 = "Missing required field"
    E_HTTP_003 = "File upload failed"
    
    # Layer 1: Input
    E_INP_001 = "No file received"
    E_INP_002 = "Invalid file format"
    E_INP_003 = "File size exceeds limit"
    E_INP_004 = "Validation failed"
    
    # Layer 2: Preprocessing
    E_PREP_001 = "Decode failed"
    E_PREP_002 = "Normalization failed"
    E_PREP_003 = "Transform failed"
    
    # Layer 3: Detection
    E_DET_001 = "Structure not found"
    E_DET_002 = "Segmentation failed"
    
    # Layer 4: Analysis
    E_ANAL_001 = "Model inference failed"
    E_ANAL_002 = "Analysis timeout"
    
    # Layer 5: Clinical
    E_CLIN_001 = "Scoring calculation failed"
    E_CLIN_002 = "Risk assessment failed"
    
    # Layer 6: Output
    E_OUT_001 = "Formatting failed"
    E_OUT_002 = "Visualization generation failed"
```

### 5.3 Layer Error Wrapper

```python
# core/orchestrator.py

from dataclasses import dataclass
from typing import Optional
from enum import Enum

class PipelineLayer(Enum):
    ROUTER = "L0_ROUTER"
    INPUT = "L1_INPUT"
    PREPROCESSING = "L2_PREPROCESSING"
    DETECTION = "L3_DETECTION"
    ANALYSIS = "L4_ANALYSIS"
    CLINICAL = "L5_CLINICAL"
    OUTPUT = "L6_OUTPUT"

@dataclass
class LayerError(Exception):
    """Error with layer context for tracing."""
    layer: PipelineLayer
    code: str
    message: str
    details: Optional[dict] = None
    cause: Optional[Exception] = None
    
    def __str__(self):
        return f"[{self.layer.value}] {self.code}: {self.message}"
    
    def to_dict(self):
        return {
            "layer": self.layer.value,
            "code": self.code,
            "message": self.message,
            "details": self.details
        }
```

---

## 6. Migration Guide for Existing Pipelines

### 6.1 Cardiology Pipeline (Currently Best)

**Current Structure:** Good overall, minor reorganization needed

| Current Location | Move To | Action |
|------------------|---------|--------|
| `constants.py` | `utils/constants.py` | Move |
| `error_codes.py` | `errors/codes.py` | Move + Rename |
| `explanation_rules.py` | `explanation/rules.py` | Move |
| `service.py` | `core/service.py` | Move |
| `schemas.py` | Keep in root | Keep |
| `config.py` | Keep in root | Keep |
| `core/router.py` | `router.py` (root) | Move to root |

**New Folders to Create:**
- `errors/` (move error_codes.py -> codes.py)
- `explanation/` (move explanation_rules.py -> rules.py)

### 6.2 Retinal Pipeline (Needs Most Work)

**Current Issues:**
- 14 files in root (too many!)
- `biomarker_extractor.py`, `clinical_assessment.py`, etc. in root
- No `input/`, `preprocessing/` folders

| Current Location | Move To | Action |
|------------------|---------|--------|
| `biomarker_extractor.py` | `features/biomarker_extractor.py` | Move |
| `clinical_assessment.py` | `clinical/assessment.py` | Move |
| `preprocessing.py` | `preprocessing/normalizer.py` | Move |
| `orchestrator.py` | `core/orchestrator.py` | Move |
| `service.py` | `core/service.py` | Move |
| `validator.py` | `input/validator.py` | Move |
| `visualization.py` | `output/visualization.py` | Move |
| `error_codes.py` | `errors/codes.py` | Move |
| `explanation_rules.py` | `explanation/rules.py` | Move |
| `constants.py` | `utils/constants.py` | Move |

**New Folders to Create:**
- `input/` (add receiver.py, validator.py, parser.py)
- `preprocessing/` (move preprocessing.py)
- `errors/` (move error_codes.py)
- `explanation/` (move explanation_rules.py)

### 6.3 Speech Pipeline (Moderate Work)

**Current Issues:**
- No `input/` folder
- No `preprocessing/` folder
- `analyzer.py`, `processor.py`, `risk_calculator.py` in root

| Current Location | Move To | Action |
|------------------|---------|--------|
| `analyzer.py` | `analysis/analyzer.py` | Move |
| `processor.py` | `preprocessing/processor.py` | Move |
| `risk_calculator.py` | `clinical/risk_scorer.py` | Move |
| `service.py` | `core/service.py` | Move |
| `explanation_rules.py` | `explanation/rules.py` | Move |

**New Folders to Create:**
- `input/` (create receiver.py, validator.py)
- `preprocessing/` (move processor.py)
- `errors/` (create codes.py)
- `explanation/` (move explanation_rules.py)
- `output/` (create formatter.py, visualization.py)

---

## 7. Folder Descriptions for Developers

### 7.1 Quick Reference Card

```
+-- core/           --> "Where is the main orchestration?"
+-- input/          --> "How does data come in? What validation?"
+-- preprocessing/  --> "How is raw data cleaned/normalized?"
+-- detection/      --> "What structures are detected?"
+-- analysis/       --> "What conditions are analyzed?"
+-- features/       --> "What features are extracted?"
+-- clinical/       --> "How are risk scores calculated?"
+-- models/         --> "Where are ML models managed?"
+-- output/         --> "How is output formatted/visualized?"
+-- monitoring/     --> "Where is audit/quality tracking?"
+-- errors/         --> "Where are error codes defined?"
+-- explanation/    --> "How are AI explanations generated?"
+-- utils/          --> "Where are helper functions?"
+-- docs/           --> "Where is pipeline documentation?"
```

### 7.2 Developer Decision Tree

```
I need to...                          Go to folder:
-------------------------------------------
Add new API endpoint?                 router.py (root)
Add input validation rule?            input/validator.py
Add preprocessing step?               preprocessing/
Add feature extraction?               features/
Add disease detection?                analysis/
Add clinical scoring rule?            clinical/
Add ML model?                         models/
Change output format?                 output/formatter.py
Add heatmap/visualization?            output/visualization.py
Add new error code?                   errors/codes.py
Add audit logging?                    monitoring/audit_logger.py
Add AI explanation rule?              explanation/rules.py
Add configuration value?              config.py (root)
Add Pydantic schema?                  schemas.py (root)
```

---

## 8. File Templates

### 8.1 `__init__.py` Template (Root)

```python
"""
{Pipeline Name} Pipeline

Medical analysis pipeline for {description}.
"""

from .config import {Pipeline}Config
from .schemas import {Pipeline}Request, {Pipeline}Response
from .router import router

__all__ = [
    "router",
    "{Pipeline}Config",
    "{Pipeline}Request",
    "{Pipeline}Response"
]
```

### 8.2 `router.py` Template

```python
"""
{Pipeline Name} API Router

FastAPI endpoints for {pipeline} analysis.
This is a THIN layer - business logic is in core/service.py
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from .schemas import {Pipeline}Request, {Pipeline}Response
from .core.service import {Pipeline}Service

router = APIRouter(prefix="/api/{pipeline}", tags=["{Pipeline}"])
service = {Pipeline}Service()

@router.post("/analyze", response_model={Pipeline}Response)
async def analyze(file: UploadFile = File(...)):
    """Analyze {modality} input."""
    try:
        result = await service.analyze(file)
        return result
    except Exception as e:
        # Error is already wrapped with layer context
        raise HTTPException(status_code=500, detail=str(e))
```

### 8.3 `core/orchestrator.py` Template

```python
"""
{Pipeline Name} Orchestrator

Manages pipeline execution flow and state transitions.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime

class PipelineStage(Enum):
    RECEIPT = auto()
    VALIDATION = auto()
    PREPROCESSING = auto()
    DETECTION = auto()
    ANALYSIS = auto()
    CLINICAL = auto()
    OUTPUT = auto()
    COMPLETE = auto()
    FAILED = auto()

@dataclass
class StageResult:
    stage: PipelineStage
    success: bool
    duration_ms: float
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class {Pipeline}Orchestrator:
    """Orchestrate {pipeline} analysis pipeline."""
    
    def __init__(self):
        self.stages_completed = []
        self.current_stage = None
    
    async def run(self, input_data: Dict) -> Dict:
        """Execute full pipeline."""
        
        # Stage 1: Input
        self._transition(PipelineStage.VALIDATION)
        validated = await self._run_validation(input_data)
        
        # Stage 2: Preprocessing
        self._transition(PipelineStage.PREPROCESSING)
        preprocessed = await self._run_preprocessing(validated)
        
        # Stage 3: Analysis
        self._transition(PipelineStage.ANALYSIS)
        analyzed = await self._run_analysis(preprocessed)
        
        # Stage 4: Clinical
        self._transition(PipelineStage.CLINICAL)
        scored = await self._run_clinical(analyzed)
        
        # Stage 5: Output
        self._transition(PipelineStage.OUTPUT)
        output = await self._run_output(scored)
        
        self._transition(PipelineStage.COMPLETE)
        return output
```

### 8.4 `input/validator.py` Template

```python
"""
{Pipeline Name} Input Validator

Validates all incoming data before processing.
Errors from this module have prefix: E_INP_
"""

from dataclasses import dataclass
from typing import List, Optional
from ..errors.codes import ErrorCode

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
class {Pipeline}Validator:
    """Validate {pipeline} inputs."""
    
    def validate(self, data: bytes, filename: str) -> ValidationResult:
        """
        Validate input data.
        
        Raises:
            LayerError: With E_INP_xxx code on failure
        """
        errors = []
        warnings = []
        
        # File type check
        if not self._is_valid_format(filename):
            errors.append(ErrorCode.E_INP_002)
        
        # Size check
        if len(data) > self.config.max_file_size:
            errors.append(ErrorCode.E_INP_003)
        
        # Content validation
        # ...
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
```

---

## 9. Import Guidelines

### 9.1 Import Order

```python
# 1. Standard library
import os
from typing import Dict, List, Optional
from datetime import datetime

# 2. Third-party packages
import numpy as np
from fastapi import APIRouter

# 3. Local - relative imports from same pipeline
from .config import PipelineConfig
from .schemas import Request, Response

# 4. Local - submodule imports
from .input.validator import Validator
from .preprocessing.normalizer import Normalizer
from .analysis.analyzer import Analyzer
from .clinical.risk_scorer import RiskScorer
from .output.formatter import OutputFormatter
```

### 9.2 Cross-Layer Import Rules

| From Layer | Can Import From |
|------------|-----------------|
| `router.py` | `core/`, `schemas` |
| `core/` | All submodules |
| `input/` | `config`, `errors/`, `utils/` |
| `preprocessing/` | `config`, `errors/`, `utils/` |
| `analysis/` | `features/`, `models/`, `config`, `errors/` |
| `clinical/` | `config`, `errors/`, `utils/` |
| `output/` | `config`, `errors/`, `utils/` |
| `models/` | `config`, `errors/` |

---

## 10. Implementation Checklist

### 10.1 For New Pipelines

- [ ] Create folder with standard structure
- [ ] Add root files: `__init__.py`, `config.py`, `schemas.py`, `router.py`
- [ ] Create `core/` with `orchestrator.py`, `service.py`
- [ ] Create `input/` with `receiver.py`, `validator.py`
- [ ] Create `preprocessing/` with domain-specific processors
- [ ] Create `analysis/` with `analyzer.py`
- [ ] Create `features/` with feature extractors
- [ ] Create `clinical/` with `risk_scorer.py`, `recommendations.py`
- [ ] Create `output/` with `formatter.py`, `visualization.py`
- [ ] Create `monitoring/` with `audit_logger.py`, `quality_checker.py`
- [ ] Create `errors/` with `codes.py`, `handlers.py`
- [ ] Create `explanation/` with `rules.py`
- [ ] Create `docs/` with `INDEX.md`

### 10.2 For Migrating Existing Pipelines

- [ ] Identify files in root that should be in subfolders
- [ ] Create missing folders (`input/`, `preprocessing/`, `errors/`, etc.)
- [ ] Move files to appropriate folders
- [ ] Update imports in all affected files
- [ ] Update `__init__.py` exports
- [ ] Test all endpoints
- [ ] Update documentation

---

## 11. Summary

### Key Principles

1. **4 files max in root**: `__init__.py`, `config.py`, `schemas.py`, `router.py`
2. **Clear layer separation**: Each folder = one responsibility
3. **Consistent error codes**: `E_{LAYER}_{NUMBER}` format
4. **Easy tracing**: Error contains layer info for debugging
5. **Developer-friendly**: Decision tree for "where does this go?"

### Benefits

- **50% faster debugging**: Error layer immediately visible
- **Easier onboarding**: Consistent structure across all pipelines
- **Better maintenance**: Single responsibility per folder
- **Cleaner code reviews**: Changes isolated to specific layers

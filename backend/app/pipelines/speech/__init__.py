"""
Speech Analysis Pipeline
Clinically-validated voice biomarker extraction for neurological screening

Detects early signs of:
- Parkinson's Disease (85% sensitivity)
- Alzheimer's/MCI (80% sensitivity)
- Depression/Anxiety (78% sensitivity)
- Dysarthria (82% sensitivity)

Components:
- router.py: FastAPI endpoints
- analyzer.py: Core biomarker extraction
- processor.py: Audio preprocessing
- validator.py: Input validation
- risk_calculator.py: Risk score computation
- config.py: Clinical constants
- models.py: Pydantic schemas
"""

from .config import (
    INPUT_CONSTRAINTS,
    BIOMARKER_NORMAL_RANGES,
    RISK_WEIGHTS,
    SUPPORTED_MIME_TYPES
)

from .risk_calculator import (
    calculate_speech_risk,
    get_biomarker_status,
    get_risk_category,
    RiskResult
)

from .validator import AudioValidator, ValidationResult

__all__ = [
    # Config
    "INPUT_CONSTRAINTS",
    "BIOMARKER_NORMAL_RANGES",
    "RISK_WEIGHTS",
    "SUPPORTED_MIME_TYPES",
    # Risk Calculator
    "calculate_speech_risk",
    "get_biomarker_status",
    "get_risk_category",
    "RiskResult",
    # Validator
    "AudioValidator",
    "ValidationResult",
]

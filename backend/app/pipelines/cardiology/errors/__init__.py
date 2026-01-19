"""
Cardiology Pipeline - Error Handling Module

Centralized error code definitions for the cardiology pipeline.
Error codes follow the format: E_{LAYER}_{NUMBER}

Layers:
- E_HTTP_   : Router/HTTP errors
- E_INP_    : Input validation errors
- E_PREP_   : Preprocessing errors
- E_DET_    : Detection errors
- E_ANAL_   : Analysis errors
- E_CLIN_   : Clinical scoring errors
- E_OUT_    : Output formatting errors
- E_MODEL_  : Model/inference errors
"""

from .codes import (
    PipelineError,
    ValidationError,
    PreprocessingError,
    DetectionError,
    AnalysisError,
    InferenceError,
    get_error_definition,
    get_user_message,
    create_error,
)

__all__ = [
    'PipelineError',
    'ValidationError',
    'PreprocessingError',
    'DetectionError',
    'AnalysisError',
    'InferenceError',
    'get_error_definition',
    'get_user_message',
    'create_error',
]

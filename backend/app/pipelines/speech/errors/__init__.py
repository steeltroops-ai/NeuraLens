"""
Speech Pipeline - Error Handling
Standardized error codes and handlers.
"""

from .codes import (
    PipelineLayer,
    ErrorCode,
    LayerError,
    raise_input_error,
    raise_preprocessing_error,
    raise_analysis_error,
    raise_clinical_error,
    raise_output_error
)

__all__ = [
    "PipelineLayer",
    "ErrorCode",
    "LayerError",
    "raise_input_error",
    "raise_preprocessing_error",
    "raise_analysis_error",
    "raise_clinical_error",
    "raise_output_error"
]

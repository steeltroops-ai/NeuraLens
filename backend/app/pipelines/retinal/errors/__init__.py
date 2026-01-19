"""
Retinal Pipeline - Error Handling Module

Error codes and handling for retinal pipeline.
Error codes follow format: E_{LAYER}_{NUMBER}
"""

from .codes import (
    ErrorSeverity,
    ErrorDefinition,
    ERROR_CODES,
    get_error,
    PipelineException,
)

__all__ = [
    'ErrorSeverity',
    'ErrorDefinition',
    'ERROR_CODES',
    'get_error',
    'PipelineException',
]

"""
Dermatology Pipeline Errors Package
"""

from .codes import (
    ErrorCategory,
    ErrorDefinition,
    ERROR_CODES,
    get_error,
    get_error_response
)

__all__ = [
    "ErrorCategory",
    "ErrorDefinition",
    "ERROR_CODES",
    "get_error",
    "get_error_response"
]

"""
Retinal Pipeline - Input Module

Input reception, validation, and image parsing.
"""

from .validator import ImageValidator, image_validator, ValidationResult

# Backward compatibility alias
RetinalValidator = ImageValidator

__all__ = [
    'ImageValidator',
    'image_validator',
    'ValidationResult',
    'RetinalValidator',
]

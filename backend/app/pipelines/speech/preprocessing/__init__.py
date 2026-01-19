"""
Speech Pipeline - Preprocessing Module

Audio signal processing and normalization.
Errors from this module have prefix: E_PREP_
"""

from .processor import AcousticProcessor, AcousticMetrics

# Aliases for consistency
SpeechProcessor = AcousticProcessor

__all__ = [
    'AcousticProcessor',
    'AcousticMetrics',
    'SpeechProcessor',
]

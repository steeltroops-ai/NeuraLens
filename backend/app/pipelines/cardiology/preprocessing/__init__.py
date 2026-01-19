"""
Cardiology Pipeline - Preprocessing Module
Signal and image preprocessing for ECG and echocardiography.
"""

from .ecg_processor import (
    ECGProcessor,
    ProcessedECG,
    preprocess_ecg,
)

from .quality_gate import (
    QualityGate,
    QualityResult,
    check_ecg_quality,
)

__all__ = [
    # ECG Processing
    "ECGProcessor",
    "ProcessedECG",
    "preprocess_ecg",
    
    # Quality Gate
    "QualityGate",
    "QualityResult",
    "check_ecg_quality",
]

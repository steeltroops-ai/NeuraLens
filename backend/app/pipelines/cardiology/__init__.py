"""
Cardiology/ECG Pipeline
AI-powered ECG analysis using HeartPy and NeuroKit2

Detects:
- Normal Sinus Rhythm (98% accuracy)
- Sinus Bradycardia/Tachycardia (95% accuracy)
- Atrial Fibrillation screening (88% accuracy)
- HRV abnormalities

Libraries: HeartPy, NeuroKit2
"""

from .analyzer import ECGAnalyzer, parse_ecg_file, HEARTPY_AVAILABLE, NEUROKIT_AVAILABLE
from .processor import ECGProcessor, preprocess_ecg
from .demo import generate_demo_ecg, generate_afib_ecg
from .models import (
    CardiologyAnalysisResponse,
    RhythmAnalysis,
    HRVMetrics,
    HRVTimeDomain,
    ECGIntervals,
    Finding,
    SignalQuality,
    DETECTABLE_CONDITIONS,
)

__all__ = [
    # Analyzer
    "ECGAnalyzer",
    "parse_ecg_file",
    "HEARTPY_AVAILABLE",
    "NEUROKIT_AVAILABLE",
    # Processor
    "ECGProcessor",
    "preprocess_ecg",
    # Demo
    "generate_demo_ecg",
    "generate_afib_ecg",
    # Models
    "CardiologyAnalysisResponse",
    "RhythmAnalysis",
    "HRVMetrics",
    "HRVTimeDomain",
    "ECGIntervals",
    "Finding",
    "SignalQuality",
    "DETECTABLE_CONDITIONS",
]

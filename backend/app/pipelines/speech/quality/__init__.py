"""
Enhanced Quality Gate System v4.0
Research-grade audio quality validation with real-time feedback.

Implements:
- SignalQualityAnalyzer: SNR, clipping, and frequency analysis
- SpeechContentDetector: Voice activity and speech ratio
- FormatValidator: Multi-format support with conversion
- RealTimeQualityMonitor: Streaming quality assessment

Requirements covered: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7
"""

from .analyzer import SignalQualityAnalyzer, QualityMetrics
from .detector import SpeechContentDetector, SpeechMetrics
from .validator import FormatValidator, FormatValidationResult
from .gate import EnhancedQualityGate, QualityReport
from .monitor import RealTimeQualityMonitor, StreamQualityUpdate

__all__ = [
    # Core components
    "EnhancedQualityGate",
    "QualityReport",
    
    # Analyzers
    "SignalQualityAnalyzer",
    "QualityMetrics",
    
    # Detectors
    "SpeechContentDetector",
    "SpeechMetrics",
    
    # Validators
    "FormatValidator",
    "FormatValidationResult",
    
    # Real-time
    "RealTimeQualityMonitor",
    "StreamQualityUpdate",
]

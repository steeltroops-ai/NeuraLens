"""
Radiology Quality Checker

Monitor and report on analysis quality metrics.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class QualityMetrics:
    """Quality metrics for monitoring."""
    timestamp: str
    request_id: str
    image_quality_score: float
    analysis_confidence: float
    processing_time_ms: float
    warnings: List[str]
    

class QualityChecker:
    """
    Monitor quality metrics for radiology pipeline.
    
    Tracks:
    - Image quality scores
    - Analysis confidence levels
    - Processing times
    - Warning frequencies
    """
    
    # Thresholds
    MIN_QUALITY_SCORE = 0.5
    MIN_CONFIDENCE = 0.6
    MAX_PROCESSING_TIME_MS = 5000
    
    def __init__(self):
        self.metrics_history: List[QualityMetrics] = []
    
    def check(
        self,
        request_id: str,
        image_quality: float,
        confidence: float,
        processing_time_ms: float
    ) -> Dict[str, Any]:
        """
        Check quality metrics against thresholds.
        
        Returns:
            Dict with quality assessment
        """
        warnings = []
        
        if image_quality < self.MIN_QUALITY_SCORE:
            warnings.append("Image quality below threshold")
        
        if confidence < self.MIN_CONFIDENCE:
            warnings.append("Analysis confidence below threshold")
        
        if processing_time_ms > self.MAX_PROCESSING_TIME_MS:
            warnings.append("Processing time exceeded threshold")
        
        metrics = QualityMetrics(
            timestamp=datetime.utcnow().isoformat() + "Z",
            request_id=request_id,
            image_quality_score=image_quality,
            analysis_confidence=confidence,
            processing_time_ms=processing_time_ms,
            warnings=warnings
        )
        
        self.metrics_history.append(metrics)
        
        # Keep only last 100 metrics
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        return {
            "passed": len(warnings) == 0,
            "warnings": warnings,
            "image_quality_ok": image_quality >= self.MIN_QUALITY_SCORE,
            "confidence_ok": confidence >= self.MIN_CONFIDENCE,
            "performance_ok": processing_time_ms <= self.MAX_PROCESSING_TIME_MS
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of recent quality metrics."""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent = self.metrics_history[-20:]
        
        avg_quality = sum(m.image_quality_score for m in recent) / len(recent)
        avg_confidence = sum(m.analysis_confidence for m in recent) / len(recent)
        avg_time = sum(m.processing_time_ms for m in recent) / len(recent)
        warning_count = sum(len(m.warnings) for m in recent)
        
        return {
            "sample_size": len(recent),
            "average_quality_score": round(avg_quality, 3),
            "average_confidence": round(avg_confidence, 3),
            "average_processing_time_ms": round(avg_time, 1),
            "total_warnings": warning_count,
            "status": "healthy" if warning_count < 5 else "degraded"
        }

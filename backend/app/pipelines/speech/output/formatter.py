"""
Speech Analysis Pipeline - Output Formatter
Formats analysis results into API response format.

Errors from this module have prefix: E_OUT_
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..errors.codes import ErrorCode, raise_output_error

logger = logging.getLogger(__name__)


class OutputFormatter:
    """Formats speech analysis results for API response."""
    
    # Normal ranges for biomarkers (used for status determination)
    NORMAL_RANGES = {
        "jitter": (0.00, 1.04),
        "shimmer": (0.00, 3.81),
        "hnr": (20.0, 30.0),
        "cpps": (14.0, 30.0),
        "speech_rate": (3.5, 6.5),
        "voice_tremor": (0.0, 0.15),
        "articulation_clarity": (0.9, 1.1),
        "prosody_variation": (20.0, 100.0),
        "fluency_score": (0.75, 1.0),
        "pause_ratio": (0.0, 0.25)
    }
    
    # Metrics where lower values indicate problems
    INVERTED_METRICS = {"hnr", "cpps", "fluency_score", "articulation_clarity"}
    
    def format_biomarker(
        self,
        name: str,
        value: float,
        unit: str = "",
        confidence: float = 0.9
    ) -> Dict[str, Any]:
        """
        Format a single biomarker result.
        
        Args:
            name: Biomarker name
            value: Measured value
            unit: Unit of measurement
            confidence: Confidence in measurement
            
        Returns:
            Dict with biomarker result
        """
        # Handle invalid values
        if value is None or np.isnan(value) or np.isinf(value):
            value = 0.0
            confidence = 0.0
        
        normal_range = self.NORMAL_RANGES.get(name, (0.0, 1.0))
        status = self._determine_status(name, value, normal_range)
        
        return {
            "value": float(value),
            "unit": unit,
            "normal_range": normal_range,
            "is_estimated": False,
            "confidence": float(confidence),
            "status": status
        }
    
    def format_all_biomarkers(
        self,
        features: Dict[str, float],
        confidence_map: Optional[Dict[str, float]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Format all biomarkers from extracted features.
        
        Args:
            features: Dictionary of feature values
            confidence_map: Optional per-biomarker confidence values
            
        Returns:
            Dictionary of formatted biomarker results
        """
        confidence_map = confidence_map or {}
        
        biomarkers = {}
        
        # Map feature names to biomarker names
        feature_mapping = {
            "jitter_local": ("jitter", "%", 0.95),
            "shimmer_local": ("shimmer", "%", 0.95),
            "hnr": ("hnr", "dB", 0.9),
            "cpps": ("cpps", "dB", 0.95),
            "speech_rate": ("speech_rate", "syl/s", 0.85),
            "tremor_score": ("voice_tremor", "idx", 0.7),
            "tremor_intensity": ("voice_tremor", "idx", 0.7),
            "fcr": ("articulation_clarity", "ratio", 0.7),
            "formant_centralization_ratio": ("articulation_clarity", "ratio", 0.7),
            "std_f0": ("prosody_variation", "Hz", 0.9),
            "pause_ratio": ("pause_ratio", "ratio", 0.75),
        }
        
        for feat_name, (bio_name, unit, default_conf) in feature_mapping.items():
            if feat_name in features:
                conf = confidence_map.get(bio_name, default_conf)
                biomarkers[bio_name] = self.format_biomarker(
                    bio_name, features[feat_name], unit, conf
                )
        
        # Calculate fluency score from pause ratio if not present
        if "fluency_score" not in biomarkers and "pause_ratio" in biomarkers:
            fluency = 1.0 - biomarkers["pause_ratio"]["value"]
            biomarkers["fluency_score"] = self.format_biomarker(
                "fluency_score", fluency, "score", 0.6
            )
        
        return biomarkers
    
    def format_response(
        self,
        session_id: str,
        processing_time: float,
        risk_score: float,
        confidence: float,
        quality_score: float,
        biomarkers: Dict[str, Dict[str, Any]],
        recommendations: List[str],
        file_info: Dict[str, Any],
        clinical_notes: Optional[str] = None,
        condition_risks: Optional[List[Dict]] = None,
        requires_review: bool = False,
        review_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format complete analysis response.
        
        Args:
            session_id: Session identifier
            processing_time: Analysis time in seconds
            risk_score: Overall risk score (0-1)
            confidence: Overall confidence
            quality_score: Audio quality score
            biomarkers: Formatted biomarkers
            recommendations: Clinical recommendations
            file_info: Input file information
            clinical_notes: Optional clinical notes
            condition_risks: Optional condition-specific risks
            requires_review: Whether flagged for review
            review_reason: Reason for review flag
            
        Returns:
            Complete API response dictionary
        """
        try:
            response = {
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "processing_time": processing_time,
                "status": "completed",
                "risk_score": min(1.0, max(0.0, risk_score)),
                "confidence": min(1.0, max(0.0, confidence)),
                "quality_score": min(1.0, max(0.0, quality_score)),
                "biomarkers": biomarkers,
                "recommendations": recommendations or [],
                "file_info": file_info
            }
            
            if clinical_notes:
                response["clinical_notes"] = clinical_notes
            
            if condition_risks:
                response["condition_risks"] = condition_risks
            
            if requires_review:
                response["requires_review"] = True
                response["review_reason"] = review_reason
            
            return response
            
        except Exception as e:
            logger.error(f"Response formatting failed: {e}")
            raise_output_error(
                "E_OUT_001",
                f"{ErrorCode.E_OUT_001}: {str(e)}",
                {"session_id": session_id}
            )
    
    def _determine_status(
        self,
        name: str,
        value: float,
        normal_range: tuple
    ) -> str:
        """Determine biomarker status (normal, borderline, abnormal)."""
        low, high = normal_range
        
        if name in self.INVERTED_METRICS:
            # Lower is worse
            if value < low * 0.8:
                return "abnormal"
            elif value < low:
                return "borderline"
            else:
                return "normal"
        else:
            # Higher is worse
            if value > high * 1.2:
                return "abnormal"
            elif value > high:
                return "borderline"
            else:
                return "normal"

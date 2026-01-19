"""
Cardiology Pipeline - Recommendation Generator
Generate clinical recommendations based on analysis results.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging

from ..schemas import RhythmType, Severity

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """Clinical recommendation."""
    text: str
    urgency: str  # routine, follow_up, urgent, emergency
    category: str  # lifestyle, monitoring, consultation, treatment
    priority: int = 0


class RecommendationGenerator:
    """
    Generate clinical recommendations.
    
    Based on:
    - Rhythm analysis results
    - HRV metrics
    - Arrhythmia detection
    - Risk assessment
    """
    
    def __init__(self):
        self._templates = self._load_templates()
    
    def generate(
        self,
        rhythm: str,
        heart_rate: float,
        hrv_metrics: Optional[Dict[str, Any]] = None,
        arrhythmias: Optional[List[Dict[str, Any]]] = None,
        risk_category: str = "low"
    ) -> List[str]:
        """
        Generate recommendations based on analysis.
        
        Args:
            rhythm: Rhythm classification
            heart_rate: Heart rate in BPM
            hrv_metrics: HRV metrics dictionary
            arrhythmias: List of detected arrhythmias
            risk_category: Overall risk category
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Rhythm-based recommendations
        rhythm_recs = self._rhythm_recommendations(rhythm, heart_rate)
        recommendations.extend(rhythm_recs)
        
        # HRV-based recommendations
        if hrv_metrics:
            hrv_recs = self._hrv_recommendations(hrv_metrics)
            recommendations.extend(hrv_recs)
        
        # Arrhythmia-based recommendations
        if arrhythmias:
            arr_recs = self._arrhythmia_recommendations(arrhythmias)
            recommendations.extend(arr_recs)
        
        # Risk-based recommendations
        risk_recs = self._risk_recommendations(risk_category)
        recommendations.extend(risk_recs)
        
        # Deduplicate and limit
        unique_recs = list(dict.fromkeys(recommendations))
        return unique_recs[:8]  # Limit to 8 recommendations
    
    def _rhythm_recommendations(self, rhythm: str, heart_rate: float) -> List[str]:
        """Generate rhythm-based recommendations."""
        recs = []
        
        if rhythm == RhythmType.NORMAL_SINUS.value:
            recs.append("Normal sinus rhythm detected")
        
        elif rhythm == RhythmType.SINUS_BRADY.value:
            if heart_rate < 50:
                recs.append("Significant bradycardia detected - medical evaluation recommended")
            else:
                recs.append("Bradycardia noted - may be normal in athletes")
        
        elif rhythm == RhythmType.SINUS_TACHY.value:
            if heart_rate > 120:
                recs.append("Elevated heart rate - consider causes such as stress, caffeine, or medical conditions")
            else:
                recs.append("Mild tachycardia noted - monitor and reduce stimulants if applicable")
        
        elif rhythm == RhythmType.AFIB.value:
            recs.append("IMPORTANT: Irregular rhythm detected consistent with atrial fibrillation")
            recs.append("Recommend consultation with cardiologist for confirmation and stroke risk assessment")
        
        return recs
    
    def _hrv_recommendations(self, hrv: Dict[str, Any]) -> List[str]:
        """Generate HRV-based recommendations."""
        recs = []
        
        time_domain = hrv.get("time_domain", {})
        rmssd = time_domain.get("rmssd_ms")
        
        if rmssd is not None:
            if rmssd < 20:
                recs.append("Low heart rate variability detected - consider stress management and sleep quality")
            elif rmssd > 80:
                recs.append("High heart rate variability - typically indicates good cardiovascular health")
        
        return recs
    
    def _arrhythmia_recommendations(self, arrhythmias: List[Dict[str, Any]]) -> List[str]:
        """Generate arrhythmia-based recommendations."""
        recs = []
        
        for arr in arrhythmias:
            arr_type = arr.get("type", "")
            urgency = arr.get("urgency", "low")
            
            if arr_type == "atrial_fibrillation":
                recs.append("Atrial fibrillation requires medical evaluation")
                recs.append("Discuss anticoagulation and rhythm control options with physician")
            
            elif arr_type == "premature_ventricular_contraction":
                count = arr.get("count", 0)
                if count > 5:
                    recs.append(f"Multiple PVCs detected ({count}) - may warrant further evaluation")
                else:
                    recs.append("Occasional PVCs detected - often benign but mention to physician")
            
            elif arr_type == "bradycardia" and urgency in ["high", "critical"]:
                recs.append("Significant bradycardia detected - seek medical evaluation")
            
            elif arr_type == "tachycardia" and urgency in ["high", "critical"]:
                recs.append("Significant tachycardia detected - seek medical evaluation if symptomatic")
        
        return recs
    
    def _risk_recommendations(self, risk_category: str) -> List[str]:
        """Generate risk-based recommendations."""
        recs = []
        
        if risk_category == "low":
            recs.append("Continue routine cardiovascular monitoring")
        elif risk_category == "moderate":
            recs.append("Consider lifestyle modifications: exercise, diet, stress management")
            recs.append("Follow-up with healthcare provider recommended")
        elif risk_category == "high":
            recs.append("Multiple risk factors identified - recommend cardiology consultation")
            recs.append("Regular monitoring and medication review recommended")
        elif risk_category == "critical":
            recs.append("IMPORTANT: Significant cardiac findings - prompt medical evaluation advised")
            recs.append("Do not delay seeking medical attention if experiencing symptoms")
        
        return recs
    
    def _load_templates(self) -> Dict[str, str]:
        """Load recommendation templates."""
        return {
            "normal": "Normal findings - continue healthy lifestyle",
            "lifestyle": "Consider lifestyle modifications for cardiovascular health",
            "followup": "Follow-up with healthcare provider recommended",
            "urgent": "Prompt medical evaluation advised",
            "emergency": "Seek emergency care if symptomatic",
        }


def generate_recommendations(
    rhythm: str,
    heart_rate: float,
    hrv_metrics: Optional[Dict[str, Any]] = None,
    arrhythmias: Optional[List[Dict[str, Any]]] = None,
    risk_category: str = "low"
) -> List[str]:
    """
    Convenience function to generate recommendations.
    
    Args:
        rhythm: Rhythm classification
        heart_rate: Heart rate in BPM
        hrv_metrics: HRV metrics dictionary
        arrhythmias: Detected arrhythmias
        risk_category: Risk category
    
    Returns:
        List of recommendation strings
    """
    generator = RecommendationGenerator()
    return generator.generate(rhythm, heart_rate, hrv_metrics, arrhythmias, risk_category)

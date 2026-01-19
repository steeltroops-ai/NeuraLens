"""
AI Explanation Rules for Cardiology/ECG Analysis
Rules and templates for explaining heart health and HRV results.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BiomarkerExplanation:
    """Template for explaining a cardiac biomarker."""
    name: str
    friendly_name: str
    unit: str
    normal_range: tuple
    normal_explanation: str
    borderline_explanation: str
    abnormal_explanation: str
    clinical_relevance: str


BIOMARKER_EXPLANATIONS: Dict[str, BiomarkerExplanation] = {
    "heart_rate": BiomarkerExplanation(
        name="heart_rate",
        friendly_name="Resting Heart Rate",
        unit="bpm",
        normal_range=(60, 100),
        normal_explanation=(
            "Your resting heart rate of {value:.0f} bpm is within the healthy range. "
            "This indicates good cardiovascular fitness."
        ),
        borderline_explanation=(
            "Your heart rate of {value:.0f} bpm is {direction} the typical range. "
            "This may be normal for you, especially if you're athletic (lower) "
            "or were nervous during the test (higher)."
        ),
        abnormal_explanation=(
            "Your heart rate of {value:.0f} bpm is outside the normal range. "
            "While this may have many explanations, discussing it with your "
            "doctor is recommended."
        ),
        clinical_relevance="Basic measure of cardiac function and fitness level"
    ),
    
    "rmssd": BiomarkerExplanation(
        name="rmssd",
        friendly_name="RMSSD (Heart Rate Variability)",
        unit="ms",
        normal_range=(20, 100),
        normal_explanation=(
            "Your RMSSD of {value:.1f} ms indicates healthy heart rate variability. "
            "This suggests good parasympathetic (rest-and-digest) nervous system activity."
        ),
        borderline_explanation=(
            "Your RMSSD of {value:.1f} ms is at the {direction} end of the normal range. "
            "This reflects your current stress and recovery balance."
        ),
        abnormal_explanation=(
            "Your RMSSD of {value:.1f} ms is {direction}. Lower HRV can be associated "
            "with stress, fatigue, or other factors. Consider stress reduction techniques."
        ),
        clinical_relevance="Reflects parasympathetic nervous system activity and recovery capacity"
    ),
    
    "sdnn": BiomarkerExplanation(
        name="sdnn",
        friendly_name="SDNN (Overall HRV)",
        unit="ms",
        normal_range=(50, 150),
        normal_explanation=(
            "Your SDNN of {value:.1f} ms shows healthy overall heart rate variability, "
            "indicating good autonomic nervous system function."
        ),
        borderline_explanation=(
            "Your SDNN of {value:.1f} ms is slightly {direction} typical. "
            "This is a general measure of heart rate variability."
        ),
        abnormal_explanation=(
            "Your SDNN of {value:.1f} ms is {direction}. This overall HRV measure "
            "can be influenced by many factors including stress, sleep, and fitness."
        ),
        clinical_relevance="Overall measure of autonomic nervous system health"
    ),
    
    "lf_hf_ratio": BiomarkerExplanation(
        name="lf_hf_ratio",
        friendly_name="LF/HF Ratio (Stress Balance)",
        unit="ratio",
        normal_range=(0.5, 2.0),
        normal_explanation=(
            "Your LF/HF ratio of {value:.2f} indicates balanced sympathetic and "
            "parasympathetic nervous system activity."
        ),
        borderline_explanation=(
            "Your LF/HF ratio of {value:.2f} is slightly elevated, which may "
            "indicate higher sympathetic (stress) activity."
        ),
        abnormal_explanation=(
            "Your LF/HF ratio of {value:.2f} suggests an imbalance in autonomic "
            "nervous system activity. Consider stress management techniques."
        ),
        clinical_relevance="Balance between sympathetic (stress) and parasympathetic (rest) activity"
    ),
}


RISK_LEVEL_MESSAGES = {
    RiskLevel.LOW: {
        "summary": "Your heart health indicators look good! Your HRV metrics suggest healthy nervous system balance.",
        "action": "Continue healthy lifestyle habits",
    },
    RiskLevel.MODERATE: {
        "summary": "Some heart metrics are worth monitoring. This could reflect stress, fatigue, or lifestyle factors.",
        "action": "Consider stress reduction and sleep optimization",
    },
    RiskLevel.HIGH: {
        "summary": "Your heart metrics suggest elevated stress or reduced variability. Clinical evaluation may be helpful.",
        "action": "Discuss these findings with your healthcare provider",
    },
    RiskLevel.CRITICAL: {
        "summary": "Significant findings in your heart metrics. Please consult a healthcare professional.",
        "action": "Schedule a cardiac evaluation",
    },
}


LIFESTYLE_RECOMMENDATIONS = {
    "stress": [
        "Practice deep breathing exercises (4-7-8 technique)",
        "Try mindfulness meditation for 10 minutes daily",
        "Take regular breaks during work",
        "Spend time in nature when possible"
    ],
    "sleep": [
        "Aim for 7-9 hours of quality sleep",
        "Maintain a consistent sleep schedule",
        "Avoid screens 1 hour before bed",
        "Keep your bedroom cool and dark"
    ],
    "exercise": [
        "Regular moderate exercise improves HRV",
        "Aim for 150 minutes of activity per week",
        "Include both cardio and strength training",
        "Don't overtrain - recovery is important"
    ],
    "nutrition": [
        "Stay well hydrated throughout the day",
        "Limit caffeine, especially after noon",
        "Reduce alcohol consumption",
        "Eat a balanced diet rich in omega-3s"
    ]
}


MANDATORY_DISCLAIMER = """
---
DISCLAIMER: This heart health screening is for informational purposes only and 
is NOT a substitute for a clinical ECG or cardiac evaluation. Heart rate 
variability can be affected by many factors. Always consult a qualified 
healthcare provider for cardiac concerns.
---
"""


class CardiologyExplanationGenerator:
    """Generates explanations for cardiology/HRV results."""
    
    def generate(self, results: dict, patient_context: Optional[dict] = None) -> str:
        sections = []
        
        sections.append(self._generate_summary(results))
        sections.append(self._generate_hrv_explanation())
        sections.append(self._generate_key_metrics(results))
        sections.append(self._generate_interpretation(results))
        sections.append(self._generate_lifestyle_tips(results))
        sections.append(MANDATORY_DISCLAIMER)
        
        return "\n\n".join(sections)
    
    def _generate_summary(self, results: dict) -> str:
        hr = results.get('heart_rate', 'N/A')
        rhythm = results.get('rhythm', 'Regular')
        
        return f"""## Your Heart Health Summary

Your heart analysis has been completed.

**Heart Rate:** {hr} bpm
**Rhythm:** {rhythm}
**Analysis Confidence:** {results.get('confidence', 0)*100:.0f}%
"""
    
    def _generate_hrv_explanation(self) -> str:
        return """### What is Heart Rate Variability (HRV)?

Heart Rate Variability measures the variation in time between heartbeats. 
Unlike what you might expect, a healthy heart doesn't beat like a metronome - 
it naturally speeds up and slows down slightly.

**Higher HRV** generally indicates:
- Good stress resilience
- Strong recovery capacity
- Healthy nervous system balance

**Lower HRV** may suggest:
- Higher stress levels
- Fatigue or overtraining
- Need for recovery
"""
    
    def _generate_key_metrics(self, results: dict) -> str:
        metrics = []
        
        for key, config in BIOMARKER_EXPLANATIONS.items():
            if key in results:
                value = results[key]
                if isinstance(value, (int, float)):
                    normal_min, normal_max = config.normal_range
                    status = "Normal" if normal_min <= value <= normal_max else "Monitor"
                    metrics.append(f"- **{config.friendly_name}:** {value:.1f} {config.unit} ({status})")
        
        if not metrics:
            metrics = ["- Metrics data not available"]
        
        return f"### Key Metrics\n\n" + "\n".join(metrics)
    
    def _generate_interpretation(self, results: dict) -> str:
        rmssd = results.get('rmssd_ms', 50)
        
        if rmssd > 50:
            interp = "Your HRV indicates good recovery and stress resilience. Keep up your healthy habits!"
        elif rmssd > 30:
            interp = "Your HRV is moderate. Consider prioritizing sleep and stress management."
        else:
            interp = "Your HRV suggests you may benefit from rest and recovery. Focus on sleep and reducing stress."
        
        return f"### What This Means for You\n\n{interp}"
    
    def _generate_lifestyle_tips(self, results: dict) -> str:
        tips = []
        tips.extend(LIFESTYLE_RECOMMENDATIONS["stress"][:2])
        tips.extend(LIFESTYLE_RECOMMENDATIONS["sleep"][:2])
        tips.extend(LIFESTYLE_RECOMMENDATIONS["exercise"][:1])
        
        tip_list = "\n".join([f"- {tip}" for tip in tips])
        return f"### Tips to Improve Heart Health\n\n{tip_list}"


def generate_cardiology_explanation(
    results: dict,
    patient_context: Optional[dict] = None
) -> str:
    """Generate patient-friendly explanation of cardiology results."""
    generator = CardiologyExplanationGenerator()
    return generator.generate(results, patient_context)

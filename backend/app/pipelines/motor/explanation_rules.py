"""
AI Explanation Rules for Motor Function Assessment
Rules and templates for explaining motor/tremor analysis results.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class TremorType(Enum):
    NONE = "none"
    PHYSIOLOGICAL = "physiological"
    ESSENTIAL = "essential"
    PARKINSONIAN = "parkinsonian"
    INTENTION = "intention"
    UNKNOWN = "unknown"


@dataclass
class BiomarkerExplanation:
    """Template for explaining a motor biomarker."""
    name: str
    friendly_name: str
    unit: str
    normal_range: tuple
    normal_explanation: str
    borderline_explanation: str
    abnormal_explanation: str
    clinical_relevance: str


BIOMARKER_EXPLANATIONS: Dict[str, BiomarkerExplanation] = {
    "tremor_score": BiomarkerExplanation(
        name="tremor_score",
        friendly_name="Tremor Intensity",
        unit="score",
        normal_range=(0, 0.2),
        normal_explanation=(
            "No significant tremor detected. Your motor control appears stable."
        ),
        borderline_explanation=(
            "Mild tremor detected ({value:.2f}). This is often normal and can be caused "
            "by caffeine, fatigue, anxiety, or simply cold hands."
        ),
        abnormal_explanation=(
            "Notable tremor detected ({value:.2f}). While tremor can have many benign causes, "
            "consistent tremor may warrant evaluation by a neurologist."
        ),
        clinical_relevance="Measures involuntary rhythmic oscillations in movement"
    ),
    
    "tremor_frequency": BiomarkerExplanation(
        name="tremor_frequency",
        friendly_name="Tremor Frequency",
        unit="Hz",
        normal_range=(0, 12),
        normal_explanation="Tremor frequency not applicable or minimal.",
        borderline_explanation=(
            "Detected tremor at {value:.1f} Hz. Different frequencies can suggest "
            "different types of tremor."
        ),
        abnormal_explanation=(
            "Tremor frequency: {value:.1f} Hz. The frequency pattern can help "
            "characterize the type of tremor if evaluation is needed."
        ),
        clinical_relevance="Frequency helps distinguish tremor types"
    ),
    
    "coordination_score": BiomarkerExplanation(
        name="coordination_score",
        friendly_name="Coordination",
        unit="/100",
        normal_range=(70, 100),
        normal_explanation=(
            "Excellent coordination! Your motor control is smooth and accurate."
        ),
        borderline_explanation=(
            "Coordination score of {value:.0f}/100 is slightly below average. "
            "This could be due to testing conditions or fatigue."
        ),
        abnormal_explanation=(
            "Coordination score of {value:.0f}/100 suggests room for improvement. "
            "Consider factors like focus, fatigue, and practice with coordination exercises."
        ),
        clinical_relevance="Measures smoothness and accuracy of intentional movements"
    ),
    
    "fatigue_index": BiomarkerExplanation(
        name="fatigue_index",
        friendly_name="Motor Fatigue",
        unit="index",
        normal_range=(0, 0.3),
        normal_explanation="Low fatigue - your motor performance remained consistent.",
        borderline_explanation=(
            "Moderate fatigue detected ({value:.2f}). Performance declined slightly "
            "over the test duration, which is common."
        ),
        abnormal_explanation=(
            "Significant fatigue ({value:.2f}) - performance declined notably during testing. "
            "Consider factors like sleep, hydration, and general fatigue."
        ),
        clinical_relevance="Measures decline in motor performance over time"
    ),
    
    "movement_smoothness": BiomarkerExplanation(
        name="movement_smoothness",
        friendly_name="Movement Smoothness",
        unit="/100",
        normal_range=(70, 100),
        normal_explanation="Smooth, fluid movements with good motor control.",
        borderline_explanation=(
            "Movement smoothness of {value:.0f}/100 shows some variation. "
            "Practice and focus can often improve this."
        ),
        abnormal_explanation=(
            "Jerky or less fluid movements detected ({value:.0f}/100). "
            "This can reflect stress, unfamiliarity with the test, or warrant follow-up."
        ),
        clinical_relevance="Measures fluidity and continuity of movements"
    ),
}


# Tremor type explanations
TREMOR_TYPE_EXPLANATIONS = {
    TremorType.NONE: {
        "name": "No Tremor",
        "description": "No significant tremor was detected during the assessment.",
        "meaning": "This is a healthy finding.",
        "action": "No action needed"
    },
    TremorType.PHYSIOLOGICAL: {
        "name": "Physiological Tremor",
        "description": "A very fine tremor that everyone has, usually invisible to the eye.",
        "meaning": "This is completely normal and not a cause for concern.",
        "action": "No action needed",
        "common_causes": ["Caffeine", "Fatigue", "Stress", "Certain medications"]
    },
    TremorType.ESSENTIAL: {
        "name": "Essential Tremor Pattern",
        "description": "A rhythmic tremor that typically occurs during movement or while holding a posture.",
        "meaning": "Essential tremor is common and often hereditary. It's usually not dangerous but can be bothersome.",
        "action": "Consider neurologist consultation if tremor affects daily activities"
    },
    TremorType.PARKINSONIAN: {
        "name": "Parkinsonian Tremor Pattern",
        "description": "A tremor pattern that occurs at rest and may decrease with movement.",
        "meaning": "This pattern can be seen in Parkinson's disease, but requires proper medical evaluation to diagnose.",
        "action": "Neurological evaluation recommended"
    },
    TremorType.INTENTION: {
        "name": "Intention Tremor Pattern",
        "description": "Tremor that increases when approaching a target.",
        "meaning": "This pattern can be related to cerebellar function.",
        "action": "Consider neurological evaluation"
    },
}


# Factors that can affect motor performance
CONFOUNDING_FACTORS = [
    "**Caffeine** - Can increase physiological tremor",
    "**Fatigue** - Tired muscles may shake more",
    "**Anxiety/Stress** - Nervous trembling is very common",
    "**Cold temperature** - Cold hands often shake",
    "**Certain medications** - Some medications cause tremor as a side effect",
    "**Low blood sugar** - Can cause shakiness",
    "**Alcohol withdrawal** - Can cause tremor",
    "**Exercise** - Recent intense exercise can cause temporary tremor"
]


RISK_LEVEL_MESSAGES = {
    "low": {
        "summary": "Your motor function assessment shows healthy, stable movement patterns.",
        "action": "No specific action needed - continue staying active!"
    },
    "moderate": {
        "summary": "Some findings worth noting, though often explained by common factors.",
        "action": "Consider lifestyle factors and retest when well-rested"
    },
    "high": {
        "summary": "Motor patterns suggest evaluation may be beneficial.",
        "action": "Consider discussing with a healthcare provider"
    },
    "critical": {
        "summary": "Significant motor findings detected.",
        "action": "Neurological evaluation recommended"
    }
}


MANDATORY_DISCLAIMER = """
---
DISCLAIMER: This motor screening is for informational purposes only and 
is NOT a clinical neurological examination. Tremor detection can be affected 
by many temporary factors. This tool cannot diagnose movement disorders. 
If you have concerns about tremor or motor function, please consult a 
neurologist for proper evaluation.
---
"""


class MotorExplanationGenerator:
    """Generates explanations for motor assessment results."""
    
    def generate(self, results: dict, patient_context: Optional[dict] = None) -> str:
        sections = []
        
        sections.append(self._generate_summary(results))
        sections.append(self._generate_tremor_analysis(results))
        sections.append(self._generate_coordination_analysis(results))
        sections.append(self._generate_factors_section())
        sections.append(self._generate_recommendations(results))
        sections.append(MANDATORY_DISCLAIMER)
        
        return "\n\n".join(sections)
    
    def _generate_summary(self, results: dict) -> str:
        risk_score = results.get('risk_score', 0) * 100
        
        if risk_score < 25:
            level = "low"
        elif risk_score < 50:
            level = "moderate"
        elif risk_score < 75:
            level = "high"
        else:
            level = "critical"
        
        msg = RISK_LEVEL_MESSAGES[level]
        
        return f"""## Your Motor Function Summary

Your motor assessment has been completed.

**Risk Level:** {level.title()}
**Confidence:** {results.get('confidence', 0)*100:.0f}%

{msg["summary"]}
"""
    
    def _generate_tremor_analysis(self, results: dict) -> str:
        tremor_score = results.get('tremor_score', 0)
        tremor_type = results.get('tremor_type', 'none')
        frequency = results.get('tremor_frequency', 0)
        
        if tremor_score < 0.1:
            finding = "No significant tremor was detected."
        elif tremor_score < 0.3:
            finding = f"Mild tremor detected (score: {tremor_score:.2f}). This is often normal."
        else:
            finding = f"Notable tremor detected (score: {tremor_score:.2f}). Consider evaluation."
        
        type_info = TREMOR_TYPE_EXPLANATIONS.get(
            TremorType(tremor_type) if tremor_type in [t.value for t in TremorType] else TremorType.UNKNOWN,
            TREMOR_TYPE_EXPLANATIONS[TremorType.NONE]
        )
        
        return f"""### Tremor Analysis

{finding}

**Pattern:** {type_info['name']}
{type_info['description']}

{f"**Frequency:** {frequency:.1f} Hz" if frequency > 0 else ""}
"""
    
    def _generate_coordination_analysis(self, results: dict) -> str:
        coord = results.get('coordination_score', 80)
        fatigue = results.get('fatigue_index', 0.1)
        
        if coord >= 80:
            coord_finding = "Excellent coordination and motor control!"
        elif coord >= 60:
            coord_finding = "Good coordination with some room for improvement."
        else:
            coord_finding = "Coordination scores below average - various factors could contribute."
        
        fatigue_finding = "Low motor fatigue" if fatigue < 0.2 else "Some motor fatigue noted during assessment"
        
        return f"""### Coordination & Fatigue

**Coordination Score:** {coord}/100
{coord_finding}

**Fatigue:** {fatigue_finding}
"""
    
    def _generate_factors_section(self) -> str:
        factors = CONFOUNDING_FACTORS[:5]
        return f"""### Factors That Can Affect Results

It's important to note that many temporary factors can influence motor performance:

{chr(10).join(factors)}

Consider retesting when well-rested and relaxed for the most accurate results.
"""
    
    def _generate_recommendations(self, results: dict) -> str:
        risk_score = results.get('risk_score', 0) * 100
        
        if risk_score < 30:
            recs = [
                "Continue with regular physical activity",
                "No immediate follow-up needed",
                "Monitor if you notice any changes in motor function"
            ]
        elif risk_score < 60:
            recs = [
                "Consider factors that might affect results (caffeine, fatigue, stress)",
                "Retest when well-rested to compare",
                "Discuss with your doctor if you have concerns"
            ]
        else:
            recs = [
                "Consider evaluation by a neurologist",
                "Keep track of when tremor occurs and what makes it better or worse",
                "Rule out medication side effects with your doctor"
            ]
        
        rec_list = "\n".join([f"- {r}" for r in recs])
        return f"### Recommendations\n\n{rec_list}"


def generate_motor_explanation(
    results: dict,
    patient_context: Optional[dict] = None
) -> str:
    """Generate patient-friendly explanation of motor assessment results."""
    generator = MotorExplanationGenerator()
    return generator.generate(results, patient_context)

"""
AI Explanation Rules for NRI (Neurological Risk Index) Fusion
Rules and templates for explaining multi-modal assessment results.
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
class ModalityContribution:
    """Describes a modality's contribution to NRI."""
    name: str
    friendly_name: str
    description: str
    weight: float
    what_it_measures: str


MODALITY_DESCRIPTIONS: Dict[str, ModalityContribution] = {
    "speech": ModalityContribution(
        name="speech",
        friendly_name="Voice Analysis",
        description="Analyzes voice biomarkers including pitch stability, amplitude, and tremor",
        weight=0.25,
        what_it_measures="Laryngeal and respiratory motor control, cognitive-motor integration"
    ),
    "retinal": ModalityContribution(
        name="retinal",
        friendly_name="Retinal Imaging",
        description="Analyzes fundus images for vascular and optic nerve health",
        weight=0.25,
        what_it_measures="Cerebrovascular health, optic nerve integrity"
    ),
    "cardiology": ModalityContribution(
        name="cardiology",
        friendly_name="Cardiac/HRV Analysis",
        description="Measures heart rate variability and autonomic function",
        weight=0.20,
        what_it_measures="Autonomic nervous system balance, cardiovascular health"
    ),
    "cognitive": ModalityContribution(
        name="cognitive",
        friendly_name="Cognitive Assessment",
        description="Tests memory, attention, processing speed, and executive function",
        weight=0.15,
        what_it_measures="Multiple cognitive domains and brain function"
    ),
    "motor": ModalityContribution(
        name="motor",
        friendly_name="Motor Function",
        description="Measures tremor, coordination, and motor control",
        weight=0.15,
        what_it_measures="Motor system integrity, cerebellar and basal ganglia function"
    ),
}


NRI_EXPLANATION = """
### What is the Neurological Risk Index (NRI)?

The NRI is a comprehensive score that combines multiple health assessments to provide 
a holistic view of neurological health. Think of it like getting a complete physical 
instead of just checking one thing.

**Why Multi-Modal Assessment?**

Different conditions can affect different parts of your nervous system:
- Some conditions first show in the voice before other symptoms appear
- The eyes can reveal vascular changes in the brain
- Heart rate variability reflects autonomic nervous system health
- Cognitive tests assess multiple brain regions
- Motor tests evaluate movement control centers

By combining these, the NRI can detect patterns that might be missed by any single test alone.

**How Scores Are Combined:**

Each modality contributes to the overall score based on:
- Its relevance to neurological health
- The confidence of its measurements
- Clinical evidence for its predictive value
"""


RISK_LEVEL_MESSAGES = {
    RiskLevel.LOW: {
        "summary": (
            "Your multi-modal assessment shows consistent low-risk findings across "
            "all tested modalities. This integrated result provides strong reassurance."
        ),
        "interpretation": "All systems evaluated appear to be functioning well.",
        "action": "Continue with routine health monitoring and healthy lifestyle."
    },
    RiskLevel.MODERATE: {
        "summary": (
            "Your NRI score shows moderate findings. Some modalities show results "
            "that may warrant attention, while others are reassuring."
        ),
        "interpretation": "Mixed findings across modalities - see detailed breakdown below.",
        "action": "Consider follow-up on specific areas of concern."
    },
    RiskLevel.HIGH: {
        "summary": (
            "Your NRI score indicates elevated findings across multiple modalities. "
            "This pattern suggests that clinical evaluation may be beneficial."
        ),
        "interpretation": "Multiple systems showing findings that warrant review.",
        "action": "Recommend consulting with healthcare provider for comprehensive evaluation."
    },
    RiskLevel.CRITICAL: {
        "summary": (
            "Your NRI score shows significant findings across multiple modalities. "
            "We recommend prompt clinical evaluation."
        ),
        "interpretation": "Consistent concerning patterns detected across assessments.",
        "action": "Please consult a healthcare provider soon."
    },
}


CONDITION_PATTERNS = {
    "parkinsons_pattern": {
        "name": "Parkinson's-Related Pattern",
        "modalities": ["speech", "motor"],
        "description": "Voice tremor combined with motor tremor patterns",
        "note": "This pattern can be seen in Parkinson's but requires proper diagnosis"
    },
    "cognitive_vascular": {
        "name": "Cognitive-Vascular Pattern",
        "modalities": ["cognitive", "retinal", "cardiology"],
        "description": "Cognitive changes with vascular markers",
        "note": "May indicate vascular contributions to cognitive health"
    },
    "autonomic_dysfunction": {
        "name": "Autonomic Dysfunction Pattern",
        "modalities": ["cardiology", "speech", "motor"],
        "description": "HRV changes with motor/voice findings",
        "note": "Suggests autonomic nervous system involvement"
    },
}


MANDATORY_DISCLAIMER = """
---
DISCLAIMER: The Neurological Risk Index (NRI) is a research screening tool that 
combines multiple assessments for informational purposes only. It is NOT a 
clinical diagnosis and cannot diagnose neurological conditions. The NRI should 
be used as one of many factors in health monitoring. Always consult qualified 
healthcare professionals for medical evaluation, diagnosis, and treatment of 
neurological conditions.
---
"""


class NRIExplanationGenerator:
    """Generates explanations for NRI fusion results."""
    
    def generate(self, results: dict, patient_context: Optional[dict] = None) -> str:
        sections = []
        
        sections.append(self._generate_overview(results))
        sections.append(NRI_EXPLANATION)
        sections.append(self._generate_overall_score(results))
        sections.append(self._generate_modality_breakdown(results))
        sections.append(self._generate_areas_of_strength(results))
        sections.append(self._generate_areas_to_monitor(results))
        sections.append(self._generate_integrated_recommendations(results))
        sections.append(MANDATORY_DISCLAIMER)
        
        return "\n\n".join(sections)
    
    def _generate_overview(self, results: dict) -> str:
        modalities = results.get('modalities', [])
        modality_count = len(modalities) if modalities else 0
        
        return f"""## Your Comprehensive Assessment

You've completed a multi-modal health screening that analyzed **{modality_count} different aspects** 
of your health to create an integrated picture of your neurological wellness.

**Modalities Analyzed:** {', '.join([MODALITY_DESCRIPTIONS.get(m, {}).friendly_name or m.title() for m in modalities]) if modalities else 'Multiple'}
"""
    
    def _generate_overall_score(self, results: dict) -> str:
        nri_score = results.get('nri_score', 25)
        confidence = results.get('confidence', 0.85) * 100
        
        if nri_score < 25:
            level = RiskLevel.LOW
        elif nri_score < 50:
            level = RiskLevel.MODERATE
        elif nri_score < 75:
            level = RiskLevel.HIGH
        else:
            level = RiskLevel.CRITICAL
        
        msg = RISK_LEVEL_MESSAGES[level]
        
        return f"""### Overall NRI Score

**Score:** {nri_score}/100
**Risk Category:** {level.value.title()}
**Confidence:** {confidence:.0f}%

{msg["summary"]}

**Interpretation:** {msg["interpretation"]}
"""
    
    def _generate_modality_breakdown(self, results: dict) -> str:
        contributions = results.get('modality_contributions', {})
        modality_scores = results.get('modality_scores', {})
        
        if not contributions and not modality_scores:
            return "### Modality Breakdown\n\nDetailed modality information not available."
        
        lines = ["### How Each Modality Contributed\n"]
        
        for modality, contribution in contributions.items():
            if modality in MODALITY_DESCRIPTIONS:
                info = MODALITY_DESCRIPTIONS[modality]
                score = modality_scores.get(modality, contribution * 100)
                
                if score < 30:
                    status = "Low Risk"
                elif score < 60:
                    status = "Moderate"
                else:
                    status = "Elevated"
                
                lines.append(f"**{info.friendly_name}**")
                lines.append(f"- Contribution: {contribution*100:.0f}%")
                lines.append(f"- Status: {status}")
                lines.append(f"- Measures: {info.what_it_measures}")
                lines.append("")
        
        return "\n".join(lines)
    
    def _generate_areas_of_strength(self, results: dict) -> str:
        modality_scores = results.get('modality_scores', {})
        
        strengths = [m for m, s in modality_scores.items() if s < 30]
        
        if not strengths:
            return "### Areas of Strength\n\nContinue maintaining your healthy habits!"
        
        strength_text = []
        for modality in strengths:
            if modality in MODALITY_DESCRIPTIONS:
                info = MODALITY_DESCRIPTIONS[modality]
                strength_text.append(f"- **{info.friendly_name}**: Results within healthy ranges")
        
        return "### Areas of Strength\n\n" + "\n".join(strength_text)
    
    def _generate_areas_to_monitor(self, results: dict) -> str:
        modality_scores = results.get('modality_scores', {})
        
        concerns = [(m, s) for m, s in modality_scores.items() if s >= 40]
        concerns.sort(key=lambda x: -x[1])
        
        if not concerns:
            return "### Areas to Monitor\n\nNo significant areas of concern at this time."
        
        concern_text = []
        for modality, score in concerns[:3]:
            if modality in MODALITY_DESCRIPTIONS:
                info = MODALITY_DESCRIPTIONS[modality]
                concern_text.append(
                    f"- **{info.friendly_name}** (Score: {score:.0f}/100): "
                    f"Consider reviewing detailed results"
                )
        
        return "### Areas to Monitor\n\n" + "\n".join(concern_text)
    
    def _generate_integrated_recommendations(self, results: dict) -> str:
        nri_score = results.get('nri_score', 25)
        modality_scores = results.get('modality_scores', {})
        
        recs = []
        
        # General recommendations based on NRI score
        if nri_score < 30:
            recs.append("Continue with your current healthy lifestyle")
            recs.append("Schedule regular health check-ups")
        elif nri_score < 60:
            recs.append("Review the detailed results for each modality")
            recs.append("Consider follow-up on areas with elevated scores")
            recs.append("Discuss any concerns with your healthcare provider")
        else:
            recs.append("Schedule an appointment with your healthcare provider")
            recs.append("Bring this report to discuss the findings")
            recs.append("Consider specialist referrals as recommended")
        
        # Modality-specific recommendations
        for modality, score in modality_scores.items():
            if score >= 50:
                if modality == "speech":
                    recs.append("Consider speech-language pathology evaluation")
                elif modality == "retinal":
                    recs.append("Schedule comprehensive eye examination")
                elif modality == "cardiology":
                    recs.append("Review cardiovascular health with your doctor")
                elif modality == "cognitive":
                    recs.append("Consider formal cognitive assessment if concerned")
                elif modality == "motor":
                    recs.append("Consider neurological consultation for motor findings")
        
        rec_list = "\n".join([f"- {r}" for r in recs[:6]])
        return f"""### Integrated Recommendations

Based on your comprehensive assessment:

{rec_list}

{RISK_LEVEL_MESSAGES[RiskLevel.MODERATE if 25 <= nri_score < 50 else RiskLevel.LOW if nri_score < 25 else RiskLevel.HIGH]["action"]}
"""


def generate_nri_explanation(
    results: dict,
    patient_context: Optional[dict] = None
) -> str:
    """Generate patient-friendly explanation of NRI fusion results."""
    generator = NRIExplanationGenerator()
    return generator.generate(results, patient_context)

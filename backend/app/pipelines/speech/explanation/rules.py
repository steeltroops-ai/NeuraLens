"""
AI Voice Explanation Rules for Speech Analysis
Rules and templates for explaining speech biomarker results to users.

These rules are used by the explain pipeline to generate 
patient-friendly explanations of clinical results.
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
    """Template for explaining a biomarker."""
    name: str
    friendly_name: str
    unit: str
    normal_range: tuple
    
    # Explanations for different statuses
    normal_explanation: str
    borderline_explanation: str
    abnormal_explanation: str
    
    # Clinical context
    clinical_relevance: str
    conditions_associated: List[str]


# Biomarker explanation templates
BIOMARKER_EXPLANATIONS: Dict[str, BiomarkerExplanation] = {
    "jitter": BiomarkerExplanation(
        name="jitter_local",
        friendly_name="Voice Pitch Stability (Jitter)",
        unit="%",
        normal_range=(0.0, 1.04),
        normal_explanation=(
            "Your voice pitch stability is excellent at {value}%, which is within the "
            "healthy range (under 1.04%). This suggests smooth vocal fold vibration."
        ),
        borderline_explanation=(
            "Your voice shows slightly elevated pitch variation at {value}%. "
            "This is borderline and may be due to fatigue, nervousness, or a mild "
            "cold. Consider re-recording when relaxed."
        ),
        abnormal_explanation=(
            "Your voice pitch variation is elevated at {value}% (normal is under 1.04%). "
            "This pattern can be associated with vocal strain or neurological conditions "
            "affecting voice control. A follow-up with a speech pathologist may be helpful."
        ),
        clinical_relevance="Measures cycle-to-cycle variation in fundamental frequency",
        conditions_associated=["Parkinson's Disease", "Vocal fold pathology", "Laryngitis"]
    ),
    
    "shimmer": BiomarkerExplanation(
        name="shimmer_local",
        friendly_name="Voice Amplitude Stability (Shimmer)",
        unit="%",
        normal_range=(0.0, 3.81),
        normal_explanation=(
            "Your voice amplitude stability is healthy at {value}%, indicating steady "
            "breath support and vocal fold closure."
        ),
        borderline_explanation=(
            "Slight amplitude variation detected at {value}%. This could indicate "
            "mild vocal fatigue or subtle changes in breath control."
        ),
        abnormal_explanation=(
            "Elevated amplitude variation at {value}% suggests potential vocal fold "
            "irregularities. This may warrant evaluation by an ENT specialist."
        ),
        clinical_relevance="Measures cycle-to-cycle variation in amplitude",
        conditions_associated=["Vocal fold nodules", "Paresis", "Dysphonia"]
    ),
    
    "hnr": BiomarkerExplanation(
        name="hnr",
        friendly_name="Voice Clarity (HNR)",
        unit="dB",
        normal_range=(20.0, 30.0),
        normal_explanation=(
            "Your voice clarity is excellent at {value} dB, indicating clear, "
            "resonant speech with minimal breathiness."
        ),
        borderline_explanation=(
            "Voice clarity at {value} dB is slightly reduced. This could be due to "
            "a cold, allergies, or voice fatigue."
        ),
        abnormal_explanation=(
            "Your voice shows increased breathiness or hoarseness (HNR: {value} dB). "
            "This pattern may indicate vocal fold changes worth discussing with a "
            "healthcare provider."
        ),
        clinical_relevance="Ratio of harmonic to noise components in voice",
        conditions_associated=["Dysphonia", "Vocal fold pathology", "Aging voice"]
    ),
    
    "cpps": BiomarkerExplanation(
        name="cpps",
        friendly_name="Voice Quality Score (CPPS)",
        unit="dB",
        normal_range=(14.0, 30.0),
        normal_explanation=(
            "Your overall voice quality score (CPPS: {value} dB) is excellent, "
            "indicating strong, clear phonation."
        ),
        borderline_explanation=(
            "Your CPPS score of {value} dB is slightly below optimal. CPPS is "
            "considered the most reliable measure of voice quality."
        ),
        abnormal_explanation=(
            "Your CPPS score of {value} dB is below the typical range. CPPS is "
            "considered the gold standard for voice quality assessment. Lower values "
            "may indicate dysphonia or voice disorders."
        ),
        clinical_relevance="Gold standard acoustic measure of overall voice quality",
        conditions_associated=["Dysphonia", "Voice disorders", "Laryngeal pathology"]
    ),
    
    "speech_rate": BiomarkerExplanation(
        name="speech_rate",
        friendly_name="Speaking Speed",
        unit="syllables/second",
        normal_range=(3.5, 6.5),
        normal_explanation=(
            "Your speaking rate of {value} syllables/second is within the healthy "
            "range (3.5-6.5), indicating normal motor speech function."
        ),
        borderline_explanation=(
            "Your speaking rate of {value} syllables/second is slightly outside "
            "the typical range. This may reflect your natural speaking style."
        ),
        abnormal_explanation=(
            "Your speaking rate of {value} syllables/second is notably different from "
            "typical. Significantly slow speech can occur with fatigue, careful speech, "
            "or in some neurological conditions. Very fast speech may affect clarity."
        ),
        clinical_relevance="Indicator of motor speech control and cognitive processing",
        conditions_associated=["Parkinson's Disease", "Cognitive decline", "Dysarthria"]
    ),
    
    "voice_tremor": BiomarkerExplanation(
        name="tremor_score",
        friendly_name="Voice Tremor",
        unit="index",
        normal_range=(0.0, 0.15),
        normal_explanation=(
            "No significant voice tremor detected ({value}). Your vocal control "
            "appears stable."
        ),
        borderline_explanation=(
            "Slight voice tremor detected at {value}. This is common with "
            "nervousness or caffeine intake and usually not concerning."
        ),
        abnormal_explanation=(
            "Voice tremor detected at {value}, which is higher than typical. "
            "Voice tremor can be associated with essential tremor, Parkinson's "
            "disease, or other neurological conditions. A neurological evaluation "
            "may be beneficial."
        ),
        clinical_relevance="Measures rhythmic oscillations in voice",
        conditions_associated=["Parkinson's Disease", "Essential tremor", "Neurological disorders"]
    ),
    
    "pause_ratio": BiomarkerExplanation(
        name="pause_ratio",
        friendly_name="Speech Pause Pattern",
        unit="ratio",
        normal_range=(0.0, 0.25),
        normal_explanation=(
            "Your speech flow is smooth with a healthy pause ratio of {value:.0%}."
        ),
        borderline_explanation=(
            "Your speech contains more pauses than typical ({value:.0%}). This may "
            "reflect careful, deliberate speech."
        ),
        abnormal_explanation=(
            "Your speech contains more pauses than typical ({value:.0%}). "
            "Increased pauses can be associated with word-finding difficulties, "
            "fatigue, or cognitive changes. Consider discussing with your physician "
            "if this is a change from your baseline."
        ),
        clinical_relevance="Measures silence proportion in connected speech",
        conditions_associated=["Cognitive decline", "Alzheimer's", "Word-finding difficulty"]
    ),
}


# Risk level communication templates
RISK_LEVEL_MESSAGES = {
    RiskLevel.LOW: {
        "summary": (
            "Your voice analysis shows low neurological risk indicators. "
            "All measured biomarkers are within healthy ranges."
        ),
        "tone": "reassuring",
        "action": "Continue routine monitoring",
        "emoji": ":-)"
    },
    RiskLevel.MODERATE: {
        "summary": (
            "Some voice biomarkers are outside typical ranges. While this "
            "doesn't necessarily indicate a problem, it may warrant attention "
            "if you notice other changes."
        ),
        "tone": "informative",
        "action": "Consider follow-up in 3-6 months",
        "emoji": ":-|"
    },
    RiskLevel.HIGH: {
        "summary": (
            "Your voice analysis shows patterns that warrant clinical attention. "
            "Several biomarkers indicate possible changes in speech motor control. "
            "We recommend consulting with a healthcare provider."
        ),
        "tone": "serious but supportive",
        "action": "Schedule specialist consultation",
        "emoji": ":-("
    },
    RiskLevel.CRITICAL: {
        "summary": (
            "Your voice analysis shows significant abnormalities in multiple "
            "biomarkers. We strongly recommend prompt evaluation by a healthcare "
            "professional. This is a screening result, not a diagnosis."
        ),
        "tone": "urgent but calm",
        "action": "Seek medical attention promptly",
        "emoji": ":-O"
    },
}


# Condition-specific explanation templates
CONDITION_EXPLANATIONS = {
    "parkinsons": {
        "name": "Parkinson's Disease Indicators",
        "pattern_detected": (
            "Your voice shows patterns sometimes seen in early Parkinson's disease: "
            "{contributing_factors}. The probability estimate is {probability:.0%} "
            "(confidence: {confidence:.0%}).\n\n"
            "IMPORTANT: This is a screening indicator, not a diagnosis. Many "
            "conditions can produce similar voice patterns. A neurological "
            "evaluation can provide clarity."
        ),
        "what_to_tell_doctor": (
            "Tell your doctor: 'My voice screening showed patterns that may be "
            "worth investigating - specifically {contributing_factors}. I'd like "
            "to discuss whether further evaluation is appropriate.'"
        ),
    },
    "cognitive_decline": {
        "name": "Cognitive Change Indicators",
        "pattern_detected": (
            "Your speech pattern shows increased pauses and slower rate, which "
            "can be associated with cognitive changes. The estimated probability "
            "is {probability:.0%}.\n\n"
            "NOTE: These patterns can also result from fatigue, medication, or "
            "simply careful speech. Consider re-testing after rest."
        ),
        "what_to_tell_doctor": (
            "Tell your doctor: 'I had a voice screening that showed some changes "
            "in my speech patterns. I'd like to discuss whether cognitive testing "
            "might be helpful.'"
        ),
    },
    "depression": {
        "name": "Mood-Related Voice Patterns",
        "pattern_detected": (
            "Your voice shows reduced prosodic variation (more monotone speech) "
            "and slower rate. These can be mood-related voice changes.\n\n"
            "If you've been feeling down or experiencing mood changes, speaking "
            "with a healthcare provider about your mental health may be helpful."
        ),
        "what_to_tell_doctor": (
            "If you're comfortable, tell your doctor: 'I've noticed my voice "
            "has less variation lately, and I want to discuss how I've been feeling.'"
        ),
    },
    "dysarthria": {
        "name": "Motor Speech Changes",
        "pattern_detected": (
            "Voice analysis indicates possible motor speech changes affecting "
            "articulation clarity. This pattern ({probability:.0%} likelihood) "
            "suggests a speech-language pathology evaluation could be beneficial."
        ),
        "what_to_tell_doctor": (
            "Tell your doctor: 'A voice screening suggested I might benefit from "
            "seeing a speech therapist. Can you provide a referral?'"
        ),
    },
}


# Mandatory disclaimer - MUST be included in every explanation
MANDATORY_DISCLAIMER = """
---
IMPORTANT DISCLAIMER: This analysis is for informational screening 
purposes only and is NOT a medical diagnosis. Voice biomarkers can be 
affected by many factors including recording conditions, fatigue, 
medications, and temporary illness. Always consult a qualified healthcare 
provider for medical advice, diagnosis, or treatment. This tool is 
intended to support, not replace, professional medical judgment.
---
"""


# Quality warnings
QUALITY_WARNINGS = {
    "low_quality": (
        "Audio quality score: {quality:.0%}. This affects measurement reliability. "
        "For best results, re-record in a quiet environment, speaking clearly "
        "at a normal volume for at least 10 seconds."
    ),
    "estimated_values": (
        "Some measurements are estimated due to audio limitations. These are "
        "marked with lower confidence scores."
    ),
    "short_duration": (
        "Recording duration was short. Longer samples (10+ seconds) provide "
        "more reliable measurements."
    ),
}


class SpeechExplanationGenerator:
    """
    Generates patient-friendly explanations of speech analysis results.
    
    Usage:
        generator = SpeechExplanationGenerator()
        explanation = generator.generate(results)
    """
    
    def __init__(self, include_research_biomarkers: bool = False):
        self.include_research = include_research_biomarkers
    
    def generate(
        self, 
        results: dict,
        patient_context: Optional[dict] = None
    ) -> str:
        """
        Generate a complete explanation of speech analysis results.
        
        Args:
            results: The EnhancedSpeechAnalysisResponse as a dict
            patient_context: Optional context (age, sex, known conditions)
            
        Returns:
            Markdown-formatted explanation
        """
        sections = []
        
        # 1. Opening summary
        sections.append(self._generate_summary(results))
        
        # 2. Key findings
        sections.append(self._generate_key_findings(results))
        
        # 3. Condition risks (if elevated)
        if results.get("condition_risks"):
            sections.append(self._generate_condition_section(results))
        
        # 4. Recommendations
        sections.append(self._generate_recommendations(results))
        
        # 5. Detailed biomarkers (optional expandable)
        sections.append(self._generate_biomarker_details(results))
        
        # 6. Quality notes (if applicable)
        if results.get("quality_score", 1.0) < 0.8:
            sections.append(self._generate_quality_notes(results))
        
        # 7. Mandatory disclaimer
        sections.append(MANDATORY_DISCLAIMER)
        
        return "\n\n".join(sections)
    
    def _generate_summary(self, results: dict) -> str:
        """Generate opening summary."""
        risk_score = results.get("risk_score", 0) * 100
        confidence = results.get("confidence", 0) * 100
        
        # Determine risk level
        if risk_score < 20:
            level = RiskLevel.LOW
        elif risk_score < 40:
            level = RiskLevel.MODERATE
        elif risk_score < 70:
            level = RiskLevel.HIGH
        else:
            level = RiskLevel.CRITICAL
        
        msg = RISK_LEVEL_MESSAGES[level]
        
        return f"""## Voice Analysis Summary

Your voice has been analyzed using 10+ clinically-validated biomarkers.

**Overall Risk Indicator:** {level.value.upper()} (Score: {risk_score:.0f}/100)
**Analysis Confidence:** {confidence:.0f}%

{msg["summary"]}
"""
    
    def _generate_key_findings(self, results: dict) -> str:
        """Generate key findings section."""
        biomarkers = results.get("biomarkers", {})
        findings = []
        
        # Find abnormal biomarkers
        for key, config in BIOMARKER_EXPLANATIONS.items():
            if key in biomarkers:
                bio = biomarkers[key]
                value = bio.get("value", 0)
                normal_min, normal_max = config.normal_range
                
                # Check if outside range
                if value < normal_min or value > normal_max:
                    findings.append({
                        "name": config.friendly_name,
                        "value": value,
                        "unit": config.unit,
                        "status": "borderline" if abs(value - normal_min) < normal_min * 0.2 else "abnormal"
                    })
        
        if not findings:
            return "### Key Findings\n\nAll voice biomarkers are within normal ranges. :)"
        
        findings_text = "\n".join([
            f"- **{f['name']}**: {f['value']:.2f} {f['unit']} ({f['status']})"
            for f in findings[:5]
        ])
        
        return f"### Key Findings\n\n{findings_text}"
    
    def _generate_condition_section(self, results: dict) -> str:
        """Generate condition-specific explanations."""
        condition_risks = results.get("condition_risks", [])
        elevated = [c for c in condition_risks if c.get("probability", 0) > 0.15]
        
        if not elevated:
            return ""
        
        sections = ["### Patterns Detected\n"]
        
        for cond in elevated:
            condition = cond.get("condition", "")
            if condition in CONDITION_EXPLANATIONS:
                template = CONDITION_EXPLANATIONS[condition]
                explanation = template["pattern_detected"].format(
                    probability=cond.get("probability", 0),
                    confidence=cond.get("confidence", 0),
                    contributing_factors=", ".join(cond.get("contributing_factors", []))
                )
                sections.append(f"**{template['name']}**\n{explanation}\n")
        
        return "\n".join(sections)
    
    def _generate_recommendations(self, results: dict) -> str:
        """Generate recommendations section."""
        recommendations = results.get("recommendations", [])
        
        if not recommendations:
            return "### Next Steps\n\nNo specific recommendations at this time. Continue routine health monitoring."
        
        rec_list = "\n".join([f"- {rec}" for rec in recommendations])
        return f"### Next Steps\n\n{rec_list}"
    
    def _generate_biomarker_details(self, results: dict) -> str:
        """Generate detailed biomarker breakdown."""
        biomarkers = results.get("biomarkers", {})
        
        details = ["### Detailed Biomarkers\n"]
        details.append("| Biomarker | Value | Normal Range | Status |")
        details.append("|-----------|-------|--------------|--------|")
        
        for key, config in BIOMARKER_EXPLANATIONS.items():
            if key in biomarkers:
                bio = biomarkers[key]
                value = bio.get("value", 0)
                normal_min, normal_max = config.normal_range
                
                if value >= normal_min and value <= normal_max:
                    status = "Normal"
                elif value < normal_min:
                    status = "Low"
                else:
                    status = "High"
                
                details.append(
                    f"| {config.friendly_name} | {value:.2f} {config.unit} | "
                    f"{normal_min}-{normal_max} {config.unit} | {status} |"
                )
        
        return "\n".join(details)
    
    def _generate_quality_notes(self, results: dict) -> str:
        """Generate quality notes if applicable."""
        quality = results.get("quality_score", 1.0)
        
        return f"""### Recording Quality Note

{QUALITY_WARNINGS["low_quality"].format(quality=quality)}
"""


# Convenience function for external use
def generate_speech_explanation(
    results: dict,
    include_research: bool = False
) -> str:
    """
    Generate a patient-friendly explanation of speech analysis results.
    
    Args:
        results: EnhancedSpeechAnalysisResponse as dict
        include_research: Whether to include research biomarkers
        
    Returns:
        Markdown-formatted explanation
    """
    generator = SpeechExplanationGenerator(include_research_biomarkers=include_research)
    return generator.generate(results)

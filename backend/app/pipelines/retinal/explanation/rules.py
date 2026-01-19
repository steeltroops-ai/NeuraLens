"""
AI Explanation Rules for Retinal Analysis
Rules and templates for explaining retinal imaging results to users.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class UrgencyLevel(Enum):
    ROUTINE = "routine"  # Annual follow-up
    SOON = "soon"        # 1-3 months
    URGENT = "urgent"    # 1-2 weeks
    EMERGENT = "emergent"  # Immediate


class RiskLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BiomarkerExplanation:
    """Template for explaining a retinal biomarker."""
    name: str
    friendly_name: str
    unit: str
    normal_range: tuple
    
    normal_explanation: str
    borderline_explanation: str
    abnormal_explanation: str
    
    clinical_relevance: str
    conditions_associated: List[str]
    anatomy_explanation: str


# Retinal biomarker explanation templates
BIOMARKER_EXPLANATIONS: Dict[str, BiomarkerExplanation] = {
    "cup_disc_ratio": BiomarkerExplanation(
        name="cup_disc_ratio",
        friendly_name="Cup-to-Disc Ratio",
        unit="ratio",
        normal_range=(0.3, 0.5),
        normal_explanation=(
            "Your optic disc appears healthy. The cup-to-disc ratio of {value:.2f} "
            "is within the normal range, suggesting healthy optic nerve structure."
        ),
        borderline_explanation=(
            "Your cup-to-disc ratio of {value:.2f} is slightly elevated. "
            "This may warrant monitoring at your next eye exam to track any changes."
        ),
        abnormal_explanation=(
            "Your cup-to-disc ratio of {value:.2f} is elevated. This measurement "
            "is important in glaucoma screening. We recommend discussing this finding "
            "with an ophthalmologist for proper evaluation."
        ),
        clinical_relevance="Measures the size of the optic cup relative to the disc. Elevated ratios may indicate glaucoma risk.",
        conditions_associated=["Glaucoma", "Optic nerve damage"],
        anatomy_explanation="The optic disc is where the nerve fibers exit your eye to form the optic nerve."
    ),
    
    "av_ratio": BiomarkerExplanation(
        name="av_ratio",
        friendly_name="Artery-to-Vein Ratio (AV Ratio)",
        unit="ratio",
        normal_range=(0.6, 0.8),
        normal_explanation=(
            "The artery-to-vein ratio of {value:.2f} indicates healthy blood vessel "
            "proportions in your retina. This is a positive sign for vascular health."
        ),
        borderline_explanation=(
            "Your AV ratio of {value:.2f} is slightly outside the typical range. "
            "This may be worth monitoring, especially if you have hypertension."
        ),
        abnormal_explanation=(
            "The AV ratio of {value:.2f} suggests possible vascular changes. "
            "Changes in this ratio can be associated with hypertension. "
            "Discussing this with your doctor, along with blood pressure monitoring, is recommended."
        ),
        clinical_relevance="Compares retinal artery to vein width. Changes may indicate hypertensive or cardiovascular changes.",
        conditions_associated=["Hypertension", "Cardiovascular disease", "Arteriosclerosis"],
        anatomy_explanation="Blood vessels in your retina can reflect your overall vascular health."
    ),
    
    "vessel_tortuosity": BiomarkerExplanation(
        name="vessel_tortuosity",
        friendly_name="Blood Vessel Tortuosity",
        unit="index",
        normal_range=(0.1, 0.3),
        normal_explanation=(
            "Your retinal blood vessels show normal patterns. The tortuosity index "
            "of {value:.2f} indicates healthy vessel structure."
        ),
        borderline_explanation=(
            "Slight increases in vessel tortuosity ({value:.2f}) detected. "
            "This may be a normal variation but is worth noting for future comparisons."
        ),
        abnormal_explanation=(
            "Increased blood vessel tortuosity ({value:.2f}) was detected. "
            "This can sometimes indicate vascular changes that warrant further evaluation. "
            "Consider discussing with an eye care provider."
        ),
        clinical_relevance="Measures how twisted or curved the blood vessels are. Increased tortuosity may indicate vascular pathology.",
        conditions_associated=["Diabetic retinopathy", "Hypertensive retinopathy", "Retinal vein occlusion"],
        anatomy_explanation="Healthy blood vessels in the retina are relatively straight as they branch from the optic disc."
    ),
    
    "rnfl_thickness": BiomarkerExplanation(
        name="rnfl_thickness",
        friendly_name="Retinal Nerve Fiber Layer (RNFL) Thickness",
        unit="microns",
        normal_range=(80, 120),
        normal_explanation=(
            "Your retinal nerve fiber layer thickness of {value:.0f} microns is within "
            "the healthy range, indicating good optic nerve health."
        ),
        borderline_explanation=(
            "Your RNFL thickness of {value:.0f} microns is at the edge of the normal range. "
            "This measurement should be monitored over time to detect any changes."
        ),
        abnormal_explanation=(
            "The RNFL thickness of {value:.0f} microns is thinner than expected. "
            "RNFL thinning can be associated with glaucoma or other optic nerve conditions. "
            "A comprehensive eye exam with OCT imaging is recommended."
        ),
        clinical_relevance="Measures thickness of nerve fiber layer. Thinning may indicate glaucoma or neurological conditions.",
        conditions_associated=["Glaucoma", "Multiple sclerosis", "Optic neuritis", "Neurological conditions"],
        anatomy_explanation="The nerve fiber layer contains the axons of retinal ganglion cells that form the optic nerve."
    ),
    
    "hemorrhage_count": BiomarkerExplanation(
        name="hemorrhage_count",
        friendly_name="Retinal Hemorrhages",
        unit="count",
        normal_range=(0, 0),
        normal_explanation=(
            "No retinal hemorrhages were detected. This is a healthy finding."
        ),
        borderline_explanation=(
            "{value:.0f} small hemorrhage(s) detected. This should be evaluated "
            "by an ophthalmologist to determine the cause and appropriate management."
        ),
        abnormal_explanation=(
            "{value:.0f} retinal hemorrhage(s) detected. Retinal hemorrhages require "
            "prompt evaluation by an ophthalmologist. If you have diabetes, this may "
            "indicate diabetic retinopathy. Please schedule an eye exam soon."
        ),
        clinical_relevance="Bleeding in the retina. May indicate diabetic retinopathy, hypertension, or other conditions.",
        conditions_associated=["Diabetic retinopathy", "Hypertensive retinopathy", "Retinal vein occlusion", "Blood disorders"],
        anatomy_explanation="Hemorrhages are areas of bleeding within the retinal tissue."
    ),
    
    "microaneurysm_count": BiomarkerExplanation(
        name="microaneurysm_count",
        friendly_name="Microaneurysms",
        unit="count",
        normal_range=(0, 0),
        normal_explanation=(
            "No microaneurysms detected. This is a healthy finding."
        ),
        borderline_explanation=(
            "A few microaneurysms ({value:.0f}) detected. This is often the earliest "
            "sign of diabetic retinopathy. Regular monitoring is recommended."
        ),
        abnormal_explanation=(
            "Multiple microaneurysms ({value:.0f}) detected. This finding is consistent "
            "with diabetic eye changes. Please consult an ophthalmologist and ensure "
            "your blood sugar is well-controlled."
        ),
        clinical_relevance="Small bulges in blood vessel walls. Often the earliest sign of diabetic retinopathy.",
        conditions_associated=["Diabetic retinopathy", "Diabetes"],
        anatomy_explanation="Microaneurysms are tiny balloon-like swellings in retinal blood vessel walls."
    ),
}


# Diabetic retinopathy grading
DR_GRADE_EXPLANATIONS = {
    "none": {
        "grade": "No Diabetic Retinopathy",
        "explanation": (
            "No signs of diabetic retinopathy were detected. This is excellent news! "
            "Continue maintaining good blood sugar control and have annual eye exams."
        ),
        "action": "Annual eye exam",
        "urgency": UrgencyLevel.ROUTINE
    },
    "mild_npdr": {
        "grade": "Mild Non-Proliferative Diabetic Retinopathy (NPDR)",
        "explanation": (
            "Early signs of diabetic eye changes detected, including some microaneurysms. "
            "At this stage, vision is typically not affected, but regular monitoring is important."
        ),
        "action": "Eye exam every 9-12 months. Focus on blood sugar and blood pressure control.",
        "urgency": UrgencyLevel.ROUTINE
    },
    "moderate_npdr": {
        "grade": "Moderate Non-Proliferative Diabetic Retinopathy",
        "explanation": (
            "Moderate diabetic eye changes detected, with more extensive retinal findings. "
            "The blood vessels supplying the retina are being affected. Careful monitoring and "
            "optimization of diabetes control is important."
        ),
        "action": "Eye exam every 6 months. Work closely with your diabetes care team.",
        "urgency": UrgencyLevel.SOON
    },
    "severe_npdr": {
        "grade": "Severe Non-Proliferative Diabetic Retinopathy",
        "explanation": (
            "Advanced diabetic eye changes detected. Many blood vessels are blocked, "
            "which increases the risk of progression to proliferative retinopathy. "
            "Close monitoring by an ophthalmologist is essential."
        ),
        "action": "Ophthalmology referral within 1-2 months. May need treatment to prevent progression.",
        "urgency": UrgencyLevel.SOON
    },
    "pdr": {
        "grade": "Proliferative Diabetic Retinopathy (PDR)",
        "explanation": (
            "Advanced diabetic retinopathy with new abnormal blood vessel growth detected. "
            "This stage carries a risk of serious vision loss if untreated. "
            "Prompt treatment can help preserve your vision."
        ),
        "action": "Urgent ophthalmology referral. Treatment (laser or injections) likely needed.",
        "urgency": UrgencyLevel.URGENT
    }
}


# Risk level communication
RISK_LEVEL_MESSAGES = {
    RiskLevel.LOW: {
        "summary": (
            "Your retinal analysis shows healthy findings with low risk indicators. "
            "Continue with routine eye care."
        ),
        "tone": "reassuring",
        "action": "Annual comprehensive eye exam",
        "emoji": ":-)"
    },
    RiskLevel.MODERATE: {
        "summary": (
            "Some findings are worth monitoring. While not necessarily concerning, "
            "discussing these with an eye care provider is recommended."
        ),
        "tone": "informative",
        "action": "Schedule an eye exam within 3-6 months",
        "emoji": ":-|"
    },
    RiskLevel.HIGH: {
        "summary": (
            "Several findings warrant clinical attention. We recommend scheduling "
            "an appointment with an ophthalmologist for thorough evaluation."
        ),
        "tone": "serious but supportive",
        "action": "Ophthalmology consultation within 1-2 months",
        "emoji": ":-("
    },
    RiskLevel.CRITICAL: {
        "summary": (
            "Significant findings detected that require prompt evaluation. "
            "Please contact an eye care provider soon, especially if you notice any vision changes."
        ),
        "tone": "urgent but calm",
        "action": "Urgent ophthalmology referral within 1-2 weeks",
        "emoji": ":-O"
    },
}


# Condition explanations for detected patterns
CONDITION_EXPLANATIONS = {
    "glaucoma_risk": {
        "name": "Glaucoma Risk Indicators",
        "description": "Patterns that may indicate increased glaucoma risk",
        "indicators": ["Elevated cup-to-disc ratio", "RNFL thinning", "Optic disc changes"],
        "action": "Comprehensive glaucoma evaluation by an ophthalmologist",
        "explanation_template": (
            "Some findings suggest possible glaucoma risk factors. Glaucoma is a condition "
            "that affects the optic nerve and can lead to vision loss if untreated. "
            "Early detection through regular eye exams is key to preserving vision."
        )
    },
    "diabetic_retinopathy": {
        "name": "Diabetic Eye Changes",
        "description": "Signs of diabetes affecting the retina",
        "indicators": ["Microaneurysms", "Hemorrhages", "Exudates", "Vessel changes"],
        "action": "Ophthalmology referral + diabetes management optimization",
        "explanation_template": (
            "Signs of diabetic changes in the retina were detected. Diabetic retinopathy "
            "is a common complication of diabetes that, if caught early and managed properly, "
            "can often be controlled to prevent vision loss."
        )
    },
    "hypertensive_changes": {
        "name": "Hypertensive Retinopathy",
        "description": "Signs that high blood pressure is affecting the retina",
        "indicators": ["AV nicking", "Vessel narrowing", "Flame hemorrhages"],
        "action": "Blood pressure monitoring + physician consultation",
        "explanation_template": (
            "Changes consistent with high blood pressure effects on the retina were noted. "
            "The retina is one of the few places where we can directly see blood vessels, "
            "making it a window into your vascular health."
        )
    },
    "macular_changes": {
        "name": "Macular Changes",
        "description": "Findings affecting the central vision area",
        "indicators": ["Drusen", "Pigment changes", "Macular thickness changes"],
        "action": "Macular evaluation by ophthalmologist",
        "explanation_template": (
            "Some changes in the macular area were detected. The macula is responsible "
            "for central and detailed vision. Changes here warrant evaluation to ensure "
            "your central vision remains healthy."
        )
    }
}


# Quality warnings
QUALITY_WARNINGS = {
    "low_quality": (
        "Image quality score: {quality:.0%}. This may affect measurement accuracy. "
        "For best results, images should be taken with proper dilation, focus, and lighting."
    ),
    "poor_visibility": (
        "Some areas of the retina were not clearly visible. A repeat image or "
        "in-person examination may be needed for complete assessment."
    ),
    "media_opacity": (
        "Some opacity (cloudiness) was detected, possibly from cataracts or other factors. "
        "This may affect our ability to see all retinal details clearly."
    )
}


# Mandatory disclaimer
MANDATORY_DISCLAIMER = """
---
IMPORTANT DISCLAIMER: This AI-powered retinal screening is for informational 
purposes only and is NOT a substitute for a comprehensive eye examination by 
a qualified eye care professional. The ability to detect certain conditions 
depends on image quality and other factors. This tool cannot diagnose eye 
diseases - only an ophthalmologist or optometrist can provide diagnosis after 
thorough examination. Regular eye exams are essential for maintaining eye health.
---
"""


class RetinalExplanationGenerator:
    """
    Generates patient-friendly explanations of retinal analysis results.
    """
    
    def __init__(self):
        pass
    
    def generate(
        self,
        results: dict,
        patient_context: Optional[dict] = None
    ) -> str:
        """Generate a complete explanation of retinal analysis results."""
        sections = []
        
        # 1. Opening summary
        sections.append(self._generate_summary(results))
        
        # 2. What we analyzed
        sections.append(self._generate_analysis_overview(results))
        
        # 3. Key findings
        sections.append(self._generate_key_findings(results))
        
        # 4. DR grading if applicable
        if results.get('dr_grade'):
            sections.append(self._generate_dr_section(results))
        
        # 5. Condition patterns
        if results.get('detected_conditions'):
            sections.append(self._generate_condition_section(results))
        
        # 6. Risk assessment
        sections.append(self._generate_risk_assessment(results))
        
        # 7. Recommendations
        sections.append(self._generate_recommendations(results))
        
        # 8. Disclaimer
        sections.append(MANDATORY_DISCLAIMER)
        
        return "\n\n".join(sections)
    
    def _generate_summary(self, results: dict) -> str:
        """Generate opening summary."""
        risk_score = results.get('risk_score', 0) * 100
        
        if risk_score < 20:
            level = RiskLevel.LOW
        elif risk_score < 40:
            level = RiskLevel.MODERATE
        elif risk_score < 70:
            level = RiskLevel.HIGH
        else:
            level = RiskLevel.CRITICAL
        
        msg = RISK_LEVEL_MESSAGES[level]
        
        return f"""## Your Eye Health Summary

Your retinal image has been analyzed using AI-powered screening technology.

**Overall Assessment:** {level.value.upper()}
**Risk Score:** {risk_score:.0f}/100
**Confidence:** {results.get('confidence', 0)*100:.0f}%

{msg["summary"]}
"""
    
    def _generate_analysis_overview(self, results: dict) -> str:
        """Explain what was analyzed."""
        return """### What We Analyzed

A fundus photograph captures the inside of your eye, including the retina, 
optic disc, and blood vessels. This AI analysis evaluates several key 
structures to screen for common eye conditions.
"""
    
    def _generate_key_findings(self, results: dict) -> str:
        """Generate key findings section."""
        biomarkers = results.get('biomarkers', {})
        findings = []
        
        for key, config in BIOMARKER_EXPLANATIONS.items():
            if key in biomarkers:
                bio = biomarkers[key]
                value = bio if isinstance(bio, (int, float)) else bio.get('value', 0)
                normal_min, normal_max = config.normal_range
                
                if value < normal_min or value > normal_max:
                    findings.append({
                        "name": config.friendly_name,
                        "value": value,
                        "unit": config.unit,
                        "explanation": config.abnormal_explanation.format(value=value)
                    })
        
        if not findings:
            return "### Key Findings\n\nAll measured biomarkers are within normal ranges. :)"
        
        findings_text = "\n\n".join([
            f"**{f['name']}**\n{f['explanation']}"
            for f in findings[:4]
        ])
        
        return f"### Key Findings\n\n{findings_text}"
    
    def _generate_dr_section(self, results: dict) -> str:
        """Generate diabetic retinopathy section."""
        dr_grade = results.get('dr_grade', 'none').lower().replace(' ', '_')
        dr_info = DR_GRADE_EXPLANATIONS.get(dr_grade, DR_GRADE_EXPLANATIONS['none'])
        
        return f"""### Diabetic Retinopathy Screening

**Grade:** {dr_info['grade']}

{dr_info['explanation']}

**Recommended Action:** {dr_info['action']}
"""
    
    def _generate_condition_section(self, results: dict) -> str:
        """Generate detected conditions section."""
        conditions = results.get('detected_conditions', [])
        
        if not conditions:
            return ""
        
        sections = ["### Detected Patterns\n"]
        
        for cond_name in conditions:
            if cond_name in CONDITION_EXPLANATIONS:
                info = CONDITION_EXPLANATIONS[cond_name]
                sections.append(f"**{info['name']}**\n{info['explanation_template']}\n")
        
        return "\n".join(sections)
    
    def _generate_risk_assessment(self, results: dict) -> str:
        """Generate risk assessment section."""
        risk_score = results.get('risk_score', 0) * 100
        
        if risk_score < 20:
            assessment = "Low risk - healthy findings"
        elif risk_score < 40:
            assessment = "Moderate - some findings to monitor"
        elif risk_score < 70:
            assessment = "Elevated - clinical evaluation recommended"
        else:
            assessment = "High - prompt evaluation needed"
        
        return f"""### Risk Assessment

**Overall Risk Level:** {assessment}
**Score:** {risk_score:.0f}/100

This score is based on all analyzed biomarkers and detected patterns.
"""
    
    def _generate_recommendations(self, results: dict) -> str:
        """Generate recommendations section."""
        risk_score = results.get('risk_score', 0) * 100
        recommendations = results.get('recommendations', [])
        
        if not recommendations:
            if risk_score < 30:
                recommendations = [
                    "Continue with annual comprehensive eye exams",
                    "Maintain good overall health habits",
                    "Wear UV-protective sunglasses outdoors"
                ]
            elif risk_score < 60:
                recommendations = [
                    "Schedule an eye exam within the next 3-6 months",
                    "Discuss these findings with your eye care provider",
                    "Monitor any changes in your vision"
                ]
            else:
                recommendations = [
                    "Contact an ophthalmologist for evaluation",
                    "Bring this report to your appointment",
                    "Seek prompt attention if you notice vision changes"
                ]
        
        rec_list = "\n".join([f"- {rec}" for rec in recommendations])
        return f"### Recommended Actions\n\n{rec_list}"


def generate_retinal_explanation(
    results: dict,
    patient_context: Optional[dict] = None
) -> str:
    """
    Generate a patient-friendly explanation of retinal analysis results.
    
    Args:
        results: RetinalAnalysisResponse as dict
        patient_context: Optional patient context
        
    Returns:
        Markdown-formatted explanation
    """
    generator = RetinalExplanationGenerator()
    return generator.generate(results, patient_context)

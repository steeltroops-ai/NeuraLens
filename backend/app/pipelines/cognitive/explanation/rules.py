"""
AI Explanation Rules for Cognitive Assessment
Rules and templates for explaining cognitive screening results.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class PerformanceLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    BELOW_AVERAGE = "below_average"
    NEEDS_ATTENTION = "needs_attention"


@dataclass
class DomainExplanation:
    """Template for explaining a cognitive domain."""
    name: str
    friendly_name: str
    description: str
    what_it_measures: str
    good_result: str
    needs_work: str
    improvement_tips: List[str]


DOMAIN_EXPLANATIONS: Dict[str, DomainExplanation] = {
    "memory": DomainExplanation(
        name="memory",
        friendly_name="Memory",
        description="Your ability to store and recall information",
        what_it_measures="How well you remember new information and retrieve stored memories",
        good_result="Your memory performance is strong! You're effectively encoding and retrieving information.",
        needs_work="Your memory score suggests this area might benefit from some targeted exercises.",
        improvement_tips=[
            "Use memory techniques like mnemonics and visualization",
            "Practice active recall rather than passive re-reading",
            "Get adequate sleep - memory consolidation happens during sleep",
            "Try memory games and puzzles",
            "Stay socially active - conversation exercises memory"
        ]
    ),
    
    "attention": DomainExplanation(
        name="attention",
        friendly_name="Attention & Focus",
        description="Your ability to concentrate and maintain focus",
        what_it_measures="How well you can focus on tasks and filter out distractions",
        good_result="Excellent focus abilities! You can concentrate effectively on tasks.",
        needs_work="Your attention score indicates room for improvement in focus and concentration.",
        improvement_tips=[
            "Practice mindfulness meditation",
            "Reduce multitasking - focus on one thing at a time",
            "Take regular short breaks (Pomodoro technique)",
            "Minimize digital distractions",
            "Ensure good sleep and exercise habits"
        ]
    ),
    
    "processing_speed": DomainExplanation(
        name="processing_speed",
        friendly_name="Processing Speed",
        description="How quickly you can process and respond to information",
        what_it_measures="The speed at which you can take in information and react",
        good_result="Your processing speed is excellent - quick and accurate responses!",
        needs_work="Processing speed can be improved with practice and certain lifestyle factors.",
        improvement_tips=[
            "Regular aerobic exercise improves processing speed",
            "Brain training games focused on speed",
            "Ensure adequate sleep",
            "Stay mentally active with new challenges",
            "Maintain good cardiovascular health"
        ]
    ),
    
    "executive_function": DomainExplanation(
        name="executive_function",
        friendly_name="Executive Function",
        description="Planning, decision-making, and mental flexibility",
        what_it_measures="Your ability to plan, organize, and adapt your thinking",
        good_result="Strong executive function - great at planning and problem-solving!",
        needs_work="Executive function skills can be strengthened with practice.",
        improvement_tips=[
            "Learn a new skill or language",
            "Play strategy games like chess",
            "Practice planning and organizing activities",
            "Try puzzles that require flexible thinking",
            "Set goals and track progress"
        ]
    ),
    
    "language": DomainExplanation(
        name="language",
        friendly_name="Language & Verbal Skills",
        description="Word finding, verbal fluency, and communication",
        what_it_measures="How easily you can find words and express yourself verbally",
        good_result="Excellent verbal abilities - strong word finding and expression!",
        needs_work="Verbal fluency can be improved with regular practice.",
        improvement_tips=[
            "Read regularly - variety of topics",
            "Do crossword puzzles and word games",
            "Engage in conversation and discussion",
            "Write regularly - journaling, letters",
            "Learn new vocabulary words"
        ]
    ),
    
    "visuospatial": DomainExplanation(
        name="visuospatial",
        friendly_name="Visual-Spatial Skills",
        description="Understanding and manipulating visual information",
        what_it_measures="Ability to understand space, directions, and visual patterns",
        good_result="Strong visual-spatial skills - good at understanding patterns and space!",
        needs_work="Visual-spatial abilities can be enhanced with targeted practice.",
        improvement_tips=[
            "Try jigsaw puzzles",
            "Practice navigation without GPS occasionally",
            "Draw or sketch regularly",
            "Play spatial games like Tetris",
            "Build things with your hands"
        ]
    ),
}


AGE_CONTEXT_MESSAGES = {
    "young": "Your scores are compared to others in your age group. Cognitive abilities are typically at their peak in early adulthood.",
    "middle": "Cognitive abilities naturally show some changes with age, but most people maintain strong function well into middle age and beyond.",
    "senior": "Some cognitive changes are normal with age. The good news is that many cognitive skills can be maintained or even improved with regular mental and physical exercise."
}


RISK_LEVEL_MESSAGES = {
    "excellent": {
        "summary": "Excellent cognitive performance across domains!",
        "action": "Keep up your brain-healthy lifestyle"
    },
    "good": {
        "summary": "Good cognitive performance - your brain is functioning well.",
        "action": "Continue engaging in mentally stimulating activities"
    },
    "average": {
        "summary": "Your cognitive performance is within typical ranges.",
        "action": "Consider brain-healthy lifestyle enhancements"
    },
    "below_average": {
        "summary": "Some cognitive areas show room for improvement.",
        "action": "Focus on targeted cognitive exercises and lifestyle factors"
    },
    "needs_attention": {
        "summary": "Cognitive scores suggest this may warrant discussion with a healthcare provider.",
        "action": "Consider professional cognitive assessment"
    }
}


BRAIN_HEALTH_TIPS = [
    "**Exercise regularly** - Physical activity improves blood flow to the brain",
    "**Get quality sleep** - Sleep is essential for memory consolidation",
    "**Stay socially active** - Social interaction exercises multiple cognitive skills",
    "**Learn new things** - Novel challenges build cognitive reserve",
    "**Manage stress** - Chronic stress can impair memory and attention",
    "**Eat a brain-healthy diet** - Mediterranean diet, omega-3s, berries, leafy greens",
    "**Limit alcohol** - Excessive alcohol can impair cognitive function",
    "**Protect your head** - Wear helmets, prevent falls",
    "**Manage health conditions** - Control blood pressure, diabetes, and heart health"
]


MANDATORY_DISCLAIMER = """
---
DISCLAIMER: This cognitive screening is for informational and wellness purposes 
only. It is NOT a clinical neuropsychological evaluation and cannot diagnose 
cognitive disorders. Scores can be affected by many factors including fatigue, 
stress, and testing conditions. If you have concerns about your cognitive 
function, please consult a healthcare provider for proper evaluation.
---
"""


class CognitiveExplanationGenerator:
    """Generates explanations for cognitive assessment results."""
    
    def generate(self, results: dict, patient_context: Optional[dict] = None) -> str:
        sections = []
        
        sections.append(self._generate_summary(results))
        sections.append(self._generate_domain_breakdown(results))
        sections.append(self._generate_strengths(results))
        sections.append(self._generate_areas_to_focus(results))
        sections.append(self._generate_brain_health_tips())
        
        if patient_context and patient_context.get('age'):
            sections.append(self._generate_age_context(patient_context['age']))
        
        sections.append(MANDATORY_DISCLAIMER)
        
        return "\n\n".join(sections)
    
    def _generate_summary(self, results: dict) -> str:
        overall = results.get('overall_score', 75)
        
        if overall >= 85:
            level = "excellent"
        elif overall >= 70:
            level = "good"
        elif overall >= 55:
            level = "average"
        elif overall >= 40:
            level = "below average"
        else:
            level = "needs attention"
        
        msg = RISK_LEVEL_MESSAGES.get(level.replace(" ", "_"), RISK_LEVEL_MESSAGES["average"])
        
        return f"""## Your Cognitive Health Summary

Your cognitive assessment has been completed.

**Overall Score:** {overall}/100
**Performance Level:** {level.title()}

{msg["summary"]}
"""
    
    def _generate_domain_breakdown(self, results: dict) -> str:
        domain_scores = results.get('domain_scores', {})
        
        if not domain_scores:
            return "### Domain Breakdown\n\nDetailed domain scores not available."
        
        lines = ["### Domain Breakdown\n"]
        
        for domain, score in domain_scores.items():
            if domain in DOMAIN_EXPLANATIONS:
                info = DOMAIN_EXPLANATIONS[domain]
                status = "Strong" if score >= 70 else "Focus Area" if score < 50 else "Average"
                lines.append(f"- **{info.friendly_name}:** {score}/100 ({status})")
                lines.append(f"  - {info.description}")
        
        return "\n".join(lines)
    
    def _generate_strengths(self, results: dict) -> str:
        domain_scores = results.get('domain_scores', {})
        strengths = [(d, s) for d, s in domain_scores.items() if s >= 70]
        
        if not strengths:
            return "### Your Strengths\n\nKeep working on building cognitive strengths!"
        
        strength_text = []
        for domain, score in sorted(strengths, key=lambda x: -x[1])[:3]:
            if domain in DOMAIN_EXPLANATIONS:
                info = DOMAIN_EXPLANATIONS[domain]
                strength_text.append(f"- **{info.friendly_name}**: {info.good_result}")
        
        return "### Your Strengths\n\n" + "\n".join(strength_text)
    
    def _generate_areas_to_focus(self, results: dict) -> str:
        domain_scores = results.get('domain_scores', {})
        focus_areas = [(d, s) for d, s in domain_scores.items() if s < 60]
        
        if not focus_areas:
            return "### Areas to Focus On\n\nGreat job - no major areas need immediate attention!"
        
        focus_text = []
        for domain, score in sorted(focus_areas, key=lambda x: x[1])[:2]:
            if domain in DOMAIN_EXPLANATIONS:
                info = DOMAIN_EXPLANATIONS[domain]
                tips = "\n".join([f"  - {tip}" for tip in info.improvement_tips[:3]])
                focus_text.append(f"**{info.friendly_name}**\n{info.needs_work}\n\n*Tips to improve:*\n{tips}")
        
        return "### Areas to Focus On\n\n" + "\n\n".join(focus_text)
    
    def _generate_brain_health_tips(self) -> str:
        tips = BRAIN_HEALTH_TIPS[:5]
        return "### Brain Health Tips\n\n" + "\n".join(tips)
    
    def _generate_age_context(self, age: int) -> str:
        if age < 40:
            context = AGE_CONTEXT_MESSAGES["young"]
        elif age < 65:
            context = AGE_CONTEXT_MESSAGES["middle"]
        else:
            context = AGE_CONTEXT_MESSAGES["senior"]
        
        return f"### Age Context\n\n{context}"


def generate_cognitive_explanation(
    results: dict,
    patient_context: Optional[dict] = None
) -> str:
    """Generate patient-friendly explanation of cognitive results."""
    generator = CognitiveExplanationGenerator()
    return generator.generate(results, patient_context)

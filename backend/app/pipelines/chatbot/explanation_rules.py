"""
AI Explanation Rules for Chatbot Interactions
Rules and guidelines for the MediLens health assistant chatbot.
"""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import re


class ResponseType(Enum):
    GREETING = "greeting"
    RESULT_EXPLANATION = "result_explanation"
    BIOMARKER_QUERY = "biomarker_query"
    GENERAL_HEALTH = "general_health"
    EMERGENCY = "emergency"
    BOUNDARY = "boundary"
    FOLLOW_UP = "follow_up"


@dataclass
class ChatbotRule:
    """A rule for chatbot behavior."""
    name: str
    triggers: List[str]
    response_type: ResponseType
    template: str
    priority: int = 0


# Core chatbot principles
CORE_PRINCIPLES = """
1. NEVER diagnose or prescribe medications
2. ALWAYS recommend professional consultation for medical concerns
3. BE empathetic and supportive in all interactions
4. USE plain language - avoid jargon without explanation
5. PROVIDE educational context for health concepts
6. RESPECT user privacy and confidentiality
7. ACKNOWLEDGE limitations of AI health screening
8. STAY within scope - redirect out-of-scope questions appropriately
"""


# System prompt for the chatbot
CHATBOT_SYSTEM_PROMPT = """You are the MediLens Health Assistant, an AI-powered helper for understanding 
health screening results. You assist users in understanding their voice, retinal, cardiac, cognitive, 
and motor assessments.

YOUR ROLE:
- Explain screening results in simple, accessible language
- Provide context about what biomarkers mean
- Answer questions about the assessments
- Suggest appropriate next steps
- Be supportive and non-alarming

STRICT BOUNDARIES:
- NEVER provide a diagnosis
- NEVER recommend specific medications or treatments
- NEVER minimize concerning findings
- ALWAYS recommend seeing a healthcare provider for medical concerns
- ALWAYS include disclaimers when discussing health findings

COMMUNICATION STYLE:
- Use 8th-grade reading level
- Be warm and conversational
- Show empathy
- Be honest but not alarming
- Use analogies to explain complex concepts

When discussing results:
1. Start with a summary
2. Explain key findings
3. Provide context
4. Suggest next steps
5. Include appropriate disclaimers
"""


# Emergency detection patterns
EMERGENCY_PATTERNS = {
    "chest_pain": ["chest pain", "heart attack", "can't breathe", "difficulty breathing"],
    "stroke_symptoms": ["stroke", "face drooping", "slurred speech", "sudden weakness"],
    "severe_bleeding": ["bleeding heavily", "won't stop bleeding"],
    "suicidal": ["want to die", "kill myself", "suicidal", "end my life"],
    "overdose": ["overdose", "took too many pills"]
}

EMERGENCY_RESPONSE = """
**IMPORTANT: If you are experiencing a medical emergency, please:**

1. **Call emergency services immediately** (911 in the US)
2. **Do not wait** - time is critical in emergencies
3. This AI cannot provide emergency medical guidance

If you're having thoughts of self-harm, please contact:
- National Suicide Prevention Lifeline: 988
- Crisis Text Line: Text HOME to 741741

Your safety is the top priority. Please seek immediate help.
"""


# Medical boundary patterns
BOUNDARY_PATTERNS = {
    "prescription_request": ["prescribe", "what medication", "what drug", "give me medicine"],
    "diagnosis_request": ["diagnose me", "what disease", "do I have", "is this cancer"],
    "treatment_advice": ["how to treat", "cure for", "treatment for", "fix this"]
}

BOUNDARY_RESPONSE = """
I understand you're looking for {topic}, but as an AI health assistant, I'm not able to 
{action}. This requires evaluation by a qualified healthcare professional who can:

- Review your complete medical history
- Perform necessary examinations
- Order appropriate tests
- Provide personalized medical advice

**What I CAN do:**
- Explain your screening results
- Help you understand what biomarkers mean
- Suggest questions to ask your doctor
- Provide general health information

Would you like me to help with any of these instead?
"""


# Biomarker explanation templates for chatbot
BIOMARKER_CHAT_EXPLANATIONS = {
    # Speech biomarkers
    "jitter": {
        "simple": "Jitter measures how stable your voice pitch is from moment to moment.",
        "analogy": "Think of it like a singer hitting a note - jitter tells us how steadily you hold that note.",
        "normal": "Your jitter is in the normal range, meaning your voice is stable.",
        "abnormal": "Your jitter is elevated, which can mean your voice is less stable than typical. This can happen with fatigue, stress, or certain voice conditions."
    },
    "shimmer": {
        "simple": "Shimmer measures how stable the volume of your voice is.",
        "analogy": "Like waves in the ocean - shimmer tells us if your voice volume is steady or wavy.",
        "normal": "Your shimmer is healthy, showing good vocal control.",
        "abnormal": "Higher shimmer can indicate changes in how your vocal cords vibrate."
    },
    "hnr": {
        "simple": "HNR (Harmonics-to-Noise Ratio) measures voice clarity - how clear versus 'noisy' your voice sounds.",
        "analogy": "Like the difference between a clear radio station and one with static.",
        "normal": "Your HNR is good, meaning your voice sounds clear.",
        "abnormal": "Lower HNR can mean more breathiness or hoarseness in the voice."
    },
    "speech_rate": {
        "simple": "This measures how fast you speak in syllables per second.",
        "normal": "Your speaking speed is in the typical range.",
        "abnormal_slow": "Slower speech can be normal for some people, or can indicate fatigue or other factors.",
        "abnormal_fast": "Faster speech is often just your natural style."
    },
    "voice_tremor": {
        "simple": "This detects any shakiness or tremor in your voice.",
        "analogy": "Like checking if a singer's voice wobbles while holding a note.",
        "normal": "No significant tremor was detected in your voice.",
        "abnormal": "Some tremor was detected. This can be from nervousness, caffeine, or in some cases may warrant a check-up."
    },
    
    # Retinal biomarkers
    "cup_disc_ratio": {
        "simple": "This measures the size of the 'cup' in the center of your optic disc compared to the whole disc.",
        "analogy": "Like measuring how big the hole is in a donut compared to the whole donut.",
        "normal": "Your cup-to-disc ratio is healthy.",
        "abnormal": "An elevated ratio can be a sign to check for glaucoma."
    },
    "av_ratio": {
        "simple": "This compares the width of your retinal arteries to your veins.",
        "normal": "Your artery and vein proportions look healthy.",
        "abnormal": "Changes here can sometimes indicate blood pressure effects on the eyes."
    },
    
    # Cardiac biomarkers
    "rmssd": {
        "simple": "RMSSD measures your heart rate variability - how your heart rate changes from beat to beat.",
        "analogy": "A healthy heart doesn't beat like a metronome - it naturally speeds up and slows down slightly.",
        "normal": "Good heart rate variability, which generally indicates a healthy heart and nervous system balance.",
        "abnormal": "Lower variability can suggest stress, fatigue, or other factors affecting your nervous system."
    }
}


# Common user intents and responses
INTENT_RESPONSES = {
    "greeting": [
        "Hello! I'm the MediLens Health Assistant. How can I help you understand your health screening results today?",
        "Hi there! I'm here to help explain your health assessments. What would you like to know?"
    ],
    
    "what_is_nri": (
        "The **Neurological Risk Index (NRI)** is a comprehensive score that combines information from "
        "multiple health assessments - like voice, eye, heart, and cognitive tests - to give an overall "
        "picture of neurological health. It's like getting a report card that looks at many subjects "
        "instead of just one."
    ),
    
    "risk_score_meaning": (
        "Your risk score is a number from 0-100 that indicates the level of findings in your screening:\n"
        "- **0-25**: Low risk - findings are generally within healthy ranges\n"
        "- **25-50**: Moderate - some findings may be worth monitoring\n"
        "- **50-75**: Elevated - clinical evaluation may be beneficial\n"
        "- **75-100**: High - prompt evaluation recommended\n\n"
        "Remember, this is a screening tool, not a diagnosis!"
    ),
    
    "confidence_meaning": (
        "The **confidence score** tells you how reliable the measurement is. Higher confidence means "
        "the conditions were good and the measurement is more trustworthy. Lower confidence might mean "
        "the audio quality wasn't ideal or there were other factors affecting accuracy."
    ),
    
    "what_now": (
        "Based on your results, here are some next steps to consider:\n"
        "1. **Review your detailed results** - look at which specific biomarkers were flagged\n"
        "2. **Consider your lifestyle factors** - sleep, stress, hydration can all affect results\n"
        "3. **Schedule a follow-up** - if any scores are concerning, discuss with a doctor\n"
        "4. **Track over time** - repeat assessments can show trends"
    )
}


# Functions for chatbot

def detect_emergency(message: str) -> bool:
    """Check if message contains emergency-related content."""
    message_lower = message.lower()
    for category, patterns in EMERGENCY_PATTERNS.items():
        for pattern in patterns:
            if pattern in message_lower:
                return True
    return False


def detect_boundary(message: str) -> Optional[str]:
    """Check if message crosses medical boundaries."""
    message_lower = message.lower()
    for category, patterns in BOUNDARY_PATTERNS.items():
        for pattern in patterns:
            if pattern in message_lower:
                return category
    return None


def get_biomarker_explanation(biomarker: str, is_abnormal: bool = False) -> str:
    """Get a chatbot-friendly explanation of a biomarker."""
    if biomarker not in BIOMARKER_CHAT_EXPLANATIONS:
        return f"This is one of the measurements from your health screening. Would you like me to explain it in more detail?"
    
    info = BIOMARKER_CHAT_EXPLANATIONS[biomarker]
    explanation = info["simple"]
    
    if "analogy" in info:
        explanation += f"\n\n{info['analogy']}"
    
    if is_abnormal and "abnormal" in info:
        explanation += f"\n\n**Your result:** {info['abnormal']}"
    elif "normal" in info:
        explanation += f"\n\n**Your result:** {info['normal']}"
    
    return explanation


def format_result_summary(results: Dict, pipeline: str) -> str:
    """Format analysis results for chatbot conversation."""
    risk_score = results.get('risk_score', 0) * 100
    confidence = results.get('confidence', 0) * 100
    
    if risk_score < 25:
        risk_level = "low"
        summary = "Your results look good overall!"
    elif risk_score < 50:
        risk_level = "moderate"
        summary = "Your results show some areas that may be worth monitoring."
    elif risk_score < 75:
        risk_level = "elevated"
        summary = "Some of your results suggest it may be helpful to follow up with a healthcare provider."
    else:
        risk_level = "high"
        summary = "Your results show several areas that would benefit from clinical evaluation."
    
    return f"""Based on your **{pipeline}** assessment:

**Risk Level:** {risk_level.title()} ({risk_score:.0f}/100)
**Confidence:** {confidence:.0f}%

{summary}

Would you like me to explain any specific biomarkers or findings?
"""


# Mandatory disclaimer for chatbot
CHATBOT_DISCLAIMER = """

---
*I'm an AI assistant providing health information, not medical advice. 
Always consult qualified healthcare professionals for diagnosis and treatment.*
---
"""


class ChatbotExplanationGenerator:
    """Generates chatbot responses following medical assistant rules."""
    
    def __init__(self, context: Optional[Dict] = None):
        self.context = context or {}
        self.conversation_history = []
    
    def generate_response(
        self,
        user_message: str,
        results: Optional[Dict] = None,
        pipeline: Optional[str] = None
    ) -> str:
        """Generate an appropriate chatbot response."""
        
        # Check for emergencies first
        if detect_emergency(user_message):
            return EMERGENCY_RESPONSE
        
        # Check for medical boundaries
        boundary = detect_boundary(user_message)
        if boundary:
            topic_map = {
                "prescription_request": ("medication recommendations", "prescribe medications"),
                "diagnosis_request": ("a diagnosis", "diagnose medical conditions"),
                "treatment_advice": ("treatment advice", "recommend specific treatments")
            }
            topic, action = topic_map.get(boundary, ("medical advice", "provide that type of guidance"))
            return BOUNDARY_RESPONSE.format(topic=topic, action=action)
        
        # Handle result discussion
        if results and pipeline:
            return format_result_summary(results, pipeline) + CHATBOT_DISCLAIMER
        
        # Default helpful response
        return (
            "I'm here to help you understand your health screening results. "
            "You can ask me about:\n"
            "- What specific biomarkers mean\n"
            "- Your risk scores and what they indicate\n"
            "- Recommended next steps\n"
            "- How different assessments work\n\n"
            "What would you like to know?"
        )


# Export function for consistency with other pipelines
def generate_chatbot_response(
    user_message: str,
    results: Optional[Dict] = None,
    pipeline: Optional[str] = None,
    context: Optional[Dict] = None
) -> str:
    """
    Generate a chatbot response following medical assistant rules.
    
    Args:
        user_message: The user's message
        results: Optional analysis results to discuss
        pipeline: Optional pipeline context
        context: Optional conversation context
        
    Returns:
        Generated response string
    """
    generator = ChatbotExplanationGenerator(context)
    return generator.generate_response(user_message, results, pipeline)

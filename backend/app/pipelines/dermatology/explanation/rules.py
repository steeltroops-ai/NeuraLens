"""
Dermatology Pipeline Explanation Rules

Rules for generating explanations.
"""

from typing import Dict, Any, List


# =============================================================================
# BIOMARKER EXPLANATIONS
# =============================================================================

BIOMARKER_EXPLANATIONS = {
    "asymmetry": {
        "name": "Asymmetry",
        "normal": "Lesion appears symmetric along both axes, which is a reassuring finding.",
        "concerning": "Lesion shows asymmetry, meaning one half does not mirror the other. This is one of the key features dermatologists look for.",
        "critical": "Significant asymmetry detected - this is an important warning sign that warrants professional evaluation."
    },
    "border": {
        "name": "Border Irregularity",
        "normal": "The lesion has a smooth, well-defined border, which is typical of benign lesions.",
        "concerning": "The border appears irregular with notches or uneven edges. Irregular borders can indicate atypical cell growth.",
        "critical": "Highly irregular border with multiple notches or poorly defined edges - this requires professional evaluation."
    },
    "color": {
        "name": "Color Variation",
        "normal": "Uniform color distribution within the lesion, typical of benign moles.",
        "concerning": "Multiple colors detected within the lesion. Color variation can indicate varying depths of pigment cells.",
        "critical": "Multiple concerning colors detected including potential blue-white veil - this is a significant finding."
    },
    "diameter": {
        "name": "Diameter",
        "normal": "Lesion diameter is within normal range (less than 6mm).",
        "concerning": "Lesion exceeds 6mm in diameter. While many benign lesions can be larger, size is one of the factors considered in evaluation.",
        "critical": "Large lesion size warrants professional evaluation, especially in combination with other concerning features."
    },
    "evolution": {
        "name": "Evolution",
        "normal": "No significant changes detected or reported.",
        "concerning": "Texture patterns suggest potential recent changes. Any evolving lesion should be monitored.",
        "critical": "Indicators of rapid change detected - evolving lesions are among the most important warning signs."
    }
}


# =============================================================================
# RISK LEVEL MESSAGES
# =============================================================================

RISK_LEVEL_MESSAGES = {
    1: {
        "title": "Critical Risk - Immediate Attention Required",
        "description": "This lesion shows multiple features that require urgent medical evaluation.",
        "urgency": "Please seek dermatology evaluation within 24-48 hours.",
        "icon": "critical"
    },
    2: {
        "title": "High Risk - Prompt Attention Needed",
        "description": "This lesion has concerning features that should be evaluated promptly.",
        "urgency": "Please schedule a dermatology appointment within 1-2 weeks.",
        "icon": "warning"
    },
    3: {
        "title": "Moderate Risk - Evaluation Recommended",
        "description": "This lesion shows some features that warrant professional evaluation.",
        "urgency": "We recommend scheduling a dermatology consultation within 1-3 months.",
        "icon": "caution"
    },
    4: {
        "title": "Low Risk - Routine Monitoring",
        "description": "This lesion appears low risk but should continue to be monitored.",
        "urgency": "Consider evaluation during your next routine skin check.",
        "icon": "info"
    },
    5: {
        "title": "Benign Appearance",
        "description": "This lesion shows characteristics consistent with benign lesions.",
        "urgency": "Continue routine skin health monitoring.",
        "icon": "success"
    }
}


# =============================================================================
# SUBTYPE DESCRIPTIONS
# =============================================================================

SUBTYPE_DESCRIPTIONS = {
    "melanoma": {
        "name": "Melanoma",
        "description": "A serious form of skin cancer that develops from pigment-producing cells (melanocytes).",
        "severity": "high",
        "action": "Requires immediate professional evaluation"
    },
    "basal_cell_carcinoma": {
        "name": "Basal Cell Carcinoma",
        "description": "The most common type of skin cancer, typically slow-growing and rarely spreads.",
        "severity": "moderate",
        "action": "Requires professional evaluation for treatment"
    },
    "squamous_cell_carcinoma": {
        "name": "Squamous Cell Carcinoma",
        "description": "A common type of skin cancer that can spread if not treated.",
        "severity": "moderate",
        "action": "Requires professional evaluation for treatment"
    },
    "actinic_keratosis": {
        "name": "Actinic Keratosis",
        "description": "A rough, scaly patch caused by sun damage. Can potentially develop into skin cancer.",
        "severity": "low",
        "action": "Monitor and consider treatment to prevent progression"
    },
    "benign_keratosis": {
        "name": "Benign Keratosis",
        "description": "A non-cancerous skin growth, such as a seborrheic keratosis.",
        "severity": "minimal",
        "action": "No immediate action required"
    },
    "dermatofibroma": {
        "name": "Dermatofibroma",
        "description": "A benign skin nodule of unknown cause, often appearing on the legs.",
        "severity": "minimal",
        "action": "No treatment necessary unless symptomatic"
    },
    "nevus": {
        "name": "Nevus (Mole)",
        "description": "A common benign growth of pigment cells, typically present from birth or early childhood.",
        "severity": "minimal",
        "action": "Monitor for changes"
    },
    "vascular_lesion": {
        "name": "Vascular Lesion",
        "description": "A skin lesion composed of blood vessels, such as a hemangioma or cherry angioma.",
        "severity": "minimal",
        "action": "No treatment necessary unless for cosmetic reasons"
    }
}


# =============================================================================
# QUALITY WARNINGS
# =============================================================================

QUALITY_WARNINGS = {
    "blur": {
        "message": "Image blur detected which may affect analysis accuracy.",
        "recommendation": "For best results, retake the photo with the lesion in sharp focus."
    },
    "lighting": {
        "message": "Lighting conditions may have affected the analysis.",
        "recommendation": "Use even, diffuse lighting for optimal results."
    },
    "hair": {
        "message": "Hair occlusion detected over the lesion area.",
        "recommendation": "Consider removing hair around the lesion for clearer imaging."
    },
    "low_confidence": {
        "message": "Analysis confidence is lower than optimal.",
        "recommendation": "Consider retaking with better image quality or seek professional evaluation."
    }
}


def get_biomarker_explanation(biomarker: str, level: str) -> str:
    """Get explanation for a biomarker at given level."""
    if biomarker in BIOMARKER_EXPLANATIONS:
        return BIOMARKER_EXPLANATIONS[biomarker].get(level, "")
    return ""


def get_risk_message(tier: int) -> Dict[str, str]:
    """Get message for risk tier."""
    return RISK_LEVEL_MESSAGES.get(tier, RISK_LEVEL_MESSAGES[5])


def get_subtype_info(subtype: str) -> Dict[str, str]:
    """Get information about lesion subtype."""
    return SUBTYPE_DESCRIPTIONS.get(subtype, {
        "name": subtype.replace("_", " ").title(),
        "description": "Skin lesion requiring evaluation.",
        "severity": "unknown",
        "action": "Consult a dermatologist"
    })

"""
Cardiology Explanation Rules (Legacy Compatibility)

This file provides backward compatibility for the explanation system.
The main implementation is in explanation/rules.py following the new architecture.
"""

# Re-export from the new architecture location
from .explanation.rules import (
    BIOMARKER_EXPLANATIONS,
    RISK_LEVEL_MESSAGES,
    LIFESTYLE_RECOMMENDATIONS,
    MANDATORY_DISCLAIMER,
    generate_cardiology_explanation,
    CardiologyExplanationGenerator,
)

__all__ = [
    "BIOMARKER_EXPLANATIONS",
    "RISK_LEVEL_MESSAGES",
    "LIFESTYLE_RECOMMENDATIONS",
    "MANDATORY_DISCLAIMER",
    "generate_cardiology_explanation",
    "CardiologyExplanationGenerator",
]

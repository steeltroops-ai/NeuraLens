"""
Retinal Pipeline - Explanation Module

AI explanation rules for retinal analysis results.

v5.0 additions:
- Enhanced explainability stack (Grad-CAM, region analysis)
- Biomarker importance scoring
- Clinical narrative generation
"""

from .rules import (
    UrgencyLevel,
    RiskLevel,
    BiomarkerExplanation,
    BIOMARKER_EXPLANATIONS,
    DR_GRADE_EXPLANATIONS,
    RISK_LEVEL_MESSAGES,
    CONDITION_EXPLANATIONS,
    QUALITY_WARNINGS,
    MANDATORY_DISCLAIMER,
    RetinalExplanationGenerator,
    generate_retinal_explanation,
)

# v5.0 Enhanced explainability
from .explainer import (
    RetinalExplainer,
    ExplanationResult,
    RegionContribution,
    BiomarkerImportance,
    GradCAMGenerator,
    RegionContributionAnalyzer,
    BiomarkerImportanceAnalyzer,
    ClinicalNarrativeGenerator,
    AnatomicalRegions,
    retinal_explainer,
)

# Backward compatibility alias
RETINAL_EXPLANATION_RULES = {
    "biomarkers": BIOMARKER_EXPLANATIONS,
    "dr_grades": DR_GRADE_EXPLANATIONS,
    "risk_levels": RISK_LEVEL_MESSAGES,
    "conditions": CONDITION_EXPLANATIONS,
    "quality_warnings": QUALITY_WARNINGS,
}

__all__ = [
    # Rules-based explanation
    'UrgencyLevel',
    'RiskLevel',
    'BiomarkerExplanation',
    'BIOMARKER_EXPLANATIONS',
    'DR_GRADE_EXPLANATIONS',
    'RISK_LEVEL_MESSAGES',
    'CONDITION_EXPLANATIONS',
    'QUALITY_WARNINGS',
    'MANDATORY_DISCLAIMER',
    'RetinalExplanationGenerator',
    'generate_retinal_explanation',
    'RETINAL_EXPLANATION_RULES',
    
    # v5.0 Enhanced Explainability
    'RetinalExplainer',
    'ExplanationResult',
    'RegionContribution',
    'BiomarkerImportance',
    'GradCAMGenerator',
    'RegionContributionAnalyzer',
    'BiomarkerImportanceAnalyzer',
    'ClinicalNarrativeGenerator',
    'AnatomicalRegions',
    'retinal_explainer',
]

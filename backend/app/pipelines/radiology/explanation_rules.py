"""
Radiology Explanation Rules (Legacy Compatibility)

This file provides backward compatibility for the explanation system.
The main implementation is in explanation/rules.py following the new architecture.
"""

# Re-export from the new architecture location
from .explanation.rules import RadiologyExplanationRules

# Legacy module-level exports for backward compatibility
CONDITION_EXPLANATIONS = RadiologyExplanationRules.CONDITION_EXPLANATIONS
RISK_TEMPLATES = RadiologyExplanationRules.RISK_TEMPLATES

# Mandatory disclaimer
MANDATORY_DISCLAIMER = """
IMPORTANT: This AI analysis is for informational screening purposes only. 
It is NOT a medical diagnosis and should NOT replace professional radiologist interpretation.
Always consult a qualified radiologist or physician for clinical decisions.
This tool is FDA Class II pending - for research and educational use only.
"""

# Risk level messages
RISK_LEVEL_MESSAGES = {
    'normal': {
        'summary': 'No significant abnormalities detected in your chest X-ray.',
        'action': 'Continue routine health monitoring as directed by your physician.'
    },
    'low': {
        'summary': 'Minor findings that are likely normal variants.',
        'action': 'No immediate action required. Follow routine screening schedule.'
    },
    'moderate': {
        'summary': 'Some findings that may warrant clinical attention.',
        'action': 'Consider follow-up with your physician for evaluation.'
    },
    'high': {
        'summary': 'Significant findings detected that require medical attention.',
        'action': 'Consult with a healthcare provider for further evaluation.'
    },
    'critical': {
        'summary': 'Critical findings detected requiring immediate attention.',
        'action': 'Seek prompt medical evaluation. Contact physician immediately.'
    }
}

# Biomarker/condition explanations for the explanation system
BIOMARKER_EXPLANATIONS = {
    'Pneumonia': {
        'friendly_name': 'Pneumonia',
        'clinical_relevance': 'Lung infection causing inflammation and fluid in air sacs',
        'normal_finding': 'No consolidation or infiltrates detected',
        'abnormal_implications': 'May require antibiotic treatment and follow-up imaging'
    },
    'Cardiomegaly': {
        'friendly_name': 'Enlarged Heart',
        'clinical_relevance': 'Heart appears larger than normal on imaging',
        'normal_finding': 'Cardiothoracic ratio within normal limits (<0.5)',
        'abnormal_implications': 'May indicate underlying cardiac condition'
    },
    'Effusion': {
        'friendly_name': 'Pleural Effusion',
        'clinical_relevance': 'Fluid accumulation around the lungs',
        'normal_finding': 'No fluid in pleural space',
        'abnormal_implications': 'May require drainage and cause investigation'
    },
    'Pneumothorax': {
        'friendly_name': 'Collapsed Lung',
        'clinical_relevance': 'Air in the pleural space causing lung collapse',
        'normal_finding': 'Lungs fully expanded',
        'abnormal_implications': 'May require immediate intervention (chest tube)'
    },
    'Nodule': {
        'friendly_name': 'Lung Nodule',
        'clinical_relevance': 'Small rounded opacity in the lung',
        'normal_finding': 'No pulmonary nodules detected',
        'abnormal_implications': 'May require CT follow-up per Fleischner guidelines'
    },
    'Mass': {
        'friendly_name': 'Lung Mass',
        'clinical_relevance': 'Large opacity (>3cm) in the lung',
        'normal_finding': 'No lung masses detected',
        'abnormal_implications': 'Requires CT and possible biopsy for characterization'
    },
    'Consolidation': {
        'friendly_name': 'Lung Consolidation',
        'clinical_relevance': 'Dense area where air spaces filled with fluid/cells',
        'normal_finding': 'Lungs clear without consolidation',
        'abnormal_implications': 'Often indicates infection or inflammation'
    },
    'Atelectasis': {
        'friendly_name': 'Lung Collapse (Partial)',
        'clinical_relevance': 'Partial deflation of the lung',
        'normal_finding': 'Lungs fully inflated',
        'abnormal_implications': 'May indicate obstruction or post-surgical changes'
    }
}

# Quality warnings
QUALITY_WARNINGS = {
    'low_resolution': 'Image resolution may affect detection accuracy.',
    'poor_positioning': 'Suboptimal positioning may obscure some findings.',
    'motion_artifact': 'Motion blur detected which may affect analysis.',
    'underexposure': 'Image appears underexposed affecting visualization.'
}


def generate_radiology_explanation(results: dict, patient_context: dict = None) -> str:
    """
    Generate a structured explanation for radiology results.
    
    Args:
        results: Analysis results from the radiology pipeline
        patient_context: Optional patient demographics and history
    
    Returns:
        Formatted explanation string
    """
    rules = RadiologyExplanationRules()
    
    primary = results.get('primary_finding', {})
    risk_level = results.get('risk_level', 'low')
    findings = results.get('findings', [])
    quality = results.get('quality_score', 1.0)
    
    explanation = rules.generate_explanation(
        primary_finding=primary.get('condition', 'No Significant Abnormality') if isinstance(primary, dict) else str(primary),
        probability=primary.get('probability', 0) if isinstance(primary, dict) else 0,
        risk_level=risk_level,
        findings=findings,
        quality_score=quality
    )
    
    return explanation.get('overview', 'Analysis complete.')

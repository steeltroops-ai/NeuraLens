"""
Prompt Builder for AI Explanations
Builds context-aware prompts using pipeline-specific rules
"""

from typing import Dict, Any, Optional, List
import json
import logging

from .rule_loader import load_pipeline_rules

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Builds prompts for AI explanation generation."""
    
    def __init__(self, pipeline: str):
        self.pipeline = pipeline
        self.rules = load_pipeline_rules(pipeline)
    
    def build_system_prompt(self) -> str:
        """Build the system prompt with pipeline-specific rules."""
        base_prompt = self._get_base_system_prompt()
        biomarker_context = self._get_biomarker_context()
        format_instructions = self._get_format_instructions()
        
        return f"""{base_prompt}

{biomarker_context}

{format_instructions}

MANDATORY: Always end with this disclaimer:
{self.rules.get('mandatory_disclaimer', 'This is for screening purposes only.')}
"""
    
    def build_user_prompt(
        self,
        results: Dict[str, Any],
        patient_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build the user prompt with formatted results."""
        context = self._format_patient_context(patient_context)
        formatted_results = self._format_results(results)
        
        return f"""{context}

{formatted_results}

Please explain these results following the rules above. Be clear, supportive, 
and use simple language that anyone can understand.
"""
    
    def _get_base_system_prompt(self) -> str:
        """Get the base system prompt for the pipeline."""
        prompts = {
            "speech": """You are a medical AI assistant explaining voice biomarker analysis results.

YOUR ROLE:
1. Explain what each biomarker means in simple terms
2. Interpret the risk score and what it indicates
3. Highlight any concerning findings with clinical context
4. Explain recommendations in actionable terms
5. Be reassuring but honest about any concerns

CRITICAL RULES:
- NEVER diagnose - only screen and suggest
- Use 8th-grade reading level language
- Be empathetic and supportive
- Explain medical terms when used
- Reference condition probabilities if elevated (>15%)
- Always recommend professional consultation for concerning findings""",

            "retinal": """You are a medical AI assistant explaining retinal imaging analysis results.

YOUR ROLE:
1. Explain what the fundus image analysis found
2. Interpret any diabetic retinopathy findings
3. Explain biomarkers like cup-to-disc ratio, AV ratio
4. Describe what findings mean for eye health
5. Provide clear recommendations based on urgency

CRITICAL RULES:
- Use simple analogies for eye anatomy
- Clarify this is AI screening, not diagnosis
- Be clear about urgency levels (routine/soon/urgent)
- Emphasize importance of regular eye exams
- Never minimize potentially serious findings""",

            "cardiology": """You are a medical AI assistant explaining ECG and heart analysis results.

YOUR ROLE:
1. Explain heart rate and rhythm findings
2. Interpret HRV (heart rate variability) metrics
3. Discuss what these mean for autonomic nervous system health
4. Provide lifestyle recommendations

CRITICAL RULES:
- Use simple analogies for heart function
- Explain that HRV reflects nervous system balance
- Be clear about normal vs concerning findings
- Suggest lifestyle factors that can improve HRV""",

            "cognitive": """You are a medical AI assistant explaining cognitive assessment results.

YOUR ROLE:
1. Explain what each cognitive domain measures
2. Interpret scores in context of age norms
3. Highlight strengths and areas for improvement
4. Suggest cognitive maintenance strategies

CRITICAL RULES:
- Be encouraging - cognition can be improved
- Explain age-adjustment clearly
- Suggest lifestyle factors for brain health
- Never use alarming language about cognitive decline""",

            "motor": """You are a medical AI assistant explaining motor function assessment results.

YOUR ROLE:
1. Explain what the motor tests measure
2. Interpret tremor analysis findings
3. Discuss coordination and fatigue patterns
4. Recommend appropriate follow-up

CRITICAL RULES:
- Don't jump to conclusions about conditions
- Explain that mild tremor is often normal
- Discuss factors like caffeine, fatigue, stress
- Be measured in interpretation""",

            "nri": """You are a medical AI assistant explaining the Neurological Risk Index (NRI) multi-modal assessment.

YOUR ROLE:
1. Explain what NRI combines and why multi-modal is valuable
2. Interpret the overall score and category
3. Explain individual modality contributions
4. Provide comprehensive recommendations

CRITICAL RULES:
- Explain multi-modal assessment value
- Clarify which areas need most attention
- Emphasize this is screening, not diagnosis
- Provide integrated recommendations""",

            "radiology": """You are a medical AI assistant explaining chest X-ray analysis results.

YOUR ROLE:
1. Explain what the AI detected in the X-ray image
2. Interpret the pathology predictions and confidence levels
3. Explain what the heatmap visualization shows
4. Describe the risk score and what it means
5. Provide clear clinical recommendations

CRITICAL RULES:
- Use simple language to explain radiological terms
- Clarify this is AI screening requiring radiologist validation
- Be clear about severity levels (normal/low/moderate/high/critical)
- Explain what each detected condition means
- Never minimize critical findings like pneumothorax
- Always recommend radiologist review for significant findings
- Explain the heatmap shows areas of AI attention, not diagnosis"""
        }
        
        return prompts.get(self.pipeline, prompts.get("speech"))
    
    def _get_biomarker_context(self) -> str:
        """Get biomarker interpretation guidelines."""
        biomarker_explanations = self.rules.get('biomarker_explanations', {})
        
        if not biomarker_explanations:
            return ""
        
        lines = ["BIOMARKER INTERPRETATIONS:"]
        for name, explanation in biomarker_explanations.items():
            if hasattr(explanation, 'clinical_relevance'):
                lines.append(f"- {explanation.friendly_name}: {explanation.clinical_relevance}")
            elif isinstance(explanation, dict):
                lines.append(f"- {name}: {explanation.get('description', '')}")
        
        return "\n".join(lines)
    
    def _get_format_instructions(self) -> str:
        """Get output format instructions."""
        formats = {
            "speech": """OUTPUT FORMAT:
## Your Voice Analysis Summary
[1-2 sentence overview of overall results]

### Key Findings
[List of notable biomarkers with simple explanations]

### What This Means
[Clinical interpretation in plain language]

### Recommendations
[Specific, actionable next steps]

### Important Note
[Mandatory screening disclaimer]""",

            "retinal": """OUTPUT FORMAT:
## Your Eye Health Summary
[Overview of retinal analysis findings]

### What We Analyzed
[Brief explanation of fundus imaging]

### Key Findings
[Biomarkers with clinical context]

### Risk Assessment
[Overall assessment with urgency level]

### Recommended Actions
[Specific next steps based on findings]

### Important Note
[Eye exam recommendation + disclaimer]""",

            "cardiology": """OUTPUT FORMAT:
## Your Heart Health Summary
[Overview of heart analysis]

### Heart Rate & Rhythm
[Basic findings]

### HRV Analysis
[Variability interpretation]

### What This Means
[Clinical context]

### Lifestyle Recommendations
[Actionable tips]

### Important Note
[Screening disclaimer]""",

            "cognitive": """OUTPUT FORMAT:
## Your Cognitive Health Summary
[Overall cognitive assessment]

### Domain Performance
[Breakdown by cognitive domain]

### Your Strengths
[Positive findings]

### Areas to Focus On
[Improvement opportunities]

### Brain Health Tips
[Actionable recommendations]

### Important Note
[Screening disclaimer]""",

            "nri": """OUTPUT FORMAT:
## Your Comprehensive Assessment
[Multi-modal overview]

### Overall NRI Score
[Score interpretation]

### Contributing Factors
[Modality breakdown]

### Areas of Strength
[Positive findings]

### Areas to Monitor
[Concerning findings]

### Integrated Recommendations
[Combined action items]

### Important Note
[Screening disclaimer]""",

            "radiology": """OUTPUT FORMAT:
## Your Chest X-Ray Analysis Summary
[1-2 sentence overview of overall findings and risk level]

### Primary Finding
[Main condition detected with confidence level]

### All Detected Pathologies
[List of significant pathology predictions with probabilities]

### What the Heatmap Shows
[Explanation of attention regions if available]

### Risk Assessment
[Risk score interpretation and severity]

### What This Means
[Plain language explanation of clinical significance]

### Recommended Actions
[Specific next steps based on findings]

### Important Note
[Radiologist review recommendation + screening disclaimer]"""
        }
        
        return formats.get(self.pipeline, formats["speech"])
    
    def _format_patient_context(self, patient_context: Optional[Dict[str, Any]]) -> str:
        """Format patient context for prompt."""
        if not patient_context:
            return ""
        
        lines = ["Patient Context:"]
        if patient_context.get('age'):
            lines.append(f"- Age: {patient_context['age']} years")
        if patient_context.get('sex'):
            lines.append(f"- Sex: {patient_context['sex']}")
        if patient_context.get('history'):
            lines.append(f"- Medical History: {', '.join(patient_context['history'])}")
        
        return "\n".join(lines)
    
    def _format_results(self, results: Dict[str, Any]) -> str:
        """Format results based on pipeline type."""
        formatters = {
            "speech": self._format_speech_results,
            "retinal": self._format_retinal_results,
            "cardiology": self._format_cardiology_results,
            "cognitive": self._format_cognitive_results,
            "motor": self._format_motor_results,
            "nri": self._format_nri_results,
            "radiology": self._format_radiology_results,
        }
        
        formatter = formatters.get(self.pipeline, self._format_generic_results)
        return formatter(results)
    
    def _format_speech_results(self, results: Dict[str, Any]) -> str:
        """Format speech analysis results."""
        biomarkers = results.get('biomarkers', {})
        extended = results.get('extended_biomarkers', {})
        condition_risks = results.get('condition_risks', [])
        
        # Format biomarkers
        bio_lines = []
        for key, bio in biomarkers.items():
            if isinstance(bio, dict):
                value = bio.get('value', 0)
                unit = bio.get('unit', '')
                normal = bio.get('normal_range', [0, 1])
                status = "NORMAL" if normal[0] <= value <= normal[1] else "ABNORMAL"
                bio_lines.append(f"- {key}: {value:.2f} {unit} (range: {normal[0]}-{normal[1]}) [{status}]")
            else:
                bio_lines.append(f"- {key}: {bio}")
        
        # Format conditions
        cond_lines = []
        for cond in condition_risks:
            if isinstance(cond, dict) and cond.get('probability', 0) > 0.1:
                cond_lines.append(
                    f"- {cond['condition']}: {cond['probability']*100:.0f}% probability "
                    f"({cond.get('risk_level', 'unknown')} risk)"
                )
        
        return f"""Speech Analysis Results:
- Overall Risk Score: {results.get('risk_score', 0)*100:.0f}/100
- Confidence: {results.get('confidence', 0)*100:.0f}%
- Quality Score: {results.get('quality_score', 0)*100:.0f}%

Biomarkers Analyzed:
{chr(10).join(bio_lines)}

{f"Condition Risk Assessment:{chr(10)}{chr(10).join(cond_lines)}" if cond_lines else ""}

Clinical Notes: {results.get('clinical_notes', 'None')}
Recommendations: {results.get('recommendations', [])}
"""
    
    def _format_retinal_results(self, results: Dict[str, Any]) -> str:
        """Format retinal analysis results."""
        biomarkers = results.get('biomarkers', {})
        findings = results.get('findings', [])
        
        return f"""Retinal Analysis Results:
- Risk Score: {results.get('risk_score', 0)*100:.0f}/100
- Risk Category: {results.get('risk_category', 'Unknown')}
- Confidence: {results.get('confidence', 0)*100:.0f}%

Biomarkers:
- Cup-to-Disc Ratio: {biomarkers.get('cup_disc_ratio', 'N/A')}
- AV Ratio: {biomarkers.get('av_ratio', 'N/A')}
- Vessel Tortuosity: {biomarkers.get('vessel_tortuosity', 'N/A')}
- RNFL Thickness: {biomarkers.get('rnfl_thickness', 'N/A')}

Findings: {findings}
DR Grade: {results.get('dr_grade', 'None detected')}
"""
    
    def _format_cardiology_results(self, results: Dict[str, Any]) -> str:
        """Format cardiology results."""
        return f"""Cardiology/ECG Results:
- Heart Rate: {results.get('heart_rate', 'N/A')} bpm
- Rhythm: {results.get('rhythm', 'N/A')}
- RMSSD: {results.get('rmssd_ms', 'N/A')} ms
- SDNN: {results.get('sdnn_ms', 'N/A')} ms
- LF/HF Ratio: {results.get('lf_hf_ratio', 'N/A')}
"""
    
    def _format_cognitive_results(self, results: Dict[str, Any]) -> str:
        """Format cognitive assessment results."""
        domains = results.get('domain_scores', {})
        
        domain_lines = [f"- {k}: {v}/100" for k, v in domains.items()]
        
        return f"""Cognitive Assessment Results:
- Overall Score: {results.get('overall_score', 'N/A')}/100
- Age-Adjusted Percentile: {results.get('age_percentile', 'N/A')}%

Domain Scores:
{chr(10).join(domain_lines)}

Strengths: {results.get('strengths', [])}
Areas to Improve: {results.get('areas_to_improve', [])}
"""
    
    def _format_motor_results(self, results: Dict[str, Any]) -> str:
        """Format motor assessment results."""
        return f"""Motor Assessment Results:
- Tremor Score: {results.get('tremor_score', 'N/A')}
- Coordination Score: {results.get('coordination_score', 'N/A')}/100
- Fatigue Index: {results.get('fatigue_index', 'N/A')}
- Dominant Hand: {results.get('dominant_hand', 'N/A')}

Tremor Analysis:
- Frequency: {results.get('tremor_frequency', 'N/A')} Hz
- Amplitude: {results.get('tremor_amplitude', 'N/A')}
- Type: {results.get('tremor_type', 'None detected')}
"""
    
    def _format_nri_results(self, results: Dict[str, Any]) -> str:
        """Format NRI fusion results."""
        contributions = results.get('modality_contributions', {})
        
        contrib_lines = [f"- {k}: {v:.0f}% contribution" for k, v in contributions.items()]
        
        return f"""NRI Fusion Results:
- NRI Score: {results.get('nri_score', 'N/A')}/100
- Risk Category: {results.get('risk_category', 'Unknown')}
- Confidence: {results.get('confidence', 0)*100:.0f}%

Modalities Analyzed: {results.get('modalities', [])}

Modality Contributions:
{chr(10).join(contrib_lines)}
"""
    
    def _format_generic_results(self, results: Dict[str, Any]) -> str:
        """Generic result formatter."""
        return f"""{self.pipeline.title()} Results:
{json.dumps(results, indent=2, default=str)}
"""
    
    def _format_radiology_results(self, results: Dict[str, Any]) -> str:
        """Format radiology/X-ray analysis results."""
        # Primary finding
        primary = results.get('primary_finding', {})
        primary_condition = primary.get('condition', 'No significant abnormality') if isinstance(primary, dict) else 'Unknown'
        primary_prob = primary.get('probability', 0) if isinstance(primary, dict) else 0
        primary_severity = primary.get('severity', 'normal') if isinstance(primary, dict) else 'normal'
        
        # All predictions
        all_preds = results.get('all_predictions', {})
        pred_lines = []
        for condition, prob in sorted(all_preds.items(), key=lambda x: x[1], reverse=True)[:8]:
            if prob > 10:
                pred_lines.append(f"- {condition}: {prob:.1f}%")
        
        # Findings
        findings = results.get('findings', [])
        if isinstance(findings, list) and len(findings) > 0:
            diagnosis = results.get('diagnosis', findings)
        else:
            diagnosis = findings
        
        finding_lines = []
        for f in (diagnosis if isinstance(diagnosis, list) else [])[:5]:
            if isinstance(f, dict):
                cond = f.get('condition', 'Unknown')
                prob = f.get('probability', f.get('confidence', 0))
                sev = f.get('severity', 'unknown')
                finding_lines.append(f"- {cond}: {prob:.0f}% probability ({sev} severity)")
        
        # Risk info
        risk_level = results.get('risk_level', 'low')
        risk_score = results.get('risk_score', 0)
        
        # Recommendations
        recommendations = results.get('recommendations', [])
        rec_text = chr(10).join([f"- {r}" for r in recommendations[:5]]) if recommendations else "- Clinical correlation recommended"
        
        return f"""Chest X-Ray Analysis Results:

Primary Finding:
- Condition: {primary_condition}
- Probability: {primary_prob:.1f}%
- Severity: {primary_severity}

Risk Assessment:
- Risk Level: {risk_level.upper()}
- Risk Score: {risk_score:.1f}/100

Top Pathology Predictions:
{chr(10).join(pred_lines) if pred_lines else "- No significant pathologies detected"}

Detailed Findings:
{chr(10).join(finding_lines) if finding_lines else "- No significant findings"}

Recommendations:
{rec_text}

Heatmap: {"Available - showing AI attention regions" if results.get('heatmap_base64') else "Not available"}
"""


def build_explanation_prompt(
    pipeline: str,
    results: Dict[str, Any],
    patient_context: Optional[Dict[str, Any]] = None
) -> tuple:
    """
    Build system and user prompts for explanation generation.
    
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    builder = PromptBuilder(pipeline)
    return builder.build_system_prompt(), builder.build_user_prompt(results, patient_context)

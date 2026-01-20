"""
Radiology Explanation Rules

Generate AI explanations for radiology findings.
"""

from typing import Dict, List, Any, Optional


class RadiologyExplanationRules:
    """
    Generate explanations for radiology analysis results.
    
    Provides medically-accurate explanations in plain language.
    """
    
    # Condition explanations
    CONDITION_EXPLANATIONS = {
        "Pneumonia": {
            "brief": "Signs of lung infection detected",
            "detailed": """
            The analysis has detected patterns consistent with pneumonia - an infection 
            in one or both lungs. The areas highlighted on the heatmap show regions 
            where the AI identified consolidation or opacity patterns typical of 
            pneumonia. This could be bacterial, viral, or fungal in origin.
            """,
            "what_it_means": "Fluid or pus has accumulated in the air sacs of your lungs",
            "implications": "May require antibiotic treatment and follow-up imaging",
            "urgency": "Should be evaluated by a physician promptly"
        },
        "Cardiomegaly": {
            "brief": "Heart appears larger than normal",
            "detailed": """
            The AI has detected an enlarged cardiac silhouette, which may indicate 
            cardiomegaly (an enlarged heart). The heart-to-chest ratio appears 
            elevated compared to normal parameters. This finding can be associated 
            with various cardiac conditions.
            """,
            "what_it_means": "Your heart appears larger than expected for your chest size",
            "implications": "May indicate underlying cardiac condition requiring evaluation",
            "urgency": "Recommend follow-up with echocardiogram"
        },
        "Effusion": {
            "brief": "Fluid detected around the lungs",
            "detailed": """
            The analysis shows evidence of pleural effusion - fluid accumulation 
            in the space between the lungs and chest wall. The affected area 
            shows characteristic blunting of the costophrenic angle.
            """,
            "what_it_means": "Fluid has collected in the space surrounding your lungs",
            "implications": "May require drainage if large, cause investigation needed",
            "urgency": "Should be evaluated by a physician"
        },
        "Pneumothorax": {
            "brief": "CRITICAL: Air detected in chest cavity",
            "detailed": """
            URGENT FINDING: The AI has detected signs of pneumothorax - air in the 
            pleural space causing partial or complete lung collapse. This is a 
            potentially serious condition that may require immediate medical attention.
            """,
            "what_it_means": "Air has leaked into the space around your lung",
            "implications": "May cause breathing difficulty, may require chest tube",
            "urgency": "URGENT: Requires immediate medical evaluation"
        },
        "Mass": {
            "brief": "Large abnormal opacity detected",
            "detailed": """
            The analysis has identified a mass lesion (opacity larger than 3cm) 
            in the lung field. While this could represent various conditions 
            including benign causes, further evaluation with CT imaging is 
            recommended to characterize the finding.
            """,
            "what_it_means": "A large area of abnormal tissue has been detected",
            "implications": "Requires CT scan for further characterization",
            "urgency": "Priority follow-up recommended"
        },
        "Nodule": {
            "brief": "Small rounded opacity detected",
            "detailed": """
            The AI has detected a pulmonary nodule - a small rounded opacity 
            in the lung. Many nodules are benign, but follow-up may be 
            recommended based on size and patient risk factors per 
            Fleischner Society guidelines.
            """,
            "what_it_means": "A small round spot has been detected in your lung",
            "implications": "May require follow-up CT scan depending on size",
            "urgency": "Follow standard pulmonary nodule guidelines"
        },
        "No Significant Abnormality": {
            "brief": "No concerning findings detected",
            "detailed": """
            The AI analysis did not identify any significant abnormalities in 
            this chest X-ray. The lungs appear clear, heart size is within 
            normal limits, and no acute cardiopulmonary processes were detected.
            """,
            "what_it_means": "Your chest X-ray appears normal",
            "implications": "No immediate follow-up imaging required",
            "urgency": "Continue routine care as directed by your physician"
        }
    }
    
    # Risk level templates
    RISK_TEMPLATES = {
        "critical": """
        CRITICAL ALERT: This analysis has identified findings that require 
        immediate medical attention. The primary finding of {condition} 
        with {probability}% probability suggests urgent evaluation is needed.
        """,
        "high": """
        This analysis has identified significant findings that warrant 
        priority medical evaluation. The primary finding of {condition} 
        ({probability}% probability) should be reviewed by a physician promptly.
        """,
        "moderate": """
        The analysis has identified findings that may require follow-up. 
        The primary finding of {condition} ({probability}% probability) 
        should be correlated with clinical symptoms and history.
        """,
        "low": """
        The analysis shows minor or incidental findings. The primary 
        finding of {condition} has a low probability and may represent 
        normal variation. Continue routine monitoring as appropriate.
        """,
        "normal": """
        This chest X-ray appears normal. No significant abnormalities 
        were detected. The lungs are clear and heart size is normal. 
        Continue routine health maintenance as directed by your physician.
        """
    }
    
    def generate_explanation(
        self,
        primary_finding: str,
        probability: float,
        risk_level: str,
        findings: List[Dict],
        quality_score: float = 1.0
    ) -> Dict[str, Any]:
        """
        Generate complete explanation for analysis results.
        
        Args:
            primary_finding: Primary condition detected
            probability: Confidence percentage
            risk_level: Overall risk level
            findings: List of all findings
            quality_score: Image quality score
        
        Returns:
            Dict with explanation components
        """
        # Get condition-specific explanation
        condition_info = self.CONDITION_EXPLANATIONS.get(
            primary_finding,
            self._get_default_explanation(primary_finding)
        )
        
        # Get risk template
        risk_template = self.RISK_TEMPLATES.get(risk_level, self.RISK_TEMPLATES["normal"])
        overview = risk_template.format(
            condition=primary_finding,
            probability=probability
        ).strip()
        
        # Build findings summary
        findings_text = self._build_findings_summary(findings)
        
        # Quality disclaimer if needed
        quality_note = None
        if quality_score < 0.7:
            quality_note = "Note: Image quality is suboptimal which may affect accuracy."
        
        return {
            "overview": overview,
            "primary_finding_brief": condition_info["brief"],
            "primary_finding_detailed": condition_info["detailed"].strip(),
            "what_it_means": condition_info.get("what_it_means", ""),
            "implications": condition_info.get("implications", ""),
            "urgency": condition_info.get("urgency", ""),
            "findings_summary": findings_text,
            "quality_note": quality_note,
            "disclaimer": "This AI analysis is for informational purposes only and should not replace professional medical evaluation."
        }
    
    def _get_default_explanation(self, condition: str) -> Dict:
        """Get default explanation for unknown conditions."""
        return {
            "brief": f"{condition} pattern detected",
            "detailed": f"The AI has detected patterns consistent with {condition}. "
                       "Please consult with a radiologist for clinical interpretation.",
            "what_it_means": "An abnormal pattern has been detected",
            "implications": "Clinical correlation recommended",
            "urgency": "Consult with your physician"
        }
    
    def _build_findings_summary(self, findings: List[Dict]) -> str:
        """Build summary of all findings."""
        if not findings:
            return "No significant findings were detected."
        
        lines = ["The analysis detected the following findings:"]
        for i, finding in enumerate(findings[:5], 1):
            condition = finding.get("condition", "Unknown")
            prob = finding.get("probability", 0)
            severity = finding.get("severity", "unknown")
            lines.append(f"{i}. {condition}: {prob}% probability ({severity} severity)")
        
        return "\n".join(lines)

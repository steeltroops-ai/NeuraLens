"""
Radiology Recommendation Generator

Generate clinical recommendations based on findings.
"""

from typing import List, Dict, Optional

from ..config import PATHOLOGY_INFO


class RecommendationGenerator:
    """
    Generate clinical recommendations based on findings.
    
    Recommendations are based on:
    - Primary finding
    - Risk level
    - Specific conditions detected
    """
    
    # Condition-specific recommendations
    CONDITION_RECOMMENDATIONS = {
        "Pneumonia": [
            "Consider antibiotic therapy if bacterial infection suspected",
            "Follow-up chest X-ray in 4-6 weeks to document resolution"
        ],
        "Cardiomegaly": [
            "Echocardiogram recommended for cardiac evaluation",
            "Consider cardiology consultation"
        ],
        "Pneumothorax": [
            "URGENT: Immediate physician review required",
            "Chest tube placement may be required"
        ],
        "Effusion": [
            "Consider thoracentesis for large effusions",
            "Evaluate underlying cause (cardiac, malignant, infectious)"
        ],
        "Mass": [
            "CT scan recommended for further characterization",
            "Consider biopsy if clinically appropriate"
        ],
        "Nodule": [
            "Follow Fleischner Society guidelines for nodule management",
            "Consider CT follow-up based on size and risk factors"
        ],
        "Consolidation": [
            "Clinical correlation for infectious vs. non-infectious etiology",
            "Consider sputum culture if infection suspected"
        ],
        "Atelectasis": [
            "Evaluate for underlying cause (obstruction, post-operative)",
            "Incentive spirometry and deep breathing exercises"
        ],
        "Edema": [
            "Evaluate for heart failure",
            "Consider diuretic therapy if cardiogenic"
        ]
    }
    
    def generate(
        self,
        primary: Optional[str],
        risk_level: str,
        findings: List[Dict]
    ) -> List[str]:
        """
        Generate clinical recommendations.
        
        Args:
            primary: Primary finding
            risk_level: Overall risk level
            findings: List of findings
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Risk-based general recommendations
        if risk_level == "critical":
            recommendations.append("URGENT: Immediate physician review required")
            recommendations.append("Consider emergency intervention if clinically indicated")
        elif risk_level == "high":
            recommendations.append("Priority consultation recommended")
            recommendations.append("Consider CT scan for further evaluation")
        elif risk_level == "moderate":
            recommendations.append("Clinical correlation advised")
            recommendations.append("Follow-up imaging may be warranted")
        elif risk_level == "low":
            recommendations.append("Minor findings noted")
            recommendations.append("Routine follow-up if clinically indicated")
        else:
            recommendations.append("No significant abnormalities detected")
            recommendations.append("Continue routine screening as indicated")
        
        recommendations.append("Correlate with clinical findings as appropriate")
        
        # Add condition-specific recommendations
        added_conditions = set()
        
        for finding in findings[:3]:  # Top 3 findings
            condition = finding.get("condition", "")
            if condition in added_conditions:
                continue
            
            if condition in self.CONDITION_RECOMMENDATIONS:
                for rec in self.CONDITION_RECOMMENDATIONS[condition][:1]:  # First rec only
                    if rec not in recommendations:
                        recommendations.append(rec)
                        added_conditions.add(condition)
        
        # Limit to 5 recommendations
        return recommendations[:5]
    
    def generate_summary(
        self,
        primary: Optional[str],
        risk_level: str
    ) -> str:
        """Generate a brief clinical summary."""
        if risk_level == "critical":
            return f"CRITICAL: {primary or 'Significant finding'} detected. Immediate review required."
        elif risk_level == "high":
            return f"HIGH RISK: {primary or 'Significant finding'} identified. Priority follow-up recommended."
        elif risk_level == "moderate":
            return f"MODERATE: {primary or 'Finding'} noted. Clinical correlation and follow-up advised."
        elif risk_level == "low":
            return f"LOW RISK: Minor findings. Routine monitoring recommended."
        else:
            return "No significant abnormalities. Lungs clear, heart normal size."

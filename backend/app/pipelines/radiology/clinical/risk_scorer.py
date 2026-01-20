"""
Radiology Risk Scorer

Calculate clinical risk scores from pathology predictions.
"""

from typing import Dict, List, Any

from ..config import (
    CRITICAL_CONDITIONS,
    HIGH_RISK_CONDITIONS,
    MODERATE_CONDITIONS,
    RadiologyConfig,
    PATHOLOGY_INFO
)


class RiskScorer:
    """
    Calculate overall risk from X-ray findings.
    
    Uses weighted scoring based on:
    - Critical conditions (highest weight)
    - High-risk conditions
    - Moderate conditions
    - Low-risk conditions
    """
    
    def calculate(self, predictions: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate risk score from predictions.
        
        Args:
            predictions: Dict of condition -> probability (0-100)
        
        Returns:
            Dict with risk_score, category, color, and findings
        """
        risk_score = 0.0
        findings = []
        
        for pathology, prob in predictions.items():
            if prob < 10:  # Below threshold
                continue
            
            finding = {
                "condition": pathology,
                "probability": prob,
                "description": PATHOLOGY_INFO.get(pathology, {}).get("description", "")
            }
            
            if pathology in CRITICAL_CONDITIONS:
                finding["severity"] = "critical"
                risk_score += prob * RadiologyConfig.CRITICAL_WEIGHT
            elif pathology in HIGH_RISK_CONDITIONS:
                finding["severity"] = "high"
                risk_score += prob * RadiologyConfig.HIGH_WEIGHT
            elif pathology in MODERATE_CONDITIONS:
                finding["severity"] = "moderate"
                risk_score += prob * RadiologyConfig.MODERATE_WEIGHT
            else:
                finding["severity"] = "low"
                risk_score += prob * RadiologyConfig.LOW_WEIGHT
            
            findings.append(finding)
        
        # Sort by probability
        findings.sort(key=lambda x: x["probability"], reverse=True)
        
        # Categorize
        if risk_score < 15:
            category = "low"
            color = "green"
        elif risk_score < 40:
            category = "moderate"
            color = "yellow"
        elif risk_score < 70:
            category = "high"
            color = "orange"
        else:
            category = "critical"
            color = "red"
        
        return {
            "risk_score": min(100, round(risk_score, 1)),
            "category": category,
            "color": color,
            "findings": findings[:5],  # Top 5 findings
            "critical_findings": [
                f for f in findings if f["severity"] == "critical"
            ]
        }
    
    def get_risk_level(self, predictions: Dict[str, float]) -> str:
        """Get simplified risk level."""
        result = self.calculate(predictions)
        return result["category"]
    
    def get_urgency(self, risk_category: str) -> str:
        """Get clinical urgency from risk category."""
        urgency_map = {
            "critical": "urgent",
            "high": "priority",
            "moderate": "routine",
            "low": "routine"
        }
        return urgency_map.get(risk_category, "routine")

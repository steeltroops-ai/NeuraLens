"""
Radiology Output Formatter

Format analysis results for API response.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

from ..config import PATHOLOGY_INFO


class OutputFormatter:
    """
    Format radiology analysis results for API response.
    
    Formats:
    - Primary finding
    - All predictions
    - Detailed findings
    - Risk assessment
    - Recommendations
    """
    
    def format_response(
        self,
        analysis_result: Dict[str, Any],
        quality_metrics: Dict[str, Any],
        stage_timings: Dict[str, float],
        request_id: str
    ) -> Dict[str, Any]:
        """
        Format complete analysis response.
        
        Args:
            analysis_result: Raw analysis results
            quality_metrics: Image quality assessment
            stage_timings: Pipeline stage timings
            request_id: Request identifier
        
        Returns:
            Formatted response dict
        """
        # Format primary finding
        primary = self.format_primary_finding(analysis_result)
        
        # Format all predictions
        predictions = analysis_result.get("all_predictions", {})
        
        # Format detailed findings
        findings = self.format_findings(analysis_result.get("findings", []))
        
        # Format recommendations
        recommendations = analysis_result.get("recommendations", [])
        
        return {
            "success": True,
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "processing_time_ms": sum(stage_timings.values()),
            
            "primary_finding": primary,
            "all_predictions": predictions,
            "findings": findings,
            
            "risk_level": analysis_result.get("risk_level", "low"),
            "risk_score": analysis_result.get("risk_score", 0),
            
            "heatmap_base64": analysis_result.get("heatmap_base64"),
            
            "quality": self.format_quality(quality_metrics),
            
            "recommendations": recommendations,
            
            "stages_completed": [
                {
                    "stage": stage,
                    "status": "success",
                    "time_ms": round(time_ms, 1)
                }
                for stage, time_ms in stage_timings.items()
            ]
        }
    
    def format_primary_finding(self, analysis_result: Dict) -> Dict:
        """Format primary finding."""
        primary_condition = analysis_result.get("primary_finding", "No Significant Abnormality")
        
        return {
            "condition": primary_condition,
            "probability": analysis_result.get("confidence", 0),
            "severity": analysis_result.get("risk_level", "normal"),
            "description": self._get_description(primary_condition)
        }
    
    def format_findings(self, findings: List[Dict]) -> List[Dict]:
        """Format detailed findings list."""
        formatted = []
        
        for i, finding in enumerate(findings):
            condition = finding.get("condition", "Unknown")
            prob = finding.get("probability", 0)
            
            formatted.append({
                "id": f"finding_{i+1:03d}",
                "condition": condition,
                "probability": prob,
                "severity": finding.get("severity", self._get_severity(prob)),
                "confidence": prob / 100,
                "location": finding.get("location"),
                "description": finding.get("description", self._get_description(condition)),
                "radiological_features": self._get_features(condition),
                "urgency": PATHOLOGY_INFO.get(condition, {}).get("urgency"),
                "is_critical": finding.get("severity") in ["critical", "high"]
            })
        
        return formatted
    
    def format_quality(self, quality_metrics: Dict) -> Dict:
        """Format quality assessment."""
        return {
            "overall_quality": quality_metrics.get("quality", "good"),
            "quality_score": quality_metrics.get("quality_score", 0.8),
            "resolution": quality_metrics.get("resolution"),
            "resolution_adequate": quality_metrics.get("resolution_ok", True),
            "positioning": "adequate",
            "exposure": "satisfactory",
            "contrast": quality_metrics.get("contrast"),
            "issues": quality_metrics.get("issues", []),
            "usable": quality_metrics.get("usable", True)
        }
    
    def _get_description(self, condition: str) -> str:
        """Get clinical description for condition."""
        if condition in ["No Significant Abnormality", "No Significant Findings"]:
            return "Lungs are clear. Heart size is normal. No acute cardiopulmonary process."
        return PATHOLOGY_INFO.get(condition, {}).get(
            "description",
            "Finding detected - clinical correlation recommended"
        )
    
    def _get_severity(self, probability: float) -> str:
        """Get severity from probability."""
        if probability >= 70:
            return "high"
        elif probability >= 40:
            return "moderate"
        elif probability >= 20:
            return "low"
        else:
            return "minimal"
    
    def _get_features(self, condition: str) -> List[str]:
        """Get radiological features for condition."""
        features_map = {
            "Pneumonia": ["Consolidation", "Air bronchograms", "Ground-glass opacity"],
            "Cardiomegaly": ["Enlarged cardiac silhouette", "Cardiothoracic ratio >0.5"],
            "Effusion": ["Costophrenic blunting", "Meniscus sign"],
            "Pneumothorax": ["Visceral pleural line", "Absent lung markings"],
            "Consolidation": ["Dense opacity", "Air bronchograms"],
            "Atelectasis": ["Volume loss", "Shifted fissures"],
            "Nodule": ["Rounded opacity", "Well-defined margins"],
            "Mass": ["Large opacity", "Irregular margins"]
        }
        return features_map.get(condition, [])

"""
Cognitive Output Formatter
"""

from datetime import datetime
from ..schemas import CognitiveResponse, CognitiveRiskAssessment, CognitiveFeatures, ClinicalRecommendation

class OutputFormatter:
    def format_response(
        self, 
        session_id: str, 
        risk: CognitiveRiskAssessment, 
        features: CognitiveFeatures,
        recommendations: list[ClinicalRecommendation],
        process_time: float
    ) -> CognitiveResponse:
        
        return CognitiveResponse(
            session_id=session_id,
            timestamp=datetime.now(),
            processing_time_ms=process_time * 1000,
            risk_assessment=risk,
            features=features,
            recommendations=recommendations,
            pipeline_version="1.0.0",
            status="success"
        )

"""
Cognitive Output Formatter
"""

from datetime import datetime, timezone
from ..schemas import CognitiveResponse, CognitiveRiskAssessment, CognitiveFeatures, ClinicalRecommendation, StageProgress, PipelineStage
from ..config import config

class OutputFormatter:
    def format_response(
        self, 
        session_id: str, 
        risk: CognitiveRiskAssessment, 
        features: CognitiveFeatures,
        recommendations: list[ClinicalRecommendation],
        process_time: float
    ) -> CognitiveResponse:
        # Build the four expected stages
        stages = [
            StageProgress(stage=PipelineStage.VALIDATING, stage_index=0, message="Validation"),
            StageProgress(stage=PipelineStage.EXTRACTING, stage_index=1, message="Feature extraction"),
            StageProgress(stage=PipelineStage.SCORING, stage_index=2, message="Risk scoring"),
            StageProgress(stage=PipelineStage.COMPLETE, stage_index=3, message="Complete")
        ]
        
        return CognitiveResponse(
            session_id=session_id,
            timestamp=datetime.now(timezone.utc),
            processing_time_ms=process_time * 1000,
            risk_assessment=risk,
            features=features,
            recommendations=recommendations,
            pipeline_version=config.VERSION,
            status="success",
            stages=stages
        )

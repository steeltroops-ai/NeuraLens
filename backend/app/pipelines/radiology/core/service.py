"""
Radiology Pipeline Service

Main service entry point that coordinates all pipeline stages.
"""

import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .orchestrator import RadiologyOrchestrator, PipelineStage, PipelineContext
from ..config import RadiologyConfig, PATHOLOGY_INFO
from ..input.validator import ImageValidator, ValidationResult
from ..preprocessing.normalizer import ImageNormalizer
from ..analysis.analyzer import XRayAnalyzer
from ..clinical.risk_scorer import RiskScorer
from ..clinical.recommendations import RecommendationGenerator
from ..output.formatter import OutputFormatter
from ..output.visualization import HeatmapGenerator
from ..errors.codes import ErrorCode
from ..errors.handlers import handle_pipeline_error

logger = logging.getLogger(__name__)


class RadiologyService:
    """
    Main radiology analysis service.
    
    Coordinates the full analysis pipeline:
    1. Input validation
    2. Image preprocessing
    3. TorchXRayVision inference
    4. Risk scoring
    5. Output formatting
    """
    
    def __init__(self):
        self.orchestrator = RadiologyOrchestrator()
        self.validator = ImageValidator()
        self.normalizer = ImageNormalizer()
        self.analyzer = XRayAnalyzer()
        self.risk_scorer = RiskScorer()
        self.recommendation_gen = RecommendationGenerator()
        self.formatter = OutputFormatter()
        self.heatmap_gen = HeatmapGenerator()
        
        self.config = RadiologyConfig()
    
    async def analyze(
        self,
        image_bytes: bytes,
        filename: str,
        content_type: Optional[str] = None,
        options: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Run complete radiology analysis pipeline.
        
        Args:
            image_bytes: Raw image data
            filename: Original filename
            content_type: MIME type
            options: Analysis options
        
        Returns:
            Complete analysis response
        """
        options = options or {}
        request_id = self.orchestrator.initialize()
        
        context = PipelineContext(
            request_id=request_id,
            image_bytes=image_bytes,
            metadata={
                "filename": filename,
                "content_type": content_type,
                "file_size_mb": len(image_bytes) / (1024 * 1024)
            }
        )
        
        try:
            # Stage 1: Receipt
            self.orchestrator.transition(PipelineStage.RECEIPT)
            receipt = self._process_receipt(context)
            
            # Stage 2: Validation
            self.orchestrator.transition(PipelineStage.VALIDATION)
            validation = self._run_validation(context)
            if not validation.is_valid:
                raise ValueError(f"Validation failed: {validation.errors}")
            
            # Stage 3: Preprocessing
            self.orchestrator.transition(PipelineStage.PREPROCESSING)
            preprocessed = self._run_preprocessing(context)
            context.preprocessed_data = preprocessed
            
            # Stage 4: Detection (anatomical)
            self.orchestrator.transition(PipelineStage.DETECTION)
            detection = self._run_detection(context)
            context.detection_results = detection
            
            # Stage 5: Analysis (pathology)
            self.orchestrator.transition(PipelineStage.ANALYSIS)
            analysis = self._run_analysis(context)
            context.analysis_results = analysis
            
            # Stage 6: Aggregation
            self.orchestrator.transition(PipelineStage.AGGREGATION)
            aggregated = self._run_aggregation(context)
            
            # Stage 7: Scoring
            self.orchestrator.transition(PipelineStage.SCORING)
            scored = self._run_scoring(context, aggregated)
            context.clinical_results = scored
            
            # Stage 8: Formatting
            self.orchestrator.transition(PipelineStage.FORMATTING)
            formatted = self._format_response(context, options)
            
            # Complete
            self.orchestrator.transition(PipelineStage.COMPLETE)
            
            return formatted
            
        except Exception as e:
            logger.error(f"[{request_id}] Pipeline failed: {e}")
            self.orchestrator.transition(
                PipelineStage.FAILED,
                success=False,
                error=str(e)
            )
            return handle_pipeline_error(e, self.orchestrator)
    
    def _process_receipt(self, context: PipelineContext) -> Dict:
        """Process and acknowledge input receipt."""
        return {
            "acknowledged": True,
            "modality_received": context.modality,
            "body_region": context.body_region,
            "is_volumetric": context.is_volumetric,
            "file_size_mb": context.metadata.get("file_size_mb", 0)
        }
    
    def _run_validation(self, context: PipelineContext) -> ValidationResult:
        """Run input validation."""
        return self.validator.validate(
            file_path=context.metadata.get("filename", "unknown"),
            file_bytes=context.image_bytes
        )
    
    def _run_preprocessing(self, context: PipelineContext) -> Dict:
        """Run image preprocessing."""
        return self.normalizer.normalize(context.image_bytes)
    
    def _run_detection(self, context: PipelineContext) -> Dict:
        """Run anatomical detection."""
        # Anatomical detection (lung segmentation etc)
        # Currently simplified - returns basic structure
        return {
            "lungs_detected": True,
            "heart_detected": True,
            "structures": ["left_lung", "right_lung", "heart", "mediastinum"]
        }
    
    def _run_analysis(self, context: PipelineContext) -> Dict:
        """Run pathology analysis."""
        result = self.analyzer.analyze(context.image_bytes)
        return {
            "primary_finding": result.primary_finding,
            "confidence": result.confidence,
            "risk_level": result.risk_level,
            "findings": result.findings,
            "all_predictions": result.all_predictions,
            "heatmap_base64": result.heatmap_base64
        }
    
    def _run_aggregation(self, context: PipelineContext) -> Dict:
        """Aggregate findings from all analyses."""
        analysis = context.analysis_results or {}
        
        # Combine all findings
        findings = analysis.get("findings", [])
        predictions = analysis.get("all_predictions", {})
        
        # Filter significant findings
        significant = [
            f for f in findings 
            if f.get("probability", 0) >= RadiologyConfig.CONFIDENCE_THRESHOLD * 100
        ]
        
        return {
            "primary_finding": analysis.get("primary_finding"),
            "confidence": analysis.get("confidence"),
            "significant_findings": significant,
            "all_predictions": predictions,
            "heatmap_base64": analysis.get("heatmap_base64")
        }
    
    def _run_scoring(self, context: PipelineContext, aggregated: Dict) -> Dict:
        """Run clinical scoring."""
        predictions = aggregated.get("all_predictions", {})
        
        # Calculate risk score
        risk_result = self.risk_scorer.calculate(predictions)
        
        # Generate recommendations
        recommendations = self.recommendation_gen.generate(
            primary=aggregated.get("primary_finding"),
            risk_level=risk_result["category"],
            findings=aggregated.get("significant_findings", [])
        )
        
        return {
            **aggregated,
            "risk_score": risk_result["risk_score"],
            "risk_level": risk_result["category"],
            "risk_color": risk_result["color"],
            "recommendations": recommendations
        }
    
    def _format_response(self, context: PipelineContext, options: Dict) -> Dict:
        """Format final response."""
        clinical = context.clinical_results or {}
        analysis = context.analysis_results or {}
        
        # Build quality metrics
        quality = self.validator.assess_quality(context.image_bytes)
        
        # Build primary finding
        primary_condition = clinical.get("primary_finding", "No Significant Abnormality")
        primary = {
            "condition": primary_condition,
            "probability": clinical.get("confidence", 0),
            "severity": clinical.get("risk_level", "normal"),
            "description": self._get_condition_description(primary_condition)
        }
        
        # Format findings
        findings = [
            {
                "id": f"finding_{i+1:03d}",
                "condition": f.get("condition"),
                "probability": f.get("probability"),
                "severity": f.get("severity", "moderate"),
                "confidence": f.get("probability", 0) / 100,
                "description": f.get("description", ""),
                "urgency": PATHOLOGY_INFO.get(f.get("condition", ""), {}).get("urgency"),
                "is_critical": f.get("severity") in ["critical", "high"]
            }
            for i, f in enumerate(clinical.get("significant_findings", []))
        ]
        
        return {
            "success": True,
            "request_id": context.request_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "processing_time_ms": int(self.orchestrator.get_total_duration()),
            
            "receipt": {
                "acknowledged": True,
                "modality_received": context.modality,
                "body_region": context.body_region,
                "is_volumetric": context.is_volumetric,
                "file_size_mb": round(context.metadata.get("file_size_mb", 0), 2)
            },
            
            "stages_completed": self.orchestrator.get_completed_stages(),
            
            "primary_finding": primary,
            "all_predictions": clinical.get("all_predictions", {}),
            "findings": findings,
            
            "risk_level": clinical.get("risk_level", "low"),
            "risk_score": clinical.get("risk_score", 0),
            
            "heatmap_base64": clinical.get("heatmap_base64"),
            
            "quality": {
                "overall_quality": quality.get("quality", "good"),
                "quality_score": quality.get("quality_score", 0.8),
                "resolution": quality.get("resolution"),
                "resolution_adequate": quality.get("resolution_ok", True),
                "contrast": quality.get("contrast"),
                "issues": quality.get("issues", []),
                "usable": quality.get("usable", True)
            },
            
            "recommendations": clinical.get("recommendations", [])
        }
    
    def _get_condition_description(self, condition: str) -> str:
        """Get clinical description for a condition."""
        if condition in ["No Significant Abnormality", "No Significant Findings"]:
            return "Lungs are clear. Heart size is normal. No acute cardiopulmonary process."
        
        return PATHOLOGY_INFO.get(condition, {}).get(
            "description",
            "Finding detected - clinical correlation recommended"
        )

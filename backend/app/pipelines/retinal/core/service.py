"""
Research-Grade Retinal Analysis Service v4.0
Comprehensive fundus image analysis pipeline for medical screening.

Integrates:
- Image preprocessing (CLAHE, color normalization)
- Anatomical detection (optic disc, macula, vessels)
- Biomarker extraction (CDR, AVR, lesions)
- Clinical risk assessment with uncertainty
- Production monitoring (quality, audit)

Author: NeuraLens Medical AI Team
Version: 4.0.0
"""

import logging
import numpy as np
import io
from datetime import datetime
from typing import Optional, Dict, Tuple, List, Any
from dataclasses import dataclass
from PIL import Image

# Pipeline modules
from ..config import (
    INPUT_CONSTRAINTS,
    QUALITY_THRESHOLDS,
    BIOMARKER_NORMAL_RANGES,
    RISK_WEIGHTS,
    RECOMMENDATIONS,
    ICD10_CODES,
    DRGrade,
    DR_GRADE_CRITERIA,
)
from ..preprocessing import (
    ImagePreprocessor,
    PreprocessingResult,
    image_preprocessor,
)
from ..features import biomarker_extractor
from ..clinical import (
    DRGrader,
    DMEAssessor,
    RiskCalculator,
    ClinicalFindingsGenerator,
    RecommendationGenerator,
    dr_grader,
    dme_assessor,
    risk_calculator,
    findings_generator,
    recommendation_generator,
)
from ..errors.codes import PipelineException, get_error
from .orchestrator import (
    AuditLogger,
    ReceiptConfirmation,
    create_execution_context,
    get_disclaimers,
)
from ..output import visualization_service

# Schemas
from ..schemas import (
    PipelineStage,
    PipelineState,
    RetinalAnalysisResponse,
    ImageQuality,
    CompleteBiomarkers,
    DiabeticRetinopathyResult,
    RiskAssessment,
    ClinicalFinding,
)

logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """Configuration for the retinal analysis service."""
    model_input_size: int = 512
    min_quality_score: float = 0.30
    
    # Feature extraction
    extract_vessels: bool = True
    extract_lesions: bool = True
    generate_heatmap: bool = True
    
    # Quality thresholds
    quality_gate_enabled: bool = True
    
    # Clinical
    use_uncertainty: bool = True
    
    # Monitoring
    enable_audit: bool = True


class ResearchGradeRetinalService:
    """
    Research-grade retinal fundus analysis service.
    
    Provides comprehensive fundus image analysis with
    clinical-grade accuracy and uncertainty quantification.
    """
    
    VERSION = "4.0.0"
    
    def __init__(self, config: Optional[ServiceConfig] = None):
        self.config = config or ServiceConfig()
        
        # Initialize components
        self.preprocessor = image_preprocessor
        self.biomarker_extractor = biomarker_extractor
        self.dr_grader = dr_grader
        self.dme_assessor = dme_assessor
        self.risk_calculator = risk_calculator
        self.findings_generator = findings_generator
        self.recommendation_generator = recommendation_generator
        self.visualization = visualization_service
        
        # Audit logger
        self.audit_logger = AuditLogger() if self.config.enable_audit else None
        
        logger.info(f"ResearchGradeRetinalService v{self.VERSION} initialized")
    
    async def analyze(
        self,
        image_bytes: bytes,
        session_id: str,
        filename: str,
        content_type: Optional[str] = None,
        patient_id: Optional[str] = None,
        patient_age: Optional[int] = None,
    ) -> RetinalAnalysisResponse:
        """
        Perform comprehensive retinal analysis.
        
        Args:
            image_bytes: Fundus image content
            session_id: Unique session identifier
            filename: Original filename
            content_type: MIME type
            patient_id: Optional patient identifier
            patient_age: Optional age for risk calculation
            
        Returns:
            RetinalAnalysisResponse with full analysis
        """
        start_time = datetime.now()
        state = PipelineState(session_id=session_id)
        
        # Create receipt
        receipt = ReceiptConfirmation(
            image_received=True,
            image_size_bytes=len(image_bytes),
            received_at=datetime.utcnow().isoformat(),
            filename=filename,
            content_type=content_type,
        )
        
        try:
            # 1. Validate input
            self._validate_input(image_bytes, content_type)
            state.stages_completed.append(PipelineStage.INPUT_VALIDATION)
            
            # 2. Preprocess image
            preprocessing_result = self._preprocess(image_bytes)
            state.stages_completed.append(PipelineStage.IMAGE_PREPROCESSING)
            
            # Update receipt with dimensions
            h, w = preprocessing_result.image.shape[:2]
            receipt.image_dimensions = (w, h)
            
            # 3. Quality gate
            image_quality = self._build_quality(preprocessing_result, image_bytes)
            state.stages_completed.append(PipelineStage.QUALITY_ASSESSMENT)
            
            if self.config.quality_gate_enabled:
                if not image_quality.is_gradable:
                    raise PipelineException(
                        "PRE_010",
                        {"quality_score": image_quality.overall_score}
                    )
            
            # 4. Extract biomarkers
            biomarkers = self._extract_biomarkers(
                preprocessing_result.image, 
                image_quality.overall_score,
                patient_age
            )
            state.stages_completed.append(PipelineStage.LESION_DETECTION)
            
            # 5. Grade DR
            dr_result = self._grade_dr(biomarkers)
            state.stages_completed.append(PipelineStage.DR_GRADING)
            
            # 6. Assess DME
            dme_result = self.dme_assessor.assess(biomarkers)
            
            # 7. Calculate risk
            risk = self._calculate_risk(biomarkers, dr_result)
            state.stages_completed.append(PipelineStage.RISK_CALCULATION)
            
            # 8. Generate findings
            findings = self._generate_findings(biomarkers, dr_result, dme_result, risk)
            state.stages_completed.append(PipelineStage.CLINICAL_ASSESSMENT)
            
            # 9. Generate heatmap
            heatmap_b64 = ""
            if self.config.generate_heatmap:
                heatmap_b64 = self._generate_heatmap(image_bytes)
                state.stages_completed.append(PipelineStage.HEATMAP_GENERATION)
            
            # 10. Build response
            end_time = datetime.now()
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            response = self._build_response(
                session_id=session_id,
                patient_id=patient_id or "ANONYMOUS",
                state=state,
                image_quality=image_quality,
                biomarkers=biomarkers,
                dr_result=dr_result,
                dme_result=dme_result,
                risk=risk,
                findings=findings,
                heatmap_b64=heatmap_b64,
                processing_time_ms=processing_time_ms,
            )
            
            # 11. Audit log
            if self.audit_logger:
                self._log_audit(
                    session_id=session_id,
                    patient_id=patient_id,
                    start_time=start_time,
                    end_time=end_time,
                    dr_grade=dr_result.grade,
                    risk_score=risk.score,
                    quality_score=image_quality.overall_score,
                )
            
            return response
            
        except PipelineException as e:
            logger.error(f"Pipeline error: {e.code} - {e.error['message']}")
            return self._build_error_response(
                session_id=session_id,
                patient_id=patient_id or "ANONYMOUS",
                state=state,
                error=e,
                start_time=start_time,
            )
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            error = PipelineException("SYS_001", {"error": str(e)})
            return self._build_error_response(
                session_id=session_id,
                patient_id=patient_id or "ANONYMOUS",
                state=state,
                error=error,
                start_time=start_time,
            )
    
    def _validate_input(self, image_bytes: bytes, content_type: Optional[str]) -> None:
        """Validate input constraints."""
        # Size check
        size_mb = len(image_bytes) / (1024 * 1024)
        if size_mb > INPUT_CONSTRAINTS["max_file_size_mb"]:
            raise PipelineException(
                "VAL_052",
                {"size_mb": size_mb, "max_mb": INPUT_CONSTRAINTS["max_file_size_mb"]}
            )
        
        if len(image_bytes) < 1000:
            raise PipelineException("VAL_050", {"size_bytes": len(image_bytes)})
        
        # Content type check
        if content_type and not content_type.startswith("image/"):
            raise PipelineException("VAL_001", {"content_type": content_type})
    
    def _preprocess(self, image_bytes: bytes) -> PreprocessingResult:
        """Run preprocessing pipeline."""
        return self.preprocessor.preprocess(image_bytes)
    
    def _build_quality(
        self, 
        result: PreprocessingResult,
        image_bytes: bytes
    ) -> ImageQuality:
        """Build ImageQuality from preprocessing result."""
        # Get dimensions
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
        file_size_mb = len(image_bytes) / (1024 * 1024)
        
        return ImageQuality(
            overall_score=result.quality_score,
            gradability=result.quality_grade,
            is_gradable=result.quality_score >= self.config.min_quality_score,
            issues=result.warnings,
            snr_db=result.snr_db,
            focus_score=result.sharpness_score,
            illumination_score=result.illumination_score,
            contrast_score=result.contrast_score,
            optic_disc_visible=True,
            macula_visible=True,
            vessel_arcades_visible=True,
            resolution=(width, height),
            file_size_mb=round(file_size_mb, 2),
            field_of_view="standard"
        )
    
    def _extract_biomarkers(
        self,
        image: np.ndarray,
        quality_score: float,
        patient_age: Optional[int]
    ) -> CompleteBiomarkers:
        """Extract all biomarkers from image."""
        return self.biomarker_extractor.extract(
            image=image,
            quality_score=quality_score,
            patient_age=patient_age,
        )
    
    def _grade_dr(self, biomarkers: CompleteBiomarkers) -> DiabeticRetinopathyResult:
        """Grade diabetic retinopathy."""
        return self.dr_grader.grade(biomarkers)
    
    def _calculate_risk(
        self,
        biomarkers: CompleteBiomarkers,
        dr_result: DiabeticRetinopathyResult
    ) -> RiskAssessment:
        """Calculate overall risk assessment."""
        return self.risk_calculator.calculate(biomarkers, dr_result)
    
    def _generate_findings(
        self,
        biomarkers: CompleteBiomarkers,
        dr_result: DiabeticRetinopathyResult,
        dme_result: Any,
        risk: RiskAssessment
    ) -> List[ClinicalFinding]:
        """Generate clinical findings."""
        return self.findings_generator.generate(biomarkers, dr_result, dme_result, risk)
    
    def _generate_heatmap(self, image_bytes: bytes) -> str:
        """Generate attention heatmap."""
        try:
            return self.visualization.generate_heatmap(image_bytes)
        except Exception as e:
            logger.warning(f"Heatmap generation failed: {e}")
            return ""
    
    def _build_response(
        self,
        session_id: str,
        patient_id: str,
        state: PipelineState,
        image_quality: ImageQuality,
        biomarkers: CompleteBiomarkers,
        dr_result: DiabeticRetinopathyResult,
        dme_result: Any,
        risk: RiskAssessment,
        findings: List[ClinicalFinding],
        heatmap_b64: str,
        processing_time_ms: int,
    ) -> RetinalAnalysisResponse:
        """Build successful response."""
        state.current_stage = PipelineStage.COMPLETED
        
        # Get recommendations
        recommendations = self.recommendation_generator.generate(dr_result, risk)
        
        # Get summary
        summary = self._generate_summary(dr_result, risk, findings)
        
        return RetinalAnalysisResponse(
            success=True,
            session_id=session_id,
            patient_id=patient_id,
            timestamp=datetime.utcnow().isoformat(),
            total_processing_time_ms=processing_time_ms,
            pipeline_state=state,
            image_quality=image_quality,
            biomarkers=biomarkers,
            diabetic_retinopathy=dr_result,
            risk_assessment=risk,
            findings=findings,
            recommendations=recommendations,
            heatmap_base64=heatmap_b64,
            clinical_summary=summary,
            disclaimers=get_disclaimers("screening"),
        )
    
    def _build_error_response(
        self,
        session_id: str,
        patient_id: str,
        state: PipelineState,
        error: PipelineException,
        start_time: datetime,
    ) -> RetinalAnalysisResponse:
        """Build error response."""
        from .schemas import PipelineError
        
        state.current_stage = PipelineStage.FAILED
        state.errors.append(PipelineError(
            stage=state.current_stage,
            error_type=error.code,
            message=error.error["message"],
            details=error.error,
        ))
        
        processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        return RetinalAnalysisResponse(
            success=False,
            session_id=session_id,
            patient_id=patient_id,
            timestamp=datetime.utcnow().isoformat(),
            total_processing_time_ms=processing_time_ms,
            pipeline_state=state,
        )
    
    def _generate_summary(
        self,
        dr_result: DiabeticRetinopathyResult,
        risk: RiskAssessment,
        findings: List[ClinicalFinding]
    ) -> str:
        """Generate clinical summary text."""
        parts = []
        
        # DR status
        if dr_result:
            grade_info = DR_GRADE_CRITERIA.get(DRGrade(dr_result.grade), {})
            parts.append(
                f"Diabetic Retinopathy: {grade_info.get('description', 'Unknown')} "
                f"(Grade {dr_result.grade})"
            )
        
        # Risk level
        if risk:
            parts.append(f"Overall Risk: {risk.category.upper()} (Score: {risk.score:.0f}/100)")
        
        # Key findings
        if findings:
            abnormal = [f for f in findings if f.severity in ["moderate", "severe"]]
            if abnormal:
                parts.append(f"Notable findings: {len(abnormal)} items requiring attention")
        
        return ". ".join(parts) + "."
    
    def _log_audit(
        self,
        session_id: str,
        patient_id: Optional[str],
        start_time: datetime,
        end_time: datetime,
        dr_grade: int,
        risk_score: float,
        quality_score: float,
    ) -> None:
        """Log audit entry."""
        if not self.audit_logger:
            return
        
        try:
            ctx = create_execution_context(session_id)
            self.audit_logger.log_session_end(ctx, True, dr_grade)
        except Exception as e:
            logger.error(f"Audit logging failed: {e}")


# Backward compatibility alias
RetinalPipelineService = ResearchGradeRetinalService

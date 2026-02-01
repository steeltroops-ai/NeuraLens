"""
Cognitive Pipeline Service - Production Grade
Orchestrates all pipeline stages with logging, timing, and error propagation.
"""

import asyncio
import time
import logging
import uuid
from datetime import datetime
from typing import List, Optional, Tuple

from ..schemas import (
    CognitiveSessionInput, CognitiveResponse, CognitiveFeatures,
    CognitiveRiskAssessment, DomainRiskDetail, RiskLevel,
    StageProgress, PipelineStage, ExplainabilityArtifact,
    TaskCompletionStatus
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database.repositories import AssessmentRepository
from app.database.models import User
from ..input.validator import CognitiveValidator
from ..features.extractor import FeatureExtractor
from ..clinical.risk_scorer import RiskScorer
from ..output.formatter import OutputFormatter
from ..errors.codes import ErrorCode, PipelineError
from ..config import config

logger = logging.getLogger(__name__)


class CognitiveService:
    """
    Production-grade cognitive pipeline orchestrator.
    
    Features:
    - Stage-level progress tracking
    - Structured logging
    - Confidence propagation
    - Graceful degradation
    """
    
    def __init__(self):
        self.validator = CognitiveValidator()
        self.extractor = FeatureExtractor()
        self.scorer = RiskScorer()
        self.formatter = OutputFormatter()
        self._request_count = 0
        self._last_request_at: Optional[datetime] = None
        self._request_lock = asyncio.Lock()
    
    async def process_session(self, data: CognitiveSessionInput, db: AsyncSession) -> CognitiveResponse:
        """
        Main entry point. Processes session through all pipeline stages.
        Returns structured response even on partial failure.
        """
        start_time = time.time()
        
        async with self._request_lock:
            self._request_count += 1
            self._last_request_at = datetime.now()
        
        stages: List[StageProgress] = []
        features: Optional[CognitiveFeatures] = None
        risk_assessment: Optional[CognitiveRiskAssessment] = None
        explainability: Optional[ExplainabilityArtifact] = None
        final_status = "success"
        error_code = None
        error_message = None
        
        logger.info(f"[COGNITIVE] Processing session: {data.session_id}")
        
        # =====================================================================
        # STAGE 1: VALIDATION
        # =====================================================================
        stage_start = time.time()
        try:
            stages.append(StageProgress(
                stage=PipelineStage.VALIDATING,
                stage_index=0,
                message="Validating session data",
                started_at=datetime.now()
            ))
            
            validation_errors = self.validator.validate(data)
            
            if validation_errors:
                logger.warning(f"[COGNITIVE] Validation failed: {validation_errors}")
                raise PipelineError(ErrorCode.E_INP_002, str(validation_errors))
            
            stages[-1].completed_at = datetime.now()
            stages[-1].duration_ms = (time.time() - stage_start) * 1000
            stages[-1].message = "Validation passed"
            logger.info(f"[COGNITIVE] Stage 1 complete: Validation ({stages[-1].duration_ms:.1f}ms)")
            
        except PipelineError as e:
            stages[-1].stage = PipelineStage.FAILED
            stages[-1].error = str(e)
            return self._build_error_response(
                data.session_id, stages, e, time.time() - start_time
            )
        
        # =====================================================================
        # STAGE 2: FEATURE EXTRACTION
        # =====================================================================
        stage_start = time.time()
        try:
            stages.append(StageProgress(
                stage=PipelineStage.EXTRACTING,
                stage_index=1,
                message="Extracting cognitive features",
                started_at=datetime.now()
            ))
            
            metrics = self.extractor.extract(data.tasks)
            
            # Aggregate domain scores
            domain_scores = {}
            valid_count = 0
            for m in metrics:
                if m.validity_flag:
                    valid_count += 1
                    domain = self._map_task_to_domain(m.task_id)
                    domain_scores[domain] = m.performance_score / 100.0
            
            if not domain_scores:
                raise PipelineError(ErrorCode.E_FEAT_003, "No valid task results")
            
            # Calculate fatigue (simplified: compare first vs last task performance)
            fatigue_index = self._calculate_fatigue(metrics)
            
            # Calculate consistency (variance in performance)
            consistency = self._calculate_consistency(metrics)
            
            features = CognitiveFeatures(
                domain_scores=domain_scores,
                raw_metrics=metrics,
                fatigue_index=fatigue_index,
                consistency_score=consistency,
                valid_task_count=valid_count,
                total_task_count=len(metrics)
            )
            
            stages[-1].completed_at = datetime.now()
            stages[-1].duration_ms = (time.time() - stage_start) * 1000
            stages[-1].message = f"Extracted {len(domain_scores)} domain scores"
            logger.info(f"[COGNITIVE] Stage 2 complete: Extraction ({stages[-1].duration_ms:.1f}ms)")
            
        except PipelineError as e:
            stages[-1].stage = PipelineStage.FAILED
            stages[-1].error = str(e)
            return self._build_error_response(
                data.session_id, stages, e, time.time() - start_time
            )
        except Exception as e:
            logger.exception("[COGNITIVE] Unexpected error in extraction")
            stages[-1].stage = PipelineStage.FAILED
            stages[-1].error = str(e)
            pe = PipelineError(ErrorCode.E_FEAT_001, str(e))
            return self._build_error_response(
                data.session_id, stages, pe, time.time() - start_time
            )
        
        # =====================================================================
        # STAGE 3: CLINICAL SCORING (with Age-Adjusted Norms)
        # =====================================================================
        stage_start = time.time()
        try:
            stages.append(StageProgress(
                stage=PipelineStage.SCORING,
                stage_index=2,
                message="Calculating clinical risk scores with normative comparison",
                started_at=datetime.now()
            ))
            
            # Extract patient demographics for age-adjusted scoring
            patient_age = None
            education_years = None
            if data.user_metadata:
                patient_age = data.user_metadata.get("age") or data.user_metadata.get("patient_age")
                education_years = data.user_metadata.get("education") or data.user_metadata.get("education_years")
                
                # Validate age
                if patient_age is not None:
                    try:
                        patient_age = int(patient_age)
                        if patient_age < 18 or patient_age > 100:
                            logger.warning(f"[COGNITIVE] Invalid patient age: {patient_age}, ignoring")
                            patient_age = None
                    except (ValueError, TypeError):
                        logger.warning(f"[COGNITIVE] Invalid patient age format: {patient_age}")
                        patient_age = None
                
                # Validate education
                if education_years is not None:
                    try:
                        education_years = int(education_years)
                        if education_years < 0 or education_years > 25:
                            logger.warning(f"[COGNITIVE] Invalid education years: {education_years}, ignoring")
                            education_years = None
                    except (ValueError, TypeError):
                        logger.warning(f"[COGNITIVE] Invalid education format: {education_years}")
                        education_years = None
            
            logger.info(f"[COGNITIVE] Scoring with demographics: age={patient_age}, education={education_years}")
            
            risk_assessment, explainability = self.scorer.score_with_explanation(
                features,
                patient_age=patient_age,
                education_years=education_years
            )
            recommendations = self.scorer.generate_recommendations(risk_assessment)
            
            stages[-1].completed_at = datetime.now()
            stages[-1].duration_ms = (time.time() - stage_start) * 1000
            stages[-1].message = f"Risk level: {risk_assessment.risk_level.value}"
            logger.info(f"[COGNITIVE] Stage 3 complete: Scoring ({stages[-1].duration_ms:.1f}ms)")
            
        except PipelineError as e:
            stages[-1].stage = PipelineStage.FAILED
            stages[-1].error = str(e)
            # Partial success - we have features but no risk
            final_status = "partial"
            error_code = e.error_def.code
            error_message = str(e)
            recommendations = []
        except Exception as e:
            logger.exception("[COGNITIVE] Unexpected error in scoring")
            stages[-1].stage = PipelineStage.FAILED
            stages[-1].error = str(e)
            final_status = "partial"
            error_code = ErrorCode.E_CLIN_001.code
            error_message = str(e)
            recommendations = []
        
        # =====================================================================
        # STAGE 4: OUTPUT GENERATION & PERSISTENCE
        # =====================================================================
        stage_start = time.time()
        stages.append(StageProgress(
            stage=PipelineStage.COMPLETE,
            stage_index=3,
            message="Generating response",
            started_at=datetime.now()
        ))
        
        total_duration_ms = (time.time() - start_time) * 1000
        
        # PERSISTENCE (Added)
        try:
            repo = AssessmentRepository(db)
            
            # Resolve patient_id (The subject of the assessment)
            patient_uuid = None
            if data.patient_id:
                try:
                    patient_uuid = uuid.UUID(data.patient_id)
                except ValueError:
                    logger.warning(f"[COGNITIVE] Invalid patient_id format: {data.patient_id}")

            # Resolve user_id (The clinician/operator)
            # TODO: Get actual logged-in user from context. For now, fallback to first admin/user.
            user_id = None
            result = await db.execute(select(User.id).limit(1))
            user_id = result.scalar_one_or_none()
            
            if user_id:
                # Create Assessment
                assessment = await repo.create_assessment(
                    user_id=user_id,
                    pipeline_type="cognitive",
                    session_id=data.session_id,
                    patient_id=patient_uuid,
                    status="completed" if final_status == "success" else final_status,
                    risk_score=risk_assessment.overall_risk_score if risk_assessment else None,
                    risk_level=risk_assessment.risk_level.value if risk_assessment else None,
                    confidence=risk_assessment.confidence_score if risk_assessment else None,
                    processing_time_ms=int(total_duration_ms),
                    results={
                        "features": features.model_dump() if features else None,
                        "explainability": explainability.model_dump() if explainability else None,
                        "recommendations": [r.model_dump() for r in recommendations]
                    }
                )
                
                # Save Cognitive Result
                if features and risk_assessment:
                    await repo.save_cognitive_result(
                        assessment_id=assessment.id,
                        overall_risk_score=risk_assessment.overall_risk_score,
                        risk_level=risk_assessment.risk_level.value,
                        confidence_score=risk_assessment.confidence_score,
                        attention_score=features.domain_scores.get("attention"),
                        memory_score=features.domain_scores.get("memory"),
                        executive_function_score=features.domain_scores.get("executive") or features.domain_scores.get("inhibition"),
                        processing_speed_score=features.domain_scores.get("processing_speed"),
                        tasks_completed=features.total_task_count,
                        valid_tasks=features.valid_task_count,
                        fatigue_index=features.fatigue_index,
                        consistency_score=features.consistency_score
                    )
                
                logger.info(f"[COGNITIVE] Results saved for session {data.session_id} (Assessment ID: {assessment.id})")
            else:
                logger.warning("[COGNITIVE] No user found to save assessment results")
                
        except Exception as e:
            logger.error(f"[COGNITIVE] Failed to save results to database: {e}")
            # Do not fail the request if persistence fails, but log it
        
        stages[-1].completed_at = datetime.now()
        stages[-1].duration_ms = (time.time() - stage_start) * 1000
        
        logger.info(f"[COGNITIVE] Pipeline complete: {total_duration_ms:.1f}ms, status={final_status}")
        
        return CognitiveResponse(
            session_id=data.session_id,
            pipeline_version=config.VERSION,
            timestamp=datetime.now(),
            processing_time_ms=total_duration_ms,
            status=final_status,
            stages=stages,
            risk_assessment=risk_assessment,
            features=features,
            recommendations=recommendations,
            explainability=explainability,
            error_code=error_code,
            error_message=error_message,
            recoverable=True
        )
    
    def _build_error_response(
        self, 
        session_id: str, 
        stages: List[StageProgress], 
        error: PipelineError,
        duration: float
    ) -> CognitiveResponse:
        """Build a structured error response"""
        return CognitiveResponse(
            session_id=session_id,
            pipeline_version=config.VERSION,
            timestamp=datetime.now(),
            processing_time_ms=duration * 1000,
            status="failed",
            stages=stages,
            risk_assessment=None,
            features=None,
            recommendations=[],
            explainability=None,
            error_code=error.error_def.code,
            error_message=str(error),
            recoverable=error.recoverable,
            retry_after_ms=error.error_def.retry_after_ms
        )
    
    def _map_task_to_domain(self, task_id: str) -> str:
        """Map task ID to cognitive domain"""
        # Working Memory
        if "n_back" in task_id or "memory" in task_id:
            return "memory"
        # Processing Speed
        elif "reaction" in task_id or "digit_symbol" in task_id:
            return "processing_speed"
        # Response Inhibition / Executive Control
        elif "go_no_go" in task_id:
            return "inhibition"
        # Attention / Cognitive Flexibility
        elif "stroop" in task_id:
            return "attention"
        # Executive Function / Task Switching
        elif "trail_making_b" in task_id:
            return "executive"
        # Visual Attention
        elif "trail_making_a" in task_id or "trail_making" in task_id:
            return "attention"
        else:
            return "general"
    
    def _calculate_fatigue(self, metrics: list) -> float:
        """Calculate fatigue index from performance decline"""
        if len(metrics) < 2:
            return 0.0
        
        valid = [m for m in metrics if m.validity_flag]
        if len(valid) < 2:
            return 0.0
        
        first_score = valid[0].performance_score
        last_score = valid[-1].performance_score
        
        if first_score == 0:
            return 0.0
        
        decline = (first_score - last_score) / first_score
        return max(0.0, min(1.0, decline))
    
    def _calculate_consistency(self, metrics: list) -> float:
        """Calculate consistency score (inverse of variance)"""
        valid = [m.performance_score for m in metrics if m.validity_flag]
        if len(valid) < 2:
            return 1.0
        
        import numpy as np
        variance = np.var(valid)
        # Normalize: low variance = high consistency
        # Assume max reasonable variance is 1000 (100^2 / 10)
        normalized = 1.0 - min(1.0, variance / 1000)
        return normalized
    
    def get_health(self) -> dict:
        """Return service health status"""
        return {
            "status": "ok",
            "service": "cognitive-pipeline",
            "version": config.VERSION,
            "request_count": self._request_count,
            "last_request_at": self._last_request_at.isoformat() if self._last_request_at else None
        }

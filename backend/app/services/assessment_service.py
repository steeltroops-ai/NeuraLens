"""
Assessment service providing CRUD operations for assessments and results
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime
import uuid
import logging

from app.models.assessment import Assessment, AssessmentResult, NRIResult
from app.services.database_service import DatabaseService
from app.schemas.assessment import (
    SpeechAnalysisResponse, RetinalAnalysisResponse, 
    MotorAssessmentResponse, CognitiveAssessmentResponse,
    NRIFusionResponse
)

logger = logging.getLogger(__name__)


class AssessmentService:
    """Service for managing assessments and results"""
    
    def __init__(self):
        self.assessment_db = DatabaseService(Assessment)
        self.result_db = DatabaseService(AssessmentResult)
        self.nri_db = DatabaseService(NRIResult)
    
    def create_assessment(
        self, 
        db: Session, 
        user_id: Optional[str] = None,
        modalities: List[str] = None,
        assessment_type: str = "full"
    ) -> Assessment:
        """Create a new assessment session"""
        try:
            assessment_data = {
                "session_id": str(uuid.uuid4()),
                "user_id": user_id,
                "modalities": modalities or ["speech", "retinal", "motor", "cognitive"],
                "assessment_type": assessment_type,
                "status": "in_progress",
                "created_at": datetime.utcnow()
            }
            
            assessment = self.assessment_db.create(db, obj_in=assessment_data)
            logger.info(f"Created assessment session: {assessment.session_id}")
            return assessment
            
        except SQLAlchemyError as e:
            logger.error(f"Error creating assessment: {str(e)}")
            raise
    
    def get_assessment(self, db: Session, session_id: str) -> Optional[Assessment]:
        """Get assessment by session ID"""
        try:
            return db.query(Assessment).filter(
                Assessment.session_id == session_id
            ).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting assessment {session_id}: {str(e)}")
            raise
    
    def get_user_assessments(
        self, 
        db: Session, 
        user_id: str,
        skip: int = 0,
        limit: int = 10
    ) -> List[Assessment]:
        """Get all assessments for a user"""
        try:
            return self.assessment_db.get_multi(
                db,
                skip=skip,
                limit=limit,
                filters={"user_id": user_id},
                order_by="created_at",
                order_desc=True
            )
        except SQLAlchemyError as e:
            logger.error(f"Error getting user assessments for {user_id}: {str(e)}")
            raise
    
    def update_assessment_status(
        self, 
        db: Session, 
        session_id: str, 
        status: str
    ) -> Optional[Assessment]:
        """Update assessment status"""
        try:
            assessment = self.get_assessment(db, session_id)
            if assessment:
                return self.assessment_db.update(
                    db, 
                    db_obj=assessment, 
                    obj_in={"status": status}
                )
            return None
        except SQLAlchemyError as e:
            logger.error(f"Error updating assessment status: {str(e)}")
            raise
    
    def store_speech_result(
        self, 
        db: Session, 
        assessment_id: int,
        result: SpeechAnalysisResponse
    ) -> AssessmentResult:
        """Store speech analysis result"""
        try:
            result_data = {
                "assessment_id": assessment_id,
                "modality": "speech",
                "risk_score": result.risk_score,
                "confidence": result.confidence,
                "processing_time": result.processing_time,
                "biomarkers": result.biomarkers.dict() if hasattr(result, 'biomarkers') else {},
                "raw_data": {
                    "timestamp": result.timestamp.isoformat(),
                    "session_id": result.session_id
                },
                "recommendations": getattr(result, 'recommendations', []),
                "file_info": getattr(result, 'file_info', {}),
                "created_at": datetime.utcnow()
            }
            
            return self.result_db.create(db, obj_in=result_data)
            
        except SQLAlchemyError as e:
            logger.error(f"Error storing speech result: {str(e)}")
            raise
    
    def store_retinal_result(
        self, 
        db: Session, 
        assessment_id: int,
        result: RetinalAnalysisResponse
    ) -> AssessmentResult:
        """Store retinal analysis result"""
        try:
            result_data = {
                "assessment_id": assessment_id,
                "modality": "retinal",
                "risk_score": result.risk_score,
                "confidence": result.confidence,
                "quality_score": getattr(result, 'quality_score', 0.0),
                "processing_time": result.processing_time,
                "biomarkers": result.biomarkers.dict() if hasattr(result, 'biomarkers') else {},
                "raw_data": {
                    "timestamp": result.timestamp.isoformat(),
                    "session_id": result.session_id,
                    "detected_conditions": getattr(result, 'detected_conditions', [])
                },
                "recommendations": getattr(result, 'recommendations', []),
                "file_info": getattr(result, 'image_info', {}),
                "created_at": datetime.utcnow()
            }
            
            return self.result_db.create(db, obj_in=result_data)
            
        except SQLAlchemyError as e:
            logger.error(f"Error storing retinal result: {str(e)}")
            raise
    
    def store_motor_result(
        self, 
        db: Session, 
        assessment_id: int,
        result: MotorAssessmentResponse
    ) -> AssessmentResult:
        """Store motor assessment result"""
        try:
            result_data = {
                "assessment_id": assessment_id,
                "modality": "motor",
                "risk_score": result.risk_score,
                "confidence": result.confidence,
                "processing_time": result.processing_time,
                "biomarkers": result.biomarkers.dict() if hasattr(result, 'biomarkers') else {},
                "raw_data": {
                    "timestamp": result.timestamp.isoformat(),
                    "session_id": result.session_id,
                    "assessment_type": result.assessment_type,
                    "movement_quality": result.movement_quality
                },
                "recommendations": getattr(result, 'recommendations', []),
                "created_at": datetime.utcnow()
            }
            
            return self.result_db.create(db, obj_in=result_data)
            
        except SQLAlchemyError as e:
            logger.error(f"Error storing motor result: {str(e)}")
            raise
    
    def store_cognitive_result(
        self, 
        db: Session, 
        assessment_id: int,
        result: CognitiveAssessmentResponse
    ) -> AssessmentResult:
        """Store cognitive assessment result"""
        try:
            result_data = {
                "assessment_id": assessment_id,
                "modality": "cognitive",
                "risk_score": result.risk_score,
                "confidence": result.confidence,
                "processing_time": result.processing_time,
                "biomarkers": result.biomarkers.dict() if hasattr(result, 'biomarkers') else {},
                "raw_data": {
                    "timestamp": result.timestamp.isoformat(),
                    "session_id": result.session_id,
                    "domain_scores": getattr(result, 'domain_scores', {}),
                    "test_battery": getattr(result, 'test_battery', [])
                },
                "recommendations": getattr(result, 'recommendations', []),
                "created_at": datetime.utcnow()
            }
            
            return self.result_db.create(db, obj_in=result_data)
            
        except SQLAlchemyError as e:
            logger.error(f"Error storing cognitive result: {str(e)}")
            raise
    
    def store_nri_result(
        self, 
        db: Session, 
        assessment_id: int,
        result: NRIFusionResponse
    ) -> NRIResult:
        """Store NRI fusion result"""
        try:
            nri_data = {
                "assessment_id": assessment_id,
                "nri_score": result.nri_score,
                "risk_category": result.risk_category,
                "confidence": result.confidence,
                "uncertainty": result.uncertainty,
                "consistency_score": result.consistency_score,
                "modality_contributions": [contrib.dict() for contrib in result.modality_contributions],
                "fusion_method": "bayesian",  # Default fusion method
                "processing_time": result.processing_time,
                "recommendations": result.recommendations,
                "follow_up_actions": result.follow_up_actions,
                "created_at": datetime.utcnow()
            }
            
            return self.nri_db.create(db, obj_in=nri_data)
            
        except SQLAlchemyError as e:
            logger.error(f"Error storing NRI result: {str(e)}")
            raise
    
    def get_assessment_results(
        self, 
        db: Session, 
        assessment_id: int
    ) -> List[AssessmentResult]:
        """Get all results for an assessment"""
        try:
            return self.result_db.get_multi(
                db,
                filters={"assessment_id": assessment_id},
                order_by="created_at"
            )
        except SQLAlchemyError as e:
            logger.error(f"Error getting assessment results: {str(e)}")
            raise
    
    def get_nri_result(
        self, 
        db: Session, 
        assessment_id: int
    ) -> Optional[NRIResult]:
        """Get NRI result for an assessment"""
        try:
            return db.query(NRIResult).filter(
                NRIResult.assessment_id == assessment_id
            ).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting NRI result: {str(e)}")
            raise


# Global service instance
assessment_service = AssessmentService()

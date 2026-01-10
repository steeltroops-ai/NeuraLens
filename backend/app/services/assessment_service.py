"""
Simplified assessment service for core CRUD operations
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from datetime import datetime
import uuid

from app.models.assessment import Assessment, AssessmentResult, NRIResult


class AssessmentService:
    """Simplified service for managing assessments and results"""
    
    def create_assessment(
        self, 
        db: Session, 
        user_id: Optional[str] = None,
        modalities: List[str] = None
    ) -> Assessment:
        """Create a new assessment session"""
        assessment = Assessment(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            modalities=modalities or ["speech", "retinal", "motor", "cognitive"],
            status="in_progress"
        )
        db.add(assessment)
        db.commit()
        db.refresh(assessment)
        return assessment
    
    def get_assessment(self, db: Session, session_id: str) -> Optional[Assessment]:
        """Get assessment by session ID"""
        return db.query(Assessment).filter(Assessment.session_id == session_id).first()
    
    def save_result(
        self, 
        db: Session, 
        assessment_id: int,
        modality: str,
        risk_score: float,
        confidence: float,
        biomarkers: Dict[str, Any]
    ) -> AssessmentResult:
        """Save assessment result"""
        result = AssessmentResult(
            assessment_id=assessment_id,
            modality=modality,
            risk_score=risk_score,
            confidence=confidence,
            biomarkers=biomarkers
        )
        db.add(result)
        db.commit()
        db.refresh(result)
        return result
    
    def save_nri_result(
        self,
        db: Session,
        assessment_id: int,
        nri_score: float,
        risk_category: str,
        confidence: float,
        modality_contributions: Dict[str, float]
    ) -> NRIResult:
        """Save NRI fusion result"""
        nri_result = NRIResult(
            assessment_id=assessment_id,
            nri_score=nri_score,
            risk_category=risk_category,
            confidence=confidence,
            modality_contributions=modality_contributions
        )
        db.add(nri_result)
        db.commit()
        db.refresh(nri_result)
        return nri_result
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

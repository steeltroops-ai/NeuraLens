"""
Database Persistence Utilities
Unified persistence layer for all pipelines to save results to Neon database.
"""

import logging
import uuid
from typing import Optional, Dict, Any
from uuid import UUID
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.database.repositories.assessment_repository import AssessmentRepository
from app.database.models import (
    Assessment,
    SpeechResult,
    CardiologyResult,
    RadiologyResult,
    DermatologyResult,
    RetinalResult,
    CognitiveResult
)

logger = logging.getLogger(__name__)


class PersistenceService:
    """Unified persistence service for all pipelines."""
    
    # Placeholder user ID for anonymous/unauthenticated requests
    DEFAULT_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000000")
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.repo = AssessmentRepository(session)
    
    @staticmethod
    def parse_patient_id(patient_id: Optional[str]) -> Optional[UUID]:
        """Parse patient_id string to UUID, handling 'ANONYMOUS' and invalid values."""
        if not patient_id or patient_id == "ANONYMOUS":
            return None
        try:
            return uuid.UUID(patient_id)
        except (ValueError, TypeError):
            logger.warning(f"Invalid patient_id format: {patient_id}")
            return None
    
    async def save_speech_assessment(
        self,
        session_id: str,
        patient_id: Optional[str] = None,
        result_data: Optional[Dict[str, Any]] = None
    ) -> Optional[UUID]:
        """
        Save speech analysis results to database.
        
        Args:
            session_id: Session identifier
            patient_id: Optional patient UUID string
            result_data: Analysis results dictionary
            
        Returns:
            Assessment ID if successful, None otherwise
        """
        try:
            pid = self.parse_patient_id(patient_id)
            
            # Create assessment record
            assessment = await self.repo.create_assessment(
                user_id=self.DEFAULT_USER_ID,
                pipeline_type="speech",
                session_id=session_id,
                patient_id=pid,
                status="completed"
            )
            
            # Create modality-specific result
            if result_data:
                speech_result = SpeechResult(
                    assessment_id=assessment.id,
                    duration_seconds=Decimal(str(result_data.get("duration_seconds", 0))),
                    sample_rate=result_data.get("sample_rate"),
                    audio_quality_score=Decimal(str(result_data.get("quality_score", 0.8))),
                    jitter=Decimal(str(result_data.get("jitter", 0))) if result_data.get("jitter") else None,
                    shimmer=Decimal(str(result_data.get("shimmer", 0))) if result_data.get("shimmer") else None,
                    hnr=Decimal(str(result_data.get("hnr", 0))) if result_data.get("hnr") else None,
                    cpps=Decimal(str(result_data.get("cpps", 0))) if result_data.get("cpps") else None,
                    speech_rate=Decimal(str(result_data.get("speech_rate", 0))) if result_data.get("speech_rate") else None,
                    parkinsons_probability=Decimal(str(result_data.get("pd_probability", 0))) if result_data.get("pd_probability") else None,
                )
                self.session.add(speech_result)
            
            await self.session.commit()
            logger.info(f"[{session_id}] DATABASE: Speech assessment saved with ID {assessment.id}")
            return assessment.id
            
        except Exception as e:
            logger.error(f"[{session_id}] DATABASE ERROR (speech): {e}")
            await self.session.rollback()
            return None
    
    async def save_cardiology_assessment(
        self,
        session_id: str,
        patient_id: Optional[str] = None,
        result_data: Optional[Dict[str, Any]] = None
    ) -> Optional[UUID]:
        """Save cardiology/ECG analysis results to database."""
        try:
            pid = self.parse_patient_id(patient_id)
            
            assessment = await self.repo.create_assessment(
                user_id=self.DEFAULT_USER_ID,
                pipeline_type="cardiology",
                session_id=session_id,
                patient_id=pid,
                status="completed"
            )
            
            if result_data:
                cardio_result = CardiologyResult(
                    assessment_id=assessment.id,
                    rhythm_classification=result_data.get("rhythm_classification"),
                    heart_rate_bpm=result_data.get("heart_rate"),
                    regularity=result_data.get("regularity"),
                    rmssd_ms=Decimal(str(result_data.get("rmssd", 0))) if result_data.get("rmssd") else None,
                    sdnn_ms=Decimal(str(result_data.get("sdnn", 0))) if result_data.get("sdnn") else None,
                    pnn50_percent=Decimal(str(result_data.get("pnn50", 0))) if result_data.get("pnn50") else None,
                    arrhythmia_detected=result_data.get("arrhythmia_detected", False),
                    arrhythmia_types=result_data.get("arrhythmia_types", []),
                )
                self.session.add(cardio_result)
            
            await self.session.commit()
            logger.info(f"[{session_id}] DATABASE: Cardiology assessment saved with ID {assessment.id}")
            return assessment.id
            
        except Exception as e:
            logger.error(f"[{session_id}] DATABASE ERROR (cardiology): {e}")
            await self.session.rollback()
            return None
    
    async def save_radiology_assessment(
        self,
        session_id: str,
        patient_id: Optional[str] = None,
        result_data: Optional[Dict[str, Any]] = None
    ) -> Optional[UUID]:
        """Save radiology/X-ray analysis results to database."""
        try:
            pid = self.parse_patient_id(patient_id)
            
            assessment = await self.repo.create_assessment(
                user_id=self.DEFAULT_USER_ID,
                pipeline_type="radiology",
                session_id=session_id,
                patient_id=pid,
                status="completed"
            )
            
            if result_data:
                radio_result = RadiologyResult(
                    assessment_id=assessment.id,
                    modality_type="chest_xray",
                    primary_condition=result_data.get("primary_condition"),
                    primary_probability=Decimal(str(result_data.get("primary_probability", 0))) if result_data.get("primary_probability") else None,
                    primary_severity=result_data.get("severity"),
                    findings=result_data.get("findings", []),
                    quality_score=Decimal(str(result_data.get("quality_score", 0.8))) if result_data.get("quality_score") else None,
                )
                self.session.add(radio_result)
            
            await self.session.commit()
            logger.info(f"[{session_id}] DATABASE: Radiology assessment saved with ID {assessment.id}")
            return assessment.id
            
        except Exception as e:
            logger.error(f"[{session_id}] DATABASE ERROR (radiology): {e}")
            await self.session.rollback()
            return None
    
    async def save_dermatology_assessment(
        self,
        session_id: str,
        patient_id: Optional[str] = None,
        result_data: Optional[Dict[str, Any]] = None
    ) -> Optional[UUID]:
        """Save dermatology/skin lesion analysis results to database."""
        try:
            pid = self.parse_patient_id(patient_id)
            
            assessment = await self.repo.create_assessment(
                user_id=self.DEFAULT_USER_ID,
                pipeline_type="dermatology",
                session_id=session_id,
                patient_id=pid,
                status="completed"
            )
            
            if result_data:
                derm_result = DermatologyResult(
                    assessment_id=assessment.id,
                    primary_classification=result_data.get("primary_classification"),
                    melanoma_suspicion=result_data.get("melanoma_classification"),
                    asymmetry_score=Decimal(str(result_data.get("asymmetry_score", 0))) if result_data.get("asymmetry_score") else None,
                    border_score=Decimal(str(result_data.get("border_score", 0))) if result_data.get("border_score") else None,
                    color_score=Decimal(str(result_data.get("color_score", 0))) if result_data.get("color_score") else None,
                    diameter_mm=Decimal(str(result_data.get("diameter_mm", 0))) if result_data.get("diameter_mm") else None,
                    body_location=result_data.get("body_location"),
                )
                self.session.add(derm_result)
            
            await self.session.commit()
            logger.info(f"[{session_id}] DATABASE: Dermatology assessment saved with ID {assessment.id}")
            return assessment.id
            
        except Exception as e:
            logger.error(f"[{session_id}] DATABASE ERROR (dermatology): {e}")
            await self.session.rollback()
            return None


# Convenience function for getting persistence service
async def get_persistence_service(db: AsyncSession) -> PersistenceService:
    """Get persistence service instance."""
    return PersistenceService(db)

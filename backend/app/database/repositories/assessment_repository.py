"""
Database Repository - Assessment Repository
Data access layer for assessments and related entities
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc
from sqlalchemy.orm import selectinload
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime, timedelta

from app.database.models import (
    Assessment,
    PipelineStage,
    BiomarkerValue,
    RetinalResult,
    SpeechResult,
    CardiologyResult,
    RadiologyResult,
    DermatologyResult,
    CognitiveResult
)


class AssessmentRepository:
    """Repository pattern for assessment data access"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_assessment(
        self,
        user_id: UUID,
        pipeline_type: str,
        session_id: str,
        organization_id: Optional[UUID] = None,
        patient_id: Optional[UUID] = None,
        **kwargs
    ) -> Assessment:
        """Create new assessment"""
        assessment = Assessment(
            user_id=user_id,
            pipeline_type=pipeline_type,
            session_id=session_id,
            organization_id=organization_id,
            patient_id=patient_id,
            status='pending',
            **kwargs
        )
        self.session.add(assessment)
        await self.session.flush()
        await self.session.refresh(assessment)
        return assessment
    
    async def get_assessment_by_id(
        self,
        assessment_id: UUID,
        load_relationships: bool = False
    ) -> Optional[Assessment]:
        """Get assessment by ID"""
        query = select(Assessment).where(Assessment.id == assessment_id)
        
        if load_relationships:
            query = query.options(
                selectinload(Assessment.biomarker_values),
                selectinload(Assessment.pipeline_stages),
                selectinload(Assessment.ai_explanations)
            )
        
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_assessment_by_session_id(self, session_id: str) -> Optional[Assessment]:
        """Get assessment by session ID"""
        query = select(Assessment).where(Assessment.session_id == session_id)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_user_assessments(
        self,
        user_id: UUID,
        pipeline_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Assessment]:
        """Get user's assessment history"""
        query = select(Assessment).where(
            and_(
                Assessment.user_id == user_id,
                Assessment.deleted_at.is_(None)
            )
        )
        
        if pipeline_type:
            query = query.where(Assessment.pipeline_type == pipeline_type)
        
        if status:
            query = query.where(Assessment.status == status)
        
        query = query.order_by(desc(Assessment.created_at)).limit(limit).offset(offset)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_assessments_requiring_review(
        self,
        organization_id: Optional[UUID] = None,
        limit: int = 100
    ) -> List[Assessment]:
        """Get assessments flagged for review"""
        query = select(Assessment).where(
            and_(
                Assessment.requires_review == True,
                Assessment.deleted_at.is_(None)
            )
        )
        
        if organization_id:
            query = query.where(Assessment.organization_id == organization_id)
        
        query = query.order_by(desc(Assessment.created_at)).limit(limit)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def update_assessment_status(
        self,
        assessment_id: UUID,
        status: str,
        **kwargs
    ) -> Assessment:
        """Update assessment status and metadata"""
        assessment = await self.get_assessment_by_id(assessment_id)
        if not assessment:
            raise ValueError(f"Assessment {assessment_id} not found")
        
        assessment.status = status
        if status == 'completed':
            assessment.completed_at = datetime.utcnow()
        
        for key, value in kwargs.items():
            setattr(assessment, key, value)
        
        await self.session.flush()
        await self.session.refresh(assessment)
        return assessment
    
    async def add_pipeline_stage(
        self,
        assessment_id: UUID,
        stage_name: str,
        stage_index: int,
        status: str = 'pending',
        **kwargs
    ) -> PipelineStage:
        """Add pipeline stage record"""
        stage = PipelineStage(
            assessment_id=assessment_id,
            stage_name=stage_name,
            stage_index=stage_index,
            status=status,
            started_at=datetime.utcnow() if status == 'running' else None,
            **kwargs
        )
        self.session.add(stage)
        await self.session.flush()
        await self.session.refresh(stage)
        return stage
    
    async def save_biomarkers(
        self,
        assessment_id: UUID,
        biomarkers: List[Dict[str, Any]]
    ) -> List[BiomarkerValue]:
        """Bulk insert biomarker values"""
        biomarker_objs = [
            BiomarkerValue(assessment_id=assessment_id, **bm)
            for bm in biomarkers
        ]
        self.session.add_all(biomarker_objs)
        await self.session.flush()
        return biomarker_objs
    
    async def get_user_biomarker_history(
        self,
        user_id: UUID,
        biomarker_name: str,
        pipeline_type: Optional[str] = None,
        days: int = 90
    ) -> List[BiomarkerValue]:
        """Get biomarker time series for user"""
        since = datetime.utcnow() - timedelta(days=days)
        
        query = select(BiomarkerValue).join(Assessment).where(
            and_(
                Assessment.user_id == user_id,
                BiomarkerValue.biomarker_name == biomarker_name,
                BiomarkerValue.created_at >= since,
                Assessment.deleted_at.is_(None)
            )
        )
        
        if pipeline_type:
            query = query.where(Assessment.pipeline_type == pipeline_type)
        
        query = query.order_by(BiomarkerValue.created_at)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def save_retinal_result(
        self,
        assessment_id: UUID,
        **kwargs
    ) -> RetinalResult:
        """Save retinal-specific results"""
        result = RetinalResult(assessment_id=assessment_id, **kwargs)
        self.session.add(result)
        await self.session.flush()
        await self.session.refresh(result)
        return result
    
    async def save_speech_result(
        self,
        assessment_id: UUID,
        **kwargs
    ) -> SpeechResult:
        """Save speech-specific results"""
        result = SpeechResult(assessment_id=assessment_id, **kwargs)
        self.session.add(result)
        await self.session.flush()
        await self.session.refresh(result)
        return result
    
    async def save_cardiology_result(
        self,
        assessment_id: UUID,
        **kwargs
    ) -> CardiologyResult:
        """Save cardiology-specific results"""
        result = CardiologyResult(assessment_id=assessment_id, **kwargs)
        self.session.add(result)
        await self.session.flush()
        await self.session.refresh(result)
        return result
    
    async def save_cognitive_result(
        self,
        assessment_id: UUID,
        **kwargs
    ) -> CognitiveResult:
        """Save cognitive-specific results"""
        result = CognitiveResult(assessment_id=assessment_id, **kwargs)
        self.session.add(result)
        await self.session.flush()
        await self.session.refresh(result)
        return result
    
    async def get_assessment_statistics(
        self,
        user_id: Optional[UUID] = None,
        organization_id: Optional[UUID] = None,
        pipeline_type: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get assessment statistics"""
        since = datetime.utcnow() - timedelta(days=days)
        
        query = select(
            func.count(Assessment.id).label('total'),
            func.avg(Assessment.risk_score).label('avg_risk'),
            func.avg(Assessment.confidence).label('avg_confidence'),
            func.avg(Assessment.processing_time_ms).label('avg_processing_time')
        ).where(
            and_(
                Assessment.created_at >= since,
                Assessment.status == 'completed',
                Assessment.deleted_at.is_(None)
            )
        )
        
        if user_id:
            query = query.where(Assessment.user_id == user_id)
        
        if organization_id:
            query = query.where(Assessment.organization_id == organization_id)
        
        if pipeline_type:
            query = query.where(Assessment.pipeline_type == pipeline_type)
        
        result = await self.session.execute(query)
        row = result.one()
        
        return {
            'total_assessments': row.total or 0,
            'avg_risk_score': float(row.avg_risk or 0),
            'avg_confidence': float(row.avg_confidence or 0),
            'avg_processing_time_ms': int(row.avg_processing_time or 0)
        }
    
    async def soft_delete_assessment(self, assessment_id: UUID) -> Assessment:
        """Soft delete assessment (HIPAA compliant)"""
        assessment = await self.get_assessment_by_id(assessment_id)
        if not assessment:
            raise ValueError(f"Assessment {assessment_id} not found")
        
        assessment.deleted_at = datetime.utcnow()
        await self.session.flush()
        await self.session.refresh(assessment)
        return assessment

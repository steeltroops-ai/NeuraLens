"""
Database Repository - Patient Repository
Data access layer for patient management
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc, func
from sqlalchemy.orm import selectinload
from typing import List, Optional
from uuid import UUID
from datetime import datetime

from app.database.models import Patient
from app.database.models import Assessment

class PatientRepository:
    """Repository pattern for patient data access"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_patient(self, full_name: str, phone_number: str, **kwargs) -> Patient:
        """Create a new patient record"""
        patient = Patient(
            full_name=full_name,
            phone_number=phone_number,
            **kwargs
        )
        self.session.add(patient)
        await self.session.flush()
        await self.session.refresh(patient)
        return patient

    async def get_patient_by_id(self, patient_id: UUID) -> Optional[Patient]:
        """Get patient by UUID"""
        query = select(Patient).where(Patient.id == patient_id)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_patient_by_phone(self, phone_number: str) -> Optional[Patient]:
        """Get patient by unique phone number"""
        query = select(Patient).where(Patient.phone_number == phone_number)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def search_patients(
        self,
        query_text: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Patient]:
        """Search patients by name or phone"""
        # Case insensitive search
        search_filter = or_(
            Patient.full_name.ilike(f"%{query_text}%"),
            Patient.phone_number.ilike(f"%{query_text}%")
        )
        
        query = select(Patient).where(search_filter)\
            .order_by(Patient.full_name)\
            .limit(limit)\
            .offset(offset)
            
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_all_patients(self, limit: int = 50, offset: int = 0) -> List[Patient]:
        """Get all patients with pagination"""
        query = select(Patient).order_by(Patient.created_at.desc()).limit(limit).offset(offset)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_patient_assessments(
        self,
        patient_id: UUID,
        limit: int = 50,
        offset: int = 0
    ) -> List[Assessment]:
        """Get assessment history for a patient"""
        query = select(Assessment).where(
            and_(
                Assessment.patient_id == patient_id,
                Assessment.deleted_at.is_(None)
            )
        ).order_by(Assessment.created_at.desc()).limit(limit).offset(offset)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def update_patient(self, patient_id: UUID, **kwargs) -> Patient:
        """Update patient details"""
        patient = await self.get_patient_by_id(patient_id)
        if not patient:
            raise ValueError(f"Patient {patient_id} not found")
        
        for key, value in kwargs.items():
            if value is not None:
                setattr(patient, key, value)
        
        await self.session.flush()
        await self.session.refresh(patient)
        return patient

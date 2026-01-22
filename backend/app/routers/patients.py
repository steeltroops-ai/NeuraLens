"""
API Router - Patient Management
Endpoints for managing patient records
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, Any, Dict
from uuid import UUID

from app.database import get_db
from app.database.repositories.patient_repository import PatientRepository
from app.schemas.patient import Patient, PatientCreate, PatientUpdate

router = APIRouter()

@router.post("/", response_model=Patient, status_code=status.HTTP_201_CREATED)
async def create_patient(
    patient: PatientCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new patient record"""
    repo = PatientRepository(db)
    
    # check for existing phone number
    if await repo.get_patient_by_phone(patient.phone_number):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Patient with phone number {patient.phone_number} already exists"
        )
        
    new_patient = await repo.create_patient(**patient.model_dump())
    return new_patient

@router.get("/", response_model=List[Patient])
async def search_patients(
    q: Optional[str] = Query(None, description="Search query (name or phone)"),
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    """Search patients by name or phone"""
    repo = PatientRepository(db)
    if q:
        return await repo.search_patients(q, limit=limit, offset=offset)
    return await repo.get_all_patients(limit=limit, offset=offset)

@router.get("/{patient_id}", response_model=Patient)
async def get_patient(
    patient_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get patient details by ID"""
    repo = PatientRepository(db)
    patient = await repo.get_patient_by_id(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient

@router.put("/{patient_id}", response_model=Patient)
async def update_patient(
    patient_id: UUID,
    patient_update: PatientUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update patient details"""
    repo = PatientRepository(db)
    patient = await repo.get_patient_by_id(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
        
    updated = await repo.update_patient(patient_id, **patient_update.model_dump(exclude_unset=True))
    return updated

@router.get("/{patient_id}/assessments")
async def get_patient_assessments(
    patient_id: UUID,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
) -> List[Dict[str, Any]]:
    """Get assessment history for a patient"""
    repo = PatientRepository(db)
    assessments = await repo.get_patient_assessments(patient_id, limit, offset)
    return assessments

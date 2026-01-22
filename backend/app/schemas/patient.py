from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from uuid import UUID

class PatientBase(BaseModel):
    full_name: str = Field(..., min_length=1, description="Full name of the patient")
    phone_number: str = Field(..., min_length=3, description="Unique phone identifier")
    age: Optional[int] = Field(None, ge=0, le=150)
    gender: Optional[str] = None
    medical_notes: Optional[str] = None

class PatientCreate(PatientBase):
    pass

class PatientUpdate(BaseModel):
    full_name: Optional[str] = None
    phone_number: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    medical_notes: Optional[str] = None

class Patient(PatientBase):
    id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

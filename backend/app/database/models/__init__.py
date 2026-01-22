"""
Database Models Package
SQLAlchemy ORM models for MediLens
"""

# Import all models to register them with Base
from .identity import User, Organization, UserProfile, Role, UserRole
from .assessment import Assessment, PipelineStage, BiomarkerValue
from .modality_results import (
    RetinalResult,
    SpeechResult,
    CardiologyResult,
    RadiologyResult,
    DermatologyResult,
    CognitiveResult
)
from .ai_and_files import (
    ChatThread,
    ChatMessage,
    AIExplanation,
    UploadedFile,
    AuditEvent
)
from .patient import Patient

__all__ = [
    # Identity & Access
    "User",
    "Organization",
    "UserProfile",
    "Role",
    "UserRole",
    
    # Assessment Core
    "Assessment",
    "PipelineStage",
    "BiomarkerValue",
    
    # Modality Results
    "RetinalResult",
    "SpeechResult",
    "CardiologyResult",
    "RadiologyResult",
    "DermatologyResult",
    "CognitiveResult",
    
    # AI & Files
    "ChatThread",
    "ChatMessage",
    "AIExplanation",
    "UploadedFile",
    "AuditEvent",
    "Patient",
]

"""
SQLAlchemy Models - Conversational AI & Files
Chat, AI Explanations, File Management
"""

from sqlalchemy import (
    Column, String, DateTime, Integer, ForeignKey,
    JSON, Text, Boolean, BigInteger
)
from sqlalchemy.dialects.postgresql import UUID, INET
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.database import Base


class ChatThread(Base):
    """Chat conversation threads"""
    __tablename__ = "chat_threads"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    title = Column(String(255))
    context = Column(JSON, default=dict)
    
    # Status
    is_active = Column(Boolean, default=True, index=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_message_at = Column(DateTime(timezone=True), index=True)
    
    # Relationships
    user = relationship("User", back_populates="chat_threads")
    messages = relationship("ChatMessage", back_populates="thread", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ChatThread(id={self.id}, user_id={self.user_id})>"


class ChatMessage(Base):
    """Individual chat messages"""
    __tablename__ = "chat_messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    thread_id = Column(UUID(as_uuid=True), ForeignKey("chat_threads.id", ondelete="CASCADE"), nullable=False, index=True)
    
    role = Column(String(20), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    
    # AI metadata
    model_used = Column(String(50))
    tokens_used = Column(Integer)
    confidence = Column(JSON)
    sources = Column(JSON, default=list)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Relationships
    thread = relationship("ChatThread", back_populates="messages")
    
    def __repr__(self):
        return f"<ChatMessage(id={self.id}, role={self.role})>"


class AIExplanation(Base):
    """AI-generated explanations for assessments"""
    __tablename__ = "ai_explanations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    assessment_id = Column(UUID(as_uuid=True), ForeignKey("assessments.id", ondelete="CASCADE"), nullable=False, index=True)
    
    explanation_text = Column(Text, nullable=False)
    
    # Voice synthesis
    voice_generated = Column(Boolean, default=False)
    voice_url = Column(Text)
    voice_duration_ms = Column(Integer)
    
    # Metadata
    model_used = Column(String(50))
    tokens_used = Column(Integer)
    generation_time_ms = Column(Integer)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    assessment = relationship("Assessment", back_populates="ai_explanations")
    
    def __repr__(self):
        return f"<AIExplanation(id={self.id}, assessment_id={self.assessment_id})>"


class UploadedFile(Base):
    """Uploaded file metadata and tracking"""
    __tablename__ = "uploaded_files"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    assessment_id = Column(UUID(as_uuid=True), ForeignKey("assessments.id", ondelete="CASCADE"), index=True)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id", ondelete="SET NULL"), index=True)
    
    # File metadata
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255))
    content_type = Column(String(100))
    file_size_bytes = Column(BigInteger)
    
    # Storage
    storage_path = Column(Text)
    storage_provider = Column(String(50), default='local')
    
    # Processing
    processing_status = Column(String(20), default='pending')
    
    # Security
    file_hash = Column(String(64), index=True)
    is_encrypted = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    deleted_at = Column(DateTime(timezone=True))
    
    # Relationships
    user = relationship("User", back_populates="uploaded_files")
    assessment = relationship("Assessment", back_populates="uploaded_files")
    patient = relationship("Patient", back_populates="uploaded_files")
    
    def __repr__(self):
        return f"<UploadedFile(id={self.id}, filename={self.filename})>"


class AuditEvent(Base):
    """Audit log for compliance"""
    __tablename__ = "audit_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Who
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), index=True)
    actor_type = Column(String(50), nullable=False)
    actor_ip = Column(INET)
    
    # What
    event_type = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(50), index=True)
    resource_id = Column(UUID(as_uuid=True), index=True)
    
    # Context
    action = Column(String(50), nullable=False)
    details = Column(JSON, default=dict)
    
    # Compliance
    hipaa_relevant = Column(Boolean, default=False, index=True)
    phi_accessed = Column(Boolean, default=False)
    
    # Session tracking
    session_id = Column(String(255))
    user_agent = Column(Text)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    def __repr__(self):
        return f"<AuditEvent(id={self.id}, event_type={self.event_type})>"

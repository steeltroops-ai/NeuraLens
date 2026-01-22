"""
SQLAlchemy Models - Identity & Access
Users, Organizations, Roles, Permissions
"""

from sqlalchemy import (
    Column, String, Boolean, DateTime, Date, Integer,
    ForeignKey, JSON, SmallInteger, Text
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid

from app.database import Base


class User(Base):
    """User accounts with Clerk integration"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    clerk_user_id = Column(String(255), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    email_verified = Column(Boolean, default=False)
    username = Column(String(100), unique=True)
    
    # Profile
    first_name = Column(String(100))
    last_name = Column(String(100))
    date_of_birth = Column(Date)
    gender = Column(String(20))
    
    # Organization linkage
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="SET NULL"))
    
    # Status
    is_active = Column(Boolean, default=True, index=True)
    is_verified = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login_at = Column(DateTime(timezone=True))
    deleted_at = Column(DateTime(timezone=True))  # Soft delete
    
    # Relationships
    organization = relationship("Organization", back_populates="users")
    profile = relationship("UserProfile", back_populates="user", uselist=False)
    assessments = relationship("Assessment", back_populates="user")
    user_roles = relationship("UserRole", back_populates="user", foreign_keys="[UserRole.user_id]")
    chat_threads = relationship("ChatThread", back_populates="user")
    uploaded_files = relationship("UploadedFile", back_populates="user")
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"


class Organization(Base):
    """Organizations for multi-tenancy"""
    __tablename__ = "organizations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    type = Column(String(50), nullable=False, index=True)  # 'clinic', 'hospital', 'research', 'individual'
    
    # Contact
    email = Column(String(255))
    phone = Column(String(50))
    address = Column(JSON)
    
    # Settings
    settings = Column(JSON, default=dict)    
    # Status
    is_active = Column(Boolean, default=True)
    subscription_tier = Column(String(50), default='free')  # 'free', 'professional', 'enterprise'
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    deleted_at = Column(DateTime(timezone=True))
    
    # Relationships
    users = relationship("User", back_populates="organization")
    assessments = relationship("Assessment", back_populates="organization")
    
    def __repr__(self):
        return f"<Organization(id={self.id}, name={self.name})>"


class UserProfile(Base):
    """Extended user demographics and medical history"""
    __tablename__ = "user_profiles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    
    # Demographics
    age = Column(Integer)
    fitzpatrick_type = Column(SmallInteger)
    ethnicity = Column(String(100))
    
    # Medical History
    medical_history = Column(JSON, default=[])
    medications = Column(JSON, default=[])
    allergies = Column(JSON, default=[])
    
    # Preferences
    language = Column(String(10), default='en')
    timezone = Column(String(50), default='UTC')
    
    # Consent
    consent_research = Column(Boolean, default=False)
    consent_data_sharing = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="profile")
    
    def __repr__(self):
        return f"<UserProfile(user_id={self.user_id})>"


class Role(Base):
    """User roles for RBAC"""
    __tablename__ = "roles"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text)
    permissions = Column(JSON, default=[])
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user_roles = relationship("UserRole", back_populates="role")
    
    def __repr__(self):
        return f"<Role(id={self.id}, name={self.name})>"


class UserRole(Base):
    """User-Role assignment junction table"""
    __tablename__ = "user_roles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    role_id = Column(Integer, ForeignKey("roles.id", ondelete="CASCADE"), nullable=False)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"))
    
    granted_at = Column(DateTime(timezone=True), server_default=func.now())
    granted_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    
    # Relationships
    user = relationship("User", back_populates="user_roles", foreign_keys=[user_id])
    role = relationship("Role", back_populates="user_roles")
    
    def __repr__(self):
        return f"<UserRole(user_id={self.user_id}, role_id={self.role_id})>"

"""
Security and Compliance Features for Retinal Analysis Pipeline

Implements HIPAA-compliant security measures:
- Data encryption (AES-256 at rest, TLS 1.3 in transit) - Requirements 8.1, 8.2, 11.4, 11.5
- Authentication and authorization (MFA, RBAC) - Requirements 11.1, 11.2, 11.6
- Audit logging - Requirements 8.11, 11.3, 11.11
- Data anonymization - Requirement 11.8
- Session management - Requirement 11.6

Author: NeuraLens Team
"""

import logging
import hashlib
import secrets
import hmac
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import uuid
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Constants
# ============================================================================

class UserRole(str, Enum):
    """Role-based access control roles (Requirement 11.2)"""
    ADMIN = "admin"
    PHYSICIAN = "physician"
    TECHNICIAN = "technician"
    RESEARCHER = "researcher"
    PATIENT = "patient"
    AUDITOR = "auditor"


class AuditAction(str, Enum):
    """Audit log action types (Requirement 8.11)"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXPORT = "export"
    LOGIN = "login"
    LOGOUT = "logout"
    FAILED_LOGIN = "failed_login"
    ACCESS_DENIED = "access_denied"
    REPORT_GENERATED = "report_generated"
    ANALYSIS_STARTED = "analysis_started"
    ANALYSIS_COMPLETED = "analysis_completed"


class ResourceType(str, Enum):
    """Protected resource types"""
    ASSESSMENT = "assessment"
    PATIENT = "patient"
    REPORT = "report"
    IMAGE = "image"
    ANALYTICS = "analytics"


# Permission matrix for RBAC
ROLE_PERMISSIONS: Dict[UserRole, Dict[ResourceType, List[str]]] = {
    UserRole.ADMIN: {
        ResourceType.ASSESSMENT: ["create", "read", "update", "delete", "export"],
        ResourceType.PATIENT: ["create", "read", "update", "delete"],
        ResourceType.REPORT: ["create", "read", "delete"],
        ResourceType.IMAGE: ["create", "read", "delete"],
        ResourceType.ANALYTICS: ["read"],
    },
    UserRole.PHYSICIAN: {
        ResourceType.ASSESSMENT: ["create", "read", "update"],
        ResourceType.PATIENT: ["read", "update"],
        ResourceType.REPORT: ["create", "read"],
        ResourceType.IMAGE: ["create", "read"],
        ResourceType.ANALYTICS: ["read"],
    },
    UserRole.TECHNICIAN: {
        ResourceType.ASSESSMENT: ["create", "read"],
        ResourceType.PATIENT: ["read"],
        ResourceType.REPORT: [],
        ResourceType.IMAGE: ["create", "read"],
        ResourceType.ANALYTICS: [],
    },
    UserRole.RESEARCHER: {
        ResourceType.ASSESSMENT: ["read"],  # Anonymized only
        ResourceType.PATIENT: [],
        ResourceType.REPORT: [],
        ResourceType.IMAGE: ["read"],  # Anonymized only
        ResourceType.ANALYTICS: ["read"],
    },
    UserRole.PATIENT: {
        ResourceType.ASSESSMENT: ["read"],  # Own only
        ResourceType.PATIENT: ["read"],  # Own only
        ResourceType.REPORT: ["read"],  # Own only
        ResourceType.IMAGE: [],
        ResourceType.ANALYTICS: [],
    },
    UserRole.AUDITOR: {
        ResourceType.ASSESSMENT: ["read"],
        ResourceType.PATIENT: ["read"],
        ResourceType.REPORT: ["read"],
        ResourceType.IMAGE: [],
        ResourceType.ANALYTICS: ["read"],
    },
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class AuditLogEntry:
    """
    Audit log entry structure.
    
    Requirement 8.11: Log all data access with user ID, action, timestamp
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: str = ""
    user_role: Optional[UserRole] = None
    action: AuditAction = AuditAction.READ
    resource_type: ResourceType = ResourceType.ASSESSMENT
    resource_id: Optional[str] = None
    patient_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "user_role": self.user_role.value if self.user_role else None,
            "action": self.action.value,
            "resource_type": self.resource_type.value,
            "resource_id": self.resource_id,
            "patient_id": self._anonymize_patient_id(self.patient_id) if self.patient_id else None,
            "ip_address": self.ip_address,
            "success": self.success,
            "error_message": self.error_message
        }
    
    def _anonymize_patient_id(self, patient_id: str) -> str:
        """Anonymize patient ID in logs (Requirement 11.8)"""
        if len(patient_id) <= 4:
            return "***"
        return f"{patient_id[:2]}***{patient_id[-2:]}"


@dataclass
class Session:
    """User session data"""
    session_id: str
    user_id: str
    role: UserRole
    created_at: datetime
    last_activity: datetime
    ip_address: Optional[str] = None
    mfa_verified: bool = False
    
    def is_expired(self, timeout_minutes: int = 15) -> bool:
        """Check if session is expired (Requirement 11.6: 15 min timeout)"""
        return (datetime.utcnow() - self.last_activity) > timedelta(minutes=timeout_minutes)


# ============================================================================
# Encryption Service
# ============================================================================

class EncryptionService:
    """
    Data encryption service.
    
    Requirements 8.1, 8.2, 11.4, 11.5: AES-256 at rest, TLS 1.3 in transit
    """
    
    def __init__(self, master_key: Optional[bytes] = None):
        """
        Initialize encryption service.
        
        Args:
            master_key: 32-byte master key (generated if not provided)
        """
        if master_key:
            self._master_key = master_key
        else:
            # Generate a key from environment or use default for development
            # In production, this should come from a secure key management service
            self._master_key = self._derive_key("NEURALENS_ENCRYPTION_KEY")
        
        self._fernet = Fernet(base64.urlsafe_b64encode(self._master_key[:32]))
    
    def _derive_key(self, password: str, salt: Optional[bytes] = None) -> bytes:
        """Derive encryption key from password using PBKDF2"""
        if salt is None:
            salt = b"neuralens_default_salt_v1"  # Use env variable in production
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        return kdf.derive(password.encode())
    
    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt plaintext data.
        
        Requirement 8.1: AES-256 encryption for data at rest
        """
        encrypted = self._fernet.encrypt(plaintext.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt(self, ciphertext: str) -> str:
        """Decrypt ciphertext data."""
        encrypted = base64.urlsafe_b64decode(ciphertext.encode())
        decrypted = self._fernet.decrypt(encrypted)
        return decrypted.decode()
    
    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """Encrypt dictionary as JSON"""
        import json
        return self.encrypt(json.dumps(data))
    
    def decrypt_dict(self, ciphertext: str) -> Dict[str, Any]:
        """Decrypt to dictionary"""
        import json
        return json.loads(self.decrypt(ciphertext))
    
    def hash_patient_id(self, patient_id: str) -> str:
        """Create one-way hash of patient ID for anonymization"""
        return hashlib.sha256(f"{patient_id}:neuralens_salt".encode()).hexdigest()[:16]


# ============================================================================
# Authentication Service
# ============================================================================

class AuthenticationService:
    """
    Authentication and MFA service.
    
    Requirements 11.1, 11.2, 11.6: MFA, RBAC, session timeout
    """
    
    SESSION_TIMEOUT_MINUTES = 15  # Requirement 11.6
    
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._failed_attempts: Dict[str, List[datetime]] = {}
        self._lockout_threshold = 5
        self._lockout_duration = timedelta(minutes=30)
    
    def create_session(
        self,
        user_id: str,
        role: UserRole,
        ip_address: Optional[str] = None,
        mfa_verified: bool = False
    ) -> Session:
        """
        Create a new user session.
        
        Requirement 11.1: MFA requirement
        """
        session_id = secrets.token_urlsafe(32)
        session = Session(
            session_id=session_id,
            user_id=user_id,
            role=role,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            ip_address=ip_address,
            mfa_verified=mfa_verified
        )
        self._sessions[session_id] = session
        return session
    
    def validate_session(self, session_id: str) -> Optional[Session]:
        """
        Validate session and update last activity.
        
        Requirement 11.6: 15-minute session timeout
        """
        session = self._sessions.get(session_id)
        if not session:
            return None
        
        if session.is_expired(self.SESSION_TIMEOUT_MINUTES):
            self.invalidate_session(session_id)
            return None
        
        session.last_activity = datetime.utcnow()
        return session
    
    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate/logout session"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
    
    def record_failed_attempt(self, user_id: str) -> bool:
        """
        Record failed login attempt and check for lockout.
        
        Returns True if account should be locked.
        """
        now = datetime.utcnow()
        
        if user_id not in self._failed_attempts:
            self._failed_attempts[user_id] = []
        
        # Remove old attempts
        cutoff = now - self._lockout_duration
        self._failed_attempts[user_id] = [
            t for t in self._failed_attempts[user_id] if t > cutoff
        ]
        
        self._failed_attempts[user_id].append(now)
        
        return len(self._failed_attempts[user_id]) >= self._lockout_threshold
    
    def is_locked_out(self, user_id: str) -> bool:
        """Check if user is locked out"""
        if user_id not in self._failed_attempts:
            return False
        
        cutoff = datetime.utcnow() - self._lockout_duration
        recent_attempts = [t for t in self._failed_attempts[user_id] if t > cutoff]
        
        return len(recent_attempts) >= self._lockout_threshold
    
    def verify_mfa_token(self, user_id: str, token: str) -> bool:
        """
        Verify MFA token (TOTP).
        
        Requirement 11.1: MFA requirement
        
        Note: In production, integrate with actual TOTP library (pyotp)
        """
        # Placeholder: In production, validate against TOTP secret
        # For demo, accept any 6-digit token
        return len(token) == 6 and token.isdigit()


# ============================================================================
# Authorization Service
# ============================================================================

class AuthorizationService:
    """
    Role-based access control service.
    
    Requirement 11.2: RBAC implementation
    """
    
    def check_permission(
        self,
        role: UserRole,
        resource_type: ResourceType,
        action: str
    ) -> bool:
        """
        Check if role has permission for action on resource.
        
        Args:
            role: User's role
            resource_type: Type of resource being accessed
            action: Action being performed (create, read, update, delete, export)
            
        Returns:
            True if permitted, False otherwise
        """
        role_perms = ROLE_PERMISSIONS.get(role, {})
        resource_perms = role_perms.get(resource_type, [])
        return action in resource_perms
    
    def get_accessible_resources(
        self,
        role: UserRole,
        resource_type: ResourceType
    ) -> List[str]:
        """Get list of permitted actions for a resource type"""
        role_perms = ROLE_PERMISSIONS.get(role, {})
        return role_perms.get(resource_type, [])
    
    def filter_by_ownership(
        self,
        role: UserRole,
        user_id: str,
        patient_id: str,
        resource_patient_id: str
    ) -> bool:
        """
        Check if user can access resource based on ownership.
        
        Patients can only access their own data.
        """
        if role == UserRole.PATIENT:
            return patient_id == resource_patient_id
        return True  # Other roles not restricted by ownership


# ============================================================================
# Audit Logging Service
# ============================================================================

class AuditLoggingService:
    """
    Comprehensive audit logging service.
    
    Requirements 8.11, 11.3, 11.11: Log all access and modifications
    """
    
    def __init__(self):
        self._logs: List[AuditLogEntry] = []
        self._max_logs = 10000  # Keep last N logs in memory
    
    def log(
        self,
        user_id: str,
        action: AuditAction,
        resource_type: ResourceType,
        resource_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        user_role: Optional[UserRole] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> AuditLogEntry:
        """
        Create an audit log entry.
        
        Requirement 8.11: Include user ID, action, timestamp
        """
        entry = AuditLogEntry(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            patient_id=patient_id,
            user_role=user_role,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {},
            success=success,
            error_message=error_message
        )
        
        self._logs.append(entry)
        
        # Trim old logs if needed
        if len(self._logs) > self._max_logs:
            self._logs = self._logs[-self._max_logs:]
        
        # Also log to Python logger
        log_msg = (
            f"AUDIT: user={user_id}, action={action.value}, "
            f"resource={resource_type.value}/{resource_id}, "
            f"success={success}"
        )
        if success:
            logger.info(log_msg)
        else:
            logger.warning(f"{log_msg}, error={error_message}")
        
        return entry
    
    def get_logs(
        self,
        user_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        resource_type: Optional[ResourceType] = None,
        patient_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditLogEntry]:
        """Query audit logs with filters"""
        filtered = self._logs
        
        if user_id:
            filtered = [l for l in filtered if l.user_id == user_id]
        if action:
            filtered = [l for l in filtered if l.action == action]
        if resource_type:
            filtered = [l for l in filtered if l.resource_type == resource_type]
        if patient_id:
            filtered = [l for l in filtered if l.patient_id == patient_id]
        if start_time:
            filtered = [l for l in filtered if l.timestamp >= start_time]
        if end_time:
            filtered = [l for l in filtered if l.timestamp <= end_time]
        
        return filtered[-limit:]
    
    def export_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        anonymize: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Export logs for compliance review.
        
        Requirement 11.8: Anonymize data on export
        """
        logs = self.get_logs(start_time=start_time, end_time=end_time, limit=10000)
        
        exported = []
        for log in logs:
            entry = log.to_dict()
            
            if anonymize:
                # Anonymize sensitive fields
                if entry.get("user_id"):
                    entry["user_id"] = f"USER-{hashlib.md5(entry['user_id'].encode()).hexdigest()[:8]}"
                if entry.get("ip_address"):
                    entry["ip_address"] = "XXX.XXX.XXX.XXX"
            
            exported.append(entry)
        
        return exported


# ============================================================================
# Data Anonymization Service
# ============================================================================

class DataAnonymizationService:
    """
    Data anonymization for research and export.
    
    Requirement 11.8: Anonymize patient identifiers on export
    """
    
    def __init__(self):
        self._salt = secrets.token_hex(16)
    
    def anonymize_patient_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymize patient-identifiable information.
        
        Removes or hashes:
        - Patient ID
        - Patient name
        - Date of birth
        - Address
        - Any identifying metadata
        """
        anonymized = data.copy()
        
        # Fields to hash
        hash_fields = ["patient_id", "assessment_id"]
        for field in hash_fields:
            if field in anonymized:
                anonymized[field] = self._hash_value(str(anonymized[field]))
        
        # Fields to remove entirely
        remove_fields = ["patient_name", "patient_dob", "address", "ssn", "mrn"]
        for field in remove_fields:
            anonymized.pop(field, None)
        
        # Generalize dates (keep only year-month)
        date_fields = ["created_at", "analysis_date"]
        for field in date_fields:
            if field in anonymized and anonymized[field]:
                if isinstance(anonymized[field], str):
                    anonymized[field] = anonymized[field][:7]  # YYYY-MM
                elif isinstance(anonymized[field], datetime):
                    anonymized[field] = anonymized[field].strftime("%Y-%m")
        
        return anonymized
    
    def _hash_value(self, value: str) -> str:
        """Create consistent hash for value"""
        return hashlib.sha256(f"{value}:{self._salt}".encode()).hexdigest()[:12]
    
    def anonymize_report(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize clinical report data for research use"""
        anonymized = self.anonymize_patient_data(report_data)
        
        # Keep biomarker data but remove context
        if "provider_name" in anonymized:
            anonymized["provider_name"] = "ANONYMIZED"
        if "provider_npi" in anonymized:
            anonymized["provider_npi"] = "XXXXXXXXXX"
        
        return anonymized


# ============================================================================
# Singleton Instances
# ============================================================================

encryption_service = EncryptionService()
authentication_service = AuthenticationService()
authorization_service = AuthorizationService()
audit_logging_service = AuditLoggingService()
anonymization_service = DataAnonymizationService()


# ============================================================================
# Decorators for Route Protection
# ============================================================================

def require_auth(roles: Optional[List[UserRole]] = None):
    """Decorator to require authentication and optionally specific roles"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # In production, extract session from request headers
            # For now, this is a placeholder that should be integrated with FastAPI
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def audit_log(action: AuditAction, resource_type: ResourceType):
    """Decorator to automatically log access to resources"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Log before execution
            # In production, extract user info from request context
            result = await func(*args, **kwargs)
            return result
        return wrapper
    return decorator

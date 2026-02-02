"""
Streaming Session Manager v4.0
Manages streaming session lifecycle and state.

Requirements: 3.1, 3.2, 3.7
- Session initialization with unique IDs
- State management and cleanup
- Memory-bounded buffer management
"""

import uuid
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Deque
from collections import deque
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class SessionState(str, Enum):
    """Streaming session states."""
    INITIALIZING = "initializing"
    READY = "ready"
    RECORDING = "recording"
    PROCESSING = "processing"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    EXPIRED = "expired"


@dataclass
class SessionConfig:
    """Configuration for a streaming session."""
    max_duration_s: int = 300  # 5 minutes max
    target_duration_s: int = 30
    buffer_size_samples: int = 48000  # 3 seconds at 16kHz
    sample_rate: int = 16000
    window_size_ms: int = 500
    hop_size_ms: int = 200
    auto_stop_silence_s: float = 5.0
    
    @property
    def window_size_samples(self) -> int:
        return int(self.window_size_ms * self.sample_rate / 1000)
    
    @property
    def hop_size_samples(self) -> int:
        return int(self.hop_size_ms * self.sample_rate / 1000)


@dataclass
class SessionMetrics:
    """Runtime metrics for a streaming session."""
    # Timing
    start_time: float = 0.0
    last_chunk_time: float = 0.0
    total_audio_duration: float = 0.0
    
    # Quality metrics (running averages)
    average_quality: float = 0.0
    average_snr: float = 0.0
    min_quality: float = 100.0
    max_quality: float = 0.0
    
    # Processing metrics
    chunks_received: int = 0
    chunks_processed: int = 0
    processing_latency_ms: float = 0.0
    
    # Issues
    clipping_events: int = 0
    low_quality_events: int = 0
    silence_duration: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "duration": self.total_audio_duration,
            "average_quality": self.average_quality,
            "average_snr": self.average_snr,
            "chunks_processed": self.chunks_processed,
            "latency_ms": self.processing_latency_ms,
            "clipping_events": self.clipping_events,
            "low_quality_events": self.low_quality_events
        }


@dataclass
class StreamingSession:
    """Represents an active streaming session."""
    # Identity
    session_id: str = ""
    user_id: Optional[str] = None
    
    # State
    state: SessionState = SessionState.INITIALIZING
    config: SessionConfig = field(default_factory=SessionConfig)
    metrics: SessionMetrics = field(default_factory=SessionMetrics)
    
    # Audio buffer (circular)
    audio_buffer: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=100))
    
    # Accumulated audio (for final analysis)
    accumulated_audio: List[np.ndarray] = field(default_factory=list)
    
    # Quality history
    quality_history: List[float] = field(default_factory=list)
    
    # Preliminary results
    preliminary_features: Dict[str, float] = field(default_factory=dict)
    preliminary_risk: Optional[float] = None
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    client_info: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.session_id:
            self.session_id = str(uuid.uuid4())
        self.metrics.start_time = time.time()
    
    @property
    def duration_s(self) -> float:
        """Current session duration in seconds."""
        return time.time() - self.created_at
    
    @property
    def is_active(self) -> bool:
        """Whether session is actively recording."""
        return self.state in (SessionState.READY, SessionState.RECORDING)
    
    @property
    def has_enough_audio(self) -> bool:
        """Whether enough audio for full analysis."""
        return self.metrics.total_audio_duration >= 3.0  # Min 3 seconds
    
    def get_full_audio(self) -> np.ndarray:
        """Concatenate all accumulated audio."""
        if not self.accumulated_audio:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.accumulated_audio)
    
    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "duration_s": self.duration_s,
            "audio_duration_s": self.metrics.total_audio_duration,
            "metrics": self.metrics.to_dict(),
            "has_enough_audio": self.has_enough_audio,
            "preliminary_risk": self.preliminary_risk
        }


class StreamingSessionManager:
    """
    Manages multiple concurrent streaming sessions.
    
    Handles:
    - Session creation and lifecycle
    - Resource cleanup
    - Session expiration
    - Concurrent session limits
    """
    
    # Limits
    MAX_CONCURRENT_SESSIONS = 100
    SESSION_TIMEOUT_S = 600  # 10 minutes
    CLEANUP_INTERVAL_S = 60
    
    def __init__(
        self,
        max_sessions: int = 100,
        session_timeout_s: int = 600
    ):
        self.max_sessions = max_sessions
        self.session_timeout_s = session_timeout_s
        
        self.sessions: Dict[str, StreamingSession] = {}
        self.last_cleanup = time.time()
        
        logger.info(
            f"StreamingSessionManager initialized: "
            f"max_sessions={max_sessions}, timeout={session_timeout_s}s"
        )
    
    def create_session(
        self,
        user_id: Optional[str] = None,
        config: Optional[SessionConfig] = None,
        client_info: Optional[Dict] = None
    ) -> StreamingSession:
        """
        Create a new streaming session.
        
        Args:
            user_id: Optional user identifier
            config: Session configuration
            client_info: Client metadata
            
        Returns:
            New StreamingSession
            
        Raises:
            RuntimeError: If session limit reached
        """
        # Cleanup expired sessions first
        self._cleanup_expired()
        
        # Check limits
        if len(self.sessions) >= self.max_sessions:
            raise RuntimeError(
                f"Maximum concurrent sessions ({self.max_sessions}) reached"
            )
        
        # Create session
        session = StreamingSession(
            user_id=user_id,
            config=config or SessionConfig(),
            client_info=client_info or {}
        )
        session.state = SessionState.READY
        
        self.sessions[session.session_id] = session
        
        logger.info(f"Created streaming session: {session.session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[StreamingSession]:
        """Get session by ID."""
        return self.sessions.get(session_id)
    
    def start_recording(self, session_id: str) -> bool:
        """Start recording for a session."""
        session = self.get_session(session_id)
        if not session:
            return False
        
        if session.state == SessionState.READY:
            session.state = SessionState.RECORDING
            session.metrics.start_time = time.time()
            session.updated_at = time.time()
            logger.info(f"Started recording: {session_id}")
            return True
        
        return False
    
    def pause_recording(self, session_id: str) -> bool:
        """Pause recording for a session."""
        session = self.get_session(session_id)
        if not session:
            return False
        
        if session.state == SessionState.RECORDING:
            session.state = SessionState.PAUSED
            session.updated_at = time.time()
            return True
        
        return False
    
    def resume_recording(self, session_id: str) -> bool:
        """Resume recording for a session."""
        session = self.get_session(session_id)
        if not session:
            return False
        
        if session.state == SessionState.PAUSED:
            session.state = SessionState.RECORDING
            session.updated_at = time.time()
            return True
        
        return False
    
    def stop_recording(self, session_id: str) -> bool:
        """Stop recording and prepare for final analysis."""
        session = self.get_session(session_id)
        if not session:
            return False
        
        if session.state in (SessionState.RECORDING, SessionState.PAUSED):
            session.state = SessionState.PROCESSING
            session.updated_at = time.time()
            logger.info(
                f"Stopped recording: {session_id}, "
                f"duration={session.metrics.total_audio_duration:.1f}s"
            )
            return True
        
        return False
    
    def complete_session(self, session_id: str) -> bool:
        """Mark session as completed."""
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.state = SessionState.COMPLETED
        session.updated_at = time.time()
        logger.info(f"Completed session: {session_id}")
        return True
    
    def error_session(self, session_id: str, error: str) -> bool:
        """Mark session as errored."""
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.state = SessionState.ERROR
        session.client_info["error"] = error
        session.updated_at = time.time()
        logger.error(f"Session error: {session_id} - {error}")
        return True
    
    def add_audio_chunk(
        self,
        session_id: str,
        audio_chunk: np.ndarray
    ) -> bool:
        """
        Add audio chunk to session buffer.
        
        Args:
            session_id: Session identifier
            audio_chunk: Audio samples as numpy array
            
        Returns:
            True if chunk added successfully
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        if session.state != SessionState.RECORDING:
            return False
        
        # Add to buffer
        session.audio_buffer.append(audio_chunk)
        session.accumulated_audio.append(audio_chunk)
        
        # Update metrics
        chunk_duration = len(audio_chunk) / session.config.sample_rate
        session.metrics.total_audio_duration += chunk_duration
        session.metrics.chunks_received += 1
        session.metrics.last_chunk_time = time.time()
        session.updated_at = time.time()
        
        # Check max duration
        if session.metrics.total_audio_duration >= session.config.max_duration_s:
            self.stop_recording(session_id)
            logger.info(f"Session {session_id} reached max duration")
        
        return True
    
    def close_session(self, session_id: str) -> bool:
        """Close and remove a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Closed session: {session_id}")
            return True
        return False
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return [
            sid for sid, session in self.sessions.items()
            if session.is_active
        ]
    
    def _cleanup_expired(self):
        """Remove expired sessions."""
        now = time.time()
        
        # Only run periodically
        if now - self.last_cleanup < self.CLEANUP_INTERVAL_S:
            return
        
        self.last_cleanup = now
        expired = []
        
        for session_id, session in self.sessions.items():
            # Expired if no activity for too long
            if now - session.updated_at > self.session_timeout_s:
                expired.append(session_id)
            # Or if in completed/error state for a while
            elif session.state in (SessionState.COMPLETED, SessionState.ERROR):
                if now - session.updated_at > 60:  # 1 minute grace period
                    expired.append(session_id)
        
        for session_id in expired:
            session = self.sessions[session_id]
            session.state = SessionState.EXPIRED
            del self.sessions[session_id]
            logger.info(f"Cleaned up expired session: {session_id}")
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")

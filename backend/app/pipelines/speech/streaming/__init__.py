"""
Real-Time Streaming Module v4.0
WebSocket-based streaming speech analysis.

Implements:
- StreamingSessionManager: Session lifecycle management
- StreamProcessor: Real-time feature extraction
- StreamingAnalyzer: Incremental analysis with quality feedback
- WebSocket handlers for client integration

Requirements: 3.1-3.8
"""

from .session import StreamingSessionManager, StreamingSession, SessionState
from .processor import StreamProcessor, ChunkResult
from .analyzer import StreamingAnalyzer, StreamingResult
from .handlers import WebSocketHandler

__all__ = [
    # Session management
    "StreamingSessionManager",
    "StreamingSession",
    "SessionState",
    
    # Processing
    "StreamProcessor",
    "ChunkResult",
    
    # Analysis
    "StreamingAnalyzer",
    "StreamingResult",
    
    # WebSocket
    "WebSocketHandler",
]

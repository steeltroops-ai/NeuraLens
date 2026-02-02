"""
WebSocket Handler v4.0
WebSocket protocol handling for streaming speech analysis.

Implements:
- Connection management
- Message serialization
- Error handling
- Session lifecycle over WebSocket
"""

import json
import time
import logging
import asyncio
import base64
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Callable, Any
from enum import Enum

import numpy as np

from .session import StreamingSessionManager, StreamingSession, SessionState
from .analyzer import StreamingAnalyzer

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """WebSocket message types."""
    # Client -> Server
    START_SESSION = "start_session"
    AUDIO_CHUNK = "audio_chunk"
    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"
    CANCEL = "cancel"
    
    # Server -> Client
    SESSION_STARTED = "session_started"
    CHUNK_RESULT = "chunk_result"
    QUALITY_UPDATE = "quality_update"
    SESSION_PAUSED = "session_paused"
    SESSION_RESUMED = "session_resumed"
    SESSION_STOPPED = "session_stopped"
    FINAL_RESULT = "final_result"
    ERROR = "error"
    
    # Bidirectional
    PING = "ping"
    PONG = "pong"


@dataclass
class WebSocketMessage:
    """Structured WebSocket message."""
    type: str
    payload: Dict
    session_id: Optional[str] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    def to_json(self) -> str:
        return json.dumps({
            "type": self.type,
            "session_id": self.session_id,
            "payload": self.payload,
            "timestamp": self.timestamp
        })
    
    @classmethod
    def from_json(cls, data: str) -> "WebSocketMessage":
        parsed = json.loads(data)
        return cls(
            type=parsed.get("type", ""),
            payload=parsed.get("payload", {}),
            session_id=parsed.get("session_id"),
            timestamp=parsed.get("timestamp", time.time())
        )


class WebSocketHandler:
    """
    WebSocket handler for real-time speech analysis.
    
    Manages the protocol for streaming audio to the analysis pipeline
    and sending back real-time feedback and results.
    """
    
    def __init__(
        self,
        session_manager: Optional[StreamingSessionManager] = None,
        sample_rate: int = 16000,
        chunk_size_ms: int = 200
    ):
        self.session_manager = session_manager or StreamingSessionManager()
        self.sample_rate = sample_rate
        self.chunk_size_ms = chunk_size_ms
        
        # Per-connection analyzers
        self.analyzers: Dict[str, StreamingAnalyzer] = {}
        
        # Message handlers
        self._handlers: Dict[str, Callable] = {
            MessageType.START_SESSION: self._handle_start_session,
            MessageType.AUDIO_CHUNK: self._handle_audio_chunk,
            MessageType.PAUSE: self._handle_pause,
            MessageType.RESUME: self._handle_resume,
            MessageType.STOP: self._handle_stop,
            MessageType.CANCEL: self._handle_cancel,
            MessageType.PING: self._handle_ping,
        }
    
    async def handle_message(
        self,
        message: str,
        send_callback: Callable[[str], Any]
    ) -> None:
        """
        Handle incoming WebSocket message.
        
        Args:
            message: Raw JSON message from client
            send_callback: Async function to send response
        """
        try:
            msg = WebSocketMessage.from_json(message)
            
            handler = self._handlers.get(msg.type)
            if handler:
                response = await handler(msg)
                if response:
                    await send_callback(response.to_json())
            else:
                logger.warning(f"Unknown message type: {msg.type}")
                await send_callback(self._error_response(
                    None, f"Unknown message type: {msg.type}"
                ).to_json())
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            await send_callback(self._error_response(
                None, "Invalid JSON message"
            ).to_json())
            
        except Exception as e:
            logger.error(f"Message handling error: {e}", exc_info=True)
            await send_callback(self._error_response(
                None, str(e)
            ).to_json())
    
    async def _handle_start_session(
        self,
        msg: WebSocketMessage
    ) -> WebSocketMessage:
        """Handle session start request."""
        try:
            # Extract config from payload
            user_id = msg.payload.get("user_id")
            client_info = msg.payload.get("client_info", {})
            
            # Create session
            session = self.session_manager.create_session(
                user_id=user_id,
                client_info=client_info
            )
            
            # Create analyzer for this session
            self.analyzers[session.session_id] = StreamingAnalyzer(
                sample_rate=self.sample_rate,
                chunk_size_ms=self.chunk_size_ms
            )
            
            # Start recording
            self.session_manager.start_recording(session.session_id)
            
            return WebSocketMessage(
                type=MessageType.SESSION_STARTED,
                session_id=session.session_id,
                payload={
                    "session": session.to_dict(),
                    "config": {
                        "sample_rate": self.sample_rate,
                        "chunk_size_ms": self.chunk_size_ms,
                        "max_duration_s": session.config.max_duration_s
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Session start failed: {e}")
            return self._error_response(None, f"Failed to start session: {e}")
    
    async def _handle_audio_chunk(
        self,
        msg: WebSocketMessage
    ) -> WebSocketMessage:
        """Handle incoming audio chunk."""
        session_id = msg.session_id
        if not session_id:
            return self._error_response(None, "Missing session_id")
        
        session = self.session_manager.get_session(session_id)
        if not session:
            return self._error_response(session_id, "Session not found")
        
        if session.state != SessionState.RECORDING:
            return self._error_response(
                session_id, 
                f"Session not recording: {session.state.value}"
            )
        
        try:
            # Decode audio data
            audio_b64 = msg.payload.get("audio")
            if not audio_b64:
                return self._error_response(session_id, "Missing audio data")
            
            audio_bytes = base64.b64decode(audio_b64)
            audio_chunk = np.frombuffer(audio_bytes, dtype=np.float32)
            
            # Add to session
            self.session_manager.add_audio_chunk(session_id, audio_chunk)
            
            # Process chunk
            analyzer = self.analyzers.get(session_id)
            if analyzer:
                result = analyzer.process_chunk(audio_chunk)
                
                # Update session metrics
                session.metrics.chunks_processed += 1
                session.metrics.processing_latency_ms = result.processing_time_ms
                
                # Update quality history
                session.quality_history.append(result.preliminary_quality_score)
                
                return WebSocketMessage(
                    type=MessageType.CHUNK_RESULT,
                    session_id=session_id,
                    payload=result.to_dict()
                )
            
            return None  # No response if no analyzer
            
        except Exception as e:
            logger.error(f"Audio chunk processing failed: {e}")
            return self._error_response(session_id, str(e))
    
    async def _handle_pause(
        self,
        msg: WebSocketMessage
    ) -> WebSocketMessage:
        """Handle pause request."""
        session_id = msg.session_id
        if not session_id:
            return self._error_response(None, "Missing session_id")
        
        if self.session_manager.pause_recording(session_id):
            return WebSocketMessage(
                type=MessageType.SESSION_PAUSED,
                session_id=session_id,
                payload={"status": "paused"}
            )
        
        return self._error_response(session_id, "Failed to pause session")
    
    async def _handle_resume(
        self,
        msg: WebSocketMessage
    ) -> WebSocketMessage:
        """Handle resume request."""
        session_id = msg.session_id
        if not session_id:
            return self._error_response(None, "Missing session_id")
        
        if self.session_manager.resume_recording(session_id):
            return WebSocketMessage(
                type=MessageType.SESSION_RESUMED,
                session_id=session_id,
                payload={"status": "recording"}
            )
        
        return self._error_response(session_id, "Failed to resume session")
    
    async def _handle_stop(
        self,
        msg: WebSocketMessage
    ) -> WebSocketMessage:
        """Handle stop and finalize request."""
        session_id = msg.session_id
        if not session_id:
            return self._error_response(None, "Missing session_id")
        
        session = self.session_manager.get_session(session_id)
        if not session:
            return self._error_response(session_id, "Session not found")
        
        # Stop recording
        self.session_manager.stop_recording(session_id)
        
        try:
            # Finalize analysis
            analyzer = self.analyzers.get(session_id)
            if analyzer:
                result = analyzer.finalize_session(session)
                
                # Complete session
                self.session_manager.complete_session(session_id)
                
                # Cleanup
                del self.analyzers[session_id]
                
                return WebSocketMessage(
                    type=MessageType.FINAL_RESULT,
                    session_id=session_id,
                    payload=result.to_dict()
                )
            
            return self._error_response(session_id, "Analyzer not found")
            
        except Exception as e:
            logger.error(f"Session finalization failed: {e}")
            self.session_manager.error_session(session_id, str(e))
            return self._error_response(session_id, str(e))
    
    async def _handle_cancel(
        self,
        msg: WebSocketMessage
    ) -> WebSocketMessage:
        """Handle cancel request."""
        session_id = msg.session_id
        if session_id:
            if session_id in self.analyzers:
                del self.analyzers[session_id]
            self.session_manager.close_session(session_id)
        
        return WebSocketMessage(
            type=MessageType.SESSION_STOPPED,
            session_id=session_id,
            payload={"status": "cancelled"}
        )
    
    async def _handle_ping(
        self,
        msg: WebSocketMessage
    ) -> WebSocketMessage:
        """Handle ping for keepalive."""
        return WebSocketMessage(
            type=MessageType.PONG,
            session_id=msg.session_id,
            payload={"pong": True}
        )
    
    def _error_response(
        self,
        session_id: Optional[str],
        error: str
    ) -> WebSocketMessage:
        """Create error response message."""
        return WebSocketMessage(
            type=MessageType.ERROR,
            session_id=session_id,
            payload={"error": error}
        )
    
    def get_session_status(self, session_id: str) -> Optional[Dict]:
        """Get current status of a session."""
        session = self.session_manager.get_session(session_id)
        if session:
            return session.to_dict()
        return None
    
    def cleanup_connection(self, session_id: Optional[str]):
        """Cleanup when WebSocket connection closes."""
        if session_id:
            if session_id in self.analyzers:
                del self.analyzers[session_id]
            
            session = self.session_manager.get_session(session_id)
            if session and session.is_active:
                # Mark as error if still active
                self.session_manager.error_session(
                    session_id, 
                    "Connection closed unexpectedly"
                )

"""
Real-Time Quality Monitor v4.0
Streaming quality assessment during live recording.

Provides immediate visual feedback and corrective suggestions
with <200ms latency updates.

Requirements: 3.3, 3.4, 4.6
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque
from collections import deque
from enum import Enum

import numpy as np

from .analyzer import SignalQualityAnalyzer, QualityMetrics
from ..core.interfaces import QualityLevel

logger = logging.getLogger(__name__)


class QualityTrend(str, Enum):
    """Quality trend direction."""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    FLUCTUATING = "fluctuating"


@dataclass
class StreamQualityUpdate:
    """Real-time quality status update."""
    # Current quality
    current_quality: float = 0.0  # 0-100
    quality_level: QualityLevel = QualityLevel.ACCEPTABLE
    
    # Trend analysis
    quality_trend: QualityTrend = QualityTrend.STABLE
    trend_history: List[float] = field(default_factory=list)
    
    # Current metrics
    snr_db: float = 0.0
    audio_level: float = 0.0  # 0-1
    clipping_detected: bool = False
    speech_detected: bool = False
    
    # Issues and feedback
    active_issues: List[str] = field(default_factory=list)
    guidance_message: Optional[str] = None
    
    # Visual indicators
    level_meter: float = 0.0  # -60 to 0 dB
    is_recording_recommended: bool = True
    
    # Timing
    timestamp: float = 0.0
    processing_latency_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for WebSocket transmission."""
        return {
            "quality": self.current_quality,
            "level": self.quality_level.value,
            "trend": self.quality_trend.value,
            "snr_db": round(self.snr_db, 1),
            "audio_level": round(self.audio_level, 3),
            "clipping": self.clipping_detected,
            "speech": self.speech_detected,
            "issues": self.active_issues,
            "guidance": self.guidance_message,
            "level_meter": round(self.level_meter, 1),
            "ok_to_record": self.is_recording_recommended,
            "latency_ms": round(self.processing_latency_ms, 1)
        }


class RealTimeQualityMonitor:
    """
    Real-time quality monitoring for streaming audio.
    
    Provides:
    - Instant quality feedback (<200ms latency)
    - Trend analysis over time
    - Corrective guidance during recording
    - Visual indicators for UI
    """
    
    # Quality thresholds
    MIN_SNR_DB = 15.0
    CLIPPING_THRESHOLD = 0.02
    MIN_AUDIO_LEVEL = 0.01
    MAX_AUDIO_LEVEL = 0.95
    
    # Analysis parameters
    WINDOW_SIZE_SAMPLES = 4800  # 300ms at 16kHz
    HISTORY_SIZE = 50  # Keep 50 recent quality samples
    
    # Update intervals
    MIN_UPDATE_INTERVAL_MS = 50
    TARGET_LATENCY_MS = 200
    
    # Guidance messages
    GUIDANCE_MESSAGES = {
        "too_quiet": "Speak louder or move closer to the microphone",
        "too_loud": "Speak softer or move away from the microphone",
        "clipping": "Audio is distorting - reduce volume",
        "noisy": "Background noise detected - find a quieter location",
        "no_speech": "No speech detected - please speak clearly",
        "good": "Audio quality is good - continue recording",
        "excellent": "Excellent audio quality",
    }
    
    def __init__(
        self,
        sample_rate: int = 16000,
        history_size: int = 50
    ):
        self.sample_rate = sample_rate
        self.history_size = history_size
        
        # Quality history for trend analysis
        self.quality_history: Deque[float] = deque(maxlen=history_size)
        self.snr_history: Deque[float] = deque(maxlen=history_size)
        
        # Session state
        self.session_start_time: float = 0.0
        self.last_update_time: float = 0.0
        self.total_chunks_processed: int = 0
        
        # Running statistics
        self.cumulative_speech_ratio: float = 0.0
        self.cumulative_quality: float = 0.0
        
        # Analyzer for detailed metrics
        self.signal_analyzer = SignalQualityAnalyzer(sample_rate=sample_rate)
    
    def start_session(self):
        """Initialize a new monitoring session."""
        self.quality_history.clear()
        self.snr_history.clear()
        self.session_start_time = time.time()
        self.last_update_time = 0.0
        self.total_chunks_processed = 0
        self.cumulative_speech_ratio = 0.0
        self.cumulative_quality = 0.0
        
        logger.info("Real-time quality monitoring session started")
    
    def process_chunk(
        self,
        audio_chunk: np.ndarray,
        timestamp: Optional[float] = None
    ) -> StreamQualityUpdate:
        """
        Process an audio chunk and return quality update.
        
        Args:
            audio_chunk: Audio samples as numpy array
            timestamp: Optional timestamp for the chunk
            
        Returns:
            StreamQualityUpdate with current quality status
        """
        start_time = time.time()
        
        if timestamp is None:
            timestamp = time.time()
        
        update = StreamQualityUpdate(timestamp=timestamp)
        
        try:
            # Quick analysis (optimized for speed)
            self._analyze_chunk(audio_chunk, update)
            
            # Update history and trends
            self._update_history(update)
            
            # Analyze trend
            update.quality_trend = self._analyze_trend()
            
            # Generate guidance
            update.guidance_message = self._generate_guidance(update)
            
            # Calculate level meter
            update.level_meter = self._calculate_level_meter(audio_chunk)
            
            # Determine if recording should continue
            update.is_recording_recommended = (
                update.current_quality >= 40 and
                not update.clipping_detected
            )
            
            self.total_chunks_processed += 1
            
        except Exception as e:
            logger.warning(f"Chunk processing failed: {e}")
            update.active_issues.append("Analysis error")
        
        # Record latency
        update.processing_latency_ms = (time.time() - start_time) * 1000
        self.last_update_time = time.time()
        
        return update
    
    def _analyze_chunk(self, audio_chunk: np.ndarray, update: StreamQualityUpdate):
        """Perform fast quality analysis on chunk."""
        if len(audio_chunk) < 100:
            update.current_quality = 0.0
            return
        
        # Audio level (RMS)
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        update.audio_level = min(1.0, rms * 10)  # Scale for display
        
        # Quick SNR estimation
        update.snr_db = self._estimate_snr_fast(audio_chunk)
        
        # Clipping detection
        peak = np.max(np.abs(audio_chunk))
        update.clipping_detected = peak > 0.99
        
        # Speech detection (simple energy threshold)
        update.speech_detected = rms > 0.02
        
        # Determine issues
        issues = []
        
        if rms < self.MIN_AUDIO_LEVEL:
            issues.append("Audio level too low")
        elif rms > self.MAX_AUDIO_LEVEL:
            issues.append("Audio level too high")
        
        if update.clipping_detected:
            issues.append("Clipping detected")
        
        if update.snr_db < self.MIN_SNR_DB:
            issues.append("High background noise")
        
        update.active_issues = issues
        
        # Calculate quality score
        update.current_quality = self._calculate_chunk_quality(update)
        
        # Determine quality level
        if update.current_quality >= 80:
            update.quality_level = QualityLevel.EXCELLENT
        elif update.current_quality >= 60:
            update.quality_level = QualityLevel.GOOD
        elif update.current_quality >= 40:
            update.quality_level = QualityLevel.ACCEPTABLE
        else:
            update.quality_level = QualityLevel.POOR
    
    def _estimate_snr_fast(self, audio_chunk: np.ndarray) -> float:
        """Fast SNR estimation for real-time use."""
        if len(audio_chunk) < 100:
            return 20.0
        
        # Simple SNR based on signal vs noise floor
        abs_audio = np.abs(audio_chunk)
        
        # Estimate noise as 10th percentile
        noise_level = np.percentile(abs_audio, 10)
        # Estimate signal as 90th percentile
        signal_level = np.percentile(abs_audio, 90)
        
        if noise_level < 1e-10:
            return 40.0  # Very clean
        
        snr_linear = signal_level / noise_level
        snr_db = 20 * np.log10(snr_linear) if snr_linear > 0 else 0.0
        
        return np.clip(snr_db, 0.0, 60.0)
    
    def _calculate_chunk_quality(self, update: StreamQualityUpdate) -> float:
        """Calculate quality score for current chunk."""
        score = 100.0
        
        # SNR component (40% weight)
        if update.snr_db < self.MIN_SNR_DB:
            snr_penalty = (self.MIN_SNR_DB - update.snr_db) / self.MIN_SNR_DB
            score -= min(40, snr_penalty * 50)
        
        # Audio level component (30% weight)
        if update.audio_level < self.MIN_AUDIO_LEVEL:
            score -= 25
        elif update.audio_level > self.MAX_AUDIO_LEVEL:
            score -= 15
        
        # Clipping penalty (30% weight)
        if update.clipping_detected:
            score -= 30
        
        # Speech detection bonus
        if update.speech_detected and len(update.active_issues) == 0:
            score = min(100, score + 5)
        
        return max(0.0, min(100.0, score))
    
    def _update_history(self, update: StreamQualityUpdate):
        """Update quality history for trend analysis."""
        self.quality_history.append(update.current_quality)
        self.snr_history.append(update.snr_db)
        
        # Update running averages
        alpha = 0.1  # Smoothing factor
        self.cumulative_quality = (
            alpha * update.current_quality + 
            (1 - alpha) * self.cumulative_quality
        )
        
        # Store trend history for UI
        update.trend_history = list(self.quality_history)[-20:]
    
    def _analyze_trend(self) -> QualityTrend:
        """Analyze quality trend from history."""
        if len(self.quality_history) < 5:
            return QualityTrend.STABLE
        
        recent = list(self.quality_history)[-10:]
        
        # Calculate trend using linear regression slope
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        
        # Calculate variance to detect fluctuation
        variance = np.var(recent)
        
        if variance > 200:  # High variance
            return QualityTrend.FLUCTUATING
        elif slope > 2:  # Improving
            return QualityTrend.IMPROVING
        elif slope < -2:  # Degrading
            return QualityTrend.DEGRADING
        else:
            return QualityTrend.STABLE
    
    def _generate_guidance(self, update: StreamQualityUpdate) -> str:
        """Generate user guidance based on current quality."""
        # Priority-based guidance
        if update.clipping_detected:
            return self.GUIDANCE_MESSAGES["clipping"]
        
        if update.audio_level < self.MIN_AUDIO_LEVEL:
            return self.GUIDANCE_MESSAGES["too_quiet"]
        
        if update.audio_level > self.MAX_AUDIO_LEVEL:
            return self.GUIDANCE_MESSAGES["too_loud"]
        
        if update.snr_db < self.MIN_SNR_DB:
            return self.GUIDANCE_MESSAGES["noisy"]
        
        if not update.speech_detected:
            return self.GUIDANCE_MESSAGES["no_speech"]
        
        if update.current_quality >= 80:
            return self.GUIDANCE_MESSAGES["excellent"]
        
        return self.GUIDANCE_MESSAGES["good"]
    
    def _calculate_level_meter(self, audio_chunk: np.ndarray) -> float:
        """Calculate audio level in dB for visual meter."""
        if len(audio_chunk) == 0:
            return -60.0
        
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        
        if rms < 1e-10:
            return -60.0
        
        db = 20 * np.log10(rms)
        return np.clip(db, -60.0, 0.0)
    
    def get_session_summary(self) -> Dict:
        """Get summary of the monitoring session."""
        if len(self.quality_history) == 0:
            return {"error": "No data collected"}
        
        quality_array = np.array(self.quality_history)
        snr_array = np.array(self.snr_history)
        
        return {
            "session_duration": time.time() - self.session_start_time,
            "chunks_processed": self.total_chunks_processed,
            "average_quality": float(np.mean(quality_array)),
            "min_quality": float(np.min(quality_array)),
            "max_quality": float(np.max(quality_array)),
            "quality_std": float(np.std(quality_array)),
            "average_snr": float(np.mean(snr_array)),
            "quality_trend": self._analyze_trend().value,
            "recording_recommended": self.cumulative_quality >= 50
        }
    
    def end_session(self) -> Dict:
        """End the monitoring session and return final summary."""
        summary = self.get_session_summary()
        logger.info(f"Quality monitoring session ended: avg={summary.get('average_quality', 0):.1f}")
        return summary

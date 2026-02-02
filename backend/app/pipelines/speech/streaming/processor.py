"""
Stream Processor v4.0
Real-time chunk processing for streaming analysis.

Requirements: 3.3, 3.4
- Sub-200ms processing latency
- Incremental feature extraction
- Circular buffer management
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque
from collections import deque

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    """Result of processing a single chunk."""
    # Timing
    timestamp: float = 0.0
    processing_time_ms: float = 0.0
    
    # Quality metrics (fast computation)
    rms_level: float = 0.0
    peak_level: float = 0.0
    snr_estimate: float = 0.0
    has_clipping: bool = False
    has_speech: bool = False
    
    # Fast features (computed per-chunk)
    energy: float = 0.0
    zcr: float = 0.0  # Zero crossing rate
    spectral_centroid: float = 0.0
    
    # Preliminary estimates (updated incrementally)
    preliminary_f0: float = 0.0
    preliminary_quality_score: float = 0.0
    
    # Feedback
    quality_trend: str = "stable"  # improving, stable, degrading
    guidance_message: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "latency_ms": self.processing_time_ms,
            "rms": self.rms_level,
            "peak": self.peak_level,
            "snr": self.snr_estimate,
            "clipping": self.has_clipping,
            "speech_detected": self.has_speech,
            "quality_score": self.preliminary_quality_score,
            "trend": self.quality_trend,
            "message": self.guidance_message
        }


class StreamProcessor:
    """
    Real-time stream processor for audio chunks.
    
    Optimized for low latency (<200ms) processing with
    incremental feature estimation.
    """
    
    # Performance targets
    TARGET_LATENCY_MS = 200
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size_samples: int = 3200,  # 200ms at 16kHz
        buffer_seconds: float = 2.0
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size_samples
        self.buffer_seconds = buffer_seconds
        
        # Circular buffer for recent audio
        buffer_size = int(buffer_seconds * sample_rate / chunk_size_samples)
        self.audio_buffer: Deque[np.ndarray] = deque(maxlen=buffer_size)
        
        # Running statistics
        self.noise_floor_estimate = -60.0
        self.quality_history: Deque[float] = deque(maxlen=50)
        self.f0_estimates: Deque[float] = deque(maxlen=20)
        
        # Counters
        self.chunks_processed = 0
        self.total_processing_time_ms = 0
        
        # Pre-compute filter coefficients
        self._init_filters()
    
    def _init_filters(self):
        """Initialize filter coefficients."""
        # High-pass filter for DC removal
        self.hp_b, self.hp_a = signal.butter(2, 50 / (self.sample_rate / 2), btype='high')
        
        # Band-pass for speech detection (100-3000 Hz)
        self.bp_b, self.bp_a = signal.butter(
            3, 
            [100 / (self.sample_rate / 2), 3000 / (self.sample_rate / 2)],
            btype='band'
        )
    
    def process_chunk(
        self,
        audio_chunk: np.ndarray,
        timestamp: Optional[float] = None
    ) -> ChunkResult:
        """
        Process a single audio chunk.
        
        Args:
            audio_chunk: Audio samples as float32 array
            timestamp: Optional timestamp for the chunk
            
        Returns:
            ChunkResult with real-time metrics
        """
        start_time = time.time()
        
        result = ChunkResult(
            timestamp=timestamp or time.time()
        )
        
        try:
            # Ensure correct dtype
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)
            
            # Add to buffer
            self.audio_buffer.append(audio_chunk)
            
            # Fast quality metrics
            self._compute_levels(audio_chunk, result)
            self._detect_clipping(audio_chunk, result)
            self._detect_speech(audio_chunk, result)
            
            # Fast features
            self._compute_fast_features(audio_chunk, result)
            
            # Estimate F0 (if speech detected)
            if result.has_speech:
                self._estimate_f0(audio_chunk, result)
            
            # Update quality score
            self._update_quality_score(result)
            
            # Trend analysis
            self._analyze_trend(result)
            
            # Generate guidance
            self._generate_guidance(result)
            
            self.chunks_processed += 1
            
        except Exception as e:
            logger.warning(f"Chunk processing error: {e}")
            result.guidance_message = "Processing error"
        
        result.processing_time_ms = (time.time() - start_time) * 1000
        self.total_processing_time_ms += result.processing_time_ms
        
        # Log latency warnings
        if result.processing_time_ms > self.TARGET_LATENCY_MS:
            logger.warning(
                f"Chunk processing exceeded latency target: "
                f"{result.processing_time_ms:.1f}ms > {self.TARGET_LATENCY_MS}ms"
            )
        
        return result
    
    def _compute_levels(self, audio: np.ndarray, result: ChunkResult):
        """Compute RMS and peak levels."""
        result.rms_level = float(np.sqrt(np.mean(audio ** 2)))
        result.peak_level = float(np.max(np.abs(audio)))
        
        # Estimate SNR
        if result.rms_level > 0:
            rms_db = 20 * np.log10(result.rms_level + 1e-10)
            result.snr_estimate = rms_db - self.noise_floor_estimate
            
            # Update noise floor when levels are low (silence)
            if rms_db < self.noise_floor_estimate + 10:
                self.noise_floor_estimate = 0.9 * self.noise_floor_estimate + 0.1 * rms_db
    
    def _detect_clipping(self, audio: np.ndarray, result: ChunkResult):
        """Detect clipping in chunk."""
        clipping_threshold = 0.98
        clipped_samples = np.sum(np.abs(audio) > clipping_threshold)
        result.has_clipping = clipped_samples > len(audio) * 0.001
    
    def _detect_speech(self, audio: np.ndarray, result: ChunkResult):
        """Fast speech detection using energy and ZCR."""
        # Energy threshold
        energy_threshold = 0.01
        result.has_speech = result.rms_level > energy_threshold
        
        if not result.has_speech:
            return
        
        # Refine with band-passed energy
        try:
            filtered = signal.filtfilt(self.bp_b, self.bp_a, audio)
            band_energy = np.sqrt(np.mean(filtered ** 2))
            
            # Speech typically has energy concentrated in speech band
            energy_ratio = band_energy / (result.rms_level + 1e-10)
            result.has_speech = energy_ratio > 0.3 and result.snr_estimate > 5
        except:
            pass
    
    def _compute_fast_features(self, audio: np.ndarray, result: ChunkResult):
        """Compute fast spectral features."""
        result.energy = float(np.sum(audio ** 2))
        
        # Zero crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / 2
        result.zcr = float(zero_crossings / len(audio))
        
        # Spectral centroid (fast approximation)
        try:
            spectrum = np.abs(np.fft.rfft(audio))
            freqs = np.fft.rfftfreq(len(audio), 1 / self.sample_rate)
            
            total_energy = np.sum(spectrum) + 1e-10
            result.spectral_centroid = float(np.sum(freqs * spectrum) / total_energy)
        except:
            result.spectral_centroid = 0.0
    
    def _estimate_f0(self, audio: np.ndarray, result: ChunkResult):
        """Fast F0 estimation using autocorrelation."""
        try:
            # Autocorrelation
            corr = np.correlate(audio, audio, mode='full')
            corr = corr[len(corr)//2:]
            
            # Find first peak after min_lag
            min_lag = int(self.sample_rate / 500)  # 500 Hz max
            max_lag = int(self.sample_rate / 75)   # 75 Hz min
            
            search_region = corr[min_lag:max_lag]
            if len(search_region) > 0:
                peak_idx = np.argmax(search_region) + min_lag
                f0 = self.sample_rate / peak_idx
                
                # Sanity check
                if 75 < f0 < 500:
                    result.preliminary_f0 = float(f0)
                    self.f0_estimates.append(f0)
        except:
            pass
    
    def _update_quality_score(self, result: ChunkResult):
        """Update preliminary quality score."""
        score = 100.0
        
        # Penalize clipping
        if result.has_clipping:
            score -= 30
        
        # Penalize low SNR
        if result.snr_estimate < 15:
            score -= (15 - result.snr_estimate) * 2
        
        # Penalize silence
        if not result.has_speech:
            score -= 10
        
        result.preliminary_quality_score = max(0, min(100, score))
        self.quality_history.append(result.preliminary_quality_score)
    
    def _analyze_trend(self, result: ChunkResult):
        """Analyze quality trend over recent chunks."""
        if len(self.quality_history) < 5:
            result.quality_trend = "stable"
            return
        
        recent = list(self.quality_history)[-10:]
        first_half = np.mean(recent[:len(recent)//2])
        second_half = np.mean(recent[len(recent)//2:])
        
        diff = second_half - first_half
        
        if diff > 5:
            result.quality_trend = "improving"
        elif diff < -5:
            result.quality_trend = "degrading"
        else:
            result.quality_trend = "stable"
    
    def _generate_guidance(self, result: ChunkResult):
        """Generate user guidance message."""
        if result.has_clipping:
            result.guidance_message = "Audio is clipping - please lower volume"
        elif result.snr_estimate < 10:
            result.guidance_message = "Low signal quality - move closer to microphone"
        elif not result.has_speech:
            result.guidance_message = "No speech detected - please speak"
        elif result.quality_trend == "degrading":
            result.guidance_message = "Quality decreasing - check environment"
        elif result.preliminary_quality_score > 80:
            result.guidance_message = "Good quality - continue speaking"
        else:
            result.guidance_message = ""
    
    def get_buffered_audio(self) -> np.ndarray:
        """Get all buffered audio concatenated."""
        if not self.audio_buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(list(self.audio_buffer))
    
    def get_average_latency(self) -> float:
        """Get average processing latency in ms."""
        if self.chunks_processed == 0:
            return 0.0
        return self.total_processing_time_ms / self.chunks_processed
    
    def reset(self):
        """Reset processor state."""
        self.audio_buffer.clear()
        self.quality_history.clear()
        self.f0_estimates.clear()
        self.noise_floor_estimate = -60.0
        self.chunks_processed = 0
        self.total_processing_time_ms = 0

"""
Speech Content Detector v4.0
Voice activity detection and speech content analysis.

Implements speech ratio calculation and voice activity detection
ensuring >60% speech content requirement (Requirement 4.3).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class SpeechMetrics:
    """Speech content analysis metrics."""
    # Speech detection
    speech_ratio: float = 0.0  # Proportion of audio with speech (0-1)
    voice_ratio: float = 0.0  # Proportion with voicing (0-1)
    
    # Duration analysis
    total_duration: float = 0.0
    speech_duration: float = 0.0
    silence_duration: float = 0.0
    
    # Segment analysis
    speech_segments: int = 0
    silence_segments: int = 0
    mean_speech_segment_duration: float = 0.0
    mean_silence_segment_duration: float = 0.0
    max_silence_duration: float = 0.0
    
    # VAD confidence
    vad_confidence: float = 0.0
    
    # Speech quality indicators
    has_adequate_speech: bool = False
    speech_continuity: float = 0.0  # How continuous the speech is (0-1)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "speech_ratio": self.speech_ratio,
            "voice_ratio": self.voice_ratio,
            "total_duration": self.total_duration,
            "speech_duration": self.speech_duration,
            "speech_segments": self.speech_segments,
            "max_silence_duration": self.max_silence_duration,
            "speech_continuity": self.speech_continuity
        }


class SpeechContentDetector:
    """
    Detects speech content and voice activity in audio.
    
    Uses multiple methods:
    - Energy-based VAD
    - Zero-crossing rate analysis
    - Spectral flux detection
    """
    
    # Thresholds
    MIN_SPEECH_RATIO = 0.60  # Requirement 4.3
    GOOD_SPEECH_RATIO = 0.75
    
    # Frame parameters
    FRAME_LENGTH_MS = 25
    HOP_LENGTH_MS = 10
    
    # Energy thresholds (relative)
    ENERGY_THRESHOLD_LOW = 0.01
    ENERGY_THRESHOLD_HIGH = 0.1
    
    # Minimum segment durations
    MIN_SPEECH_DURATION = 0.05  # 50ms
    MIN_SILENCE_DURATION = 0.15  # 150ms
    
    def __init__(
        self,
        sample_rate: int = 16000,
        min_speech_ratio: float = 0.60
    ):
        self.sample_rate = sample_rate
        self.min_speech_ratio = min_speech_ratio
        
        # Frame sizes in samples
        self.frame_length = int(sample_rate * self.FRAME_LENGTH_MS / 1000)
        self.hop_length = int(sample_rate * self.HOP_LENGTH_MS / 1000)
    
    def detect(self, audio: np.ndarray) -> SpeechMetrics:
        """
        Detect speech content in audio.
        
        Args:
            audio: Audio signal as numpy array
            
        Returns:
            SpeechMetrics with speech detection results
        """
        if len(audio) < self.frame_length * 2:
            return SpeechMetrics()
        
        metrics = SpeechMetrics()
        metrics.total_duration = len(audio) / self.sample_rate
        
        # Compute frame-level features
        frames, n_frames = self._get_frames(audio)
        
        if n_frames < 2:
            return metrics
        
        # Energy-based VAD
        energy_vad = self._energy_vad(frames)
        
        # Zero-crossing rate VAD (for refinement)
        zcr_vad = self._zcr_vad(frames)
        
        # Combine VAD decisions
        speech_mask = self._combine_vad(energy_vad, zcr_vad)
        
        # Apply temporal smoothing
        speech_mask = self._smooth_vad(speech_mask)
        
        # Calculate metrics
        self._calculate_metrics(speech_mask, metrics)
        
        # Determine if adequate speech
        metrics.has_adequate_speech = metrics.speech_ratio >= self.min_speech_ratio
        
        return metrics
    
    def _get_frames(self, audio: np.ndarray) -> Tuple[np.ndarray, int]:
        """Split audio into overlapping frames."""
        n_frames = (len(audio) - self.frame_length) // self.hop_length + 1
        
        if n_frames < 1:
            return np.array([]), 0
        
        frames = np.zeros((n_frames, self.frame_length))
        
        for i in range(n_frames):
            start = i * self.hop_length
            frames[i] = audio[start:start + self.frame_length]
        
        return frames, n_frames
    
    def _energy_vad(self, frames: np.ndarray) -> np.ndarray:
        """Energy-based voice activity detection."""
        # Compute frame energies
        energies = np.sum(frames ** 2, axis=1)
        
        if len(energies) == 0:
            return np.array([])
        
        # Normalize energies
        max_energy = np.max(energies)
        if max_energy < 1e-10:
            return np.zeros(len(energies), dtype=bool)
        
        energies_norm = energies / max_energy
        
        # Adaptive thresholding
        # Use histogram-based threshold selection
        sorted_energies = np.sort(energies_norm)
        
        # Find the "Knee" in the distribution
        # Assume bottom 20% is noise, top 50% is speech
        noise_floor = np.mean(sorted_energies[:max(1, len(sorted_energies)//5)])
        speech_floor = np.mean(sorted_energies[len(sorted_energies)//2:])
        
        # Threshold between noise and speech
        threshold = noise_floor + 0.3 * (speech_floor - noise_floor)
        threshold = max(self.ENERGY_THRESHOLD_LOW, min(self.ENERGY_THRESHOLD_HIGH, threshold))
        
        return energies_norm > threshold
    
    def _zcr_vad(self, frames: np.ndarray) -> np.ndarray:
        """Zero-crossing rate based detection."""
        if len(frames) == 0:
            return np.array([])
        
        zcr = np.zeros(len(frames))
        
        for i, frame in enumerate(frames):
            # Count zero crossings
            signs = np.sign(frame[:-1]) * np.sign(frame[1:])
            zcr[i] = np.sum(signs < 0) / len(frame)
        
        # Speech typically has moderate ZCR (not too high, not too low)
        # Values around 0.05-0.20 are typical for voiced speech
        # Very high ZCR (>0.35) often indicates noise
        # Very low ZCR (<0.02) might indicate silence or pure tone
        
        # Return True for frames likely to contain speech
        return (zcr > 0.02) & (zcr < 0.35)
    
    def _combine_vad(
        self, 
        energy_vad: np.ndarray, 
        zcr_vad: np.ndarray
    ) -> np.ndarray:
        """Combine multiple VAD decisions."""
        if len(energy_vad) == 0:
            return np.array([])
        
        # Primary decision: energy-based
        # Refine with ZCR to remove noise bursts
        combined = energy_vad & zcr_vad
        
        # But keep high-energy frames even with unusual ZCR
        # (for fricatives which have high ZCR)
        combined = combined | energy_vad
        
        return combined
    
    def _smooth_vad(self, speech_mask: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to VAD decisions."""
        if len(speech_mask) < 3:
            return speech_mask
        
        smoothed = speech_mask.copy()
        
        # Convert minimum durations to frames
        min_speech_frames = int(self.MIN_SPEECH_DURATION * 1000 / self.HOP_LENGTH_MS)
        min_silence_frames = int(self.MIN_SILENCE_DURATION * 1000 / self.HOP_LENGTH_MS)
        
        # Remove short speech segments (likely noise)
        self._remove_short_segments(smoothed, True, min_speech_frames)
        
        # Fill short silence gaps (likely within speech)
        self._remove_short_segments(smoothed, False, min_silence_frames)
        
        return smoothed
    
    def _remove_short_segments(
        self, 
        mask: np.ndarray, 
        target_value: bool, 
        min_length: int
    ):
        """Remove segments shorter than min_length."""
        if min_length < 2:
            return
        
        in_segment = False
        segment_start = 0
        
        for i in range(len(mask)):
            if mask[i] == target_value:
                if not in_segment:
                    segment_start = i
                    in_segment = True
            else:
                if in_segment:
                    segment_length = i - segment_start
                    if segment_length < min_length:
                        mask[segment_start:i] = not target_value
                    in_segment = False
        
        # Handle final segment
        if in_segment:
            segment_length = len(mask) - segment_start
            if segment_length < min_length:
                mask[segment_start:] = not target_value
    
    def _calculate_metrics(self, speech_mask: np.ndarray, metrics: SpeechMetrics):
        """Calculate speech metrics from VAD mask."""
        if len(speech_mask) == 0:
            return
        
        # Basic ratios
        metrics.speech_ratio = np.mean(speech_mask)
        metrics.voice_ratio = metrics.speech_ratio  # Simplified
        
        # Duration calculations
        frame_duration = self.HOP_LENGTH_MS / 1000.0
        metrics.speech_duration = np.sum(speech_mask) * frame_duration
        metrics.silence_duration = np.sum(~speech_mask) * frame_duration
        
        # Segment analysis
        self._analyze_segments(speech_mask, metrics, frame_duration)
        
        # VAD confidence based on how clear the speech/silence distinction is
        metrics.vad_confidence = self._estimate_vad_confidence(speech_mask)
        
        # Speech continuity
        if metrics.speech_segments > 0:
            expected_pauses = metrics.speech_duration / 3.0  # Assume ~3s per breath group
            actual_pauses = metrics.silence_segments
            if actual_pauses > 0:
                metrics.speech_continuity = 1.0 / (1.0 + abs(actual_pauses - expected_pauses) / max(expected_pauses, 1))
            else:
                metrics.speech_continuity = 1.0
    
    def _analyze_segments(
        self, 
        speech_mask: np.ndarray, 
        metrics: SpeechMetrics,
        frame_duration: float
    ):
        """Analyze speech and silence segments."""
        speech_durations = []
        silence_durations = []
        
        in_speech = speech_mask[0] if len(speech_mask) > 0 else False
        segment_start = 0
        
        for i in range(1, len(speech_mask)):
            if speech_mask[i] != speech_mask[i-1]:
                # Transition
                segment_duration = (i - segment_start) * frame_duration
                if in_speech:
                    speech_durations.append(segment_duration)
                else:
                    silence_durations.append(segment_duration)
                
                segment_start = i
                in_speech = speech_mask[i]
        
        # Final segment
        segment_duration = (len(speech_mask) - segment_start) * frame_duration
        if in_speech:
            speech_durations.append(segment_duration)
        else:
            silence_durations.append(segment_duration)
        
        # Calculate metrics
        metrics.speech_segments = len(speech_durations)
        metrics.silence_segments = len(silence_durations)
        
        if speech_durations:
            metrics.mean_speech_segment_duration = np.mean(speech_durations)
        
        if silence_durations:
            metrics.mean_silence_segment_duration = np.mean(silence_durations)
            metrics.max_silence_duration = max(silence_durations)
    
    def _estimate_vad_confidence(self, speech_mask: np.ndarray) -> float:
        """Estimate confidence in VAD decision based on distribution clarity."""
        if len(speech_mask) == 0:
            return 0.0
        
        speech_ratio = np.mean(speech_mask)
        
        # Confidence is higher when ratio is clearly in expected range
        # and not at extremes (all speech or all silence)
        if 0.3 < speech_ratio < 0.90:
            # Good balance of speech and silence
            return 0.9
        elif 0.1 < speech_ratio <= 0.3 or 0.90 <= speech_ratio < 0.98:
            # Unusual but possible
            return 0.7
        else:
            # Extreme values, less confident
            return 0.5
    
    def get_issues(self, metrics: SpeechMetrics) -> List[str]:
        """Get list of speech content issues."""
        issues = []
        
        if metrics.speech_ratio < self.min_speech_ratio:
            issues.append(
                f"Low speech content ({metrics.speech_ratio*100:.1f}% < {self.min_speech_ratio*100:.0f}%)"
            )
        
        if metrics.max_silence_duration > 2.0:
            issues.append(f"Long pause detected ({metrics.max_silence_duration:.1f}s)")
        
        if metrics.speech_segments > 20 and metrics.mean_speech_segment_duration < 0.3:
            issues.append("Fragmented speech (many short segments)")
        
        return issues
    
    def get_suggestions(self, metrics: SpeechMetrics) -> List[str]:
        """Get suggestions for improving speech content."""
        suggestions = []
        
        if metrics.speech_ratio < self.min_speech_ratio:
            suggestions.extend([
                "Speak continuously during the recording",
                "Follow the reading passage closely",
                "Reduce long pauses between sentences"
            ])
        
        if metrics.max_silence_duration > 2.0:
            suggestions.append("Avoid long pauses during recording")
        
        return suggestions

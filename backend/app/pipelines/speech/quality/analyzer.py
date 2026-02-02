"""
Signal Quality Analyzer v4.0
Comprehensive audio signal quality analysis.

Implements SNR calculation, clipping detection, and frequency content analysis
matching research-grade quality standards.

Requirements: 4.1, 4.2, 4.4
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

from scipy import signal
from scipy.fft import rfft, rfftfreq

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Comprehensive audio quality metrics."""
    # Signal-to-Noise Ratio
    snr_db: float = 0.0
    snr_method: str = "rms"  # Method used for SNR calculation
    
    # Clipping Analysis
    clipping_ratio: float = 0.0
    clipped_samples: int = 0
    total_samples: int = 0
    positive_clips: int = 0
    negative_clips: int = 0
    
    # Dynamic Range
    dynamic_range_db: float = 0.0
    peak_amplitude: float = 0.0
    rms_amplitude: float = 0.0
    crest_factor_db: float = 0.0
    
    # Frequency Analysis
    frequency_coverage: float = 0.0
    low_frequency_energy: float = 0.0  # 80-300 Hz
    mid_frequency_energy: float = 0.0  # 300-3000 Hz
    high_frequency_energy: float = 0.0  # 3000-8000 Hz
    has_adequate_bandwidth: bool = True
    
    # Noise Analysis
    noise_floor_db: float = -60.0
    background_noise_type: str = "unknown"
    noise_stationarity: float = 0.0
    
    # DC Offset
    dc_offset: float = 0.0
    has_dc_offset: bool = False
    
    # Overall quality score (0-100)
    quality_score: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "snr_db": self.snr_db,
            "clipping_ratio": self.clipping_ratio,
            "dynamic_range_db": self.dynamic_range_db,
            "frequency_coverage": self.frequency_coverage,
            "noise_floor_db": self.noise_floor_db,
            "quality_score": self.quality_score
        }


class SignalQualityAnalyzer:
    """
    Comprehensive audio signal quality analyzer.
    
    Analyzes:
    - Signal-to-Noise Ratio (SNR)
    - Clipping and distortion
    - Frequency content
    - Dynamic range
    - Noise characteristics
    """
    
    # Quality thresholds
    MIN_SNR_DB = 15.0
    GOOD_SNR_DB = 20.0
    EXCELLENT_SNR_DB = 30.0
    
    MAX_CLIPPING_RATIO = 0.05
    CLIPPING_WARNING_RATIO = 0.02
    
    MIN_FREQUENCY_HZ = 80.0
    MAX_FREQUENCY_HZ = 8000.0
    
    MIN_DYNAMIC_RANGE_DB = 20.0
    
    # Clipping threshold (proportion of max amplitude)
    CLIPPING_THRESHOLD = 0.99
    
    def __init__(
        self,
        sample_rate: int = 16000,
        min_snr_db: float = 15.0,
        max_clipping_ratio: float = 0.05
    ):
        self.sample_rate = sample_rate
        self.min_snr_db = min_snr_db
        self.max_clipping_ratio = max_clipping_ratio
    
    def analyze(self, audio: np.ndarray) -> QualityMetrics:
        """
        Perform comprehensive signal quality analysis.
        
        Args:
            audio: Audio signal as numpy array (float32, normalized -1 to 1)
            
        Returns:
            QualityMetrics with all quality measurements
        """
        if len(audio) < self.sample_rate * 0.1:  # Minimum 100ms
            return QualityMetrics(quality_score=0.0)
        
        metrics = QualityMetrics(total_samples=len(audio))
        
        # Analyze each quality dimension
        self._analyze_snr(audio, metrics)
        self._analyze_clipping(audio, metrics)
        self._analyze_dynamic_range(audio, metrics)
        self._analyze_frequency_content(audio, metrics)
        self._analyze_noise(audio, metrics)
        self._analyze_dc_offset(audio, metrics)
        
        # Calculate overall quality score
        metrics.quality_score = self._calculate_quality_score(metrics)
        
        return metrics
    
    def _analyze_snr(self, audio: np.ndarray, metrics: QualityMetrics):
        """Calculate signal-to-noise ratio using multiple methods."""
        try:
            # Method 1: RMS-based SNR estimation
            # Segment audio and estimate noise from quietest segments
            segment_length = int(self.sample_rate * 0.1)  # 100ms segments
            n_segments = len(audio) // segment_length
            
            if n_segments < 2:
                metrics.snr_db = 20.0  # Default for short audio
                return
            
            segment_rms = []
            for i in range(n_segments):
                start = i * segment_length
                end = start + segment_length
                segment = audio[start:end]
                rms = np.sqrt(np.mean(segment ** 2))
                if rms > 1e-10:
                    segment_rms.append(rms)
            
            if len(segment_rms) < 2:
                metrics.snr_db = 20.0
                return
            
            segment_rms = np.array(segment_rms)
            
            # Estimate noise from lowest 10% of segments
            noise_percentile = 10
            noise_threshold_idx = max(1, int(len(segment_rms) * noise_percentile / 100))
            sorted_rms = np.sort(segment_rms)
            noise_rms = np.mean(sorted_rms[:noise_threshold_idx])
            
            # Estimate signal from highest 50% of segments
            signal_threshold_idx = max(noise_threshold_idx + 1, int(len(segment_rms) * 0.5))
            signal_rms = np.mean(sorted_rms[signal_threshold_idx:])
            
            # Calculate SNR
            if noise_rms > 1e-10:
                snr = signal_rms / noise_rms
                metrics.snr_db = 20 * np.log10(snr) if snr > 0 else 0.0
            else:
                metrics.snr_db = 60.0  # Very clean signal
            
            # Clamp to reasonable range
            metrics.snr_db = np.clip(metrics.snr_db, 0.0, 80.0)
            metrics.snr_method = "rms_segment"
            
        except Exception as e:
            logger.warning(f"SNR calculation failed: {e}")
            metrics.snr_db = 20.0  # Safe default
    
    def _analyze_clipping(self, audio: np.ndarray, metrics: QualityMetrics):
        """Detect audio clipping/distortion."""
        try:
            # Normalize to find relative clipping
            max_abs = np.max(np.abs(audio))
            if max_abs < 1e-10:
                metrics.clipping_ratio = 0.0
                return
            
            normalized = audio / max_abs
            
            # Detect samples at or near maximum
            threshold = self.CLIPPING_THRESHOLD
            clipped_positive = np.sum(normalized >= threshold)
            clipped_negative = np.sum(normalized <= -threshold)
            
            metrics.positive_clips = int(clipped_positive)
            metrics.negative_clips = int(clipped_negative)
            metrics.clipped_samples = int(clipped_positive + clipped_negative)
            metrics.clipping_ratio = metrics.clipped_samples / len(audio)
            
            # Also check for consecutive clipped samples (true clipping)
            # Clipping usually produces flat regions
            consecutive_threshold = int(self.sample_rate * 0.001)  # 1ms
            if consecutive_threshold > 1:
                # Check for runs of clipped samples
                high_mask = normalized >= threshold
                low_mask = normalized <= -threshold
                
                # Simple run detection
                high_runs = self._count_long_runs(high_mask, consecutive_threshold)
                low_runs = self._count_long_runs(low_mask, consecutive_threshold)
                
                if high_runs + low_runs > 0:
                    # Severe clipping detected
                    metrics.clipping_ratio = max(
                        metrics.clipping_ratio, 
                        (high_runs + low_runs) * consecutive_threshold / len(audio)
                    )
            
        except Exception as e:
            logger.warning(f"Clipping detection failed: {e}")
            metrics.clipping_ratio = 0.0
    
    def _count_long_runs(self, mask: np.ndarray, min_length: int) -> int:
        """Count number of runs of True values >= min_length."""
        if not np.any(mask):
            return 0
        
        runs = 0
        current_run = 0
        
        for val in mask:
            if val:
                current_run += 1
            else:
                if current_run >= min_length:
                    runs += 1
                current_run = 0
        
        if current_run >= min_length:
            runs += 1
        
        return runs
    
    def _analyze_dynamic_range(self, audio: np.ndarray, metrics: QualityMetrics):
        """Analyze dynamic range of the signal."""
        try:
            # Peak amplitude
            metrics.peak_amplitude = float(np.max(np.abs(audio)))
            
            # RMS amplitude
            rms = np.sqrt(np.mean(audio ** 2))
            metrics.rms_amplitude = float(rms) if rms > 1e-10 else 1e-10
            
            # Crest factor (peak to RMS ratio)
            if metrics.rms_amplitude > 1e-10:
                crest_factor = metrics.peak_amplitude / metrics.rms_amplitude
                metrics.crest_factor_db = 20 * np.log10(crest_factor)
            
            # Dynamic range estimation using percentiles
            # Remove silent parts for more accurate estimation
            abs_audio = np.abs(audio)
            non_silent = abs_audio[abs_audio > np.percentile(abs_audio, 5)]
            
            if len(non_silent) > 100:
                p95 = np.percentile(non_silent, 95)
                p5 = np.percentile(non_silent, 5)
                
                if p5 > 1e-10:
                    metrics.dynamic_range_db = 20 * np.log10(p95 / p5)
                else:
                    metrics.dynamic_range_db = 60.0
            else:
                metrics.dynamic_range_db = 30.0  # Default
                
        except Exception as e:
            logger.warning(f"Dynamic range analysis failed: {e}")
            metrics.dynamic_range_db = 30.0
    
    def _analyze_frequency_content(self, audio: np.ndarray, metrics: QualityMetrics):
        """Analyze frequency content and bandwidth."""
        try:
            # Compute FFT
            n_fft = min(len(audio), 4096)
            windowed = audio[:n_fft] * np.hanning(n_fft)
            spectrum = np.abs(rfft(windowed))
            freqs = rfftfreq(n_fft, 1 / self.sample_rate)
            
            # Calculate total energy
            total_energy = np.sum(spectrum ** 2)
            if total_energy < 1e-10:
                metrics.frequency_coverage = 0.0
                return
            
            # Energy in speech-relevant bands
            # Low: 80-300 Hz (F0 range)
            low_mask = (freqs >= 80) & (freqs < 300)
            metrics.low_frequency_energy = np.sum(spectrum[low_mask] ** 2) / total_energy
            
            # Mid: 300-3000 Hz (most speech energy)
            mid_mask = (freqs >= 300) & (freqs < 3000)
            metrics.mid_frequency_energy = np.sum(spectrum[mid_mask] ** 2) / total_energy
            
            # High: 3000-8000 Hz (fricatives, sibilants)
            high_mask = (freqs >= 3000) & (freqs <= 8000)
            metrics.high_frequency_energy = np.sum(spectrum[high_mask] ** 2) / total_energy
            
            # Check if we have energy across the full speech range
            speech_mask = (freqs >= self.MIN_FREQUENCY_HZ) & (freqs <= self.MAX_FREQUENCY_HZ)
            speech_energy = np.sum(spectrum[speech_mask] ** 2) / total_energy
            
            metrics.frequency_coverage = float(speech_energy)
            
            # Check for adequate bandwidth
            # Speech should have some energy in all bands
            metrics.has_adequate_bandwidth = (
                metrics.low_frequency_energy > 0.05 and
                metrics.mid_frequency_energy > 0.2 and
                metrics.high_frequency_energy > 0.01
            )
            
        except Exception as e:
            logger.warning(f"Frequency analysis failed: {e}")
            metrics.frequency_coverage = 1.0  # Assume OK
    
    def _analyze_noise(self, audio: np.ndarray, metrics: QualityMetrics):
        """Analyze background noise characteristics."""
        try:
            # Estimate noise floor from spectral analysis
            n_fft = min(len(audio), 2048)
            hop = n_fft // 4
            n_frames = (len(audio) - n_fft) // hop + 1
            
            if n_frames < 2:
                return
            
            # Compute spectrogram
            frame_energies = []
            for i in range(n_frames):
                start = i * hop
                frame = audio[start:start + n_fft] * np.hanning(n_fft)
                energy = np.sum(frame ** 2)
                frame_energies.append(energy)
            
            frame_energies = np.array(frame_energies)
            
            # Noise floor from 5th percentile of frame energies
            if len(frame_energies) > 0:
                noise_energy = np.percentile(frame_energies, 5)
                if noise_energy > 1e-20:
                    metrics.noise_floor_db = 10 * np.log10(noise_energy)
                else:
                    metrics.noise_floor_db = -80.0
            
            # Estimate noise stationarity (variance of frame energies in quiet parts)
            quiet_threshold = np.percentile(frame_energies, 30)
            quiet_frames = frame_energies[frame_energies <= quiet_threshold]
            if len(quiet_frames) > 1:
                cv = np.std(quiet_frames) / (np.mean(quiet_frames) + 1e-10)
                metrics.noise_stationarity = 1.0 / (1.0 + cv)  # 0-1, higher is more stationary
            
        except Exception as e:
            logger.warning(f"Noise analysis failed: {e}")
    
    def _analyze_dc_offset(self, audio: np.ndarray, metrics: QualityMetrics):
        """Check for DC offset in the signal."""
        metrics.dc_offset = float(np.mean(audio))
        metrics.has_dc_offset = abs(metrics.dc_offset) > 0.01
    
    def _calculate_quality_score(self, metrics: QualityMetrics) -> float:
        """
        Calculate overall quality score (0-100).
        
        Weighted combination of quality factors.
        """
        score = 100.0
        
        # SNR component (40% weight)
        if metrics.snr_db < self.MIN_SNR_DB:
            snr_penalty = (self.MIN_SNR_DB - metrics.snr_db) / self.MIN_SNR_DB
            score -= min(40, snr_penalty * 60)
        elif metrics.snr_db < self.GOOD_SNR_DB:
            snr_penalty = (self.GOOD_SNR_DB - metrics.snr_db) / self.GOOD_SNR_DB
            score -= snr_penalty * 20
        
        # Clipping component (25% weight)
        if metrics.clipping_ratio > self.MAX_CLIPPING_RATIO:
            clip_penalty = metrics.clipping_ratio / self.MAX_CLIPPING_RATIO
            score -= min(25, clip_penalty * 30)
        elif metrics.clipping_ratio > self.CLIPPING_WARNING_RATIO:
            clip_penalty = metrics.clipping_ratio / self.MAX_CLIPPING_RATIO
            score -= clip_penalty * 10
        
        # Frequency coverage (20% weight)
        if metrics.frequency_coverage < 0.7:
            freq_penalty = (0.7 - metrics.frequency_coverage) / 0.7
            score -= freq_penalty * 20
        
        # Dynamic range (10% weight)
        if metrics.dynamic_range_db < self.MIN_DYNAMIC_RANGE_DB:
            dr_penalty = (self.MIN_DYNAMIC_RANGE_DB - metrics.dynamic_range_db) / self.MIN_DYNAMIC_RANGE_DB
            score -= dr_penalty * 10
        
        # DC offset (5% weight)
        if metrics.has_dc_offset:
            score -= 5
        
        return max(0.0, min(100.0, score))
    
    def get_quality_issues(self, metrics: QualityMetrics) -> List[str]:
        """Get list of quality issues from metrics."""
        issues = []
        
        if metrics.snr_db < self.MIN_SNR_DB:
            issues.append(f"Low SNR ({metrics.snr_db:.1f} dB < {self.MIN_SNR_DB} dB)")
        
        if metrics.clipping_ratio > self.MAX_CLIPPING_RATIO:
            issues.append(f"Audio clipping detected ({metrics.clipping_ratio*100:.1f}%)")
        
        if metrics.frequency_coverage < 0.7:
            issues.append("Insufficient frequency coverage for speech analysis")
        
        if not metrics.has_adequate_bandwidth:
            issues.append("Missing frequency bands (check microphone/format)")
        
        if metrics.dynamic_range_db < self.MIN_DYNAMIC_RANGE_DB:
            issues.append("Low dynamic range (possible compression)")
        
        if metrics.has_dc_offset:
            issues.append("DC offset detected in signal")
        
        return issues
    
    def get_quality_suggestions(self, metrics: QualityMetrics) -> List[str]:
        """Get improvement suggestions based on quality issues."""
        suggestions = []
        
        if metrics.snr_db < self.MIN_SNR_DB:
            suggestions.extend([
                "Record in a quieter environment",
                "Move closer to the microphone",
                "Use a directional or noise-canceling microphone"
            ])
        
        if metrics.clipping_ratio > self.MAX_CLIPPING_RATIO:
            suggestions.extend([
                "Reduce microphone input volume or gain",
                "Move slightly further from the microphone",
                "Speak at a moderate volume level"
            ])
        
        if not metrics.has_adequate_bandwidth:
            suggestions.extend([
                "Use a higher quality microphone",
                "Check audio recording settings (ensure 16kHz+ sample rate)",
                "Avoid heavy audio compression"
            ])
        
        return suggestions

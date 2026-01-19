"""
Cardiology Pipeline - ECG Signal Processor
Bandpass filtering, baseline removal, and signal normalization.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import logging

from ..config import PROCESSING_CONFIG

logger = logging.getLogger(__name__)

# Try importing scipy
try:
    from scipy import signal as scipy_signal
    from scipy.signal import butter, filtfilt, iirnotch
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not installed. Limited preprocessing available.")


@dataclass
class ProcessedECG:
    """Result of ECG preprocessing."""
    signal: np.ndarray
    sample_rate: int
    duration_seconds: float
    quality_score: float
    baseline_removed: bool
    filtered: bool
    normalized: bool
    original_mean: float
    original_std: float


class ECGProcessor:
    """
    ECG Signal Preprocessing Pipeline.
    
    Steps:
    1. Resampling to target sample rate (500 Hz)
    2. Baseline wander removal (high-pass filter at 0.5 Hz)
    3. Powerline noise removal (notch filter 50/60 Hz)
    4. Bandpass filtering (0.5-45 Hz)
    5. Normalization (z-score)
    """
    
    def __init__(self, target_sample_rate: int = 500):
        self.config = PROCESSING_CONFIG["ecg"]
        self.target_sample_rate = target_sample_rate
        self.low_cutoff = self.config["bandpass_low_hz"]
        self.high_cutoff = self.config["bandpass_high_hz"]
        self.filter_order = self.config["filter_order"]
    
    def preprocess(
        self,
        ecg_data: np.ndarray,
        original_sample_rate: int,
        remove_powerline: bool = True,
        powerline_freq: float = 50,
        normalize: bool = True
    ) -> ProcessedECG:
        """
        Full preprocessing pipeline.
        
        Args:
            ecg_data: Raw ECG signal
            original_sample_rate: Original sampling rate in Hz
            remove_powerline: Remove 50/60 Hz noise
            powerline_freq: Powerline frequency (50 or 60 Hz)
            normalize: Apply z-score normalization
        
        Returns:
            ProcessedECG with cleaned signal
        """
        signal = ecg_data.copy().astype(np.float64)
        
        # Store original statistics
        original_mean = np.nanmean(signal)
        original_std = np.nanstd(signal)
        
        # Handle NaN values
        if np.any(np.isnan(signal)):
            signal = self._interpolate_nans(signal)
        
        # Resample if needed
        if original_sample_rate != self.target_sample_rate:
            signal = self._resample(signal, original_sample_rate)
        
        # Calculate initial quality
        initial_snr = self._estimate_snr(signal)
        
        # Baseline wander removal
        baseline_removed = False
        if SCIPY_AVAILABLE:
            signal = self._remove_baseline(signal)
            baseline_removed = True
        
        # Powerline noise removal
        if remove_powerline and SCIPY_AVAILABLE:
            signal = self._remove_powerline_noise(signal, powerline_freq)
        
        # Bandpass filter
        filtered = False
        if SCIPY_AVAILABLE:
            signal = self._bandpass_filter(signal)
            filtered = True
        
        # Normalization
        normalized = False
        if normalize:
            signal = self._normalize(signal)
            normalized = True
        
        # Calculate final quality
        final_snr = self._estimate_snr(signal)
        quality_score = min(1.0, max(0.0, final_snr / 20))
        
        duration = len(signal) / self.target_sample_rate
        
        return ProcessedECG(
            signal=signal,
            sample_rate=self.target_sample_rate,
            duration_seconds=duration,
            quality_score=quality_score,
            baseline_removed=baseline_removed,
            filtered=filtered,
            normalized=normalized,
            original_mean=original_mean,
            original_std=original_std,
        )
    
    def _resample(self, signal: np.ndarray, original_sr: int) -> np.ndarray:
        """Resample signal to target sample rate."""
        if not SCIPY_AVAILABLE:
            # Linear interpolation fallback
            original_length = len(signal)
            target_length = int(original_length * self.target_sample_rate / original_sr)
            return np.interp(
                np.linspace(0, original_length - 1, target_length),
                np.arange(original_length),
                signal
            )
        
        num_samples = int(len(signal) * self.target_sample_rate / original_sr)
        return scipy_signal.resample(signal, num_samples)
    
    def _remove_baseline(self, signal: np.ndarray) -> np.ndarray:
        """Remove baseline wander using high-pass filter."""
        if not SCIPY_AVAILABLE:
            # Simple detrending
            return signal - np.linspace(signal[0], signal[-1], len(signal))
        
        nyquist = self.target_sample_rate / 2
        low = self.low_cutoff / nyquist
        
        b, a = butter(self.filter_order, low, btype='high')
        return filtfilt(b, a, signal)
    
    def _remove_powerline_noise(
        self,
        signal: np.ndarray,
        freq: float = 50
    ) -> np.ndarray:
        """Remove powerline interference (50/60 Hz)."""
        nyquist = self.target_sample_rate / 2
        notch_freq = freq / nyquist
        
        if notch_freq >= 1:
            return signal  # Frequency too high for sample rate
        
        b, a = iirnotch(notch_freq, Q=self.config["notch_q"])
        return filtfilt(b, a, signal)
    
    def _bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """Bandpass filter for ECG (0.5-45 Hz)."""
        nyquist = self.target_sample_rate / 2
        low = self.low_cutoff / nyquist
        high = min(self.high_cutoff / nyquist, 0.99)
        
        b, a = butter(self.filter_order, [low, high], btype='band')
        return filtfilt(b, a, signal)
    
    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signal to zero mean and unit variance."""
        mean = np.mean(signal)
        std = np.std(signal)
        
        if std > 0:
            return (signal - mean) / std
        return signal - mean
    
    def _estimate_snr(self, signal: np.ndarray) -> float:
        """Estimate signal-to-noise ratio in dB."""
        if not SCIPY_AVAILABLE or len(signal) < 100:
            return 10.0  # Default
        
        try:
            nyquist = self.target_sample_rate / 2
            cutoff = min(40 / nyquist, 0.99)
            b, a = butter(2, cutoff, btype='high')
            noise = filtfilt(b, a, signal)
            
            signal_power = np.var(signal)
            noise_power = np.var(noise)
            
            if noise_power > 0:
                return 10 * np.log10(signal_power / noise_power)
            return 20.0
        except Exception:
            return 10.0
    
    def _interpolate_nans(self, signal: np.ndarray) -> np.ndarray:
        """Interpolate NaN values in signal."""
        nans = np.isnan(signal)
        if not np.any(nans):
            return signal
        
        indices = np.arange(len(signal))
        signal[nans] = np.interp(
            indices[nans],
            indices[~nans],
            signal[~nans]
        )
        return signal


def preprocess_ecg(
    ecg_data: np.ndarray,
    sample_rate: int = 500,
    remove_powerline: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Convenience function to preprocess ECG.
    
    Args:
        ecg_data: Raw ECG signal
        sample_rate: Sampling rate in Hz
        remove_powerline: Remove powerline noise
        normalize: Apply z-score normalization
    
    Returns:
        Tuple of (processed_signal, quality_score)
    """
    processor = ECGProcessor(target_sample_rate=sample_rate)
    result = processor.preprocess(
        ecg_data,
        original_sample_rate=sample_rate,
        remove_powerline=remove_powerline,
        normalize=normalize,
    )
    return result.signal, result.quality_score

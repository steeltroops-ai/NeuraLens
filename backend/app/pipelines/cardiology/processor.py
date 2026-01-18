"""
Cardiology Pipeline - ECG Signal Preprocessing
Bandpass filtering, baseline removal, and noise reduction
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import logging

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
class ProcessedSignal:
    """Preprocessed ECG signal result"""
    signal: np.ndarray
    sample_rate: int
    duration_seconds: float
    quality_score: float
    baseline_removed: bool
    filtered: bool


class ECGProcessor:
    """
    ECG Signal Preprocessing
    
    Pipeline:
    1. Resampling to target sample rate
    2. Baseline wander removal (high-pass filter)
    3. Powerline noise removal (notch filter 50/60 Hz)
    4. Bandpass filtering (0.5-45 Hz for ECG)
    5. Normalization
    """
    
    # ECG frequency range
    LOW_CUTOFF = 0.5   # Hz - removes baseline wander
    HIGH_CUTOFF = 45   # Hz - removes high-frequency noise
    FILTER_ORDER = 4
    
    def __init__(self, target_sample_rate: int = 500):
        self.target_sample_rate = target_sample_rate
    
    def preprocess(
        self,
        ecg_data: np.ndarray,
        original_sample_rate: int,
        remove_powerline: bool = True,
        powerline_freq: float = 50
    ) -> ProcessedSignal:
        """
        Full preprocessing pipeline
        
        Args:
            ecg_data: Raw ECG signal
            original_sample_rate: Original sampling rate in Hz
            remove_powerline: Remove 50/60 Hz noise
            powerline_freq: Powerline frequency (50 or 60 Hz)
            
        Returns:
            ProcessedSignal with cleaned ECG
        """
        signal = ecg_data.copy().astype(np.float64)
        
        # Resample if needed
        if original_sample_rate != self.target_sample_rate:
            signal = self._resample(signal, original_sample_rate)
        
        # Calculate initial quality
        initial_snr = self._estimate_snr(signal)
        
        # Remove baseline wander
        signal = self._remove_baseline(signal)
        
        # Remove powerline noise
        if remove_powerline and SCIPY_AVAILABLE:
            signal = self._remove_powerline_noise(signal, powerline_freq)
        
        # Bandpass filter
        if SCIPY_AVAILABLE:
            signal = self._bandpass_filter(signal)
        
        # Normalize
        signal = self._normalize(signal)
        
        # Calculate final quality
        final_snr = self._estimate_snr(signal)
        quality_score = min(1.0, final_snr / 20)  # Normalize to 0-1
        
        duration = len(signal) / self.target_sample_rate
        
        return ProcessedSignal(
            signal=signal,
            sample_rate=self.target_sample_rate,
            duration_seconds=duration,
            quality_score=quality_score,
            baseline_removed=True,
            filtered=SCIPY_AVAILABLE
        )
    
    def _resample(self, signal: np.ndarray, original_sr: int) -> np.ndarray:
        """Resample signal to target sample rate"""
        if not SCIPY_AVAILABLE:
            # Simple linear interpolation fallback
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
        """Remove baseline wander using high-pass filter"""
        if not SCIPY_AVAILABLE:
            # Simple detrending
            return signal - np.linspace(signal[0], signal[-1], len(signal))
        
        # High-pass filter at 0.5 Hz
        nyquist = self.target_sample_rate / 2
        low = self.LOW_CUTOFF / nyquist
        
        b, a = butter(self.FILTER_ORDER, low, btype='high')
        return filtfilt(b, a, signal)
    
    def _remove_powerline_noise(
        self,
        signal: np.ndarray,
        freq: float = 50
    ) -> np.ndarray:
        """Remove powerline interference (50/60 Hz)"""
        nyquist = self.target_sample_rate / 2
        notch_freq = freq / nyquist
        
        # Notch filter with Q=30
        b, a = iirnotch(notch_freq, Q=30)
        return filtfilt(b, a, signal)
    
    def _bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """Bandpass filter for ECG (0.5-45 Hz)"""
        nyquist = self.target_sample_rate / 2
        low = self.LOW_CUTOFF / nyquist
        high = self.HIGH_CUTOFF / nyquist
        
        b, a = butter(self.FILTER_ORDER, [low, high], btype='band')
        return filtfilt(b, a, signal)
    
    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signal to zero mean and unit variance"""
        mean = np.mean(signal)
        std = np.std(signal)
        
        if std > 0:
            return (signal - mean) / std
        return signal - mean
    
    def _estimate_snr(self, signal: np.ndarray) -> float:
        """Estimate signal-to-noise ratio"""
        # Simple SNR estimation using signal variance
        # High-frequency component assumed to be noise
        if not SCIPY_AVAILABLE:
            return 10.0  # Default
        
        # High-pass filter to get noise estimate
        nyquist = self.target_sample_rate / 2
        b, a = butter(2, 40 / nyquist, btype='high')
        noise = filtfilt(b, a, signal)
        
        signal_power = np.var(signal)
        noise_power = np.var(noise)
        
        if noise_power > 0:
            return 10 * np.log10(signal_power / noise_power)
        return 20.0


def preprocess_ecg(
    ecg_data: np.ndarray,
    sample_rate: int = 500,
    remove_powerline: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Convenience function to preprocess ECG
    
    Args:
        ecg_data: Raw ECG signal
        sample_rate: Sampling rate in Hz
        remove_powerline: Remove powerline noise
        
    Returns:
        Tuple of (processed_signal, quality_score)
    """
    processor = ECGProcessor(target_sample_rate=sample_rate)
    result = processor.preprocess(
        ecg_data,
        original_sample_rate=sample_rate,
        remove_powerline=remove_powerline
    )
    return result.signal, result.quality_score

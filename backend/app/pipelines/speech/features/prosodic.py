"""
Prosodic Feature Extractor
Extracts temporal, rhythmic, and prosodic features from speech.

Features:
- Speech rate (syllables/second)
- Pause patterns (ratio, duration, frequency)
- F0 dynamics (contour, modulation)
- Intensity dynamics
- Tremor analysis (4-12Hz band)
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.signal import find_peaks
import librosa

logger = logging.getLogger(__name__)


@dataclass
class TremorAnalysis:
    """Voice tremor analysis in pathological frequency bands."""
    tremor_detected: bool = False
    tremor_score: float = 0.0           # Overall tremor intensity (0-1)
    dominant_frequency: float = 0.0      # Peak tremor frequency (Hz)
    parkinsonian_power: float = 0.0      # Power in 4-6Hz band
    essential_tremor_power: float = 0.0  # Power in 4-12Hz band
    tremor_type: str = "none"           # Classification
    
    def to_dict(self) -> Dict:
        return {
            "tremor_detected": self.tremor_detected,
            "tremor_score": self.tremor_score,
            "tremor_dominant_freq": self.dominant_frequency,
            "tremor_pd_power": self.parkinsonian_power,
            "tremor_et_power": self.essential_tremor_power,
            "tremor_type": self.tremor_type
        }


@dataclass
class PauseMetrics:
    """Pause pattern analysis."""
    pause_ratio: float = 0.0            # Non-speech / total duration
    pause_count: int = 0                # Number of pauses
    mean_pause_duration: float = 0.0    # Average pause length (s)
    max_pause_duration: float = 0.0     # Longest pause (s)
    pause_rate: float = 0.0             # Pauses per second
    
    def to_dict(self) -> Dict:
        return {
            "pause_ratio": self.pause_ratio,
            "pause_count": self.pause_count,
            "mean_pause_duration": self.mean_pause_duration,
            "max_pause_duration": self.max_pause_duration,
            "pause_rate": self.pause_rate
        }


@dataclass
class ProsodicFeatures:
    """Complete prosodic feature set."""
    # Speech rate
    speech_rate: float = 0.0            # Syllables per second
    articulation_rate: float = 0.0       # Rate during speech segments only
    
    # Pause patterns
    pauses: PauseMetrics = None
    
    # F0 dynamics
    f0_cv: float = 0.0                  # Coefficient of variation
    f0_slope: float = 0.0               # Overall pitch trend
    f0_excursion_rate: float = 0.0      # Rate of pitch changes
    
    # Intensity dynamics
    intensity_mean: float = 0.0         # Mean intensity (dB)
    intensity_std: float = 0.0          # Intensity variation
    intensity_range: float = 0.0        # Dynamic range (dB)
    
    # Tremor
    tremor: TremorAnalysis = None
    
    # Rhythm
    rhythm_regularity: float = 0.0      # Syllable timing regularity
    
    def __post_init__(self):
        if self.pauses is None:
            self.pauses = PauseMetrics()
        if self.tremor is None:
            self.tremor = TremorAnalysis()
    
    def to_dict(self) -> Dict:
        result = {
            "speech_rate": self.speech_rate,
            "articulation_rate": self.articulation_rate,
            "f0_cv": self.f0_cv,
            "f0_slope": self.f0_slope,
            "f0_excursion_rate": self.f0_excursion_rate,
            "intensity_mean": self.intensity_mean,
            "intensity_std": self.intensity_std,
            "intensity_range": self.intensity_range,
            "rhythm_regularity": self.rhythm_regularity
        }
        result.update(self.pauses.to_dict())
        result.update(self.tremor.to_dict())
        return result


class ProsodicFeatureExtractor:
    """
    Extract prosodic and temporal features from speech.
    
    These features capture rhythm, timing, and modulation patterns
    that are clinically relevant for neurological assessment.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: int = 2048,
        hop_length: int = 512,
        silence_threshold_db: float = -40
    ):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.silence_threshold_db = silence_threshold_db
    
    def extract(
        self, 
        audio: np.ndarray,
        f0_contour: Optional[np.ndarray] = None
    ) -> ProsodicFeatures:
        """
        Extract prosodic features from audio.
        
        Args:
            audio: Normalized audio array
            f0_contour: Optional pre-computed F0 contour
            
        Returns:
            ProsodicFeatures dataclass
        """
        features = ProsodicFeatures()
        
        try:
            # Speech rate
            features.speech_rate, syllable_times = self._estimate_speech_rate(audio)
            
            # Pause analysis
            features.pauses = self._analyze_pauses(audio)
            
            # Articulation rate (speech rate excluding pauses)
            speech_duration = len(audio) / self.sample_rate * (1 - features.pauses.pause_ratio)
            if speech_duration > 0:
                syllables = max(1, int(features.speech_rate * len(audio) / self.sample_rate))
                features.articulation_rate = syllables / speech_duration
            
            # F0 dynamics
            if f0_contour is not None and len(f0_contour) > 0:
                features.f0_cv = self._calculate_cv(f0_contour)
                features.f0_slope = self._calculate_slope(f0_contour)
                features.f0_excursion_rate = self._calculate_excursion_rate(f0_contour)
            
            # Intensity analysis
            rms = librosa.feature.rms(
                y=audio, 
                frame_length=self.frame_length, 
                hop_length=self.hop_length
            )[0]
            
            if len(rms) > 0:
                rms_db = librosa.amplitude_to_db(rms + 1e-10)
                features.intensity_mean = float(np.mean(rms_db))
                features.intensity_std = float(np.std(rms_db))
                features.intensity_range = float(np.max(rms_db) - np.min(rms_db))
            
            # Tremor analysis
            features.tremor = self._analyze_tremor(audio, f0_contour)
            
            # Rhythm regularity
            if len(syllable_times) > 2:
                intervals = np.diff(syllable_times)
                features.rhythm_regularity = 1.0 - min(1.0, np.std(intervals) / (np.mean(intervals) + 1e-6))
            
        except Exception as e:
            logger.error(f"Prosodic feature extraction failed: {e}")
            
        return features
    
    def _estimate_speech_rate(self, audio: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Estimate syllables per second using envelope peak detection.
        
        Returns:
            (speech_rate, syllable_onset_times)
        """
        if len(audio) == 0:
            return 0.0, np.array([])
        
        try:
            # Get RMS envelope
            rms = librosa.feature.rms(
                y=audio, 
                frame_length=self.frame_length, 
                hop_length=self.hop_length
            )[0]
            
            if len(rms) < 3:
                return 0.0, np.array([])
            
            # Low-pass filter to smooth envelope
            # Syllable rate is typically 3-7 Hz
            b, a = signal.butter(4, 10.0 / (self.sample_rate / self.hop_length / 2), btype='low')
            rms_smooth = signal.filtfilt(b, a, rms)
            
            # Find peaks (syllable nuclei)
            threshold = np.mean(rms_smooth) * 0.5
            min_distance = int(0.1 * self.sample_rate / self.hop_length)  # Min 100ms between syllables
            
            peaks, _ = find_peaks(rms_smooth, height=threshold, distance=min_distance)
            
            # Convert to times
            duration = len(audio) / self.sample_rate
            syllable_times = peaks * self.hop_length / self.sample_rate
            
            speech_rate = len(peaks) / duration if duration > 0 else 0.0
            
            return float(speech_rate), syllable_times
            
        except Exception as e:
            logger.warning(f"Speech rate estimation failed: {e}")
            return 0.0, np.array([])
    
    def _analyze_pauses(self, audio: np.ndarray) -> PauseMetrics:
        """Analyze pause patterns in speech."""
        metrics = PauseMetrics()
        
        try:
            # Get RMS envelope
            rms = librosa.feature.rms(
                y=audio, 
                frame_length=self.frame_length, 
                hop_length=self.hop_length
            )[0]
            
            # Convert to dB
            rms_db = librosa.amplitude_to_db(rms + 1e-10)
            
            # Identify silent frames
            silence_mask = rms_db < self.silence_threshold_db
            
            # Find pause regions (contiguous silent frames)
            pauses = []
            in_pause = False
            pause_start = 0
            
            for i, is_silent in enumerate(silence_mask):
                if is_silent and not in_pause:
                    pause_start = i
                    in_pause = True
                elif not is_silent and in_pause:
                    pause_end = i
                    pause_duration = (pause_end - pause_start) * self.hop_length / self.sample_rate
                    if pause_duration > 0.15:  # Only count pauses > 150ms
                        pauses.append(pause_duration)
                    in_pause = False
            
            # Calculate metrics
            total_duration = len(audio) / self.sample_rate
            
            if pauses:
                metrics.pause_count = len(pauses)
                metrics.pause_ratio = sum(pauses) / total_duration if total_duration > 0 else 0.0
                metrics.mean_pause_duration = float(np.mean(pauses))
                metrics.max_pause_duration = float(np.max(pauses))
                metrics.pause_rate = len(pauses) / total_duration if total_duration > 0 else 0.0
                
        except Exception as e:
            logger.warning(f"Pause analysis failed: {e}")
            
        return metrics
    
    def _analyze_tremor(
        self, 
        audio: np.ndarray,
        f0_contour: Optional[np.ndarray] = None
    ) -> TremorAnalysis:
        """
        Analyze voice tremor in pathological frequency bands.
        
        Clinical basis:
        - Parkinsonian resting tremor: 4-6 Hz
        - Essential tremor: 4-12 Hz
        - Physiological tremor: 8-12 Hz (normal, low amplitude)
        """
        tremor = TremorAnalysis()
        
        try:
            # Use amplitude envelope for tremor detection
            rms = librosa.feature.rms(
                y=audio, 
                frame_length=512, 
                hop_length=128
            )[0]
            
            if len(rms) < 50:
                return tremor
            
            # Sampling rate of envelope
            envelope_sr = self.sample_rate / 128
            
            # Remove mean and detrend
            rms_centered = signal.detrend(rms - np.mean(rms))
            
            # FFT analysis
            n = len(rms_centered)
            spectrum = np.abs(np.fft.rfft(rms_centered))
            freqs = np.fft.rfftfreq(n, 1/envelope_sr)
            
            # Total power for normalization
            total_power = np.sum(spectrum**2) + 1e-10
            
            # Parkinsonian tremor band (4-6 Hz)
            pd_mask = (freqs >= 4) & (freqs <= 6)
            pd_power = np.sum(spectrum[pd_mask]**2)
            
            # Essential tremor band (4-12 Hz)
            et_mask = (freqs >= 4) & (freqs <= 12)
            et_power = np.sum(spectrum[et_mask]**2)
            
            # Calculate tremor score
            tremor.tremor_score = float(et_power / total_power)
            tremor.parkinsonian_power = float(pd_power / total_power)
            tremor.essential_tremor_power = float(et_power / total_power)
            
            # Find dominant frequency in tremor band
            tremor_band_mask = (freqs >= 3) & (freqs <= 15)
            if np.any(tremor_band_mask):
                tremor_spectrum = spectrum * tremor_band_mask
                peak_idx = np.argmax(tremor_spectrum)
                tremor.dominant_frequency = float(freqs[peak_idx])
            
            # Classification
            tremor.tremor_detected = tremor.tremor_score > 0.10
            tremor.tremor_type = self._classify_tremor(
                tremor.dominant_frequency, 
                tremor.tremor_score
            )
            
        except Exception as e:
            logger.warning(f"Tremor analysis failed: {e}")
            
        return tremor
    
    def _classify_tremor(self, freq: float, amplitude: float) -> str:
        """Classify tremor type based on frequency and amplitude."""
        if amplitude < 0.05:
            return "none"
        elif 4 <= freq <= 6:
            return "parkinsonian_resting"
        elif 4 <= freq <= 12:
            return "essential_postural"
        elif freq > 8:
            return "physiological_normal"
        else:
            return "unclassified"
    
    def _calculate_cv(self, values: np.ndarray) -> float:
        """Calculate coefficient of variation."""
        values = values[~np.isnan(values) & (values > 0)]
        if len(values) == 0:
            return 0.0
        mean = np.mean(values)
        if mean == 0:
            return 0.0
        return float(np.std(values) / mean)
    
    def _calculate_slope(self, values: np.ndarray) -> float:
        """Calculate linear slope of time series."""
        values = values[~np.isnan(values)]
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        try:
            slope, _ = np.polyfit(x, values, 1)
            return float(slope)
        except:
            return 0.0
    
    def _calculate_excursion_rate(self, values: np.ndarray) -> float:
        """Calculate rate of significant excursions (direction changes)."""
        values = values[~np.isnan(values)]
        if len(values) < 3:
            return 0.0
        
        # Count sign changes in first derivative
        diff = np.diff(values)
        sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
        
        return float(sign_changes / len(values))

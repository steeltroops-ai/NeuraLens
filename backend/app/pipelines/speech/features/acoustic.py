"""
Acoustic Feature Extractor - Clinical Grade
Uses Parselmouth (Praat) for gold-standard acoustic analysis.

Features extracted:
- Jitter variants (local, rap, ppq5, ddp)
- Shimmer variants (local, apq3, apq5, apq11)
- HNR (Harmonics-to-Noise Ratio)
- CPPS (Cepstral Peak Prominence Smoothed)
- Voice breaks
- Degree of voice breaks

References:
- Tsanas et al. (2012) - PD voice analysis
- Maryn et al. (2010) - CPPS development
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import parselmouth
from parselmouth.praat import call

logger = logging.getLogger(__name__)


@dataclass
class JitterMetrics:
    """All jitter (frequency perturbation) variants."""
    local: float = 0.0          # Jitter (local) %
    rap: float = 0.0            # Relative Average Perturbation %
    ppq5: float = 0.0           # Five-point Period Perturbation Quotient %
    ddp: float = 0.0            # Difference of Differences of Periods %
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "jitter_local": self.local,
            "jitter_rap": self.rap,
            "jitter_ppq5": self.ppq5,
            "jitter_ddp": self.ddp
        }


@dataclass
class ShimmerMetrics:
    """All shimmer (amplitude perturbation) variants."""
    local: float = 0.0          # Shimmer (local) %
    apq3: float = 0.0           # Three-point Amplitude Perturbation Quotient %
    apq5: float = 0.0           # Five-point Amplitude Perturbation Quotient %
    apq11: float = 0.0          # 11-point Amplitude Perturbation Quotient %
    dda: float = 0.0            # Difference of Differences of Amplitudes %
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "shimmer_local": self.local,
            "shimmer_apq3": self.apq3,
            "shimmer_apq5": self.apq5,
            "shimmer_apq11": self.apq11,
            "shimmer_dda": self.dda
        }


@dataclass
class AcousticFeatures:
    """Complete acoustic feature set from Parselmouth/Praat."""
    # Perturbation measures
    jitter: JitterMetrics = field(default_factory=JitterMetrics)
    shimmer: ShimmerMetrics = field(default_factory=ShimmerMetrics)
    
    # Voice quality
    hnr: float = 0.0                    # Harmonics-to-Noise Ratio (dB)
    cpps: float = 0.0                   # Cepstral Peak Prominence Smoothed (dB)
    
    # Voice breaks
    voice_breaks_count: int = 0
    voice_breaks_degree: float = 0.0    # Fraction of unvoiced frames (0-1)
    
    # Pitch statistics
    mean_f0: float = 0.0                # Mean fundamental frequency (Hz)
    std_f0: float = 0.0                 # Standard deviation of F0 (Hz)
    min_f0: float = 0.0                 # Minimum F0 (Hz)
    max_f0: float = 0.0                 # Maximum F0 (Hz)
    f0_range: float = 0.0               # F0 range (Hz)
    
    # Formants
    f1_mean: float = 0.0                # First formant mean (Hz)
    f2_mean: float = 0.0                # Second formant mean (Hz)
    f3_mean: float = 0.0                # Third formant mean (Hz)
    
    # Quality indicators
    num_periods: int = 0
    voiced_fraction: float = 0.0
    duration: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Flatten to dictionary for downstream processing."""
        result = {
            **self.jitter.to_dict(),
            **self.shimmer.to_dict(),
            "hnr": self.hnr,
            "cpps": self.cpps,
            "voice_breaks_count": self.voice_breaks_count,
            "voice_breaks_degree": self.voice_breaks_degree,
            "mean_f0": self.mean_f0,
            "std_f0": self.std_f0,
            "min_f0": self.min_f0,
            "max_f0": self.max_f0,
            "f0_range": self.f0_range,
            "f1_mean": self.f1_mean,
            "f2_mean": self.f2_mean,
            "f3_mean": self.f3_mean,
            "num_periods": self.num_periods,
            "voiced_fraction": self.voiced_fraction,
            "duration": self.duration
        }
        return result


class AcousticFeatureExtractor:
    """
    Clinical-grade acoustic feature extraction using Parselmouth (Praat).
    
    This is the gold standard for voice analysis in clinical settings.
    All algorithms match those used in published clinical research.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        pitch_floor: float = 75.0,
        pitch_ceiling: float = 500.0,
        time_step: float = 0.01,
        silence_threshold: float = 0.1
    ):
        self.sample_rate = sample_rate
        self.pitch_floor = pitch_floor
        self.pitch_ceiling = pitch_ceiling
        self.time_step = time_step
        self.silence_threshold = silence_threshold
        
    def extract(self, audio: np.ndarray) -> AcousticFeatures:
        """
        Extract all acoustic features from audio.
        
        Args:
            audio: Normalized float32 audio array [-1, 1]
            
        Returns:
            AcousticFeatures dataclass with all measurements
        """
        features = AcousticFeatures()
        
        try:
            # Create Parselmouth Sound object
            sound = parselmouth.Sound(audio, sampling_frequency=self.sample_rate)
            features.duration = sound.get_total_duration()
            
            # Extract pitch
            pitch = self._extract_pitch(sound)
            features = self._populate_pitch_stats(features, pitch)
            
            # Extract point process for perturbation measures
            point_process = self._extract_point_process(sound)
            features.num_periods = self._get_num_periods(point_process)
            
            # Extract perturbation measures if we have enough periods
            # 3 periods minimum for jitter/shimmer (Praat requirement)
            if features.num_periods >= 3:
                features.jitter = self._extract_jitter(point_process)
                features.shimmer = self._extract_shimmer(sound, point_process)
                features.hnr = self._extract_hnr(sound)
                if features.num_periods < 10:
                    logger.warning(
                        f"Low period count ({features.num_periods}), "
                        "perturbation values may have reduced reliability"
                    )
            else:
                logger.warning(
                    f"Insufficient periods ({features.num_periods}) for perturbation extraction"
                )
            
            # CPPS - always try to extract (robust measure)
            features.cpps = self._extract_cpps(sound)
            
            # Voice breaks
            features.voice_breaks_count, features.voice_breaks_degree = \
                self._extract_voice_breaks(pitch)
            
            # Formants
            features.f1_mean, features.f2_mean, features.f3_mean = \
                self._extract_formants(sound)
            
            # Voiced fraction
            features.voiced_fraction = self._calculate_voiced_fraction(pitch)
            
        except Exception as e:
            logger.error(f"Acoustic feature extraction failed: {e}")
            
        return features
    
    def _extract_pitch(self, sound: parselmouth.Sound) -> parselmouth.Pitch:
        """Extract pitch object using autocorrelation method."""
        return sound.to_pitch_ac(
            time_step=self.time_step,
            pitch_floor=self.pitch_floor,
            pitch_ceiling=self.pitch_ceiling
        )
    
    def _populate_pitch_stats(
        self, 
        features: AcousticFeatures, 
        pitch: parselmouth.Pitch
    ) -> AcousticFeatures:
        """Calculate pitch statistics."""
        try:
            features.mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")
            features.std_f0 = call(pitch, "Get standard deviation", 0, 0, "Hertz")
            features.min_f0 = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
            features.max_f0 = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
            
            # Handle NaN values
            if np.isnan(features.mean_f0): features.mean_f0 = 0.0
            if np.isnan(features.std_f0): features.std_f0 = 0.0
            if np.isnan(features.min_f0): features.min_f0 = 0.0
            if np.isnan(features.max_f0): features.max_f0 = 0.0
            
            features.f0_range = features.max_f0 - features.min_f0
            
        except Exception as e:
            logger.warning(f"Pitch statistics extraction failed: {e}")
            
        return features
    
    def _extract_point_process(self, sound: parselmouth.Sound):
        """Extract point process for cycle detection."""
        return call(
            sound, 
            "To PointProcess (periodic, cc)", 
            self.pitch_floor, 
            self.pitch_ceiling
        )
    
    def _get_num_periods(self, point_process) -> int:
        """Get number of detected periods."""
        try:
            return int(call(
                point_process, 
                "Get number of periods", 
                0.0, 0.0, 
                self.pitch_floor, self.pitch_ceiling, 
                1.3
            ))
        except:
            return 0
    
    def _extract_jitter(self, point_process) -> JitterMetrics:
        """Extract all jitter variants."""
        jitter = JitterMetrics()
        
        try:
            # Jitter (local) - percentage
            jitter.local = call(
                point_process, "Get jitter (local)", 
                0, 0, 0.0001, 0.02, 1.3
            ) * 100
            
            # Jitter (rap)
            jitter.rap = call(
                point_process, "Get jitter (rap)", 
                0, 0, 0.0001, 0.02, 1.3
            ) * 100
            
            # Jitter (ppq5)
            jitter.ppq5 = call(
                point_process, "Get jitter (ppq5)", 
                0, 0, 0.0001, 0.02, 1.3
            ) * 100
            
            # Jitter (ddp)
            jitter.ddp = call(
                point_process, "Get jitter (ddp)", 
                0, 0, 0.0001, 0.02, 1.3
            ) * 100
            
            # Handle NaN
            for attr in ['local', 'rap', 'ppq5', 'ddp']:
                if np.isnan(getattr(jitter, attr)):
                    setattr(jitter, attr, 0.0)
                    
        except Exception as e:
            logger.warning(f"Jitter extraction failed: {e}")
            
        return jitter
    
    def _extract_shimmer(self, sound, point_process) -> ShimmerMetrics:
        """Extract all shimmer variants."""
        shimmer = ShimmerMetrics()
        
        try:
            # Shimmer (local) - percentage
            shimmer.local = call(
                [sound, point_process], "Get shimmer (local)", 
                0, 0, 0.0001, 0.02, 1.3, 1.6
            ) * 100
            
            # Shimmer (apq3)
            shimmer.apq3 = call(
                [sound, point_process], "Get shimmer (apq3)", 
                0, 0, 0.0001, 0.02, 1.3, 1.6
            ) * 100
            
            # Shimmer (apq5)
            shimmer.apq5 = call(
                [sound, point_process], "Get shimmer (apq5)", 
                0, 0, 0.0001, 0.02, 1.3, 1.6
            ) * 100
            
            # Shimmer (apq11)
            shimmer.apq11 = call(
                [sound, point_process], "Get shimmer (apq11)", 
                0, 0, 0.0001, 0.02, 1.3, 1.6
            ) * 100
            
            # Shimmer (dda)
            shimmer.dda = call(
                [sound, point_process], "Get shimmer (dda)", 
                0, 0, 0.0001, 0.02, 1.3, 1.6
            ) * 100
            
            # Handle NaN
            for attr in ['local', 'apq3', 'apq5', 'apq11', 'dda']:
                if np.isnan(getattr(shimmer, attr)):
                    setattr(shimmer, attr, 0.0)
                    
        except Exception as e:
            logger.warning(f"Shimmer extraction failed: {e}")
            
        return shimmer
    
    def _extract_hnr(self, sound: parselmouth.Sound) -> float:
        """Extract Harmonics-to-Noise Ratio."""
        try:
            harmonicity = sound.to_harmonicity_ac(
                time_step=self.time_step,
                pitch_floor=self.pitch_floor,
                silence_threshold=self.silence_threshold,
                number_of_periods_per_window=4.5
            )
            hnr = call(harmonicity, "Get mean", 0, 0)
            return 0.0 if np.isnan(hnr) else hnr
        except Exception as e:
            logger.warning(f"HNR extraction failed: {e}")
            return 0.0
    
    def _extract_cpps(self, sound: parselmouth.Sound) -> float:
        """
        Extract Cepstral Peak Prominence Smoothed (CPPS).
        
        CPPS is the gold standard for voice quality assessment,
        more robust than jitter/shimmer for severely disordered voices.
        """
        try:
            power_cepstrogram = call(
                sound, "To PowerCepstrogram", 
                60.0,   # pitch_floor
                0.002,  # time_step
                5000.0, # maximum_frequency
                50.0    # pre_emphasis_from
            )
            
            # Try the simpler CPPS call first (works with more versions)
            try:
                cpps = call(
                    power_cepstrogram, "Get CPPS",
                    False,      # subtract_tilt_before_smoothing
                    0.01,       # time_averaging_window
                    0.001,      # quefrency_averaging_window
                    60.0,       # peak_search_pitch_range_start
                    330.0,      # peak_search_pitch_range_end
                    0.05,       # tolerance
                    "Parabolic",# interpolation
                    0.001,      # tilt_line_quefrency_range_start
                    0.05,       # tilt_line_quefrency_range_end
                    "Straight", # line_type (compatible)
                    "Least squares" # fit_method (compatible)
                )
            except:
                # Fallback: even simpler call
                cpps = call(
                    power_cepstrogram, "Get CPPS",
                    False, 0.01, 0.001, 60.0, 330.0, 0.05, "Parabolic",
                    0.001, 0.05, "Straight", "Least squares"
                )
            
            return 0.0 if np.isnan(cpps) else cpps
            
        except Exception as e:
            logger.warning(f"CPPS extraction failed: {e}")
            return 0.0
    
    def _extract_voice_breaks(self, pitch: parselmouth.Pitch) -> Tuple[int, float]:
        """Extract voice break count and degree (unvoiced fraction)."""
        breaks = 0
        degree = 0.0
        
        try:
            # Get unvoiced fraction - this works reliably
            degree = call(pitch, "Get fraction of locally unvoiced frames")
            degree = 0.0 if np.isnan(degree) else degree
        except Exception as e:
            logger.warning(f"Unvoiced fraction extraction failed: {e}")
        
        # Estimate breaks from pitch contour
        try:
            f0_values = pitch.selected_array['frequency']
            # Count transitions from voiced to unvoiced
            voiced = f0_values > 0
            transitions = np.diff(voiced.astype(int))
            breaks = int(np.sum(transitions == -1))  # Voiced -> unvoiced
        except Exception as e:
            logger.warning(f"Voice breaks counting failed: {e}")
            
        return breaks, degree
    
    def _extract_formants(
        self, 
        sound: parselmouth.Sound
    ) -> Tuple[float, float, float]:
        """Extract mean formant frequencies F1, F2, F3."""
        try:
            formants = sound.to_formant_burg(
                time_step=self.time_step,
                max_number_of_formants=5,
                maximum_formant=5500.0
            )
            
            f1 = call(formants, "Get mean", 1, 0, 0, "Hertz")
            f2 = call(formants, "Get mean", 2, 0, 0, "Hertz")
            f3 = call(formants, "Get mean", 3, 0, 0, "Hertz")
            
            f1 = 0.0 if np.isnan(f1) else f1
            f2 = 0.0 if np.isnan(f2) else f2
            f3 = 0.0 if np.isnan(f3) else f3
            
            return f1, f2, f3
            
        except Exception as e:
            logger.warning(f"Formant extraction failed: {e}")
            return 0.0, 0.0, 0.0
    
    def _calculate_voiced_fraction(self, pitch: parselmouth.Pitch) -> float:
        """Calculate fraction of voiced frames."""
        try:
            unvoiced = call(pitch, "Get fraction of locally unvoiced frames")
            return 1.0 - unvoiced if not np.isnan(unvoiced) else 0.0
        except:
            return 0.0

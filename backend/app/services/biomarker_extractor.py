"""
Biomarker Extractor Service for Speech Analysis Pipeline
Extracts neurological biomarkers from audio features with fallback support.

Feature: speech-pipeline-fix
**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6**
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExtractedBiomarkers:
    """
    All extracted voice biomarkers with estimation tracking.
    
    Each biomarker includes:
    - value: The calculated or estimated value
    - is_estimated: Whether the value is a fallback default
    """
    jitter: float  # 0-1 normalized (fundamental frequency variation)
    shimmer: float  # 0-1 normalized (amplitude variation)
    hnr: float  # 0-30 dB typical (Harmonics-to-Noise Ratio)
    speech_rate: float  # syllables per second (0.5-10 range)
    pause_ratio: float  # 0-1 (proportion of silence to total duration)
    fluency_score: float  # 0-1 (speech fluency measure)
    voice_tremor: float  # 0-1 (tremor intensity)
    articulation_clarity: float  # 0-1 (clarity of articulation)
    prosody_variation: float  # 0-1 (prosodic richness)
    estimated_flags: Dict[str, bool] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize estimated_flags if not provided"""
        if not self.estimated_flags:
            self.estimated_flags = {
                'jitter': False,
                'shimmer': False,
                'hnr': False,
                'speech_rate': False,
                'pause_ratio': False,
                'fluency_score': False,
                'voice_tremor': False,
                'articulation_clarity': False,
                'prosody_variation': False
            }


class BiomarkerExtractor:
    """
    Extracts neurological biomarkers from audio features.
    
    Provides robust extraction with clinically-validated fallback values
    when extraction fails for specific metrics.
    
    **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6**
    """
    
    # Clinically-validated default values based on healthy population norms
    DEFAULT_VALUES: Dict[str, float] = {
        'jitter': 0.02,  # ~2% is typical for healthy adults
        'shimmer': 0.05,  # ~5% is typical for healthy adults
        'hnr': 15.0,  # 15 dB is typical for healthy adults
        'speech_rate': 4.5,  # ~4.5 syllables/second is average
        'pause_ratio': 0.3,  # ~30% pause time is typical
        'fluency_score': 0.7,  # Moderate fluency as default
        'voice_tremor': 0.1,  # Low tremor as default
        'articulation_clarity': 0.7,  # Moderate clarity as default
        'prosody_variation': 0.5  # Moderate prosody as default
    }
    
    # Valid ranges for each biomarker
    VALID_RANGES: Dict[str, Tuple[float, float]] = {
        'jitter': (0.0, 1.0),
        'shimmer': (0.0, 1.0),
        'hnr': (0.0, 30.0),
        'speech_rate': (0.5, 10.0),
        'pause_ratio': (0.0, 1.0),
        'fluency_score': (0.0, 1.0),
        'voice_tremor': (0.0, 1.0),
        'articulation_clarity': (0.0, 1.0),
        'prosody_variation': (0.0, 1.0)
    }
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the BiomarkerExtractor.
        
        Args:
            sample_rate: Audio sample rate in Hz (default 16000)
        """
        self.sample_rate = sample_rate
        self._librosa = None
    
    @property
    def librosa(self):
        """Lazy load librosa to avoid import overhead"""
        if self._librosa is None:
            import librosa
            self._librosa = librosa
        return self._librosa
    
    def _clamp_to_range(self, value: float, metric_name: str) -> float:
        """
        Clamp a value to its valid range.
        
        Args:
            value: The value to clamp
            metric_name: Name of the metric for range lookup
            
        Returns:
            Clamped value within valid range
        """
        min_val, max_val = self.VALID_RANGES.get(metric_name, (0.0, 1.0))
        return max(min_val, min(max_val, value))
    
    def _use_fallback(self, metric_name: str) -> Tuple[float, bool]:
        """
        Get fallback value for a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Tuple of (default_value, is_estimated=True)
        """
        return self.DEFAULT_VALUES.get(metric_name, 0.5), True

    def calculate_jitter(
        self, 
        audio_data: np.ndarray,
        f0_contour: Optional[np.ndarray] = None
    ) -> Tuple[float, bool]:
        """
        Calculate jitter (fundamental frequency variation) using YIN algorithm.
        
        Jitter measures the cycle-to-cycle variation in fundamental frequency (F0).
        High jitter values can indicate voice disorders or neurological conditions.
        
        Args:
            audio_data: Audio samples as numpy array
            f0_contour: Pre-computed F0 contour (optional, will compute if not provided)
            
        Returns:
            Tuple of (jitter_value, is_estimated)
            - jitter_value: Normalized to 0-1 range
            - is_estimated: True if fallback value was used
            
        **Validates: Requirements 4.1**
        """
        try:
            if len(audio_data) == 0:
                logger.warning("Empty audio data for jitter calculation")
                return self._use_fallback('jitter')
            
            # Extract F0 using YIN algorithm if not provided
            if f0_contour is None:
                f0_contour = self.librosa.yin(
                    audio_data,
                    fmin=50,  # Minimum F0 (Hz)
                    fmax=400,  # Maximum F0 (Hz)
                    sr=self.sample_rate
                )
            
            # Filter out unvoiced frames (F0 = 0 or NaN)
            f0_voiced = f0_contour[f0_contour > 0]
            f0_voiced = f0_voiced[~np.isnan(f0_voiced)]
            
            if len(f0_voiced) < 2:
                logger.warning("Insufficient voiced frames for jitter calculation")
                return self._use_fallback('jitter')
            
            # Calculate period-to-period variation (jitter)
            # Convert F0 to periods (T = 1/F0)
            periods = 1.0 / (f0_voiced + 1e-8)
            
            # Calculate absolute jitter (mean absolute difference between consecutive periods)
            period_diffs = np.abs(np.diff(periods))
            mean_period = np.mean(periods)
            
            if mean_period <= 0:
                return self._use_fallback('jitter')
            
            # Relative jitter (normalized by mean period)
            jitter_raw = np.mean(period_diffs) / mean_period
            
            # Normalize to 0-1 range
            # Typical jitter values are 0.5-2% for healthy voices
            # Values above 5% indicate significant voice disorder
            jitter_normalized = self._clamp_to_range(
                min(1.0, jitter_raw / 0.1),  # 10% jitter maps to 1.0
                'jitter'
            )
            
            return jitter_normalized, False
            
        except Exception as e:
            logger.error(f"Jitter calculation failed: {e}")
            return self._use_fallback('jitter')

    def calculate_shimmer(
        self, 
        audio_data: np.ndarray,
        frame_size_ms: float = 25.0
    ) -> Tuple[float, bool]:
        """
        Calculate shimmer (amplitude variation) using frame-by-frame energy analysis.
        
        Shimmer measures the cycle-to-cycle variation in amplitude.
        High shimmer values can indicate voice disorders or neurological conditions.
        
        Args:
            audio_data: Audio samples as numpy array
            frame_size_ms: Frame size in milliseconds (default 25ms)
            
        Returns:
            Tuple of (shimmer_value, is_estimated)
            - shimmer_value: Normalized to 0-1 range
            - is_estimated: True if fallback value was used
            
        **Validates: Requirements 4.2**
        """
        try:
            if len(audio_data) == 0:
                logger.warning("Empty audio data for shimmer calculation")
                return self._use_fallback('shimmer')
            
            # Calculate frame parameters
            frame_size = int(frame_size_ms * self.sample_rate / 1000)
            hop_size = frame_size // 2  # 50% overlap
            
            if len(audio_data) < frame_size:
                logger.warning("Audio too short for shimmer calculation")
                return self._use_fallback('shimmer')
            
            # Calculate frame energies (RMS amplitude)
            frame_energies = []
            for i in range(0, len(audio_data) - frame_size, hop_size):
                frame = audio_data[i:i + frame_size]
                # RMS energy
                energy = np.sqrt(np.mean(frame ** 2))
                if energy > 0:  # Only include voiced frames
                    frame_energies.append(energy)
            
            if len(frame_energies) < 2:
                logger.warning("Insufficient frames for shimmer calculation")
                return self._use_fallback('shimmer')
            
            frame_energies = np.array(frame_energies)
            
            # Filter out very low energy frames (likely silence)
            energy_threshold = np.percentile(frame_energies, 20)
            voiced_energies = frame_energies[frame_energies > energy_threshold]
            
            if len(voiced_energies) < 2:
                return self._use_fallback('shimmer')
            
            # Calculate shimmer (mean absolute difference between consecutive amplitudes)
            amplitude_diffs = np.abs(np.diff(voiced_energies))
            mean_amplitude = np.mean(voiced_energies)
            
            if mean_amplitude <= 0:
                return self._use_fallback('shimmer')
            
            # Relative shimmer (normalized by mean amplitude)
            shimmer_raw = np.mean(amplitude_diffs) / mean_amplitude
            
            # Normalize to 0-1 range
            # Typical shimmer values are 3-5% for healthy voices
            # Values above 10% indicate significant voice disorder
            shimmer_normalized = self._clamp_to_range(
                min(1.0, shimmer_raw / 0.2),  # 20% shimmer maps to 1.0
                'shimmer'
            )
            
            return shimmer_normalized, False
            
        except Exception as e:
            logger.error(f"Shimmer calculation failed: {e}")
            return self._use_fallback('shimmer')

    def calculate_hnr(
        self, 
        audio_data: np.ndarray,
        n_fft: int = 2048
    ) -> Tuple[float, bool]:
        """
        Calculate Harmonics-to-Noise Ratio (HNR) using spectral analysis.
        
        HNR measures the ratio of harmonic (periodic) energy to noise energy.
        Higher HNR indicates clearer voice quality. Typical values are 0-30 dB.
        
        Args:
            audio_data: Audio samples as numpy array
            n_fft: FFT window size (default 2048)
            
        Returns:
            Tuple of (hnr_value, is_estimated)
            - hnr_value: In dB range (0-30 typical)
            - is_estimated: True if fallback value was used
            
        **Validates: Requirements 4.3**
        """
        try:
            if len(audio_data) == 0:
                logger.warning("Empty audio data for HNR calculation")
                return self._use_fallback('hnr')
            
            if len(audio_data) < n_fft:
                logger.warning("Audio too short for HNR calculation")
                return self._use_fallback('hnr')
            
            # Compute STFT
            stft = self.librosa.stft(audio_data, n_fft=n_fft)
            magnitude = np.abs(stft)
            
            # Get frequency bins
            freqs = self.librosa.fft_frequencies(sr=self.sample_rate, n_fft=n_fft)
            
            # Extract F0 to identify harmonic frequencies
            f0 = self.librosa.yin(
                audio_data,
                fmin=50,
                fmax=400,
                sr=self.sample_rate
            )
            
            # Get median F0 for voiced frames
            f0_voiced = f0[f0 > 0]
            if len(f0_voiced) == 0:
                logger.warning("No voiced frames detected for HNR calculation")
                return self._use_fallback('hnr')
            
            median_f0 = np.median(f0_voiced)
            
            # Identify harmonic bins (F0 and its harmonics)
            harmonic_energy = 0.0
            noise_energy = 0.0
            
            # Consider harmonics up to 4000 Hz
            max_harmonic_freq = 4000
            num_harmonics = int(max_harmonic_freq / median_f0)
            
            for frame_idx in range(magnitude.shape[1]):
                frame_mag = magnitude[:, frame_idx]
                
                # Calculate harmonic energy (energy at F0 and harmonics)
                for h in range(1, num_harmonics + 1):
                    harmonic_freq = h * median_f0
                    # Find closest frequency bin
                    bin_idx = np.argmin(np.abs(freqs - harmonic_freq))
                    
                    # Sum energy in a small window around the harmonic
                    window_size = 3
                    start_bin = max(0, bin_idx - window_size)
                    end_bin = min(len(frame_mag), bin_idx + window_size + 1)
                    harmonic_energy += np.sum(frame_mag[start_bin:end_bin] ** 2)
                
                # Total energy in speech band (50-4000 Hz)
                speech_band = (freqs >= 50) & (freqs <= 4000)
                total_energy = np.sum(frame_mag[speech_band] ** 2)
                
                # Noise energy is total minus harmonic
                noise_energy += max(0, total_energy - harmonic_energy)
            
            if noise_energy <= 0:
                # Very clean signal, return high HNR
                return self._clamp_to_range(30.0, 'hnr'), False
            
            # Calculate HNR in dB
            hnr_db = 10 * np.log10(harmonic_energy / (noise_energy + 1e-10))
            
            # Clamp to valid range
            hnr_clamped = self._clamp_to_range(hnr_db, 'hnr')
            
            return hnr_clamped, False
            
        except Exception as e:
            logger.error(f"HNR calculation failed: {e}")
            return self._use_fallback('hnr')

    def calculate_speech_rate(
        self, 
        audio_data: np.ndarray,
        speech_segments: Optional[List[Tuple[int, int]]] = None
    ) -> Tuple[float, bool]:
        """
        Calculate speech rate in syllables per second using energy peak detection.
        
        Speech rate is estimated by detecting energy peaks that correspond to
        syllable nuclei (vowels).
        
        Args:
            audio_data: Audio samples as numpy array
            speech_segments: Pre-computed speech segments (optional)
            
        Returns:
            Tuple of (speech_rate, is_estimated)
            - speech_rate: Syllables per second (0.5-10 range)
            - is_estimated: True if fallback value was used
            
        **Validates: Requirements 4.4**
        """
        try:
            if len(audio_data) == 0:
                logger.warning("Empty audio data for speech rate calculation")
                return self._use_fallback('speech_rate')
            
            # Calculate total duration
            total_duration = len(audio_data) / self.sample_rate
            
            if total_duration < 0.5:
                logger.warning("Audio too short for speech rate calculation")
                return self._use_fallback('speech_rate')
            
            # Calculate frame energies for syllable detection
            frame_size = int(0.025 * self.sample_rate)  # 25ms frames
            hop_size = int(0.010 * self.sample_rate)    # 10ms hop
            
            # Calculate RMS energy per frame
            energies = []
            for i in range(0, len(audio_data) - frame_size, hop_size):
                frame = audio_data[i:i + frame_size]
                energy = np.sqrt(np.mean(frame ** 2))
                energies.append(energy)
            
            if len(energies) < 3:
                return self._use_fallback('speech_rate')
            
            energies = np.array(energies)
            
            # Smooth the energy contour
            from scipy.ndimage import gaussian_filter1d
            smoothed_energies = gaussian_filter1d(energies, sigma=2)
            
            # Find peaks (syllable nuclei)
            # Peaks should be above a threshold and have minimum distance
            energy_threshold = np.percentile(smoothed_energies, 40)
            min_distance = int(0.1 * self.sample_rate / hop_size)  # ~100ms between syllables
            
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(
                smoothed_energies,
                height=energy_threshold,
                distance=max(1, min_distance)
            )
            
            num_syllables = len(peaks)
            
            if num_syllables == 0:
                return self._use_fallback('speech_rate')
            
            # Calculate speech rate
            # If speech segments provided, use speech duration only
            if speech_segments and len(speech_segments) > 0:
                speech_duration = sum(
                    (end - start) / self.sample_rate 
                    for start, end in speech_segments
                )
                if speech_duration > 0:
                    speech_rate = num_syllables / speech_duration
                else:
                    speech_rate = num_syllables / total_duration
            else:
                speech_rate = num_syllables / total_duration
            
            # Clamp to valid range
            speech_rate_clamped = self._clamp_to_range(speech_rate, 'speech_rate')
            
            return speech_rate_clamped, False
            
        except Exception as e:
            logger.error(f"Speech rate calculation failed: {e}")
            return self._use_fallback('speech_rate')
    
    def calculate_pause_ratio(
        self, 
        audio_data: np.ndarray,
        speech_segments: Optional[List[Tuple[int, int]]] = None
    ) -> Tuple[float, bool]:
        """
        Calculate pause ratio as the proportion of silence to total audio duration.
        
        Uses Voice Activity Detection (VAD) to identify speech vs silence segments.
        
        Args:
            audio_data: Audio samples as numpy array
            speech_segments: Pre-computed speech segments from VAD (optional)
            
        Returns:
            Tuple of (pause_ratio, is_estimated)
            - pause_ratio: 0-1 (proportion of silence)
            - is_estimated: True if fallback value was used
            
        **Validates: Requirements 4.5**
        """
        try:
            if len(audio_data) == 0:
                logger.warning("Empty audio data for pause ratio calculation")
                return self._use_fallback('pause_ratio')
            
            total_duration = len(audio_data) / self.sample_rate
            
            if total_duration < 0.5:
                logger.warning("Audio too short for pause ratio calculation")
                return self._use_fallback('pause_ratio')
            
            # If speech segments provided, use them directly
            if speech_segments and len(speech_segments) > 0:
                speech_duration = sum(
                    (end - start) / self.sample_rate 
                    for start, end in speech_segments
                )
                pause_duration = total_duration - speech_duration
                pause_ratio = max(0.0, pause_duration / total_duration)
                return self._clamp_to_range(pause_ratio, 'pause_ratio'), False
            
            # Otherwise, perform energy-based VAD
            frame_size = int(0.025 * self.sample_rate)  # 25ms frames
            hop_size = int(0.010 * self.sample_rate)    # 10ms hop
            
            # Calculate frame energies
            energies = []
            for i in range(0, len(audio_data) - frame_size, hop_size):
                frame = audio_data[i:i + frame_size]
                energy = np.sum(frame ** 2)
                energies.append(energy)
            
            if len(energies) == 0:
                return self._use_fallback('pause_ratio')
            
            energies = np.array(energies)
            
            # Adaptive threshold for voice activity
            energy_threshold = np.percentile(energies, 30)
            
            # Count voiced vs unvoiced frames
            voiced_frames = np.sum(energies > energy_threshold)
            total_frames = len(energies)
            
            # Calculate pause ratio
            speech_ratio = voiced_frames / total_frames
            pause_ratio = 1.0 - speech_ratio
            
            # Clamp to valid range
            pause_ratio_clamped = self._clamp_to_range(pause_ratio, 'pause_ratio')
            
            return pause_ratio_clamped, False
            
        except Exception as e:
            logger.error(f"Pause ratio calculation failed: {e}")
            return self._use_fallback('pause_ratio')

    def calculate_fluency_score(
        self, 
        audio_data: np.ndarray,
        speech_segments: Optional[List[Tuple[int, int]]] = None
    ) -> Tuple[float, bool]:
        """
        Calculate fluency score based on speech continuity and rhythm.
        
        Args:
            audio_data: Audio samples as numpy array
            speech_segments: Pre-computed speech segments (optional)
            
        Returns:
            Tuple of (fluency_score, is_estimated)
        """
        try:
            if len(audio_data) == 0:
                return self._use_fallback('fluency_score')
            
            # Get speech segments if not provided
            if speech_segments is None or len(speech_segments) == 0:
                speech_segments = self._detect_speech_segments(audio_data)
            
            if len(speech_segments) == 0:
                return self._use_fallback('fluency_score')
            
            # Calculate fluency components
            total_duration = len(audio_data) / self.sample_rate
            total_speech_time = sum(
                (end - start) / self.sample_rate 
                for start, end in speech_segments
            )
            
            # Speech continuity (fewer, longer segments = better fluency)
            avg_segment_length = total_speech_time / len(speech_segments)
            continuity_score = min(1.0, avg_segment_length / 2.0)  # 2s = perfect
            
            # Segment count penalty (too many segments = poor fluency)
            segment_penalty = max(0.0, 1.0 - len(speech_segments) / 20.0)
            
            # Speech ratio (more speech = better fluency)
            speech_ratio = total_speech_time / total_duration if total_duration > 0 else 0.5
            
            # Combine scores
            fluency = (continuity_score * 0.4 + segment_penalty * 0.3 + speech_ratio * 0.3)
            
            return self._clamp_to_range(fluency, 'fluency_score'), False
            
        except Exception as e:
            logger.error(f"Fluency score calculation failed: {e}")
            return self._use_fallback('fluency_score')
    
    def calculate_voice_tremor(
        self, 
        audio_data: np.ndarray
    ) -> Tuple[float, bool]:
        """
        Calculate voice tremor intensity from spectral analysis.
        
        Tremor is detected by looking for regular oscillations in the 4-12 Hz range.
        
        Args:
            audio_data: Audio samples as numpy array
            
        Returns:
            Tuple of (tremor_value, is_estimated)
        """
        try:
            if len(audio_data) == 0:
                return self._use_fallback('voice_tremor')
            
            # Extract F0 contour
            f0 = self.librosa.yin(
                audio_data,
                fmin=50,
                fmax=400,
                sr=self.sample_rate
            )
            
            f0_voiced = f0[f0 > 0]
            if len(f0_voiced) < 10:
                return self._use_fallback('voice_tremor')
            
            # Calculate F0 variation (tremor indicator)
            f0_std = np.std(f0_voiced)
            f0_mean = np.mean(f0_voiced)
            
            # Coefficient of variation
            f0_cv = f0_std / (f0_mean + 1e-8)
            
            # Also check spectral tremor in 4-12 Hz range
            stft = self.librosa.stft(audio_data, n_fft=2048)
            magnitude = np.abs(stft)
            freqs = self.librosa.fft_frequencies(sr=self.sample_rate, n_fft=2048)
            
            # Tremor frequency range
            tremor_mask = (freqs >= 4) & (freqs <= 12)
            
            if np.any(tremor_mask):
                tremor_energy = np.mean(magnitude[tremor_mask, :])
                total_energy = np.mean(magnitude)
                spectral_tremor = tremor_energy / (total_energy + 1e-8)
            else:
                spectral_tremor = 0.0
            
            # Combine tremor indicators
            tremor_score = (f0_cv * 0.6 + spectral_tremor * 5.0 * 0.4)
            tremor_normalized = min(1.0, tremor_score)
            
            return self._clamp_to_range(tremor_normalized, 'voice_tremor'), False
            
        except Exception as e:
            logger.error(f"Voice tremor calculation failed: {e}")
            return self._use_fallback('voice_tremor')
    
    def calculate_articulation_clarity(
        self, 
        audio_data: np.ndarray
    ) -> Tuple[float, bool]:
        """
        Calculate articulation clarity from spectral features.
        
        Clear articulation is indicated by well-defined spectral content
        and good high-frequency energy (consonants).
        
        Args:
            audio_data: Audio samples as numpy array
            
        Returns:
            Tuple of (clarity_value, is_estimated)
        """
        try:
            if len(audio_data) == 0:
                return self._use_fallback('articulation_clarity')
            
            # Spectral centroid (higher = clearer consonants)
            spectral_centroid = self.librosa.feature.spectral_centroid(
                y=audio_data, sr=self.sample_rate
            )[0]
            
            # Spectral rolloff (indicates high-frequency content)
            spectral_rolloff = self.librosa.feature.spectral_rolloff(
                y=audio_data, sr=self.sample_rate
            )[0]
            
            # MFCC delta (articulation dynamics)
            mfcc = self.librosa.feature.mfcc(
                y=audio_data, sr=self.sample_rate, n_mfcc=13
            )
            mfcc_delta = self.librosa.feature.delta(mfcc)
            
            # Calculate clarity components
            centroid_mean = np.mean(spectral_centroid)
            rolloff_mean = np.mean(spectral_rolloff)
            delta_energy = np.mean(np.abs(mfcc_delta))
            
            # Normalize and combine
            clarity_from_centroid = min(1.0, centroid_mean / 2000.0)
            clarity_from_rolloff = min(1.0, rolloff_mean / 4000.0)
            clarity_from_dynamics = min(1.0, delta_energy * 2.0)
            
            clarity = (
                clarity_from_centroid * 0.4 + 
                clarity_from_rolloff * 0.3 + 
                clarity_from_dynamics * 0.3
            )
            
            return self._clamp_to_range(clarity, 'articulation_clarity'), False
            
        except Exception as e:
            logger.error(f"Articulation clarity calculation failed: {e}")
            return self._use_fallback('articulation_clarity')
    
    def calculate_prosody_variation(
        self, 
        audio_data: np.ndarray
    ) -> Tuple[float, bool]:
        """
        Calculate prosodic variation (pitch and energy dynamics).
        
        Healthy prosody shows moderate variation in pitch and energy.
        
        Args:
            audio_data: Audio samples as numpy array
            
        Returns:
            Tuple of (prosody_value, is_estimated)
        """
        try:
            if len(audio_data) == 0:
                return self._use_fallback('prosody_variation')
            
            # Extract F0
            f0 = self.librosa.yin(
                audio_data,
                fmin=50,
                fmax=400,
                sr=self.sample_rate
            )
            
            f0_voiced = f0[f0 > 0]
            if len(f0_voiced) < 5:
                return self._use_fallback('prosody_variation')
            
            # F0 variation
            f0_range = np.max(f0_voiced) - np.min(f0_voiced)
            f0_std = np.std(f0_voiced)
            
            # Normalize F0 variation (moderate variation is good)
            f0_variation = min(1.0, (f0_range / 100.0 + f0_std / 30.0) / 2.0)
            
            # Energy variation
            frame_size = int(0.025 * self.sample_rate)
            hop_size = int(0.010 * self.sample_rate)
            
            energies = []
            for i in range(0, len(audio_data) - frame_size, hop_size):
                frame = audio_data[i:i + frame_size]
                energy = np.sqrt(np.mean(frame ** 2))
                energies.append(energy)
            
            if len(energies) > 1:
                energy_std = np.std(energies)
                energy_mean = np.mean(energies)
                energy_cv = energy_std / (energy_mean + 1e-8)
                energy_variation = min(1.0, energy_cv * 2.0)
            else:
                energy_variation = 0.5
            
            # Combine prosody measures
            prosody = (f0_variation * 0.6 + energy_variation * 0.4)
            
            return self._clamp_to_range(prosody, 'prosody_variation'), False
            
        except Exception as e:
            logger.error(f"Prosody variation calculation failed: {e}")
            return self._use_fallback('prosody_variation')
    
    def _detect_speech_segments(
        self, 
        audio_data: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        Detect speech segments using energy-based VAD.
        
        Args:
            audio_data: Audio samples as numpy array
            
        Returns:
            List of (start_sample, end_sample) tuples
        """
        frame_size = int(0.025 * self.sample_rate)  # 25ms
        hop_size = int(0.010 * self.sample_rate)    # 10ms
        
        # Calculate frame energies
        energies = []
        for i in range(0, len(audio_data) - frame_size, hop_size):
            frame = audio_data[i:i + frame_size]
            energy = np.sum(frame ** 2)
            energies.append(energy)
        
        if len(energies) == 0:
            return []
        
        energies = np.array(energies)
        
        # Adaptive threshold
        threshold = np.percentile(energies, 30)
        
        # Find speech segments
        speech_segments = []
        in_speech = False
        segment_start = 0
        
        for i, energy in enumerate(energies):
            frame_start = i * hop_size
            
            if energy > threshold and not in_speech:
                segment_start = frame_start
                in_speech = True
            elif energy <= threshold and in_speech:
                speech_segments.append((segment_start, frame_start))
                in_speech = False
        
        if in_speech:
            speech_segments.append((segment_start, len(audio_data)))
        
        return speech_segments
    
    async def extract_all(
        self, 
        audio_data: np.ndarray,
        speech_segments: Optional[List[Tuple[int, int]]] = None
    ) -> ExtractedBiomarkers:
        """
        Extract all biomarkers from audio data with fallback support.
        
        Args:
            audio_data: Audio samples as numpy array (should be 16kHz mono)
            speech_segments: Pre-computed speech segments from VAD (optional)
            
        Returns:
            ExtractedBiomarkers with all metrics and estimation flags
            
        **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6**
        """
        estimated_flags = {}
        
        # Extract F0 once for reuse
        f0_contour = None
        try:
            if len(audio_data) > 0:
                f0_contour = self.librosa.yin(
                    audio_data,
                    fmin=50,
                    fmax=400,
                    sr=self.sample_rate
                )
        except Exception as e:
            logger.warning(f"F0 extraction failed: {e}")
        
        # Detect speech segments if not provided
        if speech_segments is None:
            speech_segments = self._detect_speech_segments(audio_data)
        
        # Calculate each biomarker
        jitter, jitter_est = self.calculate_jitter(audio_data, f0_contour)
        estimated_flags['jitter'] = jitter_est
        
        shimmer, shimmer_est = self.calculate_shimmer(audio_data)
        estimated_flags['shimmer'] = shimmer_est
        
        hnr, hnr_est = self.calculate_hnr(audio_data)
        estimated_flags['hnr'] = hnr_est
        
        speech_rate, sr_est = self.calculate_speech_rate(audio_data, speech_segments)
        estimated_flags['speech_rate'] = sr_est
        
        pause_ratio, pr_est = self.calculate_pause_ratio(audio_data, speech_segments)
        estimated_flags['pause_ratio'] = pr_est
        
        fluency_score, fs_est = self.calculate_fluency_score(audio_data, speech_segments)
        estimated_flags['fluency_score'] = fs_est
        
        voice_tremor, vt_est = self.calculate_voice_tremor(audio_data)
        estimated_flags['voice_tremor'] = vt_est
        
        articulation_clarity, ac_est = self.calculate_articulation_clarity(audio_data)
        estimated_flags['articulation_clarity'] = ac_est
        
        prosody_variation, pv_est = self.calculate_prosody_variation(audio_data)
        estimated_flags['prosody_variation'] = pv_est
        
        return ExtractedBiomarkers(
            jitter=jitter,
            shimmer=shimmer,
            hnr=hnr,
            speech_rate=speech_rate,
            pause_ratio=pause_ratio,
            fluency_score=fluency_score,
            voice_tremor=voice_tremor,
            articulation_clarity=articulation_clarity,
            prosody_variation=prosody_variation,
            estimated_flags=estimated_flags
        )


# Singleton instance for use across the application
biomarker_extractor = BiomarkerExtractor()

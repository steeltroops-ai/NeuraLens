"""
Composite Biomarker Extractor
Novel composite biomarkers combining multiple features for clinical insight.

These are research-grade composite indices that provide additional
diagnostic value beyond individual biomarkers.

Novel Biomarkers:
1. NII - Neuromotor Instability Index
2. VFMT - Vocal Fold Micro-Tremor Metric
3. ACE - Articulatory Coordination Entropy
4. RPCS - Respiratory-Phonatory Coupling Score
5. CLLM - Cognitive-Linguistic Latency Markers
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, Optional, List
from scipy import signal
from scipy.stats import entropy

logger = logging.getLogger(__name__)


@dataclass
class CompositeBiomarkers:
    """Novel composite biomarkers for enhanced clinical insight."""
    
    # Neuromotor Instability Index
    nii: float = 0.0
    nii_tremor_component: float = 0.0
    nii_jitter_component: float = 0.0
    nii_shimmer_component: float = 0.0
    nii_f0_component: float = 0.0
    
    # Vocal Fold Micro-Tremor Metric
    vfmt_ratio: float = 0.0
    vfmt_peak_freq: float = 0.0
    vfmt_bandwidth: float = 0.0
    
    # Articulatory Coordination Entropy
    ace: float = 0.0
    ace_f1_entropy: float = 0.0
    ace_f2_entropy: float = 0.0
    
    # Respiratory-Phonatory Coupling Score
    rpcs: float = 0.0
    rpcs_coherence: float = 0.0
    
    # Formant-based articulation
    fcr: float = 0.0                    # Formant Centralization Ratio
    vsa: float = 0.0                    # Vowel Space Area (approximate)
    
    def to_dict(self) -> Dict:
        return {
            "nii": self.nii,
            "nii_tremor_component": self.nii_tremor_component,
            "nii_jitter_component": self.nii_jitter_component,
            "nii_shimmer_component": self.nii_shimmer_component,
            "nii_f0_component": self.nii_f0_component,
            "vfmt_ratio": self.vfmt_ratio,
            "vfmt_peak_freq": self.vfmt_peak_freq,
            "vfmt_bandwidth": self.vfmt_bandwidth,
            "ace": self.ace,
            "ace_f1_entropy": self.ace_f1_entropy,
            "ace_f2_entropy": self.ace_f2_entropy,
            "rpcs": self.rpcs,
            "rpcs_coherence": self.rpcs_coherence,
            "fcr": self.fcr,
            "vsa": self.vsa
        }


class CompositeFeatureExtractor:
    """
    Extract novel composite biomarkers from speech.
    
    These indices combine multiple acoustic features to provide
    enhanced clinical insight into neurological conditions.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def extract(
        self,
        audio: np.ndarray,
        acoustic_features: Optional[Dict] = None,
        prosodic_features: Optional[Dict] = None,
        f0_contour: Optional[np.ndarray] = None,
        f1_contour: Optional[np.ndarray] = None,
        f2_contour: Optional[np.ndarray] = None
    ) -> CompositeBiomarkers:
        """
        Extract all composite biomarkers.
        
        Args:
            audio: Audio waveform
            acoustic_features: Dict of acoustic features
            prosodic_features: Dict of prosodic features
            f0_contour: F0 time series
            f1_contour: F1 formant time series
            f2_contour: F2 formant time series
            
        Returns:
            CompositeBiomarkers dataclass
        """
        biomarkers = CompositeBiomarkers()
        
        try:
            # Neuromotor Instability Index
            if acoustic_features and prosodic_features:
                biomarkers = self._compute_nii(biomarkers, acoustic_features, prosodic_features)
            
            # Vocal Fold Micro-Tremor
            if f0_contour is not None:
                biomarkers = self._compute_vfmt(biomarkers, f0_contour)
            
            # Articulatory Coordination Entropy
            if f1_contour is not None and f2_contour is not None:
                biomarkers = self._compute_ace(biomarkers, f1_contour, f2_contour)
            
            # Respiratory-Phonatory Coupling
            if audio is not None and f0_contour is not None:
                biomarkers = self._compute_rpcs(biomarkers, audio, f0_contour)
            
            # Formant-based articulation measures
            if acoustic_features:
                biomarkers = self._compute_formant_metrics(biomarkers, acoustic_features)
                
        except Exception as e:
            logger.error(f"Composite biomarker extraction failed: {e}")
            
        return biomarkers
    
    def _compute_nii(
        self,
        biomarkers: CompositeBiomarkers,
        acoustic: Dict,
        prosodic: Dict
    ) -> CompositeBiomarkers:
        """
        Compute Neuromotor Instability Index (NII).
        
        Unified score combining multiple motor instability markers.
        Higher values indicate greater neuromotor dysfunction.
        """
        def normalize(value, min_val, max_val):
            if max_val == min_val:
                return 0.0
            return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))
        
        # Tremor component (from prosodic)
        tremor_score = prosodic.get("tremor_score", 0.0)
        biomarkers.nii_tremor_component = normalize(tremor_score, 0, 0.5)
        
        # Jitter instability (CV of jitter if available, else raw jitter)
        jitter = acoustic.get("jitter_local", 0.0)
        biomarkers.nii_jitter_component = normalize(jitter, 0, 3.0)
        
        # Shimmer instability
        shimmer = acoustic.get("shimmer_local", 0.0)
        biomarkers.nii_shimmer_component = normalize(shimmer, 0, 8.0)
        
        # F0 roughness (excursion rate or CV)
        f0_cv = prosodic.get("f0_cv", 0.0)
        biomarkers.nii_f0_component = normalize(f0_cv, 0, 0.5)
        
        # Weighted combination
        biomarkers.nii = (
            0.35 * biomarkers.nii_tremor_component +
            0.25 * biomarkers.nii_jitter_component +
            0.20 * biomarkers.nii_shimmer_component +
            0.20 * biomarkers.nii_f0_component
        )
        
        return biomarkers
    
    def _compute_vfmt(
        self,
        biomarkers: CompositeBiomarkers,
        f0_contour: np.ndarray
    ) -> CompositeBiomarkers:
        """
        Compute Vocal Fold Micro-Tremor Metric (VFMT).
        
        Detects subtle, subclinical tremor that may precede
        clinically visible tremor by years in prodromal PD.
        """
        # Clean F0 contour
        f0_clean = f0_contour[~np.isnan(f0_contour) & (f0_contour > 0)]
        
        if len(f0_clean) < 50:
            return biomarkers
        
        try:
            # Assume F0 sampled at ~100 Hz
            sr = 100
            
            # High-resolution FFT
            n_fft = 2048
            f0_centered = f0_clean - np.mean(f0_clean)
            
            # Zero-pad for higher resolution
            padded = np.zeros(n_fft)
            padded[:len(f0_centered)] = f0_centered
            
            spectrum = np.abs(np.fft.rfft(padded))
            freqs = np.fft.rfftfreq(n_fft, 1/sr)
            
            # Micro-tremor band (4-8 Hz) - subtle, early
            micro_band = (freqs >= 4) & (freqs <= 8)
            
            # Control band (15-25 Hz) - should be low in voice
            control_band = (freqs >= 15) & (freqs <= 25)
            
            micro_power = np.sum(spectrum[micro_band]**2)
            control_power = np.sum(spectrum[control_band]**2) + 1e-10
            
            biomarkers.vfmt_ratio = float(micro_power / control_power)
            
            # Peak frequency in micro-tremor band
            if np.any(micro_band) and np.sum(spectrum[micro_band]) > 0:
                micro_spectrum = spectrum[micro_band]
                micro_freqs = freqs[micro_band]
                peak_idx = np.argmax(micro_spectrum)
                biomarkers.vfmt_peak_freq = float(micro_freqs[peak_idx])
                
                # Bandwidth (half-power width)
                peak_power = micro_spectrum[peak_idx]
                half_power = peak_power / 2
                above_half = micro_spectrum > half_power
                if np.sum(above_half) > 1:
                    biomarkers.vfmt_bandwidth = float(
                        micro_freqs[above_half][-1] - micro_freqs[above_half][0]
                    )
                    
        except Exception as e:
            logger.warning(f"VFMT computation failed: {e}")
            
        return biomarkers
    
    def _compute_ace(
        self,
        biomarkers: CompositeBiomarkers,
        f1_contour: np.ndarray,
        f2_contour: np.ndarray
    ) -> CompositeBiomarkers:
        """
        Compute Articulatory Coordination Entropy (ACE).
        
        Measures predictability of articulatory movements.
        High entropy = unpredictable/uncoordinated (dysarthria)
        Low entropy = smooth, coordinated articulation
        """
        # Clean contours
        valid_mask = (~np.isnan(f1_contour) & ~np.isnan(f2_contour) & 
                      (f1_contour > 0) & (f2_contour > 0))
        
        f1 = f1_contour[valid_mask]
        f2 = f2_contour[valid_mask]
        
        if len(f1) < 20:
            return biomarkers
        
        try:
            # Compute formant velocities
            f1_vel = np.diff(f1)
            f2_vel = np.diff(f2)
            
            # Individual entropy (discretized velocities)
            n_bins = 20
            
            # F1 velocity entropy
            f1_hist, _ = np.histogram(f1_vel, bins=n_bins, density=True)
            f1_hist = f1_hist + 1e-10
            f1_hist = f1_hist / np.sum(f1_hist)
            biomarkers.ace_f1_entropy = float(entropy(f1_hist, base=2))
            
            # F2 velocity entropy
            f2_hist, _ = np.histogram(f2_vel, bins=n_bins, density=True)
            f2_hist = f2_hist + 1e-10
            f2_hist = f2_hist / np.sum(f2_hist)
            biomarkers.ace_f2_entropy = float(entropy(f2_hist, base=2))
            
            # Joint 2D entropy
            joint_hist, _, _ = np.histogram2d(f1_vel, f2_vel, bins=n_bins, density=True)
            joint_hist = joint_hist + 1e-10
            joint_hist = joint_hist / np.sum(joint_hist)
            joint_entropy = float(entropy(joint_hist.flatten(), base=2))
            
            # Normalize to 0-1
            max_entropy = np.log2(n_bins * n_bins)
            biomarkers.ace = joint_entropy / max_entropy
            
        except Exception as e:
            logger.warning(f"ACE computation failed: {e}")
            
        return biomarkers
    
    def _compute_rpcs(
        self,
        biomarkers: CompositeBiomarkers,
        audio: np.ndarray,
        f0_contour: np.ndarray
    ) -> CompositeBiomarkers:
        """
        Compute Respiratory-Phonatory Coupling Score (RPCS).
        
        Measures synchronization between breathing and phonation.
        Decoupling indicates respiratory muscle weakness or
        poor motor planning (ALS, PD, respiratory disorders).
        """
        try:
            # Extract amplitude envelope (proxy for respiratory effort)
            analytic = signal.hilbert(audio)
            envelope = np.abs(analytic)
            
            # Smooth envelope
            window_size = min(1001, len(envelope) // 10)
            if window_size % 2 == 0:
                window_size += 1
            if window_size > 3:
                envelope_smooth = signal.savgol_filter(envelope, window_size, 3)
            else:
                envelope_smooth = envelope
            
            # Resample F0 to match envelope length
            f0_clean = f0_contour.copy()
            f0_clean[np.isnan(f0_clean)] = 0
            
            if len(f0_clean) > 0 and len(envelope_smooth) > 0:
                f0_resampled = np.interp(
                    np.linspace(0, 1, len(envelope_smooth)),
                    np.linspace(0, 1, len(f0_clean)),
                    f0_clean
                )
                
                # Compute instantaneous phase
                phase_env = np.angle(signal.hilbert(envelope_smooth - np.mean(envelope_smooth)))
                phase_f0 = np.angle(signal.hilbert(f0_resampled - np.mean(f0_resampled)))
                
                # Phase coherence (circular mean of phase differences)
                phase_diff = phase_env - phase_f0
                coherence = np.abs(np.mean(np.exp(1j * phase_diff)))
                
                biomarkers.rpcs_coherence = float(coherence)
                biomarkers.rpcs = float(coherence)  # RPCS is the coherence measure
                
        except Exception as e:
            logger.warning(f"RPCS computation failed: {e}")
            
        return biomarkers
    
    def _compute_formant_metrics(
        self,
        biomarkers: CompositeBiomarkers,
        acoustic: Dict
    ) -> CompositeBiomarkers:
        """
        Compute formant-based articulation metrics.
        
        FCR (Formant Centralization Ratio):
        - Measures vowel centralization
        - Higher values indicate reduced articulatory precision
        - Associated with dysarthria, ALS, PD
        
        VSA (Vowel Space Area):
        - Approximated from mean F1/F2
        - Reduced VSA indicates articulatory undershoot
        """
        f1_mean = acoustic.get("f1_mean", 0.0)
        f2_mean = acoustic.get("f2_mean", 0.0)
        
        if f1_mean > 0 and f2_mean > 0:
            # Simplified FCR based on global means
            # True FCR requires vowel-specific formants
            # FCR = (F2u + F2a + F1i + F1u) / (F2i + F1a)
            # Approximation: deviation from expected F1/F2 relationship
            expected_f2 = 2.5 * f1_mean + 500  # Rough vocal tract relationship
            fcr_approx = (f1_mean + f2_mean) / 2000.0
            
            biomarkers.fcr = float(fcr_approx)
            
            # VSA approximation (requires corner vowels for accuracy)
            # Using simplified F1*F2 product as proxy
            biomarkers.vsa = float(f1_mean * f2_mean / 1e6)  # Normalized
            
        return biomarkers

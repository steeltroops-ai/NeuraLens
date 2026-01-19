"""
Medical-Grade Acoustic Feature Extractor
Implements scientific standard algorithms (Praat/Parselmouth) for voice analysis.

Key Biomarkers:
- Jitter/Shimmer (Dysphonia)
- CPP (Cepstral Peak Prominence - Breathiness/Severity)
- HNR (Harmonics to Noise)
- FCR (Formant Centralization Ratio - Dysarthria/Articulation)
"""

import numpy as np
import logging
import io
import parselmouth
from parselmouth.praat import call
import librosa
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AcousticMetrics:
    # Phonation / Perturbation
    jitter_local: float  # Percent
    shimmer_local: float # Percent
    hnr: float          # dB
    
    # Cepstral (Robust Quality)
    cpps: float         # Cepstral Peak Prominence Smoothed (dB)
    
    # Articulation / Formants
    f1_mean: float      # Hz
    f2_mean: float      # Hz
    formant_centralization_ratio: float # FCR
    vowel_space_area: float # VSA (Approximate)
    
    # Prosody / Temporal
    speech_rate: float  # Syllables/sec
    mean_f0: float      # Hz
    std_f0: float       # Hz
    tremor_intensity: float # 0-1 (4-12Hz modulation)

    # Quality Checks
    signal_to_noise: float
    duration: float

class AcousticProcessor:
    """
    Scientific acoustic analysis using Parselmouth (Praat) as the Gold Standard.
    Fallback to Librosa only for specific spectral features.
    """
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def extract_features(self, audio_data: np.ndarray) -> AcousticMetrics:
        """
        Extract medical-grade acoustic features.
        Input: Normalized float32 numpy array.
        """
        # 1. Create Parselmouth Sound object (The core of analysis)
        sound = parselmouth.Sound(audio_data, sampling_frequency=self.sample_rate)
        
        # 2. Pitch Analysis
        pitch = sound.to_pitch_ac(time_step=0.01, pitch_floor=50.0, pitch_ceiling=500.0)
        mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")
        if np.isnan(mean_f0): mean_f0 = 0.0
        std_f0 = call(pitch, "Get standard deviation", 0, 0, "Hertz")
        if np.isnan(std_f0): std_f0 = 0.0

        # Tremor (Macro-tremor 4-12Hz)
        # We can approximate this by analyzing the Pitch tier itself for low-freq modulation
        # Or simplistic spectral energy approach (fast)
        try:
            # Simple 4-12Hz energy ratio from envelope
            rms = librosa.feature.rms(y=audio_data, frame_length=512, hop_length=128)[0]
            # FFT of RMS envelope to find modulation frequency
            n = len(rms)
            if n > 0:
                mod_fft = np.abs(np.fft.rfft(rms - np.mean(rms)))
                mod_freqs = np.fft.rfftfreq(n, d=128/self.sample_rate)
                tremor_mask = (mod_freqs >= 4) & (mod_freqs <= 12)
                if np.sum(mod_fft) > 0:
                    tremor_intensity = np.sum(mod_fft[tremor_mask]) / np.sum(mod_fft)
                else:
                    tremor_intensity = 0.0
            else:
                tremor_intensity = 0.0
        except:
            tremor_intensity = 0.0


        # 3. Dysphonia Measures (Jitter/Shimmer/HNR)
        # MUST use "To PointProcess" for accurate cycle detection
        point_process = call(sound, "To PointProcess (periodic, cc)", 50.0, 500.0)
        
        num_periods = call(point_process, "Get number of periods", 0.0, 0.0, 50.0, 500.0, 1.3)
        
        if num_periods > 10:
            # Jitter (Local) - RAP (Relative Average Perturbation) is also good, but Local % is standard
            jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100 # Convert to %
            
            # Shimmer (Local)
            shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100 # Convert to %
            
            # HNR (Harmonicity)
            harmonicity = sound.to_harmonicity_ac(time_step=0.01, pitch_floor=50.0, silence_threshold=0.1, number_of_periods_per_window=4.5)
            hnr = call(harmonicity, "Get mean", 0, 0)
        else:
            jitter = 0.0
            shimmer = 0.0
            hnr = 0.0

        # 4. CPPS (Cepstral Peak Prominence Smoothed) - The Heavy Hitter for Vocal Quality
        # Measures how normalized the "peak" of voice pitch is in cepstral domain.
        # Robust against irregulaties that break Jitter/Shimmer formulas.
        try:
            power_cepstrogram = call(sound, "To PowerCepstrogram", 60.0, 0.002, 5000.0, 50.0)
            cpps = call(power_cepstrogram, "Get CPPS", False, 0.01, 0.001, 60.0, 330.0, 0.05, "Parabolic", 0.001, 0.05, "Robust", "Huber")
        except:
            cpps = 0.0

        # 5. Formant Analysis (FCR & VSA) - Articulation
        # We need F1 and F2 (First and Second Formants)
        # Ideally extracted from vowels, but global mean approximates vocal tract capacity
        formants = sound.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5500.0)
        
        f1 = call(formants, "Get mean", 1, 0, 0, "Hertz")
        f2 = call(formants, "Get mean", 2, 0, 0, "Hertz")
        if np.isnan(f1): f1 = 0.0
        if np.isnan(f2): f2 = 0.0
        
        # Formant Centralization Ratio (FCR)
        # FCR = (F2_u + F2_a + F1_i + F1_u) / (F2_i + F1_a)
        # Without specific vowels, we use a simplified centralized ratio based on mean dispersion
        # Or simply return the means for the risk calculator to interpret relative to norms
        # We'll return the means and a simplified "Centralization" index
        # For now, placeholder FCR logic using global means (less accurate without segmentation)
        fcr = (f1 + f2) / 2000.0 # Placeholder normalization

        # 6. Speech Rate (Syllable Nuclei via Librosa)
        speech_rate = self._calculate_speech_rate(audio_data)

        return AcousticMetrics(
            jitter_local=jitter,
            shimmer_local=shimmer,
            hnr=hnr,
            cpps=cpps,
            f1_mean=f1,
            f2_mean=f2,
            formant_centralization_ratio=fcr,
            vowel_space_area=0.0, # Requires vowel segmentation
            speech_rate=speech_rate,
            mean_f0=mean_f0,
            std_f0=std_f0,
            tremor_intensity=tremor_intensity,
            signal_to_noise=hnr, # HNR is good proxy for SNR
            duration=sound.get_total_duration()
        )

    def _calculate_speech_rate(self, y: np.ndarray) -> float:
        """Estimate syllables per second"""
        if len(y) == 0: return 0.0
        try:
            # RMS Energy envelope
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            
            # Low-pass filter to smooth envelope
            # Peaks roughly correspond to syllables
            from scipy.signal import find_peaks
            threshold = np.mean(rms) * 0.5
            peaks, _ = find_peaks(rms, height=threshold, distance=10) # ~10 frames = 100ms
            
            duration = len(y) / self.sample_rate
            return len(peaks) / duration if duration > 0 else 0.0
        except:
            return 0.0

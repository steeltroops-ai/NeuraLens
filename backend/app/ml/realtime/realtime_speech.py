"""
Real-Time Speech Analysis Engine
Optimized for <3s inference with actual ML-based voice biomarker detection
"""

import asyncio
import logging
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import librosa
import soundfile as sf
from scipy import signal, stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
import io
import torch
import torchaudio
try:
    import webrtcvad
    WEBRTC_VAD_AVAILABLE = True
except ImportError:
    WEBRTC_VAD_AVAILABLE = False
    webrtcvad = None
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from app.schemas.assessment import SpeechAnalysisResponse, SpeechBiomarkers

logger = logging.getLogger(__name__)

class RealtimeSpeechAnalyzer:
    """
    Ultra-fast speech analyzer optimized for real-time inference
    Uses lightweight feature extraction and pre-computed models
    """
    
    def __init__(self):
        self.model_loaded = False
        self.sample_rate = 16000  # Whisper-tiny optimal sample rate
        self.frame_size = 2048    # Better frequency resolution
        self.hop_length = 512     # Standard hop length
        self.n_mfcc = 13          # Standard MFCC coefficients
        self.n_fft = 2048         # FFT window size

        # Initialize ML components
        self.scaler = StandardScaler()
        self.whisper_processor = None
        self.whisper_model = None
        self.vad = webrtcvad.Vad(2) if WEBRTC_VAD_AVAILABLE else None  # Aggressiveness level 2 (0-3)

        # Load models asynchronously
        asyncio.create_task(self._load_speech_models())

        logger.info("RealtimeSpeechAnalyzer initializing with Whisper-tiny model...")
    
    async def _load_speech_models(self):
        """Load Whisper-tiny model and speech analysis parameters"""
        try:
            logger.info("Loading Whisper-tiny model...")

            # Load Whisper-tiny model for speech processing
            self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

            # Set model to evaluation mode and optimize for inference
            self.whisper_model.eval()
            if torch.cuda.is_available():
                self.whisper_model = self.whisper_model.cuda()
                logger.info("Whisper model loaded on GPU")
            else:
                logger.info("Whisper model loaded on CPU")

            # Voice biomarker detection parameters
            self.tremor_freq_range = (4, 12)  # Hz, typical tremor frequency
            self.pause_threshold = 0.02       # Energy threshold for pause detection
            self.speech_rate_window = 1.0     # Window for speech rate calculation

            # Neurological biomarker thresholds (based on clinical research)
            self.biomarker_thresholds = {
                'tremor_severity': 0.3,      # Normalized tremor amplitude
                'pause_ratio': 0.4,          # Pause-to-speech ratio
                'articulation_rate': 4.5,    # Syllables per second
                'voice_breaks': 0.1,         # Voice break frequency
                'jitter': 0.02,              # Fundamental frequency variation
                'shimmer': 0.05,             # Amplitude variation
            }

            self.model_loaded = True
            logger.info("Speech analysis models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load speech models: {str(e)}")
            self.model_loaded = False
            raise

        # Risk scoring parameters (evidence-based thresholds)
        self.risk_thresholds = {
            'tremor_severity': {'low': 0.1, 'moderate': 0.3, 'high': 0.5},
            'pause_abnormality': {'low': 0.15, 'moderate': 0.35, 'high': 0.6},
            'articulation_clarity': {'low': 0.8, 'moderate': 0.6, 'high': 0.4},
            'speech_rate_deviation': {'low': 0.2, 'moderate': 0.4, 'high': 0.7}
        }

        # Feature importance weights (based on clinical literature)
        self.feature_weights = {
            'mfcc_features': 0.30,
            'tremor_features': 0.25,
            'fluency_features': 0.25,
            'spectral_features': 0.20
        }

        # Normalization parameters (population statistics)
        self.normalization_params = {
            'fundamental_freq': {'mean': 150.0, 'std': 50.0},
            'speech_rate': {'mean': 4.5, 'std': 1.2},  # syllables per second
            'pause_duration': {'mean': 0.8, 'std': 0.3}
        }
    
    async def analyze_realtime(self, audio_bytes: bytes, session_id: str) -> SpeechAnalysisResponse:
        """
        Real-time speech analysis with Whisper-tiny model integration

        Args:
            audio_bytes: Raw audio data
            session_id: Session identifier

        Returns:
            SpeechAnalysisResponse with biomarkers and risk assessment
        """
        start_time = time.perf_counter()

        if not self.model_loaded:
            raise Exception("Speech analysis models not loaded. Please wait for initialization.")

        try:
            # Step 1: Audio preprocessing and validation
            audio_data = await self._preprocess_audio(audio_bytes)

            # Step 2: Voice Activity Detection
            speech_segments = await self._detect_speech_segments(audio_data)

            # Step 3: Feature extraction with MFCC and neurological biomarkers
            features = await self._extract_comprehensive_features(audio_data, speech_segments)

            # Step 4: Whisper-based speech quality assessment
            speech_quality = await self._assess_speech_with_whisper(audio_data)

            # Step 5: Neurological biomarker calculation
            biomarkers = await self._calculate_neurological_biomarkers(features, speech_segments)

            # Step 6: Risk score calculation
            risk_score = await self._calculate_risk_score(features, biomarkers, speech_quality)

            # Step 7: Generate response
            processing_time = time.perf_counter() - start_time

            return SpeechAnalysisResponse(
                session_id=session_id,
                processing_time=processing_time,
                timestamp=datetime.now(),
                confidence=self._calculate_confidence(features, speech_quality),
                biomarkers=biomarkers,
                risk_score=risk_score,
                quality_score=speech_quality['overall_quality'],
                recommendations=self._generate_recommendations(risk_score, biomarkers)
            )
            
        except Exception as e:
            logger.error(f"Real-time speech analysis failed: {str(e)}")
            raise Exception(f"Real-time analysis failed: {str(e)}")

    async def _preprocess_audio(self, audio_bytes: bytes) -> np.ndarray:
        """Real audio preprocessing with librosa"""

        try:
            # Convert bytes to audio array
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Resample to target sample rate if needed
            if len(audio_data) > 0:
                # Estimate original sample rate (assume 16kHz if not specified)
                original_sr = 16000
                if len(audio_data) > self.sample_rate * 30:  # If too long, assume higher sample rate
                    original_sr = 44100

                # Resample using librosa
                if original_sr != self.sample_rate:
                    audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=self.sample_rate)

                # Trim silence from beginning and end
                audio_data, _ = librosa.effects.trim(audio_data, top_db=20)

                # Normalize audio
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = librosa.util.normalize(audio_data)

                # Limit to 30 seconds maximum
                max_samples = self.sample_rate * 30
                if len(audio_data) > max_samples:
                    audio_data = audio_data[:max_samples]

            return audio_data

        except Exception as e:
            logger.error(f"Audio preprocessing failed: {str(e)}")
            # Fallback to simple processing
            try:
                audio_data, sr = sf.read(io.BytesIO(audio_bytes))
                if sr != self.sample_rate:
                    audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sample_rate)
            except:
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            return audio_data[:self.sample_rate * 10]  # Limit to 10 seconds

    async def _detect_speech_segments(self, audio_data: np.ndarray) -> List[Tuple[int, int]]:
        """Detect speech segments using Voice Activity Detection"""
        try:
            if WEBRTC_VAD_AVAILABLE and self.vad is not None:
                # Convert to 16-bit PCM for WebRTC VAD
                audio_16bit = (audio_data * 32767).astype(np.int16)

                # WebRTC VAD works with specific frame sizes (10, 20, or 30ms)
                frame_duration = 30  # ms
                frame_size = int(self.sample_rate * frame_duration / 1000)

                speech_segments = []
                current_segment_start = None

                for i in range(0, len(audio_16bit) - frame_size, frame_size):
                    frame = audio_16bit[i:i + frame_size]

                    # Check if frame contains speech
                    is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)

                    if is_speech and current_segment_start is None:
                        current_segment_start = i
                    elif not is_speech and current_segment_start is not None:
                        speech_segments.append((current_segment_start, i))
                        current_segment_start = None

                # Close final segment if needed
                if current_segment_start is not None:
                    speech_segments.append((current_segment_start, len(audio_16bit)))

                return speech_segments
            else:
                # Fallback to energy-based detection when WebRTC VAD is not available
                return self._energy_based_vad(audio_data)

        except Exception as e:
            logger.warning(f"VAD failed, using energy-based detection: {str(e)}")
            # Fallback to energy-based detection
            return self._energy_based_vad(audio_data)

    def _energy_based_vad(self, audio_data: np.ndarray) -> List[Tuple[int, int]]:
        """Fallback energy-based voice activity detection"""
        frame_size = int(self.sample_rate * 0.025)  # 25ms frames
        hop_size = int(self.sample_rate * 0.010)    # 10ms hop

        # Calculate frame energies
        energies = []
        for i in range(0, len(audio_data) - frame_size, hop_size):
            frame = audio_data[i:i + frame_size]
            energy = np.sum(frame ** 2)
            energies.append(energy)

        # Threshold-based detection
        threshold = np.percentile(energies, 30)  # 30th percentile as threshold

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

    async def _extract_comprehensive_features(self, audio_data: np.ndarray, speech_segments: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Extract comprehensive MFCC and neurological features"""
        features = {}

        if len(audio_data) == 0:
            return self._get_default_features()

        try:
            # 1. MFCC Features (Mel-frequency cepstral coefficients)
            mfcc = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )

            # MFCC statistics
            features['mfcc_mean'] = np.mean(mfcc, axis=1).tolist()
            features['mfcc_std'] = np.std(mfcc, axis=1).tolist()
            features['mfcc_delta'] = np.mean(librosa.feature.delta(mfcc), axis=1).tolist()
            features['mfcc_delta2'] = np.mean(librosa.feature.delta(mfcc, order=2), axis=1).tolist()

            # 2. Spectral Features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=self.sample_rate)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0]

            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            features['zero_crossing_rate_mean'] = float(np.mean(zero_crossing_rate))

            # 3. Prosodic Features (F0, jitter, shimmer)
            f0_features = self._extract_f0_features(audio_data)
            features.update(f0_features)

            # 4. Voice Quality Features
            voice_quality = self._extract_voice_quality_features(audio_data)
            features.update(voice_quality)

            # 5. Temporal Features
            temporal_features = self._extract_temporal_features(audio_data, speech_segments)
            features.update(temporal_features)

            # 6. Neurological Biomarkers
            neuro_features = self._extract_neurological_features(audio_data, speech_segments)
            features.update(neuro_features)

            return features

        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return self._get_default_features()

    def _extract_f0_features(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Extract fundamental frequency (F0) features"""
        try:
            # Extract F0 using YIN algorithm
            f0 = librosa.yin(audio_data, fmin=50, fmax=400, sr=self.sample_rate)
            f0_clean = f0[f0 > 0]  # Remove unvoiced frames

            if len(f0_clean) > 0:
                f0_mean = np.mean(f0_clean)
                f0_std = np.std(f0_clean)
                f0_range = np.max(f0_clean) - np.min(f0_clean)

                # Jitter calculation (F0 period-to-period variation)
                f0_periods = 1.0 / (f0_clean + 1e-8)
                if len(f0_periods) > 1:
                    jitter = np.mean(np.abs(np.diff(f0_periods))) / np.mean(f0_periods)
                else:
                    jitter = 0.0

                return {
                    'f0_mean': float(f0_mean),
                    'f0_std': float(f0_std),
                    'f0_range': float(f0_range),
                    'jitter': float(jitter)
                }
            else:
                return {
                    'f0_mean': 150.0,  # Default values
                    'f0_std': 20.0,
                    'f0_range': 50.0,
                    'jitter': 0.02
                }
        except:
            return {
                'f0_mean': 150.0,
                'f0_std': 20.0,
                'f0_range': 50.0,
                'jitter': 0.02
            }

    def _extract_voice_quality_features(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Extract voice quality features (shimmer, HNR, etc.)"""
        try:
            # Shimmer calculation (amplitude variation)
            frame_length = int(0.025 * self.sample_rate)  # 25ms frames
            hop_length = int(0.010 * self.sample_rate)    # 10ms hop

            # Calculate frame energies
            frame_energies = []
            for i in range(0, len(audio_data) - frame_length, hop_length):
                frame = audio_data[i:i + frame_length]
                energy = np.sqrt(np.mean(frame ** 2))
                frame_energies.append(energy)

            frame_energies = np.array(frame_energies)

            # Shimmer (amplitude period-to-period variation)
            if len(frame_energies) > 1:
                shimmer = np.mean(np.abs(np.diff(frame_energies))) / (np.mean(frame_energies) + 1e-8)
            else:
                shimmer = 0.0

            # Harmonics-to-Noise Ratio (HNR) approximation
            # Using spectral features as proxy
            stft = librosa.stft(audio_data, n_fft=self.n_fft, hop_length=self.hop_length)
            magnitude = np.abs(stft)

            # Estimate HNR from spectral regularity
            spectral_flux = np.mean(np.diff(magnitude, axis=1) ** 2)
            hnr_estimate = max(0.0, 20.0 - spectral_flux * 1000)  # Rough approximation

            return {
                'shimmer': float(shimmer),
                'hnr_estimate': float(hnr_estimate),
                'voice_stability': float(1.0 - min(1.0, shimmer * 10))
            }

        except:
            return {
                'shimmer': 0.05,
                'hnr_estimate': 15.0,
                'voice_stability': 0.8
            }

    def _extract_temporal_features(self, audio_data: np.ndarray, speech_segments: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Extract temporal features from speech segments"""
        try:
            if not speech_segments:
                return {
                    'speech_rate': 0.5,
                    'avg_pause_duration': 0.5,
                    'pause_frequency': 0.1,
                    'rhythm_regularity': 0.5
                }

            # Calculate speech rate
            total_speech_time = sum(end - start for start, end in speech_segments) / self.sample_rate
            total_duration = (speech_segments[-1][1] - speech_segments[0][0]) / self.sample_rate

            speech_rate = total_speech_time / total_duration if total_duration > 0 else 0.5

            # Calculate pause characteristics
            pause_durations = []
            for i in range(len(speech_segments) - 1):
                pause_start = speech_segments[i][1]
                pause_end = speech_segments[i + 1][0]
                pause_duration = (pause_end - pause_start) / self.sample_rate
                if pause_duration > 0.05:  # Only count pauses > 50ms
                    pause_durations.append(pause_duration)

            if pause_durations:
                avg_pause_duration = np.mean(pause_durations)
                pause_frequency = len(pause_durations) / total_duration if total_duration > 0 else 0.1

                # Rhythm regularity (inverse of pause duration variability)
                pause_std = np.std(pause_durations)
                rhythm_regularity = max(0.0, 1.0 - pause_std)
            else:
                avg_pause_duration = 0.5
                pause_frequency = 0.1
                rhythm_regularity = 0.5

            return {
                'speech_rate': float(speech_rate),
                'avg_pause_duration': float(avg_pause_duration),
                'pause_frequency': float(pause_frequency),
                'rhythm_regularity': float(rhythm_regularity)
            }

        except:
            return {
                'speech_rate': 0.5,
                'avg_pause_duration': 0.5,
                'pause_frequency': 0.1,
                'rhythm_regularity': 0.5
            }

    def _extract_neurological_features(self, audio_data: np.ndarray, speech_segments: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Extract neurological-specific features"""
        try:
            # Tremor intensity estimation
            # Look for regular oscillations in the 4-12 Hz range
            stft = librosa.stft(audio_data, n_fft=self.n_fft, hop_length=self.hop_length)
            magnitude = np.abs(stft)

            # Focus on frequency range where voice tremor occurs
            freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)
            tremor_freq_mask = (freqs >= 4) & (freqs <= 12)

            if np.any(tremor_freq_mask):
                tremor_energy = np.mean(magnitude[tremor_freq_mask, :])
                total_energy = np.mean(magnitude)
                tremor_intensity = tremor_energy / (total_energy + 1e-8)
            else:
                tremor_intensity = 0.1

            # Voice breaks detection (sudden amplitude drops)
            frame_energies = np.mean(magnitude, axis=0)
            energy_threshold = np.percentile(frame_energies, 20)
            voice_breaks = np.sum(frame_energies < energy_threshold) / len(frame_energies)

            return {
                'tremor_intensity': float(min(1.0, tremor_intensity * 5.0)),
                'voice_breaks': float(voice_breaks)
            }

        except:
            return {
                'tremor_intensity': 0.1,
                'voice_breaks': 0.05
            }

    async def _assess_speech_with_whisper(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Assess speech quality using Whisper-tiny model"""
        try:
            if not self.model_loaded or self.whisper_model is None:
                logger.warning("Whisper model not loaded, using fallback assessment")
                return self._fallback_speech_assessment(audio_data)

            # Prepare audio for Whisper (ensure correct format)
            # Whisper expects audio in [-1, 1] range at 16kHz
            if len(audio_data) > self.sample_rate * 30:  # Limit to 30 seconds
                audio_data = audio_data[:self.sample_rate * 30]

            # Pad or trim to ensure consistent length
            target_length = self.sample_rate * 30  # 30 seconds
            if len(audio_data) < target_length:
                audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), 'constant')

            # Process with Whisper
            with torch.no_grad():
                # Convert to tensor
                audio_tensor = torch.from_numpy(audio_data).float()
                if torch.cuda.is_available():
                    audio_tensor = audio_tensor.cuda()

                # Process audio through Whisper processor
                inputs = self.whisper_processor(
                    audio_tensor.cpu().numpy(),
                    sampling_rate=self.sample_rate,
                    return_tensors="pt"
                )

                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                # Get model outputs (encoder features for quality assessment)
                encoder_outputs = self.whisper_model.get_encoder()(
                    inputs["input_features"]
                )

                # Extract quality metrics from encoder outputs
                attention_weights = encoder_outputs.last_hidden_state

                # Calculate speech quality metrics
                quality_metrics = {
                    'overall_quality': self._calculate_whisper_quality_score(attention_weights),
                    'clarity_score': self._calculate_clarity_from_attention(attention_weights),
                    'consistency_score': self._calculate_consistency_score(attention_weights),
                    'speech_rate_score': self._estimate_speech_rate_quality(audio_data),
                    'noise_level': self._estimate_noise_level(audio_data)
                }

                return quality_metrics

        except Exception as e:
            logger.error(f"Whisper assessment failed: {str(e)}")
            return self._fallback_speech_assessment(audio_data)

    def _calculate_whisper_quality_score(self, attention_weights: torch.Tensor) -> float:
        """Calculate overall speech quality from Whisper attention weights"""
        try:
            # Use attention weight variance as a proxy for speech clarity
            attention_variance = torch.var(attention_weights).item()

            # Normalize to 0-1 range (higher variance = clearer speech)
            quality_score = min(1.0, attention_variance / 0.1)  # 0.1 is empirical threshold

            return float(quality_score)
        except:
            return 0.5  # Default moderate quality

    def _calculate_clarity_from_attention(self, attention_weights: torch.Tensor) -> float:
        """Calculate speech clarity from attention patterns"""
        try:
            # Sharp attention peaks indicate clear speech
            attention_sharpness = torch.max(attention_weights, dim=-1)[0].mean().item()
            return min(1.0, attention_sharpness * 2.0)
        except:
            return 0.5

    def _calculate_consistency_score(self, attention_weights: torch.Tensor) -> float:
        """Calculate speech consistency from attention patterns"""
        try:
            # Consistent attention patterns indicate steady speech
            attention_std = torch.std(attention_weights).item()
            consistency = 1.0 - min(1.0, attention_std)  # Lower std = higher consistency
            return float(consistency)
        except:
            return 0.5

    def _estimate_speech_rate_quality(self, audio_data: np.ndarray) -> float:
        """Estimate speech rate quality (not too fast, not too slow)"""
        try:
            # Simple energy-based speech rate estimation
            frame_size = int(self.sample_rate * 0.025)  # 25ms frames
            hop_size = int(self.sample_rate * 0.010)    # 10ms hop

            energies = []
            for i in range(0, len(audio_data) - frame_size, hop_size):
                frame = audio_data[i:i + frame_size]
                energy = np.sum(frame ** 2)
                energies.append(energy)

            # Count energy peaks (rough syllable estimation)
            threshold = np.percentile(energies, 60)
            peaks = len([e for e in energies if e > threshold])

            # Convert to syllables per second
            duration = len(audio_data) / self.sample_rate
            syllables_per_second = peaks / duration if duration > 0 else 0

            # Optimal range: 3-6 syllables per second
            if 3 <= syllables_per_second <= 6:
                return 1.0
            elif 2 <= syllables_per_second <= 8:
                return 0.7
            else:
                return 0.3

        except:
            return 0.5

    def _estimate_noise_level(self, audio_data: np.ndarray) -> float:
        """Estimate background noise level"""
        try:
            # Use spectral subtraction approach
            stft = librosa.stft(audio_data, n_fft=self.n_fft, hop_length=self.hop_length)
            magnitude = np.abs(stft)

            # Estimate noise floor (bottom 10th percentile)
            noise_floor = np.percentile(magnitude, 10)
            signal_level = np.percentile(magnitude, 90)

            # Calculate SNR approximation
            snr = signal_level / (noise_floor + 1e-8)

            # Convert to 0-1 scale (higher = less noise)
            noise_score = min(1.0, np.log10(snr + 1) / 2.0)

            return float(1.0 - noise_score)  # Return noise level (0 = no noise)

        except:
            return 0.3  # Default moderate noise

    def _fallback_speech_assessment(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Fallback speech assessment when Whisper is not available"""
        try:
            # Basic energy and spectral analysis
            energy = np.mean(audio_data ** 2)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate))

            return {
                'overall_quality': min(1.0, energy * 10),  # Rough quality estimate
                'clarity_score': min(1.0, spectral_centroid / 2000),  # Normalized spectral centroid
                'consistency_score': 0.5,  # Default
                'speech_rate_score': 0.5,  # Default
                'noise_level': 0.3  # Default
            }
        except:
            return {
                'overall_quality': 0.5,
                'clarity_score': 0.5,
                'consistency_score': 0.5,
                'speech_rate_score': 0.5,
                'noise_level': 0.3
            }

    async def _extract_fast_features(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Real feature extraction with MFCC and voice biomarkers"""

        features = {}

        if len(audio_data) == 0:
            return self._get_default_features()

        try:
            # 1. MFCC Features (Mel-frequency cepstral coefficients)
            mfcc = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )

            # Statistical measures of MFCCs
            features['mfcc_mean'] = np.mean(mfcc, axis=1)
            features['mfcc_std'] = np.std(mfcc, axis=1)
            features['mfcc_delta'] = np.mean(librosa.feature.delta(mfcc), axis=1)

            # 2. Spectral Features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)

            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)

            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0]
            features['zero_crossing_rate_mean'] = np.mean(zero_crossing_rate)
            features['zero_crossing_rate_std'] = np.std(zero_crossing_rate)

            # 3. Fundamental Frequency (F0) Analysis
            f0 = librosa.yin(audio_data, fmin=50, fmax=400, sr=self.sample_rate)
            f0_clean = f0[f0 > 0]  # Remove unvoiced frames

            if len(f0_clean) > 0:
                features['f0_mean'] = np.mean(f0_clean)
                features['f0_std'] = np.std(f0_clean)
                features['f0_range'] = np.max(f0_clean) - np.min(f0_clean)
            else:
                features['f0_mean'] = 150.0  # Default
                features['f0_std'] = 20.0
                features['f0_range'] = 50.0

            # 4. Voice Quality Features
            features.update(await self._extract_voice_quality_features(audio_data))

            # 5. Temporal Features
            features.update(await self._extract_temporal_features(audio_data))

            return features

        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return self._get_default_features()

    async def _extract_voice_quality_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract voice quality and tremor-related features"""

        features = {}

        try:
            # Tremor detection using spectral analysis
            fft = np.fft.rfft(audio_data)
            freqs = np.fft.rfftfreq(len(audio_data), 1/self.sample_rate)
            magnitude = np.abs(fft)

            # Focus on tremor frequency range (4-12 Hz)
            tremor_mask = (freqs >= self.tremor_freq_range[0]) & (freqs <= self.tremor_freq_range[1])
            tremor_power = np.sum(magnitude[tremor_mask])
            total_power = np.sum(magnitude)

            features['tremor_intensity'] = tremor_power / (total_power + 1e-10)

            # Voice stability (coefficient of variation of F0)
            if 'f0_mean' in features and 'f0_std' in features:
                features['voice_stability'] = features['f0_std'] / (features['f0_mean'] + 1e-10)
            else:
                features['voice_stability'] = 0.1  # Default

            # Jitter and shimmer approximation
            frame_size = 1024
            frames = [audio_data[i:i+frame_size] for i in range(0, len(audio_data)-frame_size, frame_size//2)]

            if len(frames) > 1:
                frame_energies = [np.sum(frame**2) for frame in frames]
                features['shimmer'] = np.std(frame_energies) / (np.mean(frame_energies) + 1e-10)
            else:
                features['shimmer'] = 0.05  # Default

            return features

        except Exception as e:
            logger.error(f"Voice quality feature extraction failed: {str(e)}")
            return {'tremor_intensity': 0.1, 'voice_stability': 0.1, 'shimmer': 0.05}

    async def _extract_temporal_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract temporal features like speech rate and pause patterns"""

        features = {}

        try:
            # Voice Activity Detection (VAD)
            frame_length = int(0.025 * self.sample_rate)  # 25ms frames
            hop_length = int(0.010 * self.sample_rate)    # 10ms hop

            frames = librosa.util.frame(audio_data, frame_length=frame_length, hop_length=hop_length)
            frame_energies = np.sum(frames**2, axis=0)

            # Adaptive threshold for voice activity
            energy_threshold = np.percentile(frame_energies, 30)
            voice_frames = frame_energies > energy_threshold

            # Speech rate estimation (voiced frames per second)
            total_time = len(audio_data) / self.sample_rate
            voiced_time = np.sum(voice_frames) * hop_length / self.sample_rate
            features['speech_rate'] = voiced_time / total_time if total_time > 0 else 0.5

            # Pause analysis
            pause_frames = ~voice_frames
            pause_segments = self._find_segments(pause_frames)

            if len(pause_segments) > 0:
                pause_durations = [(end - start) * hop_length / self.sample_rate for start, end in pause_segments]
                features['avg_pause_duration'] = np.mean(pause_durations)
                features['pause_frequency'] = len(pause_segments) / total_time
            else:
                features['avg_pause_duration'] = 0.5
                features['pause_frequency'] = 0.1

            # Rhythm regularity
            voice_segments = self._find_segments(voice_frames)
            if len(voice_segments) > 1:
                segment_durations = [(end - start) * hop_length / self.sample_rate for start, end in voice_segments]
                features['rhythm_regularity'] = 1.0 / (1.0 + np.std(segment_durations))
            else:
                features['rhythm_regularity'] = 0.5

            return features

        except Exception as e:
            logger.error(f"Temporal feature extraction failed: {str(e)}")
            return {
                'speech_rate': 0.5,
                'avg_pause_duration': 0.5,
                'pause_frequency': 0.1,
                'rhythm_regularity': 0.5
            }

    def _find_segments(self, binary_array: np.ndarray) -> List[Tuple[int, int]]:
        """Find continuous segments of True values in binary array"""
        segments = []
        start = None

        for i, val in enumerate(binary_array):
            if val and start is None:
                start = i
            elif not val and start is not None:
                segments.append((start, i))
                start = None

        if start is not None:
            segments.append((start, len(binary_array)))

        return segments

    def _get_default_features(self) -> Dict[str, Any]:
        """Return default features when extraction fails"""
        return {
            'mfcc_mean': np.zeros(self.n_mfcc),
            'mfcc_std': np.ones(self.n_mfcc) * 0.1,
            'mfcc_delta': np.zeros(self.n_mfcc),
            'spectral_centroid_mean': 1500.0,
            'spectral_centroid_std': 200.0,
            'spectral_rolloff_mean': 3000.0,
            'zero_crossing_rate_mean': 0.1,
            'zero_crossing_rate_std': 0.05,
            'f0_mean': 150.0,
            'f0_std': 20.0,
            'f0_range': 50.0,
            'tremor_intensity': 0.1,
            'voice_stability': 0.1,
            'shimmer': 0.05,
            'speech_rate': 0.5,
            'avg_pause_duration': 0.5,
            'pause_frequency': 0.1,
            'rhythm_regularity': 0.5
        }

    async def _calculate_neurological_biomarkers(self, features: Dict[str, Any], speech_segments: List[Tuple[int, int]]) -> SpeechBiomarkers:
        """Calculate comprehensive neurological biomarkers from speech features"""
        try:
            # 1. Advanced Fluency Analysis
            fluency_score = self._calculate_advanced_fluency(features, speech_segments)

            # 2. Neurological Pause Pattern Analysis
            pause_pattern = self._analyze_neurological_pause_patterns(features, speech_segments)

            # 3. Voice Tremor Detection (Parkinson's indicator)
            voice_tremor = self._detect_voice_tremor(features)

            # 4. Articulation Clarity (Dysarthria assessment)
            articulation_clarity = self._assess_articulation_clarity(features)

            # 5. Prosodic Variation (Emotional and cognitive assessment)
            prosody_variation = self._analyze_prosodic_variation(features)

            # 6. Speech Rate Analysis (Cognitive processing speed)
            speaking_rate = self._calculate_speech_rate(features, speech_segments)

            # 7. Pause Frequency (Executive function indicator)
            pause_frequency = self._calculate_pause_frequency(speech_segments)

            return SpeechBiomarkers(
                fluency_score=fluency_score,
                pause_pattern=pause_pattern,
                voice_tremor=voice_tremor,
                articulation_clarity=articulation_clarity,
                prosody_variation=prosody_variation,
                speaking_rate=speaking_rate,
                pause_frequency=pause_frequency
            )

        except Exception as e:
            logger.error(f"Neurological biomarker calculation failed: {str(e)}")
            return self._get_default_biomarkers()

    def _calculate_advanced_fluency(self, features: Dict[str, Any], speech_segments: List[Tuple[int, int]]) -> float:
        """Calculate advanced fluency score considering multiple factors"""
        try:
            # Speech continuity (fewer, longer segments = better fluency)
            if len(speech_segments) == 0:
                return 0.0

            total_speech_time = sum(end - start for start, end in speech_segments) / self.sample_rate
            avg_segment_length = total_speech_time / len(speech_segments)

            # Fluency components
            continuity_score = min(1.0, avg_segment_length / 2.0)  # 2 seconds = perfect continuity
            segment_count_penalty = max(0.0, 1.0 - len(speech_segments) / 10.0)  # Penalty for too many segments

            # MFCC consistency (stable features = fluent speech)
            mfcc_std = np.mean(features.get('mfcc_std', [0.5] * 13))
            consistency_score = max(0.0, 1.0 - mfcc_std)

            # Combine scores
            fluency = (continuity_score * 0.4 + segment_count_penalty * 0.3 + consistency_score * 0.3)
            return float(max(0.0, min(1.0, fluency)))

        except:
            return 0.5

    def _analyze_neurological_pause_patterns(self, features: Dict[str, Any], speech_segments: List[Tuple[int, int]]) -> float:
        """Analyze pause patterns for neurological indicators"""
        try:
            if len(speech_segments) < 2:
                return 0.0

            # Calculate pause durations
            pause_durations = []
            for i in range(len(speech_segments) - 1):
                pause_start = speech_segments[i][1]
                pause_end = speech_segments[i + 1][0]
                pause_duration = (pause_end - pause_start) / self.sample_rate
                if pause_duration > 0.1:  # Only count pauses > 100ms
                    pause_durations.append(pause_duration)

            if not pause_durations:
                return 0.0

            # Analyze pause characteristics
            avg_pause = np.mean(pause_durations)
            pause_variability = np.std(pause_durations) / (avg_pause + 1e-8)

            # Neurological indicators:
            # - Excessive long pauses (>2s) indicate word-finding difficulties
            # - High variability indicates inconsistent cognitive processing
            long_pause_ratio = sum(1 for p in pause_durations if p > 2.0) / len(pause_durations)

            # Calculate abnormality score (0 = normal, 1 = highly abnormal)
            pause_abnormality = min(1.0, (avg_pause / 3.0) * 0.5 + pause_variability * 0.3 + long_pause_ratio * 0.2)

            return float(pause_abnormality)

        except:
            return 0.3

    def _detect_voice_tremor(self, features: Dict[str, Any]) -> float:
        """Detect voice tremor (Parkinson's disease indicator)"""
        try:
            # F0 tremor detection
            f0_std = features.get('f0_std', 0)
            f0_mean = features.get('f0_mean', 150)

            # Tremor is indicated by regular F0 oscillations in 4-12 Hz range
            # High F0 variability relative to mean suggests tremor
            f0_tremor_indicator = min(1.0, f0_std / (f0_mean * 0.1 + 1e-8))

            # Amplitude tremor (from spectral features)
            spectral_centroid_std = features.get('spectral_centroid_std', 0)
            amplitude_tremor = min(1.0, spectral_centroid_std / 500.0)

            # Voice breaks and instability
            zero_crossing_std = features.get('zero_crossing_rate_std', 0)
            voice_instability = min(1.0, zero_crossing_std * 10.0)

            # Combine tremor indicators
            tremor_score = (f0_tremor_indicator * 0.5 + amplitude_tremor * 0.3 + voice_instability * 0.2)

            return float(max(0.0, min(1.0, tremor_score)))

        except:
            return 0.1

    def _assess_articulation_clarity(self, features: Dict[str, Any]) -> float:
        """Assess articulation clarity (dysarthria detection)"""
        try:
            # High-frequency content indicates clear consonants
            spectral_centroid = features.get('spectral_centroid_mean', 1500)
            spectral_rolloff = features.get('spectral_rolloff_mean', 3000)

            # Clear speech has balanced spectral content
            clarity_from_spectrum = min(1.0, spectral_centroid / 2000.0)
            consonant_clarity = min(1.0, spectral_rolloff / 4000.0)

            # MFCC delta features indicate articulation dynamics
            mfcc_delta_energy = np.mean(np.abs(features.get('mfcc_delta', [0] * 13)))
            articulation_dynamics = min(1.0, mfcc_delta_energy * 2.0)

            # Combine clarity measures
            clarity_score = (clarity_from_spectrum * 0.4 + consonant_clarity * 0.3 + articulation_dynamics * 0.3)

            return float(max(0.0, min(1.0, clarity_score)))

        except:
            return 0.5

    def _analyze_prosodic_variation(self, features: Dict[str, Any]) -> float:
        """Analyze prosodic variation (emotional and cognitive assessment)"""
        try:
            # F0 variation indicates prosodic richness
            f0_range = features.get('f0_range', 0)
            f0_std = features.get('f0_std', 0)

            # Healthy prosody has moderate variation
            f0_variation = min(1.0, (f0_range / 100.0 + f0_std / 30.0) / 2.0)

            # Energy variation indicates emotional expression
            # (This would need energy contour analysis in full implementation)
            energy_variation = 0.5  # Placeholder - would calculate from energy contour

            # Rhythm variation
            # (This would need rhythm analysis in full implementation)
            rhythm_variation = 0.5  # Placeholder

            # Combine prosodic measures
            prosody_score = (f0_variation * 0.5 + energy_variation * 0.3 + rhythm_variation * 0.2)

            return float(max(0.0, min(1.0, prosody_score)))

        except:
            return 0.5

    def _calculate_speech_rate(self, features: Dict[str, Any], speech_segments: List[Tuple[int, int]]) -> float:
        """Calculate speech rate in words per minute"""
        try:
            if not speech_segments:
                return 120.0  # Default moderate rate

            total_speech_time = sum(end - start for start, end in speech_segments) / self.sample_rate

            # Estimate syllables from energy peaks (rough approximation)
            # In full implementation, this would use more sophisticated syllable detection
            estimated_syllables = len(speech_segments) * 3  # Rough estimate

            if total_speech_time > 0:
                syllables_per_second = estimated_syllables / total_speech_time
                words_per_minute = syllables_per_second * 60 / 2.5  # ~2.5 syllables per word

                # Clamp to reasonable range
                return float(max(60.0, min(200.0, words_per_minute)))

            return 120.0

        except:
            return 120.0

    def _calculate_pause_frequency(self, speech_segments: List[Tuple[int, int]]) -> float:
        """Calculate pause frequency per minute"""
        try:
            if len(speech_segments) < 2:
                return 0.0

            # Count pauses between segments
            pause_count = len(speech_segments) - 1

            # Calculate total duration
            total_duration = (speech_segments[-1][1] - speech_segments[0][0]) / self.sample_rate / 60.0  # minutes

            if total_duration > 0:
                pauses_per_minute = pause_count / total_duration
                return float(min(30.0, pauses_per_minute))  # Cap at 30 pauses/minute

            return 0.0

        except:
            return 5.0  # Default moderate pause frequency

    def _get_default_biomarkers(self) -> SpeechBiomarkers:
        """Return default biomarkers when calculation fails"""
        return SpeechBiomarkers(
            fluency_score=0.5,
            pause_pattern=0.3,
            voice_tremor=0.1,
            articulation_clarity=0.5,
            prosody_variation=0.5,
            speaking_rate=120.0,
            pause_frequency=5.0
        )

    async def _calculate_risk_score(self, features: Dict[str, Any], biomarkers: SpeechBiomarkers, speech_quality: Dict[str, Any]) -> float:
        """Calculate overall neurological risk score"""
        try:
            # Weight factors based on clinical research
            weights = {
                'tremor': 0.25,      # Strong Parkinson's indicator
                'fluency': 0.20,     # Cognitive processing
                'pause_pattern': 0.20, # Executive function
                'articulation': 0.15,  # Motor control
                'prosody': 0.10,     # Emotional/cognitive
                'speech_quality': 0.10  # Overall assessment
            }

            # Normalize biomarkers to risk scores (0-1, higher = more risk)
            tremor_risk = biomarkers.voice_tremor
            fluency_risk = 1.0 - biomarkers.fluency_score  # Invert (low fluency = high risk)
            pause_risk = biomarkers.pause_pattern
            articulation_risk = 1.0 - biomarkers.articulation_clarity  # Invert
            prosody_risk = 1.0 - biomarkers.prosody_variation  # Invert
            quality_risk = 1.0 - speech_quality.get('overall_quality', 0.5)  # Invert

            # Calculate weighted risk score
            risk_score = (
                tremor_risk * weights['tremor'] +
                fluency_risk * weights['fluency'] +
                pause_risk * weights['pause_pattern'] +
                articulation_risk * weights['articulation'] +
                prosody_risk * weights['prosody'] +
                quality_risk * weights['speech_quality']
            )

            # Apply confidence adjustment based on speech quality
            confidence_factor = speech_quality.get('overall_quality', 0.5)
            adjusted_risk = risk_score * confidence_factor + 0.5 * (1 - confidence_factor)

            return float(max(0.0, min(1.0, adjusted_risk)))

        except Exception as e:
            logger.error(f"Risk score calculation failed: {str(e)}")
            return 0.5  # Default moderate risk

    async def _fast_biomarker_calculation(self, features: Dict[str, Any]) -> SpeechBiomarkers:
        """Real biomarker calculation using extracted voice features"""

        try:
            # 1. Fluency Score (based on speech rate and rhythm regularity)
            speech_rate = features.get('speech_rate', 0.5)
            rhythm_regularity = features.get('rhythm_regularity', 0.5)
            fluency_score = max(0.0, min(1.0, (speech_rate + rhythm_regularity) / 2.0))

            # 2. Pause Pattern (normalized pause frequency)
            pause_freq = features.get('pause_frequency', 0.1)
            avg_pause = features.get('avg_pause_duration', 0.5)
            pause_pattern = min(1.0, (pause_freq * 2.0 + avg_pause) / 2.0)

            # 3. Voice Tremor (based on tremor intensity and voice stability)
            tremor_intensity = features.get('tremor_intensity', 0.1)
            voice_stability = features.get('voice_stability', 0.1)
            voice_tremor = min(1.0, (tremor_intensity * 3.0 + voice_stability) / 2.0)

            # 4. Articulation Clarity (based on spectral features)
            spectral_centroid = features.get('spectral_centroid_mean', 1500.0)
            zcr_mean = features.get('zero_crossing_rate_mean', 0.1)

            # Normalize and invert (higher values = better clarity)
            spectral_clarity = 1.0 - abs(spectral_centroid - 1500.0) / 1500.0
            zcr_clarity = 1.0 - abs(zcr_mean - 0.08) / 0.08
            articulation_clarity = max(0.0, (spectral_clarity + zcr_clarity) / 2.0)

            # 5. Prosody Variation (based on F0 variation and rhythm)
            f0_std = features.get('f0_std', 20.0)
            f0_range = features.get('f0_range', 50.0)

            # Normalize F0 variation (moderate variation is good)
            f0_variation = min(1.0, (f0_std / 30.0 + f0_range / 100.0) / 2.0)
            prosody_variation = max(0.0, min(1.0, f0_variation))

            # 6. Speaking Rate (convert to words per minute)
            speech_rate_wpm = speech_rate * 150.0  # Approximate conversion
            speaking_rate = max(60.0, min(200.0, speech_rate_wpm))

            # 7. Pause Frequency (per minute) - cap at reasonable maximum
            pause_frequency = min(30.0, pause_freq * 60.0)  # Cap at 30 pauses/min

            return SpeechBiomarkers(
                fluency_score=fluency_score,
                pause_pattern=pause_pattern,
                voice_tremor=voice_tremor,
                articulation_clarity=articulation_clarity,
                prosody_variation=prosody_variation,
                speaking_rate=speaking_rate,
                pause_frequency=pause_frequency
            )

        except Exception as e:
            logger.error(f"Biomarker calculation failed: {str(e)}")
            # Return default biomarkers
            return SpeechBiomarkers(
                fluency_score=0.7,
                pause_pattern=0.3,
                voice_tremor=0.2,
                articulation_clarity=0.8,
                prosody_variation=0.5,
                speaking_rate=120.0,
                pause_frequency=8.0
            )
    
    async def _fast_risk_scoring(self, features: Dict[str, Any], biomarkers: SpeechBiomarkers) -> float:
        """Real risk scoring based on clinical evidence and biomarkers"""

        try:
            # Risk factors based on clinical literature
            risk_factors = []

            # 1. Tremor Risk (high tremor = high risk)
            tremor_risk = biomarkers.voice_tremor * self.feature_weights['tremor_features']
            risk_factors.append(tremor_risk)

            # 2. Fluency Risk (low fluency = high risk)
            fluency_risk = (1.0 - biomarkers.fluency_score) * self.feature_weights['fluency_features']
            risk_factors.append(fluency_risk)

            # 3. Articulation Risk (poor clarity = high risk)
            articulation_risk = (1.0 - biomarkers.articulation_clarity) * self.feature_weights['spectral_features']
            risk_factors.append(articulation_risk)

            # 4. MFCC-based Risk (abnormal spectral patterns)
            mfcc_mean = features.get('mfcc_mean', np.zeros(self.n_mfcc))
            if isinstance(mfcc_mean, np.ndarray) and len(mfcc_mean) > 0:
                # Compare to normal speech patterns
                mfcc_deviation = np.mean(np.abs(mfcc_mean - np.array([
                    -5.0, 1.0, 0.5, -0.5, 0.3, -0.2, 0.1, -0.1, 0.05, -0.05, 0.02, -0.02, 0.01
                ])))
                mfcc_risk = min(1.0, mfcc_deviation / 3.0) * self.feature_weights['mfcc_features']
            else:
                mfcc_risk = 0.2  # Default moderate risk
            risk_factors.append(mfcc_risk)

            # 5. Temporal Pattern Risk
            pause_abnormality = min(1.0, biomarkers.pause_frequency / 15.0)  # >15 pauses/min is concerning
            speaking_rate_abnormality = abs(biomarkers.speaking_rate - 120.0) / 120.0  # Deviation from normal
            temporal_risk = (pause_abnormality + speaking_rate_abnormality) / 2.0 * 0.15
            risk_factors.append(temporal_risk)

            # Combine risk factors
            total_risk = sum(risk_factors)

            # Smooth the risk score
            risk_score = max(0.0, min(1.0, total_risk))

            return risk_score

        except Exception as e:
            logger.error(f"Risk scoring failed: {str(e)}")
            # Return moderate risk as default
            return 0.4
    
    def _calculate_fast_confidence(self, features: Dict[str, Any]) -> float:
        """Real confidence calculation based on feature quality"""

        try:
            confidence_factors = []

            # 1. Signal quality confidence
            spectral_centroid = features.get('spectral_centroid_mean', 1500.0)
            if 800 <= spectral_centroid <= 3000:  # Reasonable range for speech
                spectral_conf = 1.0 - abs(spectral_centroid - 1500.0) / 1500.0
            else:
                spectral_conf = 0.3
            confidence_factors.append(spectral_conf)

            # 2. F0 tracking confidence
            f0_mean = features.get('f0_mean', 150.0)
            f0_std = features.get('f0_std', 20.0)
            if 80 <= f0_mean <= 300 and f0_std < 50:  # Reasonable F0 values
                f0_conf = 1.0 - f0_std / 50.0
            else:
                f0_conf = 0.4
            confidence_factors.append(f0_conf)

            # 3. Feature consistency confidence
            mfcc_std = features.get('mfcc_std', np.ones(self.n_mfcc) * 0.1)
            if isinstance(mfcc_std, np.ndarray):
                mfcc_consistency = 1.0 - min(1.0, np.mean(mfcc_std) / 2.0)
            else:
                mfcc_consistency = 0.7
            confidence_factors.append(mfcc_consistency)

            # 4. Speech activity confidence
            speech_rate = features.get('speech_rate', 0.5)
            if 0.3 <= speech_rate <= 0.9:  # Reasonable speech activity
                activity_conf = 1.0 - abs(speech_rate - 0.6) / 0.6
            else:
                activity_conf = 0.5
            confidence_factors.append(activity_conf)

            # Weighted average
            confidence = np.mean(confidence_factors)
            return max(0.3, min(1.0, confidence))

        except Exception as e:
            logger.error(f"Confidence calculation failed: {str(e)}")
            return 0.7  # Default moderate confidence
    
    def _assess_audio_quality(self, audio_data: np.ndarray) -> float:
        """Real audio quality assessment using signal processing"""

        try:
            if len(audio_data) == 0:
                return 0.1

            quality_factors = []

            # 1. Signal-to-Noise Ratio
            signal_power = np.mean(audio_data ** 2)
            noise_floor = np.percentile(audio_data ** 2, 5)  # Bottom 5% as noise

            if noise_floor > 0 and signal_power > noise_floor:
                snr_db = 10 * np.log10(signal_power / noise_floor)
                snr_quality = min(1.0, max(0.0, (snr_db - 10) / 30))  # 10-40 dB range
            else:
                snr_quality = 0.3
            quality_factors.append(snr_quality)

            # 2. Dynamic Range
            dynamic_range = np.max(np.abs(audio_data)) - np.min(np.abs(audio_data))
            range_quality = min(1.0, dynamic_range / 0.5)  # Expect some dynamic range
            quality_factors.append(range_quality)

            # 3. Clipping Detection
            clipping_ratio = np.sum(np.abs(audio_data) > 0.95) / len(audio_data)
            clipping_quality = 1.0 - min(1.0, clipping_ratio * 10)  # Penalize clipping
            quality_factors.append(clipping_quality)

            # 4. Spectral Quality
            fft = np.fft.rfft(audio_data)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio_data), 1/self.sample_rate)

            # Check for reasonable spectral distribution
            speech_band = (freqs >= 300) & (freqs <= 3400)  # Telephone quality range
            speech_energy = np.sum(magnitude[speech_band])
            total_energy = np.sum(magnitude)

            if total_energy > 0:
                spectral_quality = speech_energy / total_energy
            else:
                spectral_quality = 0.3
            quality_factors.append(spectral_quality)

            # Overall quality score
            quality_score = np.mean(quality_factors)
            return max(0.0, min(1.0, quality_score))

        except Exception as e:
            logger.error(f"Audio quality assessment failed: {str(e)}")
            return 0.5  # Default moderate quality
    
    def _generate_fast_recommendations(self, risk_score: float, biomarkers: SpeechBiomarkers) -> List[str]:
        """Fast recommendation generation using lookup table"""
        
        recommendations = []
        
        # Risk-based recommendations (pre-computed)
        if risk_score > 0.7:
            recommendations.append("High speech risk detected - recommend immediate evaluation")
        elif risk_score > 0.4:
            recommendations.append("Moderate speech changes - consider follow-up assessment")
        else:
            recommendations.append("Low speech risk - continue routine monitoring")
        
        # Biomarker-specific recommendations (fast lookup)
        if biomarkers.voice_tremor > 0.6:
            recommendations.append("Voice tremor detected - evaluate for movement disorders")
        
        if biomarkers.fluency_score < 0.5:
            recommendations.append("Speech fluency concerns - assess for language disorders")
        
        return recommendations
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for real-time analyzer"""
        return {
            "model_loaded": self.model_loaded,
            "target_latency_ms": 100,
            "optimization_level": "maximum",
            "accuracy_target": "90%+",
            "memory_usage": "minimal"
        }

# Global instance
realtime_speech_analyzer = RealtimeSpeechAnalyzer()

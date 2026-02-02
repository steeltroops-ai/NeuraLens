"""
Speech Pipeline v4.0 Unit Tests
Tests the new v4 components: quality gate, features pipeline, clinical, streaming.

Run with: pytest tests/pipelines/speech/test_v4_pipeline.py -v
"""

import pytest
import numpy as np
import io
import wave
from typing import Dict


def generate_test_audio(duration_sec: float = 5.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate synthetic voice-like audio for testing."""
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, dtype=np.float32)
    
    # Base frequency with slight variation (simulate F0)
    f0 = 150 + 20 * np.sin(2 * np.pi * 3 * t)  # ~150 Hz with modulation
    phase = np.cumsum(2 * np.pi * f0 / sample_rate)
    audio = 0.5 * np.sin(phase)
    
    # Add harmonics
    audio += 0.25 * np.sin(2 * phase)  # 2nd harmonic
    audio += 0.12 * np.sin(3 * phase)  # 3rd harmonic
    audio += 0.06 * np.sin(4 * phase)  # 4th harmonic
    
    # Add amplitude modulation (simulate speech envelope)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t) ** 2
    audio = audio * envelope
    
    # Add small amount of noise
    audio += 0.02 * np.random.randn(n_samples).astype(np.float32)
    
    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-10) * 0.8
    
    return audio.astype(np.float32)


def generate_test_wav_bytes(duration_sec: float = 5.0, sample_rate: int = 16000) -> bytes:
    """Generate WAV file bytes from test audio."""
    audio = generate_test_audio(duration_sec, sample_rate)
    audio_int16 = (audio * 32767).astype(np.int16)
    
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    
    buffer.seek(0)
    return buffer.read()


class TestSignalQualityAnalyzer:
    """Test SignalQualityAnalyzer component."""
    
    def test_analyzer_creation(self):
        from app.pipelines.speech.quality.analyzer import SignalQualityAnalyzer
        
        analyzer = SignalQualityAnalyzer(sample_rate=16000)
        assert analyzer.sample_rate == 16000
    
    def test_analyze_normal_audio(self):
        from app.pipelines.speech.quality.analyzer import SignalQualityAnalyzer
        
        analyzer = SignalQualityAnalyzer()
        audio = generate_test_audio(duration_sec=3.0)
        
        metrics = analyzer.analyze(audio)
        
        assert metrics.quality_score >= 0
        assert metrics.quality_score <= 100
        assert metrics.snr_db is not None
        assert metrics.clipping_ratio is not None
    
    def test_detect_clipping(self):
        from app.pipelines.speech.quality.analyzer import SignalQualityAnalyzer
        
        analyzer = SignalQualityAnalyzer()
        
        # Create clipped audio
        audio = generate_test_audio(duration_sec=2.0)
        audio = audio * 2  # Amplify to clip
        audio = np.clip(audio, -1.0, 1.0)
        
        metrics = analyzer.analyze(audio)
        
        assert metrics.clipping_ratio > 0


class TestSpeechContentDetector:
    """Test SpeechContentDetector component."""
    
    def test_detector_creation(self):
        from app.pipelines.speech.quality.detector import SpeechContentDetector
        
        detector = SpeechContentDetector(sample_rate=16000)
        assert detector.sample_rate == 16000
    
    def test_detect_speech_in_audio(self):
        from app.pipelines.speech.quality.detector import SpeechContentDetector
        
        detector = SpeechContentDetector()
        audio = generate_test_audio(duration_sec=3.0)
        
        metrics = detector.detect(audio)
        
        assert metrics.speech_ratio >= 0
        assert metrics.speech_ratio <= 1
        assert metrics.total_duration > 0
    
    def test_detect_silence(self):
        from app.pipelines.speech.quality.detector import SpeechContentDetector
        
        detector = SpeechContentDetector(min_speech_ratio=0.5)
        
        # Create silent audio
        audio = np.zeros(16000 * 3, dtype=np.float32)
        
        metrics = detector.detect(audio)
        
        assert metrics.speech_ratio < 0.1
        assert not metrics.has_adequate_speech


class TestFormatValidator:
    """Test FormatValidator component."""
    
    def test_validator_creation(self):
        from app.pipelines.speech.quality.validator import FormatValidator
        
        validator = FormatValidator()
        assert validator.target_sample_rate == 16000
    
    def test_validate_wav_format(self):
        from app.pipelines.speech.quality.validator import FormatValidator
        
        validator = FormatValidator()
        wav_bytes = generate_test_wav_bytes(duration_sec=3.0)
        
        result = validator.validate(wav_bytes, filename="test.wav")
        
        assert result.is_valid
        assert result.format_detected == "wav"
        assert result.converted_audio is not None
    
    def test_detect_format_from_magic_bytes(self):
        from app.pipelines.speech.quality.validator import FormatValidator
        
        validator = FormatValidator()
        wav_bytes = generate_test_wav_bytes()
        
        # WAV files start with RIFF
        assert wav_bytes[:4] == b'RIFF'
        
        result = validator.validate(wav_bytes)
        assert result.format_detected == "wav"


class TestEnhancedQualityGate:
    """Test EnhancedQualityGate component."""
    
    def test_gate_creation(self):
        from app.pipelines.speech.quality.gate import EnhancedQualityGate
        
        gate = EnhancedQualityGate(
            sample_rate=16000,
            min_snr_db=15.0,
            max_clipping_ratio=0.05
        )
        assert gate.min_snr_db == 15.0
    
    def test_validate_good_audio(self):
        from app.pipelines.speech.quality.gate import EnhancedQualityGate
        
        gate = EnhancedQualityGate()
        wav_bytes = generate_test_wav_bytes(duration_sec=5.0)
        
        report = gate.validate_audio_bytes(wav_bytes, filename="test.wav")
        
        assert report.quality_score >= 0
        assert report.format_result is not None
    
    def test_validate_audio_array(self):
        from app.pipelines.speech.quality.gate import EnhancedQualityGate
        
        gate = EnhancedQualityGate()
        audio = generate_test_audio(duration_sec=5.0)
        
        report = gate.validate_audio_array(audio)
        
        assert report.quality_score >= 0
        assert report.signal_metrics is not None
        assert report.speech_metrics is not None


class TestUnifiedFeatureExtractor:
    """Test UnifiedFeatureExtractor component."""
    
    def test_extractor_creation(self):
        from app.pipelines.speech.features.pipeline import UnifiedFeatureExtractor
        
        extractor = UnifiedFeatureExtractor(sample_rate=16000)
        assert extractor.sample_rate == 16000
    
    def test_extract_features(self):
        from app.pipelines.speech.features.pipeline import UnifiedFeatureExtractor
        
        extractor = UnifiedFeatureExtractor()
        audio = generate_test_audio(duration_sec=5.0)
        
        result = extractor.extract(audio)
        
        # extract() returns FeatureExtractionResult
        assert result.features is not None
        assert len(result.features) > 0
    
    def test_extract_full_features(self):
        """Test extract_full which returns UnifiedFeatures."""
        from app.pipelines.speech.features.pipeline import UnifiedFeatureExtractor
        
        extractor = UnifiedFeatureExtractor()
        audio = generate_test_audio(duration_sec=5.0)
        
        result = extractor.extract_full(audio)
        
        # extract_full returns UnifiedFeatures
        assert result.acoustic is not None
        assert result.prosodic is not None
    
    def test_extract_contours(self):
        from app.pipelines.speech.features.pipeline import UnifiedFeatureExtractor
        
        extractor = UnifiedFeatureExtractor()
        audio = generate_test_audio(duration_sec=3.0)
        
        # extract_contours is a private method, test via extract_full
        result = extractor.extract_full(audio)
        
        assert result.f0_contour is not None
        assert len(result.f0_contour) > 0


class TestUncertaintyEstimator:
    """Test UncertaintyEstimator component."""
    
    def test_estimator_creation(self):
        from app.pipelines.speech.clinical.uncertainty import UncertaintyEstimator
        
        estimator = UncertaintyEstimator(n_samples=10)
        assert estimator.n_samples == 10
    
    def test_estimate_uncertainty(self):
        from app.pipelines.speech.clinical.uncertainty import UncertaintyEstimator
        
        estimator = UncertaintyEstimator(n_samples=20)
        
        features = {
            "jitter": 0.02,
            "shimmer": 0.04,
            "hnr": 20.0,
            "cpps": 15.0
        }
        
        def mock_score_fn(f):
            return 30.0 + np.random.randn() * 5
        
        result = estimator.estimate(features, mock_score_fn)
        
        assert result.mean_score >= 0
        assert result.std_score >= 0
        assert result.ci_95 is not None
        assert len(result.ci_95) == 2


class TestRiskExplainer:
    """Test RiskExplainer component."""
    
    def test_explainer_creation(self):
        from app.pipelines.speech.clinical.explainer import RiskExplainer
        
        explainer = RiskExplainer(n_perturbations=10)
        assert explainer.n_perturbations == 10
    
    def test_explain_risk(self):
        from app.pipelines.speech.clinical.explainer import RiskExplainer
        
        explainer = RiskExplainer(n_perturbations=15)
        
        features = {
            "jitter": 0.05,
            "shimmer": 0.08,
            "hnr": 15.0,
            "speech_rate": 3.5
        }
        
        normal_ranges = {
            "jitter": (0.0, 0.04),
            "shimmer": (0.0, 0.06),
            "hnr": (18.0, 25.0),
            "speech_rate": (3.5, 6.0)
        }
        
        def mock_score_fn(f):
            score = 20.0
            score += (f.get("jitter", 0) - 0.02) * 500
            score += (f.get("shimmer", 0) - 0.04) * 300
            return max(0, min(100, score))
        
        result = explainer.explain(features, mock_score_fn, normal_ranges)
        
        # RiskExplanation has 'contributions' not 'feature_contributions'
        assert result.contributions is not None
        assert len(result.contributions) > 0


class TestNormativeDataManager:
    """Test NormativeDataManager component."""
    
    def test_manager_creation(self):
        from app.pipelines.speech.clinical.normative import NormativeDataManager
        
        manager = NormativeDataManager()
        assert len(manager.biomarker_data) > 0
    
    def test_get_reference(self):
        from app.pipelines.speech.clinical.normative import NormativeDataManager
        
        manager = NormativeDataManager()
        
        # Use jitter_local (actual feature name in normative data)
        ref = manager.get_reference("jitter_local")
        
        assert ref is not None
        assert ref.mean is not None
        assert ref.std is not None
    
    def test_calculate_z_score(self):
        from app.pipelines.speech.clinical.normative import NormativeDataManager
        
        manager = NormativeDataManager()
        
        # Normal jitter_local value (use actual feature name)
        z = manager.calculate_z_score("jitter_local", 0.5)
        
        assert z is not None
        assert -5 < z < 5  # Should be within reasonable range


class TestStreamProcessor:
    """Test StreamProcessor component."""
    
    def test_processor_creation(self):
        from app.pipelines.speech.streaming.processor import StreamProcessor
        
        processor = StreamProcessor(sample_rate=16000)
        assert processor.sample_rate == 16000
    
    def test_process_chunk(self):
        from app.pipelines.speech.streaming.processor import StreamProcessor
        
        processor = StreamProcessor()
        chunk = generate_test_audio(duration_sec=0.2)
        
        result = processor.process_chunk(chunk)
        
        assert result.processing_time_ms >= 0
        assert result.rms_level >= 0
        assert result.preliminary_quality_score >= 0


class TestStreamingSessionManager:
    """Test StreamingSessionManager component."""
    
    def test_create_session(self):
        from app.pipelines.speech.streaming.session import StreamingSessionManager
        
        manager = StreamingSessionManager()
        session = manager.create_session(user_id="test_user")
        
        assert session.session_id is not None
        assert session.user_id == "test_user"
    
    def test_session_lifecycle(self):
        from app.pipelines.speech.streaming.session import (
            StreamingSessionManager, SessionState
        )
        
        manager = StreamingSessionManager()
        session = manager.create_session()
        
        assert session.state == SessionState.READY
        
        manager.start_recording(session.session_id)
        session = manager.get_session(session.session_id)
        assert session.state == SessionState.RECORDING
        
        manager.stop_recording(session.session_id)
        session = manager.get_session(session.session_id)
        assert session.state == SessionState.PROCESSING
    
    def test_add_audio_chunk(self):
        from app.pipelines.speech.streaming.session import StreamingSessionManager
        
        manager = StreamingSessionManager()
        session = manager.create_session()
        manager.start_recording(session.session_id)
        
        chunk = generate_test_audio(duration_sec=0.2)
        result = manager.add_audio_chunk(session.session_id, chunk)
        
        assert result == True
        session = manager.get_session(session.session_id)
        assert session.metrics.chunks_received == 1


class TestIntegration:
    """Integration tests for complete pipeline flow."""
    
    def test_quality_to_features_flow(self):
        """Test quality gate -> feature extraction flow."""
        from app.pipelines.speech.quality.gate import EnhancedQualityGate
        from app.pipelines.speech.features.pipeline import UnifiedFeatureExtractor
        
        gate = EnhancedQualityGate()
        extractor = UnifiedFeatureExtractor()
        
        wav_bytes = generate_test_wav_bytes(duration_sec=5.0)
        
        # Quality gate
        quality_report = gate.validate_audio_bytes(wav_bytes, filename="test.wav")
        assert quality_report.format_result.converted_audio is not None
        
        # Feature extraction (extract returns FeatureExtractionResult)
        audio = quality_report.format_result.converted_audio
        result = extractor.extract(audio)
        
        # Check result has features dict
        assert result.features is not None
        assert len(result.features) > 0
    
    def test_streaming_flow(self):
        """Test complete streaming session flow."""
        from app.pipelines.speech.streaming.session import StreamingSessionManager
        from app.pipelines.speech.streaming.processor import StreamProcessor
        
        manager = StreamingSessionManager()
        processor = StreamProcessor()
        
        # Create session
        session = manager.create_session()
        manager.start_recording(session.session_id)
        
        # Process chunks - need at least 3 seconds (15 chunks x 0.2s = 3s)
        num_chunks = 20  # 4 seconds total
        for i in range(num_chunks):
            chunk = generate_test_audio(duration_sec=0.2)
            manager.add_audio_chunk(session.session_id, chunk)
            result = processor.process_chunk(chunk)
            assert result.processing_time_ms < 200  # Sub-200ms target
        
        # Stop and check
        manager.stop_recording(session.session_id)
        session = manager.get_session(session.session_id)
        
        assert session.metrics.chunks_received == num_chunks
        assert session.has_enough_audio  # Requires >= 3 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

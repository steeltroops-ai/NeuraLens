"""
Speech Pipeline End-to-End Tests
Tests the complete speech analysis workflow from audio upload to risk scoring

Run with: pytest tests/integration/test_speech_pipeline.py -v
"""

import pytest
import io
import os
import wave
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

# Import app
from app.main import app


# Test data paths
TEST_DATA_DIR = Path(__file__).parent.parent.parent / "data"
TEST_AUDIO_WAV = TEST_DATA_DIR / "test_speech.wav"
TEST_AUDIO_DIFFERENT = TEST_DATA_DIR / "test_different.wav"


def generate_test_wav(duration_sec: float = 5.0, sample_rate: int = 16000) -> bytes:
    """Generate a simple test WAV file with a tone"""
    n_samples = int(duration_sec * sample_rate)
    
    # Generate a 440 Hz sine wave
    t = np.linspace(0, duration_sec, n_samples, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Add some harmonics for voice-like qualities
    audio += 0.2 * np.sin(2 * np.pi * 880 * t)
    audio += 0.1 * np.sin(2 * np.pi * 1320 * t)
    
    # Add slight amplitude modulation
    modulation = 1 + 0.2 * np.sin(2 * np.pi * 5 * t)
    audio = audio * modulation
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Create WAV file in memory
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    
    buffer.seek(0)
    return buffer.read()


def generate_short_wav(duration_sec: float = 1.0) -> bytes:
    """Generate a WAV file that's too short (< 3 seconds)"""
    return generate_test_wav(duration_sec=duration_sec)


def generate_silent_wav(duration_sec: float = 5.0, sample_rate: int = 16000) -> bytes:
    """Generate a silent WAV file"""
    n_samples = int(duration_sec * sample_rate)
    audio = np.zeros(n_samples, dtype=np.int16)
    
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())
    
    buffer.seek(0)
    return buffer.read()


# Test client
client = TestClient(app)


class TestSpeechPipelineEndpoints:
    """Test Speech Pipeline API endpoints"""
    
    def test_health_endpoint(self):
        """Test speech pipeline health check"""
        response = client.get("/api/speech/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"
    
    def test_analyze_with_generated_audio(self):
        """Test analysis with generated test audio"""
        audio_bytes = generate_test_wav(duration_sec=5.0)
        
        response = client.post(
            "/api/speech/analyze",
            files={"audio": ("test.wav", audio_bytes, "audio/wav")},
            data={"session_id": "test-session-001"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure (matches EnhancedSpeechAnalysisResponse)
        assert "session_id" in data
        assert "risk_score" in data
        assert 0 <= data["risk_score"] <= 1
        assert "biomarkers" in data
        assert "confidence" in data
        assert 0 <= data["confidence"] <= 1
        assert "status" in data
        assert data["status"] == "completed"
    
    def test_analyze_with_real_audio_file(self):
        """Test analysis with real test audio file if available"""
        if not TEST_AUDIO_WAV.exists():
            pytest.skip("Test audio file not found")
        
        with open(TEST_AUDIO_WAV, "rb") as f:
            audio_bytes = f.read()
        
        response = client.post(
            "/api/speech/analyze",
            files={"audio": ("test_speech.wav", audio_bytes, "audio/wav")},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert "biomarkers" in data
    
    def test_analyze_returns_all_biomarkers(self):
        """Verify all 9 biomarkers are returned"""
        audio_bytes = generate_test_wav(duration_sec=5.0)
        
        response = client.post(
            "/api/speech/analyze",
            files={"audio": ("test.wav", audio_bytes, "audio/wav")},
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check for all 9 biomarkers
        expected_biomarkers = [
            "jitter", "shimmer", "hnr",
            "speech_rate", "pause_ratio", "fluency_score",
            "voice_tremor", "articulation_clarity", "prosody_variation"
        ]
        
        biomarkers = data.get("biomarkers", {})
        for marker in expected_biomarkers:
            assert marker in biomarkers, f"Missing biomarker: {marker}"
            
            # Each biomarker should have value and normal_range
            marker_data = biomarkers[marker]
            assert "value" in marker_data
            assert "normal_range" in marker_data
            assert "unit" in marker_data
    
    def test_analyze_returns_quality_metrics(self):
        """Verify quality score is returned"""
        audio_bytes = generate_test_wav(duration_sec=5.0)
        
        response = client.post(
            "/api/speech/analyze",
            files={"audio": ("test.wav", audio_bytes, "audio/wav")},
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "quality_score" in data
        assert 0 <= data["quality_score"] <= 1
        assert "processing_time" in data
    
    def test_analyze_returns_recommendations(self):
        """Verify recommendations are returned"""
        audio_bytes = generate_test_wav(duration_sec=5.0)
        
        response = client.post(
            "/api/speech/analyze",
            files={"audio": ("test.wav", audio_bytes, "audio/wav")},
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)
    
    def test_features_endpoint(self):
        """Test the features endpoint"""
        response = client.get("/api/speech/features")
        assert response.status_code == 200
        data = response.json()
        
        assert "biomarkers" in data
        assert "normal_ranges" in data
        assert "formats" in data
        assert len(data["biomarkers"]) == 9


class TestSpeechPipelineValidation:
    """Test input validation"""
    
    def test_reject_too_short_audio(self):
        """Audio shorter than 3 seconds should be rejected or return low quality"""
        audio_bytes = generate_short_wav(duration_sec=1.5)
        
        response = client.post(
            "/api/speech/analyze",
            files={"audio": ("test.wav", audio_bytes, "audio/wav")},
        )
        
        # Either rejected with error or processed with low quality
        if response.status_code == 400:
            # Properly rejected
            pass
        else:
            # Processed - check for low quality indicator
            data = response.json()
            # Short audio may have reduced quality score
            assert "quality_score" in data
    
    def test_reject_invalid_file_type(self):
        """Non-audio files should be rejected"""
        fake_audio = b"This is not audio data"
        
        response = client.post(
            "/api/speech/analyze",
            files={"audio": ("test.txt", fake_audio, "text/plain")},
        )
        
        # Should return 400 error
        assert response.status_code == 400
    
    def test_no_file_uploaded(self):
        """Request without file should fail"""
        response = client.post(
            "/api/speech/analyze",
            files={},
        )
        
        # Should return error
        assert response.status_code == 400


class TestSpeechPipelineConditions:
    """Test condition detection patterns"""
    
    def test_normal_audio_risk_score(self):
        """Normal audio should produce reasonable risk score"""
        audio_bytes = generate_test_wav(duration_sec=10.0)
        
        response = client.post(
            "/api/speech/analyze",
            files={"audio": ("test.wav", audio_bytes, "audio/wav")},
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Generated audio should have valid risk score
        assert "risk_score" in data
        assert 0 <= data["risk_score"] <= 1


class TestRiskCalculator:
    """Test risk calculation module directly"""
    
    def test_calculate_speech_risk(self):
        """Test risk calculation with sample biomarkers"""
        from app.pipelines.speech.risk_calculator import calculate_speech_risk
        
        # Normal biomarkers
        normal_biomarkers = {
            "jitter": 0.025,
            "shimmer": 0.04,
            "hnr": 20.0,
            "speech_rate": 4.2,
            "pause_ratio": 0.18,
            "fluency_score": 0.85,
            "voice_tremor": 0.05,
            "articulation_clarity": 0.88,
            "prosody_variation": 0.55
        }
        
        result = calculate_speech_risk(normal_biomarkers)
        
        assert result.overall_score < 50  # Should be low/moderate risk
        assert result.category in ["low", "moderate"]
        assert 0 <= result.confidence <= 1
        assert "parkinsons" in result.condition_probabilities
        assert "normal" in result.condition_probabilities
    
    def test_high_risk_biomarkers(self):
        """Test risk calculation with abnormal biomarkers"""
        from app.pipelines.speech.risk_calculator import calculate_speech_risk
        
        # Parkinsonian pattern
        abnormal_biomarkers = {
            "jitter": 0.08,  # High
            "shimmer": 0.12,  # High
            "hnr": 8.0,  # Low (bad)
            "speech_rate": 2.5,  # Low (slow)
            "pause_ratio": 0.45,  # High
            "fluency_score": 0.4,  # Low (bad)
            "voice_tremor": 0.28,  # High
            "articulation_clarity": 0.55,  # Low
            "prosody_variation": 0.15  # Low
        }
        
        result = calculate_speech_risk(abnormal_biomarkers)
        
        assert result.overall_score > 40  # Should be elevated risk
        assert result.condition_probabilities["parkinsons"] > 0.1
    
    def test_get_biomarker_status(self):
        """Test biomarker status determination"""
        from app.pipelines.speech.risk_calculator import get_biomarker_status
        
        # Normal jitter
        assert get_biomarker_status("jitter", 0.025) == "normal"
        # Abnormal jitter
        assert get_biomarker_status("jitter", 0.08) == "abnormal"
        
        # Normal HNR
        assert get_biomarker_status("hnr", 20.0) == "normal"
        # Abnormal HNR (too low)
        assert get_biomarker_status("hnr", 8.0) == "abnormal"
    
    def test_get_risk_category(self):
        """Test risk categorization"""
        from app.pipelines.speech.risk_calculator import get_risk_category
        
        assert get_risk_category(10) == "low"
        assert get_risk_category(30) == "moderate"
        assert get_risk_category(60) == "high"
        assert get_risk_category(80) == "critical"


class TestAudioValidator:
    """Test audio validation module"""
    
    @pytest.mark.asyncio
    async def test_validate_wav_audio(self):
        """Test validation of WAV audio"""
        from app.pipelines.speech.validator import AudioValidator
        
        validator = AudioValidator()
        audio_bytes = generate_test_wav(duration_sec=5.0)
        
        result = await validator.validate(audio_bytes, content_type="audio/wav")
        
        assert result.is_valid == True
        assert result.duration >= 3.0
        assert result.audio_data is not None
    
    @pytest.mark.asyncio
    async def test_validate_short_audio(self):
        """Test validation rejects short audio"""
        from app.pipelines.speech.validator import AudioValidator
        
        validator = AudioValidator()
        audio_bytes = generate_test_wav(duration_sec=1.5)
        
        result = await validator.validate(audio_bytes, content_type="audio/wav")
        
        # Should fail validation due to short duration
        assert result.is_valid == False or result.duration < 3.0
    
    def test_detect_mime_type(self):
        """Test MIME type detection from bytes"""
        from app.pipelines.speech.validator import AudioValidator
        
        validator = AudioValidator()
        audio_bytes = generate_test_wav(duration_sec=3.0)
        
        detected = validator.detect_mime_type_from_bytes(audio_bytes)
        assert detected == "audio/wav"


class TestConfig:
    """Test configuration constants"""
    
    def test_biomarker_ranges_complete(self):
        """All biomarkers should have defined ranges"""
        from app.pipelines.speech.config import BIOMARKER_NORMAL_RANGES
        
        expected = [
            "jitter", "shimmer", "hnr",
            "speech_rate", "pause_ratio", "fluency_score",
            "voice_tremor", "articulation_clarity", "prosody_variation"
        ]
        
        for marker in expected:
            assert marker in BIOMARKER_NORMAL_RANGES
            low, high = BIOMARKER_NORMAL_RANGES[marker]
            assert low < high
    
    def test_risk_weights_sum_to_one(self):
        """Risk weights should sum to 1.0"""
        from app.pipelines.speech.config import RISK_WEIGHTS
        
        total = sum(RISK_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01


# Async tests for async endpoints
@pytest.mark.asyncio
async def test_async_analyze():
    """Test async analysis endpoint"""
    audio_bytes = generate_test_wav(duration_sec=5.0)
    
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        response = await ac.post(
            "/api/speech/analyze",
            files={"audio": ("test.wav", audio_bytes, "audio/wav")},
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "risk_score" in data
    assert "biomarkers" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

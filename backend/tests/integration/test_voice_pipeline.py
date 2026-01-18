"""
Voice Pipeline End-to-End Tests
Tests ElevenLabs TTS integration for speaking medical results

Run with: pytest tests/integration/test_voice_pipeline.py -v
"""

import pytest
import base64
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from app.main import app


# Test client
client = TestClient(app)


class TestVoicePipelineEndpoints:
    """Test Voice Pipeline API endpoints"""
    
    def test_health_endpoint(self):
        """Test voice pipeline health check"""
        response = client.get("/api/voice/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["module"] == "voice"
        assert "provider" in data
        assert "elevenlabs_available" in data
        assert "gtts_available" in data
        assert "default_voice" in data
    
    def test_voices_endpoint(self):
        """Test listing available voices"""
        response = client.get("/api/voice/voices")
        assert response.status_code == 200
        data = response.json()
        assert "voices" in data
        assert "provider" in data
        assert "default" in data
        assert isinstance(data["voices"], list)
        
        # Should have predefined voices
        if len(data["voices"]) > 0:
            voice = data["voices"][0]
            assert "id" in voice
            assert "name" in voice
            assert "voice_id" in voice
    
    def test_speak_endpoint(self):
        """Test basic text-to-speech"""
        response = client.post(
            "/api/voice/speak",
            json={
                "text": "Hello, this is a test.",
                "voice": "rachel"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        assert "audio_base64" in data
        assert data["format"] == "mp3"
        assert "voice_used" in data
        assert "characters_used" in data
        assert data["characters_used"] > 0
        assert "provider" in data
        
        # Verify audio is valid base64
        try:
            audio_bytes = base64.b64decode(data["audio_base64"])
            assert len(audio_bytes) > 0
        except Exception:
            pytest.fail("audio_base64 is not valid base64")
    
    def test_speak_with_medical_text(self):
        """Test TTS with medical terminology"""
        response = client.post(
            "/api/voice/speak",
            json={
                "text": "Your HRV is 45ms and BP is 120/80 mmHg. NRI score is 23.",
                "voice": "rachel"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "audio_base64" in data
    
    def test_speak_raw_audio(self):
        """Test getting raw audio file instead of base64"""
        response = client.post(
            "/api/voice/speak/audio",
            json={
                "text": "Testing raw audio response.",
                "voice": "rachel"
            }
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"
        assert len(response.content) > 0
        
        # Check for MP3 magic bytes (ID3 or frame sync)
        content = response.content
        assert content[:3] == b'ID3' or content[:2] in [b'\xff\xfb', b'\xff\xfa', b'\xff\xf3']
    
    def test_speak_empty_text_fails(self):
        """Empty text should fail validation"""
        response = client.post(
            "/api/voice/speak",
            json={"text": ""}
        )
        assert response.status_code == 422  # Validation error
    
    def test_speak_too_long_text_fails(self):
        """Text over 5000 chars should fail validation"""
        long_text = "a" * 5001
        response = client.post(
            "/api/voice/speak",
            json={"text": long_text}
        )
        assert response.status_code == 422  # Validation error
    
    def test_speak_different_voices(self):
        """Test TTS with different voices"""
        voices = ["rachel", "josh", "bella", "adam", "george"]
        
        for voice in voices:
            response = client.post(
                "/api/voice/speak",
                json={
                    "text": f"Testing {voice} voice.",
                    "voice": voice
                }
            )
            assert response.status_code == 200, f"Failed for voice: {voice}"


class TestExplainEndpoints:
    """Test explanation endpoints"""
    
    def test_explain_term(self):
        """Test explaining a medical term"""
        response = client.post(
            "/api/voice/explain/term",
            json={
                "term": "jitter",
                "context": "speech_analysis",
                "include_audio": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        assert data["term"] == "jitter"
        assert "explanation" in data
        assert len(data["explanation"]) > 20  # Meaningful explanation
        
        # Audio should be included
        if data.get("audio_base64"):
            assert data["duration_seconds"] is not None
    
    def test_explain_term_without_audio(self):
        """Test explaining term without audio"""
        response = client.post(
            "/api/voice/explain/term",
            json={
                "term": "shimmer",
                "include_audio": False
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        assert "explanation" in data
        assert data["audio_base64"] is None
    
    def test_explain_unknown_term(self):
        """Unknown terms should get generic explanation"""
        response = client.post(
            "/api/voice/explain/term",
            json={"term": "unknownterm123"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "biomarker" in data["explanation"].lower()
    
    def test_explain_result_speech(self):
        """Test explaining speech analysis result"""
        response = client.post(
            "/api/voice/explain/result",
            json={
                "pipeline": "speech",
                "result": {
                    "risk_score": 0.23,
                    "biomarkers": {}
                },
                "voice": "rachel"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "audio_base64" in data
    
    def test_explain_result_retinal(self):
        """Test explaining retinal analysis result"""
        response = client.post(
            "/api/voice/explain/result",
            json={
                "pipeline": "retinal",
                "result": {
                    "risk_score": 15,
                    "conditions": []
                }
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
    
    def test_explain_result_nri(self):
        """Test explaining NRI fusion result"""
        response = client.post(
            "/api/voice/explain/result",
            json={
                "pipeline": "nri",
                "result": {
                    "nri_score": 28,
                    "risk_category": "low"
                }
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
    
    def test_explain_result_invalid_pipeline(self):
        """Invalid pipeline should fail"""
        response = client.post(
            "/api/voice/explain/result",
            json={
                "pipeline": "invalid_pipeline",
                "result": {}
            }
        )
        assert response.status_code == 422  # Validation error


class TestCaching:
    """Test audio caching functionality"""
    
    def test_usage_endpoint(self):
        """Test usage statistics endpoint"""
        response = client.get("/api/voice/usage")
        assert response.status_code == 200
        data = response.json()
        
        assert "elevenlabs" in data
        assert "gtts" in data
        assert "cache" in data
    
    def test_cache_clear(self):
        """Test cache clearing"""
        response = client.post("/api/voice/cache/clear")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
    
    def test_cached_response(self):
        """Test that identical requests use cache"""
        text = "This is a test for caching functionality."
        
        # First request
        response1 = client.post(
            "/api/voice/speak",
            json={"text": text, "voice": "rachel"}
        )
        assert response1.status_code == 200
        data1 = response1.json()
        
        # Second identical request should be cached
        response2 = client.post(
            "/api/voice/speak",
            json={"text": text, "voice": "rachel"}
        )
        assert response2.status_code == 200
        data2 = response2.json()
        
        # Both should succeed
        assert data1["success"] == True
        assert data2["success"] == True
        
        # Second should be cached and faster
        if data2.get("cached"):
            assert data2["cached"] == True


class TestTextProcessor:
    """Test text preprocessing for better TTS"""
    
    def test_preprocess_medical_abbreviations(self):
        """Test that medical abbreviations are expanded"""
        from app.pipelines.voice.processor import preprocess_for_speech
        
        text = "Your BP is 120/80 mmHg and HR is 72 bpm."
        processed = preprocess_for_speech(text)
        
        # Abbreviations should be expanded
        assert "blood pressure" in processed.lower()
        assert "heart rate" in processed.lower()
        assert "beats per minute" in processed.lower()
    
    def test_preprocess_numbers(self):
        """Test number formatting for speech"""
        from app.pipelines.voice.processor import format_numbers_for_speech
        
        text = "Risk is 85% with score of 0.75"
        processed = format_numbers_for_speech(text)
        
        assert "percent" in processed.lower()
        assert "point" in processed.lower()
    
    def test_get_medical_explanation(self):
        """Test medical term explanations"""
        from app.pipelines.voice.processor import get_medical_explanation
        
        explanation = get_medical_explanation("jitter")
        assert len(explanation) > 20
        assert "variation" in explanation.lower() or "pitch" in explanation.lower()
        
        explanation = get_medical_explanation("hnr")
        assert "harmonic" in explanation.lower() or "noise" in explanation.lower()


class TestAudioCache:
    """Test AudioCache class directly"""
    
    def test_cache_set_get(self):
        """Test basic cache operations"""
        from app.pipelines.voice.cache import AudioCache
        
        cache = AudioCache(max_size_mb=1)
        
        text = "Test text"
        voice_id = "test_voice"
        speed = 1.0
        audio = b"fake audio bytes"
        
        # Set
        success = cache.set(text, voice_id, speed, audio, "test")
        assert success == True
        
        # Get
        result = cache.get(text, voice_id, speed)
        assert result is not None
        assert result[0] == audio
        assert result[1] == "test"
    
    def test_cache_miss(self):
        """Test cache miss"""
        from app.pipelines.voice.cache import AudioCache
        
        cache = AudioCache()
        result = cache.get("nonexistent", "voice", 1.0)
        assert result is None
    
    def test_cache_stats(self):
        """Test cache statistics"""
        from app.pipelines.voice.cache import AudioCache
        
        cache = AudioCache()
        stats = cache.get_stats()
        
        assert "entries" in stats
        assert "size_bytes" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate_percent" in stats


class TestUsageTracker:
    """Test UsageTracker class"""
    
    def test_track_usage(self):
        """Test usage tracking"""
        from app.pipelines.voice.cache import UsageTracker
        
        tracker = UsageTracker()
        tracker.track_usage(100, "elevenlabs")
        tracker.track_usage(50, "gtts")
        
        usage = tracker.get_usage()
        assert usage["elevenlabs"]["chars_used"] >= 100
        assert usage["gtts"]["chars_used"] >= 50
    
    def test_should_use_fallback(self):
        """Test fallback decision logic"""
        from app.pipelines.voice.cache import UsageTracker
        
        tracker = UsageTracker()
        
        # Should not use fallback initially
        assert tracker.should_use_fallback() == False
        
        # Simulate heavy usage
        tracker.monthly_chars["elevenlabs"] = 9500
        assert tracker.should_use_fallback() == True


class TestConvenienceFunctions:
    """Test convenience functions for cross-pipeline integration"""
    
    @pytest.mark.asyncio
    async def test_speak_text_function(self):
        """Test speak_text convenience function"""
        from app.pipelines.voice import speak_text
        
        audio_b64 = await speak_text("Testing convenience function")
        
        assert audio_b64 is not None
        assert len(audio_b64) > 0
        
        # Should be valid base64
        audio_bytes = base64.b64decode(audio_b64)
        assert len(audio_bytes) > 0
    
    @pytest.mark.asyncio
    async def test_speak_result_function(self):
        """Test speak_result convenience function"""
        from app.pipelines.voice import speak_result
        
        audio_b64 = await speak_result("speech", {"risk_score": 0.25})
        
        assert audio_b64 is not None
    
    @pytest.mark.asyncio
    async def test_speak_llm_explanation_function(self):
        """Test speak_llm_explanation convenience function"""
        from app.pipelines.voice import speak_llm_explanation
        
        llm_text = "Based on your analysis, your neurological health appears to be within normal range."
        audio_b64 = await speak_llm_explanation(llm_text)
        
        assert audio_b64 is not None
    
    @pytest.mark.asyncio
    async def test_get_audio_bytes_function(self):
        """Test get_audio_bytes convenience function"""
        from app.pipelines.voice import get_audio_bytes
        
        audio_bytes = await get_audio_bytes("Testing audio bytes")
        
        assert audio_bytes is not None
        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) > 0


# Async endpoint tests
@pytest.mark.asyncio
async def test_async_speak():
    """Test async speak endpoint"""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        response = await ac.post(
            "/api/voice/speak",
            json={"text": "Async test", "voice": "rachel"}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True


@pytest.mark.asyncio
async def test_async_explain_term():
    """Test async explain term endpoint"""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        response = await ac.post(
            "/api/voice/explain/term",
            json={"term": "tremor", "include_audio": False}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert "explanation" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

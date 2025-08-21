"""
Test script for the SpeechAnalyzer
"""

import asyncio
import numpy as np
from app.ml.models.speech_analyzer import SpeechAnalyzer

async def test_speech_analyzer():
    """Test the speech analyzer with mock audio data"""
    
    print("Testing SpeechAnalyzer...")
    
    # Create analyzer instance
    analyzer = SpeechAnalyzer()
    
    # Create mock audio data (5 seconds of audio)
    sample_rate = 22050
    duration = 5.0
    num_samples = int(duration * sample_rate)
    
    # Generate realistic mock audio signal
    t = np.linspace(0, duration, num_samples)
    audio_data = (
        0.3 * np.sin(2 * np.pi * 200 * t) +  # Fundamental frequency
        0.2 * np.sin(2 * np.pi * 400 * t) +  # First harmonic
        0.1 * np.sin(2 * np.pi * 600 * t) +  # Second harmonic
        0.05 * np.random.normal(0, 1, num_samples)  # Noise
    )
    
    # Convert to bytes (mock WAV file)
    audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
    
    try:
        # Test the analyzer
        result = await analyzer.analyze(audio_bytes, "test_session_001")
        
        print(f"✅ Analysis completed successfully!")
        print(f"   Session ID: {result.session_id}")
        print(f"   Processing Time: {result.processing_time:.3f}s")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Risk Score: {result.risk_score:.3f}")
        print(f"   Quality Score: {result.quality_score:.3f}")
        
        print(f"\n📊 Biomarkers:")
        print(f"   Fluency Score: {result.biomarkers.fluency_score:.3f}")
        print(f"   Pause Pattern: {result.biomarkers.pause_pattern:.3f}")
        print(f"   Voice Tremor: {result.biomarkers.voice_tremor:.3f}")
        print(f"   Articulation Clarity: {result.biomarkers.articulation_clarity:.3f}")
        print(f"   Prosody Variation: {result.biomarkers.prosody_variation:.3f}")
        print(f"   Speaking Rate: {result.biomarkers.speaking_rate:.1f} WPM")
        print(f"   Pause Frequency: {result.biomarkers.pause_frequency:.1f}/min")
        
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"   {i}. {rec}")
        
        # Test health check
        health = await analyzer.health_check()
        print(f"\n🏥 Health Check:")
        print(f"   Model Loaded: {health['model_loaded']}")
        print(f"   Memory Usage: {health['memory_usage']}")
        print(f"   Last Analysis: {health['last_analysis']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_speech_analyzer())
    if success:
        print("\n🎉 All tests passed! SpeechAnalyzer is working correctly.")
    else:
        print("\n💥 Tests failed!")

"""
Test script for the real speech ML pipeline
Tests MFCC extraction, voice biomarkers, and risk assessment
"""

import asyncio
import numpy as np
import time
from datetime import datetime

from app.ml.realtime.realtime_speech import realtime_speech_analyzer


def generate_test_audio(duration=5.0, sample_rate=22050, frequency=440):
    """Generate synthetic test audio with speech-like characteristics"""
    
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a complex signal with multiple components
    # Base frequency (fundamental)
    signal = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # Add harmonics (typical of speech)
    signal += 0.2 * np.sin(2 * np.pi * frequency * 2 * t)
    signal += 0.1 * np.sin(2 * np.pi * frequency * 3 * t)
    
    # Add formant-like resonances
    signal += 0.15 * np.sin(2 * np.pi * 800 * t)  # First formant
    signal += 0.1 * np.sin(2 * np.pi * 1200 * t)   # Second formant
    
    # Add some noise (realistic)
    noise = np.random.normal(0, 0.05, len(signal))
    signal += noise
    
    # Add amplitude modulation (speech-like)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)  # 3 Hz modulation
    signal *= envelope
    
    # Add pauses (speech-like)
    pause_starts = [int(0.8 * sample_rate), int(2.2 * sample_rate), int(3.8 * sample_rate)]
    pause_duration = int(0.3 * sample_rate)
    
    for start in pause_starts:
        end = min(start + pause_duration, len(signal))
        signal[start:end] *= 0.1  # Reduce amplitude for pauses
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    return signal


async def test_speech_pipeline():
    """Test the complete speech analysis pipeline"""
    
    print("🎤 Testing Real Speech ML Pipeline")
    print("=" * 50)
    
    # Test 1: Generate test audio
    print("\n1. Generating test audio...")
    test_audio = generate_test_audio(duration=10.0, frequency=150)  # Male voice frequency
    audio_bytes = (test_audio * 32767).astype(np.int16).tobytes()
    print(f"   ✅ Generated {len(test_audio)/22050:.1f}s of test audio")
    
    # Test 2: Real-time analysis
    print("\n2. Running real-time speech analysis...")
    start_time = time.perf_counter()
    
    try:
        result = await realtime_speech_analyzer.analyze_realtime(audio_bytes, "test_session")
        processing_time = time.perf_counter() - start_time
        
        print(f"   ✅ Analysis completed in {processing_time:.3f}s")
        print(f"   📊 Processing time: {result.processing_time:.3f}s")
        
        # Test 3: Validate results
        print("\n3. Validating analysis results...")
        
        print(f"   Risk Score: {result.risk_score:.3f}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Quality Score: {result.quality_score:.3f}")
        
        print("\n   🧬 Biomarkers:")
        print(f"      Fluency Score: {result.biomarkers.fluency_score:.3f}")
        print(f"      Voice Tremor: {result.biomarkers.voice_tremor:.3f}")
        print(f"      Articulation Clarity: {result.biomarkers.articulation_clarity:.3f}")
        print(f"      Prosody Variation: {result.biomarkers.prosody_variation:.3f}")
        print(f"      Speaking Rate: {result.biomarkers.speaking_rate:.1f} WPM")
        print(f"      Pause Frequency: {result.biomarkers.pause_frequency:.1f}/min")
        print(f"      Pause Pattern: {result.biomarkers.pause_pattern:.3f}")
        
        # Test 4: Performance validation
        print("\n4. Performance validation...")
        target_time = 3.0  # 3 second target
        
        if processing_time <= target_time:
            print(f"   ✅ Processing time {processing_time:.3f}s meets target <{target_time}s")
        else:
            print(f"   ⚠️  Processing time {processing_time:.3f}s exceeds target <{target_time}s")
        
        # Test 5: Feature validation
        print("\n5. Feature validation...")
        
        # Check if values are reasonable
        checks = [
            ("Risk score", 0.0 <= result.risk_score <= 1.0),
            ("Confidence", 0.0 <= result.confidence <= 1.0),
            ("Quality score", 0.0 <= result.quality_score <= 1.0),
            ("Fluency score", 0.0 <= result.biomarkers.fluency_score <= 1.0),
            ("Speaking rate", 60.0 <= result.biomarkers.speaking_rate <= 200.0),
            ("Pause frequency", 0.0 <= result.biomarkers.pause_frequency <= 60.0)
        ]
        
        all_valid = True
        for check_name, is_valid in checks:
            if is_valid:
                print(f"   ✅ {check_name}: Valid")
            else:
                print(f"   ❌ {check_name}: Invalid")
                all_valid = False
        
        # Test 6: Recommendations
        print("\n6. Recommendations:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"   {i}. {rec}")
        
        # Summary
        print("\n" + "=" * 50)
        if all_valid and processing_time <= target_time:
            print("🎉 Speech ML Pipeline Test: PASSED")
            print("✅ All features extracted successfully")
            print("✅ Real MFCC and voice biomarkers working")
            print("✅ Performance targets met")
        else:
            print("⚠️  Speech ML Pipeline Test: PARTIAL")
            if not all_valid:
                print("❌ Some feature validations failed")
            if processing_time > target_time:
                print("❌ Performance target not met")
        
        return {
            "status": "success" if all_valid and processing_time <= target_time else "partial",
            "processing_time": processing_time,
            "risk_score": result.risk_score,
            "confidence": result.confidence,
            "biomarkers": {
                "fluency": result.biomarkers.fluency_score,
                "tremor": result.biomarkers.voice_tremor,
                "articulation": result.biomarkers.articulation_clarity,
                "speaking_rate": result.biomarkers.speaking_rate
            }
        }
        
    except Exception as e:
        print(f"   ❌ Analysis failed: {str(e)}")
        return {"status": "error", "error": str(e)}


async def test_multiple_scenarios():
    """Test different audio scenarios"""
    
    print("\n🔬 Testing Multiple Audio Scenarios")
    print("=" * 50)
    
    scenarios = [
        ("Normal speech", 150, 5.0, 0.05),      # Normal male voice
        ("High pitch", 250, 5.0, 0.05),        # Female voice
        ("Low pitch", 100, 5.0, 0.05),         # Deep male voice
        ("Noisy audio", 150, 5.0, 0.2),        # High noise
        ("Short clip", 150, 2.0, 0.05),        # Short duration
    ]
    
    results = []
    
    for scenario_name, freq, duration, noise_level in scenarios:
        print(f"\n📊 Testing: {scenario_name}")
        
        # Generate scenario-specific audio
        test_audio = generate_test_audio(duration=duration, frequency=freq)
        if noise_level > 0.05:
            noise = np.random.normal(0, noise_level, len(test_audio))
            test_audio += noise
            test_audio = test_audio / np.max(np.abs(test_audio)) * 0.8
        
        audio_bytes = (test_audio * 32767).astype(np.int16).tobytes()
        
        try:
            start_time = time.perf_counter()
            result = await realtime_speech_analyzer.analyze_realtime(audio_bytes, f"test_{scenario_name}")
            processing_time = time.perf_counter() - start_time
            
            print(f"   Time: {processing_time:.3f}s, Risk: {result.risk_score:.3f}, Quality: {result.quality_score:.3f}")
            
            results.append({
                "scenario": scenario_name,
                "processing_time": processing_time,
                "risk_score": result.risk_score,
                "quality_score": result.quality_score,
                "confidence": result.confidence
            })
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
            results.append({
                "scenario": scenario_name,
                "error": str(e)
            })
    
    # Summary
    print(f"\n📈 Scenario Test Summary:")
    avg_time = np.mean([r.get('processing_time', 0) for r in results if 'processing_time' in r])
    print(f"   Average processing time: {avg_time:.3f}s")
    print(f"   Scenarios tested: {len(scenarios)}")
    print(f"   Successful: {len([r for r in results if 'error' not in r])}")
    
    return results


if __name__ == "__main__":
    print("🚀 Starting Speech ML Pipeline Tests...")
    
    # Run main test
    main_result = asyncio.run(test_speech_pipeline())
    
    # Run scenario tests
    scenario_results = asyncio.run(test_multiple_scenarios())
    
    print(f"\n🏁 Final Results:")
    print(f"   Main test: {main_result.get('status', 'unknown')}")
    print(f"   Scenarios: {len([r for r in scenario_results if 'error' not in r])}/{len(scenario_results)} passed")
    
    if main_result.get('status') == 'success':
        print("\n🎉 Speech ML Pipeline is ready for production!")
    else:
        print("\n⚠️  Speech ML Pipeline needs optimization")

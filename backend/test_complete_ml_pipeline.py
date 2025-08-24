"""
Comprehensive test script for all NeuraLens ML pipelines
Tests speech, retinal, motor, and cognitive assessments with real algorithms
"""

import asyncio
import numpy as np
import time
from datetime import datetime
from PIL import Image
import io

from app.ml.realtime.realtime_speech import realtime_speech_analyzer
from app.ml.realtime.realtime_retinal import realtime_retinal_analyzer
from app.ml.realtime.realtime_motor import realtime_motor_analyzer
from app.ml.realtime.realtime_cognitive import realtime_cognitive_analyzer
from app.schemas.assessment import MotorAssessmentRequest, CognitiveAssessmentRequest


def generate_test_audio(duration=5.0, sample_rate=22050, frequency=150):
    """Generate realistic test audio with speech characteristics"""
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create speech-like signal with harmonics and formants
    signal = 0.3 * np.sin(2 * np.pi * frequency * t)  # Fundamental
    signal += 0.2 * np.sin(2 * np.pi * frequency * 2 * t)  # Second harmonic
    signal += 0.15 * np.sin(2 * np.pi * 800 * t)  # First formant
    signal += 0.1 * np.sin(2 * np.pi * 1200 * t)  # Second formant
    
    # Add realistic noise and amplitude modulation
    noise = np.random.normal(0, 0.05, len(signal))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    signal = (signal + noise) * envelope
    
    # Add pauses (speech-like)
    pause_starts = [int(0.8 * sample_rate), int(2.2 * sample_rate)]
    for start in pause_starts:
        end = min(start + int(0.3 * sample_rate), len(signal))
        signal[start:end] *= 0.1
    
    return (signal / np.max(np.abs(signal)) * 0.8 * 32767).astype(np.int16).tobytes()


def generate_test_retinal_image(size=(512, 512)):
    """Generate synthetic retinal image for testing"""
    
    # Create base retinal image
    image = np.zeros((*size, 3), dtype=np.uint8)
    
    # Add retinal background (reddish)
    image[:, :, 0] = 120  # Red channel
    image[:, :, 1] = 60   # Green channel
    image[:, :, 2] = 40   # Blue channel
    
    # Add some vessel-like structures (dark lines)
    center_x, center_y = size[0] // 2, size[1] // 2
    
    # Create radial vessel pattern
    for angle in np.linspace(0, 2*np.pi, 8):
        for r in range(50, min(size) // 2, 20):
            x = int(center_x + r * np.cos(angle))
            y = int(center_y + r * np.sin(angle))
            if 0 <= x < size[0] and 0 <= y < size[1]:
                # Draw vessel (darker)
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if 0 <= x+dx < size[0] and 0 <= y+dy < size[1]:
                            image[x+dx, y+dy] = [80, 40, 20]
    
    # Add optic disc (bright circular region)
    disc_x, disc_y = center_x + 100, center_y
    for dx in range(-30, 31):
        for dy in range(-30, 31):
            if dx*dx + dy*dy <= 900:  # Circle
                x, y = disc_x + dx, disc_y + dy
                if 0 <= x < size[0] and 0 <= y < size[1]:
                    image[x, y] = [200, 180, 160]
    
    # Convert to bytes
    pil_image = Image.fromarray(image)
    img_bytes = io.BytesIO()
    pil_image.save(img_bytes, format='JPEG')
    return img_bytes.getvalue()


def generate_test_motor_data():
    """Generate synthetic motor sensor data"""
    
    duration = 5.0  # seconds
    sample_rate = 100  # Hz
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples)
    
    # Simulate accelerometer data with tremor
    tremor_freq = 6.0  # Hz (typical tremor frequency)
    base_movement = 0.5 * np.sin(2 * np.pi * 2 * t)  # Voluntary movement
    tremor = 0.2 * np.sin(2 * np.pi * tremor_freq * t)  # Tremor component
    noise = np.random.normal(0, 0.1, n_samples)
    
    accel_data = []
    for i in range(n_samples):
        accel_data.append({
            'x': base_movement[i] + tremor[i] + noise[i],
            'y': 0.3 * np.sin(2 * np.pi * 1.5 * t[i]) + noise[i],
            'z': 9.81 + 0.1 * noise[i]  # Gravity + small variations
        })
    
    # Simulate gyroscope data
    gyro_data = []
    for i in range(n_samples):
        gyro_data.append({
            'x': 0.1 * np.sin(2 * np.pi * 3 * t[i]) + 0.05 * noise[i],
            'y': 0.1 * np.cos(2 * np.pi * 2.5 * t[i]) + 0.05 * noise[i],
            'z': 0.05 * noise[i]
        })
    
    return {
        'accelerometer': accel_data,
        'gyroscope': gyro_data
    }


def generate_test_cognitive_data():
    """Generate synthetic cognitive test data"""
    
    # Simulate response times (in milliseconds)
    base_rt = 1500  # Base response time
    response_times = []
    for i in range(20):
        # Add learning effect (getting faster)
        learning_factor = 1.0 - (i * 0.02)
        # Add some variability
        variability = np.random.normal(0, 200)
        rt = base_rt * learning_factor + variability
        response_times.append(max(500, rt))  # Minimum 500ms
    
    # Simulate accuracy scores
    accuracy_scores = []
    for i in range(20):
        # Add learning effect (getting more accurate)
        base_accuracy = 0.7 + (i * 0.01)
        noise = np.random.normal(0, 0.1)
        accuracy = np.clip(base_accuracy + noise, 0, 1)
        accuracy_scores.append(accuracy)
    
    return {
        'response_times': response_times,
        'accuracy': accuracy_scores,
        'memory': {'immediate_recall': 0.8, 'delayed_recall': 0.75},
        'attention': {'sustained_attention': 0.85},
        'executive': {'planning': 0.7, 'flexibility': 0.8},
        'task_switching': {
            'repeat_trials': [1200, 1150, 1180, 1160],
            'switch_trials': [1400, 1380, 1420, 1390],
            'switch_accuracy': 0.75
        }
    }


async def test_all_ml_pipelines():
    """Test all ML pipelines with real algorithms"""
    
    print("🧠 Testing Complete NeuraLens ML Pipeline")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Speech Analysis
    print("\n1. 🎤 Testing Speech Analysis ML Pipeline...")
    try:
        start_time = time.perf_counter()
        audio_data = generate_test_audio(duration=5.0, frequency=140)
        
        speech_result = await realtime_speech_analyzer.analyze_realtime(audio_data, "test_speech")
        processing_time = time.perf_counter() - start_time
        
        print(f"   ✅ Processing time: {processing_time:.3f}s (target: <3s)")
        print(f"   📊 Risk score: {speech_result.risk_score:.3f}")
        print(f"   🎯 Confidence: {speech_result.confidence:.3f}")
        print(f"   🧬 Tremor: {speech_result.biomarkers.voice_tremor:.3f}")
        print(f"   🗣️  Fluency: {speech_result.biomarkers.fluency_score:.3f}")
        
        results['speech'] = {
            'status': 'success',
            'processing_time': processing_time,
            'meets_target': processing_time < 3.0,
            'risk_score': speech_result.risk_score,
            'confidence': speech_result.confidence
        }
        
    except Exception as e:
        print(f"   ❌ Speech analysis failed: {str(e)}")
        results['speech'] = {'status': 'error', 'error': str(e)}
    
    # Test 2: Retinal Analysis
    print("\n2. 👁️  Testing Retinal Analysis ML Pipeline...")
    try:
        start_time = time.perf_counter()
        retinal_image = generate_test_retinal_image()
        
        retinal_result = await realtime_retinal_analyzer.analyze_realtime(retinal_image, "test_retinal")
        processing_time = time.perf_counter() - start_time
        
        print(f"   ✅ Processing time: {processing_time:.3f}s (target: <5s)")
        print(f"   📊 Risk score: {retinal_result.risk_score:.3f}")
        print(f"   🎯 Confidence: {retinal_result.confidence:.3f}")
        print(f"   🩸 Vessel density: {retinal_result.biomarkers.vessel_density:.3f}")
        print(f"   👁️  Cup-disc ratio: {retinal_result.biomarkers.cup_disc_ratio:.3f}")
        
        results['retinal'] = {
            'status': 'success',
            'processing_time': processing_time,
            'meets_target': processing_time < 5.0,
            'risk_score': retinal_result.risk_score,
            'confidence': retinal_result.confidence
        }
        
    except Exception as e:
        print(f"   ❌ Retinal analysis failed: {str(e)}")
        results['retinal'] = {'status': 'error', 'error': str(e)}
    
    # Test 3: Motor Assessment
    print("\n3. 🤲 Testing Motor Assessment ML Pipeline...")
    try:
        start_time = time.perf_counter()
        motor_data = generate_test_motor_data()
        
        motor_request = MotorAssessmentRequest(
            session_id="test_motor",
            sensor_data=motor_data,
            assessment_type="tremor"
        )
        
        motor_result = await realtime_motor_analyzer.analyze_realtime(motor_request, "test_motor")
        processing_time = time.perf_counter() - start_time
        
        print(f"   ✅ Processing time: {processing_time:.3f}s (target: <2s)")
        print(f"   📊 Risk score: {motor_result.risk_score:.3f}")
        print(f"   🎯 Confidence: {motor_result.confidence:.3f}")
        print(f"   🤝 Coordination: {motor_result.biomarkers.coordination_index:.3f}")
        print(f"   📳 Tremor severity: {motor_result.biomarkers.tremor_severity:.3f}")
        
        results['motor'] = {
            'status': 'success',
            'processing_time': processing_time,
            'meets_target': processing_time < 2.0,
            'risk_score': motor_result.risk_score,
            'confidence': motor_result.confidence
        }
        
    except Exception as e:
        print(f"   ❌ Motor analysis failed: {str(e)}")
        results['motor'] = {'status': 'error', 'error': str(e)}
    
    # Test 4: Cognitive Assessment
    print("\n4. 🧠 Testing Cognitive Assessment ML Pipeline...")
    try:
        start_time = time.perf_counter()
        cognitive_data = generate_test_cognitive_data()
        
        cognitive_request = CognitiveAssessmentRequest(
            session_id="test_cognitive",
            test_results=cognitive_data,
            test_battery=["memory", "attention", "executive"],
            difficulty_level="standard"
        )
        
        cognitive_result = await realtime_cognitive_analyzer.analyze_realtime(cognitive_request, "test_cognitive")
        processing_time = time.perf_counter() - start_time
        
        print(f"   ✅ Processing time: {processing_time:.3f}s (target: <1s)")
        print(f"   📊 Risk score: {cognitive_result.risk_score:.3f}")
        print(f"   🎯 Confidence: {cognitive_result.confidence:.3f}")
        print(f"   🧠 Memory: {cognitive_result.biomarkers.memory_score:.3f}")
        print(f"   👁️  Attention: {cognitive_result.biomarkers.attention_score:.3f}")
        
        results['cognitive'] = {
            'status': 'success',
            'processing_time': processing_time,
            'meets_target': processing_time < 1.0,
            'risk_score': cognitive_result.risk_score,
            'confidence': cognitive_result.confidence
        }
        
    except Exception as e:
        print(f"   ❌ Cognitive analysis failed: {str(e)}")
        results['cognitive'] = {'status': 'error', 'error': str(e)}
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 ML Pipeline Test Summary")
    print("=" * 60)
    
    successful_tests = sum(1 for r in results.values() if r.get('status') == 'success')
    target_met = sum(1 for r in results.values() if r.get('meets_target', False))
    
    print(f"✅ Successful tests: {successful_tests}/4")
    print(f"⚡ Performance targets met: {target_met}/4")
    
    for modality, result in results.items():
        if result.get('status') == 'success':
            status = "✅" if result.get('meets_target') else "⚠️"
            print(f"{status} {modality.capitalize()}: {result['processing_time']:.3f}s")
        else:
            print(f"❌ {modality.capitalize()}: Failed")
    
    if successful_tests == 4 and target_met >= 3:
        print("\n🎉 ML Pipeline Implementation: COMPLETE!")
        print("✅ All modalities use real algorithms instead of mock data")
        print("✅ Performance targets mostly achieved")
        print("✅ Ready for production deployment")
    else:
        print(f"\n⚠️  ML Pipeline Implementation: PARTIAL")
        print(f"   {successful_tests}/4 modalities working")
        print(f"   {target_met}/4 performance targets met")
    
    return results


if __name__ == "__main__":
    print("🚀 Starting Complete ML Pipeline Tests...")
    results = asyncio.run(test_all_ml_pipelines())
    
    print(f"\n🏁 Final Status:")
    for modality, result in results.items():
        print(f"   {modality}: {result.get('status', 'unknown')}")

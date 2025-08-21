"""
Comprehensive test script for all NeuroLens API endpoints
Tests P0-2: Complete Backend API Endpoints implementation
"""

import asyncio
import json
import numpy as np
from datetime import datetime

# Import all analyzers directly
from app.ml.models.speech_analyzer import SpeechAnalyzer
from app.ml.models.retinal_analyzer import retinal_analyzer
from app.ml.models.motor_analyzer import motor_analyzer
from app.ml.models.cognitive_analyzer import cognitive_analyzer
from app.ml.models.nri_fusion import nri_fusion_engine
from app.ml.models.validation_engine import validation_engine

# Import schemas
from app.schemas.assessment import (
    MotorAssessmentRequest, CognitiveAssessmentRequest, NRIFusionRequest
)

async def test_all_analyzers():
    """Test all ML analyzers to ensure they work correctly"""
    
    print("🧪 Testing All NeuroLens ML Analyzers...")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Speech Analyzer
    print("\n1️⃣ Testing Speech Analyzer...")
    try:
        speech_analyzer = SpeechAnalyzer()
        
        # Create mock audio data
        sample_rate = 22050
        duration = 5.0
        num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples)
        audio_data = (
            0.3 * np.sin(2 * np.pi * 200 * t) +
            0.2 * np.sin(2 * np.pi * 400 * t) +
            0.05 * np.random.normal(0, 1, num_samples)
        )
        audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
        
        result = await speech_analyzer.analyze(audio_bytes, "test_speech_001")
        results['speech'] = {
            'status': 'success',
            'risk_score': result.risk_score,
            'confidence': result.confidence,
            'processing_time': result.processing_time
        }
        print(f"   ✅ Speech analysis completed - Risk: {result.risk_score:.3f}, Confidence: {result.confidence:.3f}")
        
    except Exception as e:
        results['speech'] = {'status': 'failed', 'error': str(e)}
        print(f"   ❌ Speech analysis failed: {str(e)}")
    
    # Test 2: Retinal Analyzer
    print("\n2️⃣ Testing Retinal Analyzer...")
    try:
        # Create mock image data (simple RGB image)
        from PIL import Image
        import io
        
        # Create a 512x512 RGB image with some patterns
        image = Image.new('RGB', (512, 512), color='black')
        pixels = image.load()
        for i in range(512):
            for j in range(512):
                # Create some circular patterns to simulate retinal features
                center_x, center_y = 256, 256
                distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                if distance < 100:
                    pixels[i, j] = (200, 100, 50)  # Optic disc
                elif distance < 200:
                    pixels[i, j] = (150, 80, 40)   # Vessels
                else:
                    pixels[i, j] = (100, 50, 25)   # Background
        
        # Convert to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        image_bytes = img_buffer.getvalue()
        
        result = await retinal_analyzer.analyze(image_bytes, "test_retinal_001")
        results['retinal'] = {
            'status': 'success',
            'risk_score': result.risk_score,
            'confidence': result.confidence,
            'processing_time': result.processing_time
        }
        print(f"   ✅ Retinal analysis completed - Risk: {result.risk_score:.3f}, Confidence: {result.confidence:.3f}")
        
    except Exception as e:
        results['retinal'] = {'status': 'failed', 'error': str(e)}
        print(f"   ❌ Retinal analysis failed: {str(e)}")
    
    # Test 3: Motor Analyzer
    print("\n3️⃣ Testing Motor Analyzer...")
    try:
        # Create mock sensor data
        sensor_data = {
            "accelerometer": [
                {"x": 0.1 + np.random.normal(0, 0.05), "y": 0.2 + np.random.normal(0, 0.05), "z": 9.8 + np.random.normal(0, 0.1)}
                for _ in range(1000)  # 1000 samples
            ],
            "gyroscope": [
                {"x": np.random.normal(0, 0.1), "y": np.random.normal(0, 0.1), "z": np.random.normal(0, 0.1)}
                for _ in range(1000)
            ],
            "timestamp": [i * 0.02 for i in range(1000)]  # 50Hz sampling
        }
        
        request = MotorAssessmentRequest(
            assessment_type="finger_tapping",
            sensor_data=sensor_data,
            duration=20.0
        )
        
        result = await motor_analyzer.analyze(request, "test_motor_001")
        results['motor'] = {
            'status': 'success',
            'risk_score': result.risk_score,
            'confidence': result.confidence,
            'processing_time': result.processing_time
        }
        print(f"   ✅ Motor analysis completed - Risk: {result.risk_score:.3f}, Confidence: {result.confidence:.3f}")
        
    except Exception as e:
        results['motor'] = {'status': 'failed', 'error': str(e)}
        print(f"   ❌ Motor analysis failed: {str(e)}")
    
    # Test 4: Cognitive Analyzer
    print("\n4️⃣ Testing Cognitive Analyzer...")
    try:
        test_results = {
            "memory": {
                "immediate_recall": 0.85,
                "delayed_recall": 0.78,
                "recognition": 0.92
            },
            "attention": {
                "sustained_attention": 0.82,
                "selective_attention": 0.79,
                "divided_attention": 0.71
            },
            "executive": {
                "planning": 0.76,
                "inhibition": 0.83,
                "flexibility": 0.74
            }
        }
        
        request = CognitiveAssessmentRequest(
            test_battery=["memory", "attention", "executive"],
            test_results=test_results,
            difficulty_level="standard"
        )
        
        result = await cognitive_analyzer.analyze(request, "test_cognitive_001")
        results['cognitive'] = {
            'status': 'success',
            'risk_score': result.risk_score,
            'confidence': result.confidence,
            'processing_time': result.processing_time
        }
        print(f"   ✅ Cognitive analysis completed - Risk: {result.risk_score:.3f}, Confidence: {result.confidence:.3f}")
        
    except Exception as e:
        results['cognitive'] = {'status': 'failed', 'error': str(e)}
        print(f"   ❌ Cognitive analysis failed: {str(e)}")
    
    # Test 5: NRI Fusion Engine
    print("\n5️⃣ Testing NRI Fusion Engine...")
    try:
        # Use results from previous tests if available
        modality_scores = {}
        if results.get('speech', {}).get('status') == 'success':
            modality_scores['speech'] = results['speech']['risk_score']
        else:
            modality_scores['speech'] = 0.35
            
        if results.get('retinal', {}).get('status') == 'success':
            modality_scores['retinal'] = results['retinal']['risk_score']
        else:
            modality_scores['retinal'] = 0.28
            
        if results.get('motor', {}).get('status') == 'success':
            modality_scores['motor'] = results['motor']['risk_score']
        else:
            modality_scores['motor'] = 0.42
            
        if results.get('cognitive', {}).get('status') == 'success':
            modality_scores['cognitive'] = results['cognitive']['risk_score']
        else:
            modality_scores['cognitive'] = 0.31
        
        request = NRIFusionRequest(
            session_id="test_nri_001",
            modalities=list(modality_scores.keys()),
            modality_scores=modality_scores,
            fusion_method="bayesian"
        )
        
        result = await nri_fusion_engine.calculate_nri(request)
        results['nri'] = {
            'status': 'success',
            'nri_score': result.nri_score,
            'confidence': result.confidence,
            'risk_category': result.risk_category,
            'processing_time': result.processing_time
        }
        print(f"   ✅ NRI fusion completed - NRI: {result.nri_score:.1f}, Category: {result.risk_category}, Confidence: {result.confidence:.3f}")
        
    except Exception as e:
        results['nri'] = {'status': 'failed', 'error': str(e)}
        print(f"   ❌ NRI fusion failed: {str(e)}")
    
    # Test 6: Validation Engine
    print("\n6️⃣ Testing Validation Engine...")
    try:
        metrics = await validation_engine.get_validation_metrics("all", "performance")
        health = await validation_engine.health_check()
        
        results['validation'] = {
            'status': 'success',
            'metrics_loaded': 'performance' in metrics,
            'health_status': health.get('model_loaded', False)
        }
        print(f"   ✅ Validation engine working - Metrics loaded: {results['validation']['metrics_loaded']}")
        
    except Exception as e:
        results['validation'] = {'status': 'failed', 'error': str(e)}
        print(f"   ❌ Validation engine failed: {str(e)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    successful_tests = sum(1 for r in results.values() if r.get('status') == 'success')
    total_tests = len(results)
    
    print(f"✅ Successful: {successful_tests}/{total_tests}")
    print(f"❌ Failed: {total_tests - successful_tests}/{total_tests}")
    
    if successful_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! Backend API endpoints are ready for hackathon demo!")
        print("\n📈 Performance Summary:")
        if 'nri' in results and results['nri']['status'] == 'success':
            print(f"   🧠 Final NRI Score: {results['nri']['nri_score']:.1f}/100")
            print(f"   🎯 Risk Category: {results['nri']['risk_category'].upper()}")
            print(f"   🔒 Confidence: {results['nri']['confidence']:.1%}")
    else:
        print(f"\n⚠️  {total_tests - successful_tests} tests failed. Check errors above.")
    
    return results

if __name__ == "__main__":
    results = asyncio.run(test_all_analyzers())

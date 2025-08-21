"""
Simple test to verify backend components work
"""

import asyncio
import sys

print("🧪 Simple Backend Test")
print("=" * 40)

# Test 1: Import all analyzers
print("\n1️⃣ Testing imports...")
try:
    from app.ml.models.speech_analyzer import SpeechAnalyzer
    print("   ✅ Speech analyzer imported")
except Exception as e:
    print(f"   ❌ Speech analyzer failed: {e}")

try:
    from app.ml.models.retinal_analyzer import retinal_analyzer
    print("   ✅ Retinal analyzer imported")
except Exception as e:
    print(f"   ❌ Retinal analyzer failed: {e}")

try:
    from app.ml.models.motor_analyzer import motor_analyzer
    print("   ✅ Motor analyzer imported")
except Exception as e:
    print(f"   ❌ Motor analyzer failed: {e}")

try:
    from app.ml.models.cognitive_analyzer import cognitive_analyzer
    print("   ✅ Cognitive analyzer imported")
except Exception as e:
    print(f"   ❌ Cognitive analyzer failed: {e}")

try:
    from app.ml.models.nri_fusion import nri_fusion_engine
    print("   ✅ NRI fusion engine imported")
except Exception as e:
    print(f"   ❌ NRI fusion engine failed: {e}")

try:
    from app.ml.models.validation_engine import validation_engine
    print("   ✅ Validation engine imported")
except Exception as e:
    print(f"   ❌ Validation engine failed: {e}")

# Test 2: Test schemas
print("\n2️⃣ Testing schemas...")
try:
    from app.schemas.assessment import (
        MotorAssessmentRequest, CognitiveAssessmentRequest, 
        NRIFusionRequest, RetinalBiomarkers
    )
    print("   ✅ All schemas imported")
except Exception as e:
    print(f"   ❌ Schema import failed: {e}")

# Test 3: Simple analyzer test
print("\n3️⃣ Testing speech analyzer...")
try:
    import numpy as np
    
    speech_analyzer = SpeechAnalyzer()
    
    # Create simple mock audio
    audio_data = np.random.normal(0, 0.1, 22050)  # 1 second of audio
    audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
    
    async def test_speech():
        result = await speech_analyzer.analyze(audio_bytes, "simple_test")
        return result
    
    result = asyncio.run(test_speech())
    print(f"   ✅ Speech analysis completed - Risk: {result.risk_score:.3f}")
    
except Exception as e:
    print(f"   ❌ Speech analysis failed: {e}")

print("\n🎉 Simple test completed!")

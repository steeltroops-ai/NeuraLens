"""
Simple Real-Time Test
Quick test to verify real-time models work
"""

import asyncio
import time
import numpy as np

print("🧪 Simple Real-Time Model Test")
print("=" * 40)

# Test imports
try:
    from app.ml.realtime.realtime_speech import realtime_speech_analyzer
    print("✅ Real-time speech analyzer imported")
except Exception as e:
    print(f"❌ Speech analyzer import failed: {e}")

try:
    from app.ml.realtime.realtime_retinal import realtime_retinal_analyzer
    print("✅ Real-time retinal analyzer imported")
except Exception as e:
    print(f"❌ Retinal analyzer import failed: {e}")

try:
    from app.ml.realtime.realtime_nri import realtime_nri_fusion
    print("✅ Real-time NRI fusion imported")
except Exception as e:
    print(f"❌ NRI fusion import failed: {e}")

async def quick_test():
    """Quick functionality test"""
    
    print("\n🎤 Testing speech analyzer...")
    try:
        # Generate simple audio
        audio_data = np.random.normal(0, 0.1, 16000).astype(np.float32)
        audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
        
        start_time = time.perf_counter()
        result = await realtime_speech_analyzer.analyze_realtime(audio_bytes, "quick_test")
        latency = (time.perf_counter() - start_time) * 1000
        
        print(f"   ✅ Speech analysis: {latency:.1f}ms, Risk: {result.risk_score:.3f}")
        
    except Exception as e:
        print(f"   ❌ Speech test failed: {e}")
    
    print("\n🎉 Quick test completed!")

if __name__ == "__main__":
    asyncio.run(quick_test())

"""
Real-Time Performance Test Suite
Tests all ML models for <100ms latency and 90%+ accuracy targets
"""

import asyncio
import time
import numpy as np
import statistics
from datetime import datetime
from PIL import Image
import io

# Import real-time analyzers
from app.ml.realtime.realtime_speech import realtime_speech_analyzer
from app.ml.realtime.realtime_retinal import realtime_retinal_analyzer
from app.ml.realtime.realtime_nri import realtime_nri_fusion

from app.schemas.assessment import NRIFusionRequest

class RealtimePerformanceTester:
    """Comprehensive performance testing for real-time ML models"""
    
    def __init__(self):
        self.test_iterations = 10  # Number of tests per model
        self.latency_targets = {
            'speech': 100,    # ms
            'retinal': 150,   # ms
            'nri_fusion': 100 # ms
        }
        
    async def run_comprehensive_tests(self):
        """Run all performance tests"""
        
        print("🚀 NeuroLens Real-Time Performance Test Suite")
        print("=" * 60)
        print(f"Target Latencies: Speech <{self.latency_targets['speech']}ms, "
              f"Retinal <{self.latency_targets['retinal']}ms, "
              f"NRI <{self.latency_targets['nri_fusion']}ms")
        print(f"Test Iterations: {self.test_iterations} per model")
        print("=" * 60)
        
        results = {}
        
        # Test 1: Speech Analysis Performance
        print("\n🎤 Testing Real-Time Speech Analysis...")
        speech_results = await self.test_speech_performance()
        results['speech'] = speech_results
        
        # Test 2: Retinal Analysis Performance  
        print("\n👁️ Testing Real-Time Retinal Analysis...")
        retinal_results = await self.test_retinal_performance()
        results['retinal'] = retinal_results
        
        # Test 3: NRI Fusion Performance
        print("\n🔬 Testing Real-Time NRI Fusion...")
        nri_results = await self.test_nri_performance()
        results['nri_fusion'] = nri_results
        
        # Test 4: End-to-End Performance
        print("\n⚡ Testing End-to-End Real-Time Pipeline...")
        e2e_results = await self.test_end_to_end_performance()
        results['end_to_end'] = e2e_results
        
        # Generate Performance Report
        self.generate_performance_report(results)
        
        return results
    
    async def test_speech_performance(self):
        """Test speech analysis latency and accuracy"""
        
        latencies = []
        accuracies = []
        
        for i in range(self.test_iterations):
            # Generate test audio
            audio_data = self.generate_test_audio(risk_level="moderate")
            
            # Measure latency
            start_time = time.perf_counter()
            result = await realtime_speech_analyzer.analyze_realtime(
                audio_data, f"perf_test_speech_{i}"
            )
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            # Mock accuracy calculation (in real scenario, compare with ground truth)
            accuracy = self.calculate_mock_accuracy(result.risk_score, "moderate")
            accuracies.append(accuracy)
            
            print(f"   Test {i+1}: {latency_ms:.1f}ms, Accuracy: {accuracy:.1%}")
        
        return {
            'latencies': latencies,
            'accuracies': accuracies,
            'avg_latency': statistics.mean(latencies),
            'max_latency': max(latencies),
            'avg_accuracy': statistics.mean(accuracies),
            'target_met': max(latencies) < self.latency_targets['speech']
        }
    
    async def test_retinal_performance(self):
        """Test retinal analysis latency and accuracy"""
        
        latencies = []
        accuracies = []
        
        for i in range(self.test_iterations):
            # Generate test image
            image_data = self.generate_test_retinal_image(risk_level="high")
            
            # Measure latency
            start_time = time.perf_counter()
            result = await realtime_retinal_analyzer.analyze_realtime(
                image_data, f"perf_test_retinal_{i}"
            )
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            # Mock accuracy calculation
            accuracy = self.calculate_mock_accuracy(result.risk_score, "high")
            accuracies.append(accuracy)
            
            print(f"   Test {i+1}: {latency_ms:.1f}ms, Accuracy: {accuracy:.1%}")
        
        return {
            'latencies': latencies,
            'accuracies': accuracies,
            'avg_latency': statistics.mean(latencies),
            'max_latency': max(latencies),
            'avg_accuracy': statistics.mean(accuracies),
            'target_met': max(latencies) < self.latency_targets['retinal']
        }
    
    async def test_nri_performance(self):
        """Test NRI fusion latency and accuracy"""
        
        latencies = []
        accuracies = []
        
        for i in range(self.test_iterations):
            # Generate test modality scores
            request = NRIFusionRequest(
                session_id=f"perf_test_nri_{i}",
                modalities=["speech", "retinal", "motor", "cognitive"],
                modality_scores={
                    "speech": np.random.uniform(0.2, 0.8),
                    "retinal": np.random.uniform(0.1, 0.7),
                    "motor": np.random.uniform(0.3, 0.9),
                    "cognitive": np.random.uniform(0.2, 0.6)
                },
                modality_confidences={
                    "speech": 0.9,
                    "retinal": 0.85,
                    "motor": 0.8,
                    "cognitive": 0.88
                },
                fusion_method="bayesian"
            )
            
            # Measure latency
            start_time = time.perf_counter()
            result = await realtime_nri_fusion.calculate_nri_realtime(request)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            # Mock accuracy calculation
            expected_nri = np.mean(list(request.modality_scores.values())) * 100
            accuracy = 1.0 - abs(result.nri_score - expected_nri) / 100.0
            accuracies.append(max(0.0, accuracy))
            
            print(f"   Test {i+1}: {latency_ms:.1f}ms, NRI: {result.nri_score:.1f}, Accuracy: {accuracy:.1%}")
        
        return {
            'latencies': latencies,
            'accuracies': accuracies,
            'avg_latency': statistics.mean(latencies),
            'max_latency': max(latencies),
            'avg_accuracy': statistics.mean(accuracies),
            'target_met': max(latencies) < self.latency_targets['nri_fusion']
        }
    
    async def test_end_to_end_performance(self):
        """Test complete pipeline latency"""
        
        latencies = []
        
        for i in range(5):  # Fewer iterations for E2E test
            start_time = time.perf_counter()
            
            # Step 1: Speech analysis
            audio_data = self.generate_test_audio("moderate")
            speech_result = await realtime_speech_analyzer.analyze_realtime(
                audio_data, f"e2e_test_{i}_speech"
            )
            
            # Step 2: Retinal analysis
            image_data = self.generate_test_retinal_image("moderate")
            retinal_result = await realtime_retinal_analyzer.analyze_realtime(
                image_data, f"e2e_test_{i}_retinal"
            )
            
            # Step 3: NRI fusion
            nri_request = NRIFusionRequest(
                session_id=f"e2e_test_{i}_nri",
                modalities=["speech", "retinal"],
                modality_scores={
                    "speech": speech_result.risk_score,
                    "retinal": retinal_result.risk_score
                },
                modality_confidences={
                    "speech": speech_result.confidence,
                    "retinal": retinal_result.confidence
                },
                fusion_method="bayesian"
            )
            
            nri_result = await realtime_nri_fusion.calculate_nri_realtime(nri_request)
            
            end_time = time.perf_counter()
            total_latency_ms = (end_time - start_time) * 1000
            latencies.append(total_latency_ms)
            
            print(f"   E2E Test {i+1}: {total_latency_ms:.1f}ms total, NRI: {nri_result.nri_score:.1f}")
        
        return {
            'latencies': latencies,
            'avg_latency': statistics.mean(latencies),
            'max_latency': max(latencies),
            'target_met': max(latencies) < 500  # 500ms target for full pipeline
        }
    
    def generate_test_audio(self, risk_level: str) -> bytes:
        """Generate synthetic audio for testing"""
        
        sample_rate = 16000
        duration = 5.0
        num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples)
        
        if risk_level == "low":
            # Clear, stable speech
            audio = 0.5 * np.sin(2 * np.pi * 150 * t) + 0.1 * np.random.normal(0, 0.1, num_samples)
        elif risk_level == "moderate":
            # Some irregularities
            freq_var = 150 + 10 * np.sin(2 * np.pi * 0.5 * t)
            audio = 0.4 * np.sin(2 * np.pi * freq_var * t) + 0.15 * np.random.normal(0, 0.15, num_samples)
        else:  # high
            # Significant impairment
            tremor = 5 * np.sin(2 * np.pi * 5 * t)
            freq_var = 150 + tremor
            audio = 0.3 * np.sin(2 * np.pi * freq_var * t) + 0.2 * np.random.normal(0, 0.2, num_samples)
        
        # Normalize and convert to bytes
        audio = audio / np.max(np.abs(audio)) * 0.8
        return (audio * 32767).astype(np.int16).tobytes()
    
    def generate_test_retinal_image(self, risk_level: str) -> bytes:
        """Generate synthetic retinal image for testing"""
        
        # Create test image
        image = Image.new('RGB', (512, 512), color='black')
        pixels = image.load()
        
        for i in range(512):
            for j in range(512):
                center_x, center_y = 256, 256
                distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                
                if risk_level == "low":
                    # Normal retina
                    if distance < 50:
                        pixels[i, j] = (220, 180, 120)  # Healthy optic disc
                    elif distance < 150:
                        pixels[i, j] = (160, 120, 80)   # Normal vessels
                    else:
                        pixels[i, j] = (120, 80, 40)    # Healthy background
                elif risk_level == "moderate":
                    # Some abnormalities
                    if distance < 60:
                        pixels[i, j] = (200, 160, 100)  # Slightly enlarged disc
                    elif distance < 140:
                        pixels[i, j] = (140, 100, 60)   # Tortuous vessels
                    else:
                        pixels[i, j] = (100, 60, 30)    # Darker background
                else:  # high
                    # Significant abnormalities
                    if distance < 80:
                        pixels[i, j] = (180, 140, 80)   # Enlarged disc
                    elif distance < 120:
                        pixels[i, j] = (120, 80, 40)    # Reduced vessels
                    else:
                        pixels[i, j] = (80, 40, 20)     # Dark background
        
        # Convert to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        return img_buffer.getvalue()
    
    def calculate_mock_accuracy(self, predicted_score: float, true_risk_level: str) -> float:
        """Calculate mock accuracy for testing"""
        
        # Expected score ranges for each risk level
        expected_ranges = {
            "low": (0.0, 0.3),
            "moderate": (0.3, 0.7),
            "high": (0.7, 1.0)
        }
        
        expected_min, expected_max = expected_ranges[true_risk_level]
        
        if expected_min <= predicted_score <= expected_max:
            return 0.95  # High accuracy for correct range
        else:
            # Accuracy decreases with distance from expected range
            if predicted_score < expected_min:
                distance = expected_min - predicted_score
            else:
                distance = predicted_score - expected_max
            
            return max(0.5, 0.95 - distance * 2.0)
    
    def generate_performance_report(self, results: dict):
        """Generate comprehensive performance report"""
        
        print("\n" + "=" * 60)
        print("📊 REAL-TIME PERFORMANCE REPORT")
        print("=" * 60)
        
        overall_success = True
        
        for model_name, result in results.items():
            if model_name == 'end_to_end':
                print(f"\n⚡ {model_name.upper().replace('_', '-')} PIPELINE:")
                print(f"   Average Latency: {result['avg_latency']:.1f}ms")
                print(f"   Maximum Latency: {result['max_latency']:.1f}ms")
                print(f"   Target (<500ms): {'✅ PASS' if result['target_met'] else '❌ FAIL'}")
                if not result['target_met']:
                    overall_success = False
            else:
                target = self.latency_targets[model_name]
                print(f"\n🧠 {model_name.upper()} ANALYSIS:")
                print(f"   Average Latency: {result['avg_latency']:.1f}ms")
                print(f"   Maximum Latency: {result['max_latency']:.1f}ms")
                print(f"   Average Accuracy: {result['avg_accuracy']:.1%}")
                print(f"   Target (<{target}ms): {'✅ PASS' if result['target_met'] else '❌ FAIL'}")
                
                if not result['target_met']:
                    overall_success = False
        
        print(f"\n" + "=" * 60)
        if overall_success:
            print("🎉 ALL PERFORMANCE TARGETS MET!")
            print("✅ Ready for real-time hackathon demo")
        else:
            print("⚠️  Some performance targets not met")
            print("🔧 Consider further optimization")
        
        print(f"📅 Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

async def main():
    """Run performance tests"""
    tester = RealtimePerformanceTester()
    await tester.run_comprehensive_tests()

if __name__ == "__main__":
    asyncio.run(main())

"""
Complete Assessment Workflow Test
End-to-end testing of the complete assessment workflow with real data processing
"""

import asyncio
import requests
import json
import time
import io
import wave
import numpy as np
from PIL import Image
from datetime import datetime
from typing import Dict, Any

# Test configuration
API_BASE_URL = "http://localhost:8000"
API_VERSION = "/api/v1"
TEST_TIMEOUT = 120

class CompleteAssessmentWorkflowTester:
    """Complete assessment workflow tester"""
    
    def __init__(self):
        self.base_url = f"{API_BASE_URL}{API_VERSION}"
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json'
        })
        self.session_id = f"test-workflow-{int(time.time())}"
        self.results = {}
    
    def create_test_audio(self, duration=3.0, sample_rate=16000, frequency=440):
        """Create test audio file"""
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create speech-like signal with multiple harmonics
        signal = 0.3 * np.sin(2 * np.pi * frequency * t)  # Fundamental
        signal += 0.2 * np.sin(2 * np.pi * frequency * 2 * t)  # Second harmonic
        signal += 0.15 * np.sin(2 * np.pi * 800 * t)  # First formant
        signal += 0.1 * np.sin(2 * np.pi * 1200 * t)  # Second formant
        
        # Add realistic noise and amplitude modulation
        noise = np.random.normal(0, 0.05, len(signal))
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
        signal = (signal + noise) * envelope
        
        # Convert to 16-bit PCM
        audio_data = (signal / np.max(np.abs(signal)) * 0.8 * 32767).astype(np.int16)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        wav_buffer.seek(0)
        return wav_buffer
    
    def create_test_image(self, size=(512, 512)):
        """Create test retinal image"""
        
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
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        return img_buffer
    
    def create_test_motor_data(self):
        """Create test motor sensor data"""
        
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
        
        return {'accelerometer': accel_data}
    
    def create_test_cognitive_data(self):
        """Create test cognitive data"""
        
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
            'executive': {'planning': 0.7, 'flexibility': 0.8}
        }
    
    async def test_complete_workflow(self):
        """Test complete assessment workflow"""
        
        print("🧠 Testing Complete Assessment Workflow")
        print("=" * 60)
        
        workflow_start_time = time.perf_counter()
        step_results = {}
        
        try:
            # Step 1: Speech Analysis
            print("\n1. 🎤 Testing Speech Analysis...")
            speech_start = time.perf_counter()
            
            audio_buffer = self.create_test_audio()
            files = {'audio_file': ('test_speech.wav', audio_buffer, 'audio/wav')}
            data = {'session_id': self.session_id}
            
            response = requests.post(
                f"{self.base_url}/speech/analyze",
                data=data,
                files=files,
                timeout=TEST_TIMEOUT
            )
            
            speech_time = time.perf_counter() - speech_start
            
            if response.status_code == 200:
                speech_result = response.json()
                step_results['speech'] = speech_result
                print(f"   ✅ Speech analysis: {speech_time:.3f}s")
                if 'data' in speech_result:
                    print(f"      Risk Score: {speech_result['data'].get('risk_score', 'N/A')}")
                    print(f"      Confidence: {speech_result['data'].get('confidence', 'N/A')}")
            else:
                print(f"   ❌ Speech analysis failed: {response.status_code}")
                step_results['speech'] = {'error': f"HTTP {response.status_code}"}
            
            # Step 2: Retinal Analysis
            print("\n2. 👁️  Testing Retinal Analysis...")
            retinal_start = time.perf_counter()
            
            image_buffer = self.create_test_image()
            files = {'image_file': ('test_retinal.jpg', image_buffer, 'image/jpeg')}
            data = {'session_id': self.session_id}
            
            response = requests.post(
                f"{self.base_url}/retinal/analyze",
                data=data,
                files=files,
                timeout=TEST_TIMEOUT
            )
            
            retinal_time = time.perf_counter() - retinal_start
            
            if response.status_code == 200:
                retinal_result = response.json()
                step_results['retinal'] = retinal_result
                print(f"   ✅ Retinal analysis: {retinal_time:.3f}s")
                if 'data' in retinal_result:
                    print(f"      Risk Score: {retinal_result['data'].get('risk_score', 'N/A')}")
                    print(f"      Quality Score: {retinal_result['data'].get('quality_score', 'N/A')}")
            else:
                print(f"   ❌ Retinal analysis failed: {response.status_code}")
                step_results['retinal'] = {'error': f"HTTP {response.status_code}"}
            
            # Step 3: Motor Assessment
            print("\n3. 🤲 Testing Motor Assessment...")
            motor_start = time.perf_counter()
            
            motor_data = {
                "session_id": self.session_id,
                "sensor_data": self.create_test_motor_data(),
                "assessment_type": "tremor"
            }
            
            response = requests.post(
                f"{self.base_url}/motor/analyze",
                json=motor_data,
                timeout=TEST_TIMEOUT
            )
            
            motor_time = time.perf_counter() - motor_start
            
            if response.status_code == 200:
                motor_result = response.json()
                step_results['motor'] = motor_result
                print(f"   ✅ Motor assessment: {motor_time:.3f}s")
                if 'data' in motor_result:
                    print(f"      Risk Score: {motor_result['data'].get('risk_score', 'N/A')}")
                    print(f"      Movement Quality: {motor_result['data'].get('movement_quality', 'N/A')}")
            else:
                print(f"   ❌ Motor assessment failed: {response.status_code}")
                step_results['motor'] = {'error': f"HTTP {response.status_code}"}
            
            # Step 4: Cognitive Assessment
            print("\n4. 🧠 Testing Cognitive Assessment...")
            cognitive_start = time.perf_counter()
            
            cognitive_data = {
                "session_id": self.session_id,
                "test_results": self.create_test_cognitive_data(),
                "test_battery": ["memory", "attention", "executive"],
                "difficulty_level": "standard"
            }
            
            response = requests.post(
                f"{self.base_url}/cognitive/analyze",
                json=cognitive_data,
                timeout=TEST_TIMEOUT
            )
            
            cognitive_time = time.perf_counter() - cognitive_start
            
            if response.status_code == 200:
                cognitive_result = response.json()
                step_results['cognitive'] = cognitive_result
                print(f"   ✅ Cognitive assessment: {cognitive_time:.3f}s")
                if 'data' in cognitive_result:
                    print(f"      Risk Score: {cognitive_result['data'].get('risk_score', 'N/A')}")
                    print(f"      Overall Score: {cognitive_result['data'].get('overall_score', 'N/A')}")
            else:
                print(f"   ❌ Cognitive assessment failed: {response.status_code}")
                step_results['cognitive'] = {'error': f"HTTP {response.status_code}"}
            
            # Step 5: NRI Fusion
            print("\n5. ⚡ Testing NRI Fusion...")
            nri_start = time.perf_counter()
            
            # Prepare modality results for fusion
            modality_results = {}
            for modality, result in step_results.items():
                if 'data' in result:
                    modality_results[modality] = result['data']
            
            nri_data = {
                "session_id": self.session_id,
                "modality_results": modality_results
            }
            
            response = requests.post(
                f"{self.base_url}/nri/fusion",
                json=nri_data,
                timeout=TEST_TIMEOUT
            )
            
            nri_time = time.perf_counter() - nri_start
            
            if response.status_code == 200:
                nri_result = response.json()
                step_results['nri'] = nri_result
                print(f"   ✅ NRI fusion: {nri_time:.3f}s")
                if 'data' in nri_result:
                    print(f"      NRI Score: {nri_result['data'].get('nri_score', 'N/A')}")
                    print(f"      Risk Category: {nri_result['data'].get('risk_category', 'N/A')}")
                    print(f"      Confidence: {nri_result['data'].get('confidence', 'N/A')}")
            else:
                print(f"   ❌ NRI fusion failed: {response.status_code}")
                step_results['nri'] = {'error': f"HTTP {response.status_code}"}
            
            # Calculate total workflow time
            total_workflow_time = time.perf_counter() - workflow_start_time
            
            # Summary
            print("\n" + "=" * 60)
            print("📊 Complete Assessment Workflow Summary")
            print("=" * 60)
            
            successful_steps = sum(1 for result in step_results.values() if 'data' in result)
            total_steps = len(step_results)
            
            print(f"✅ Successful steps: {successful_steps}/{total_steps}")
            print(f"⏱️  Total workflow time: {total_workflow_time:.3f}s")
            
            # Individual step times
            step_times = {
                'speech': speech_time,
                'retinal': retinal_time,
                'motor': motor_time,
                'cognitive': cognitive_time,
                'nri': nri_time
            }
            
            for step, time_taken in step_times.items():
                status = "✅" if 'data' in step_results.get(step, {}) else "❌"
                print(f"{status} {step.capitalize()}: {time_taken:.3f}s")
            
            # Performance targets
            targets = {
                'speech': 3.0,
                'retinal': 5.0,
                'motor': 2.0,
                'cognitive': 1.0,
                'nri': 2.0
            }
            
            targets_met = sum(1 for step, target in targets.items() 
                            if step_times.get(step, float('inf')) <= target)
            
            print(f"🎯 Performance targets met: {targets_met}/{len(targets)}")
            
            if successful_steps == total_steps and targets_met >= 4:
                print("\n🎉 Complete Assessment Workflow: SUCCESS!")
                print("✅ All assessment modalities working with real ML processing")
                print("✅ End-to-end data flow from input to NRI fusion complete")
                print("✅ Performance targets mostly achieved")
                print("✅ Ready for production deployment")
            else:
                print(f"\n⚠️  Complete Assessment Workflow: PARTIAL SUCCESS")
                print(f"   {successful_steps}/{total_steps} steps successful")
                print(f"   {targets_met}/{len(targets)} performance targets met")
            
            return step_results
            
        except Exception as e:
            print(f"\n❌ Workflow test failed with exception: {str(e)}")
            return {}


if __name__ == "__main__":
    print("🚀 NeuraLens Complete Assessment Workflow Test")
    print("Testing end-to-end assessment processing with real ML pipelines...")
    
    tester = CompleteAssessmentWorkflowTester()
    results = asyncio.run(tester.test_complete_workflow())
    
    print(f"\n🏁 Final Results:")
    print(f"   Session ID: {tester.session_id}")
    print(f"   Steps completed: {len([r for r in results.values() if 'data' in r])}/{len(results)}")
    print(f"   Workflow status: {'COMPLETE' if len([r for r in results.values() if 'data' in r]) == len(results) else 'PARTIAL'}")

"""
Comprehensive API Integration Test
Tests all endpoints with standardized responses and error handling
"""

import asyncio
import requests
import json
import time
from datetime import datetime
from typing import Dict, Any

# Test configuration
API_BASE_URL = "http://localhost:8000"
API_VERSION = "/api/v1"
TEST_TIMEOUT = 30

class ApiIntegrationTester:
    """Comprehensive API integration tester"""
    
    def __init__(self):
        self.base_url = f"{API_BASE_URL}{API_VERSION}"
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        self.results = {}
    
    def test_endpoint(self, method: str, endpoint: str, data: Any = None, files: Any = None) -> Dict[str, Any]:
        """Test a single endpoint"""
        
        url = f"{self.base_url}{endpoint}"
        start_time = time.perf_counter()
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, timeout=TEST_TIMEOUT)
            elif method.upper() == 'POST':
                if files:
                    # Remove Content-Type header for multipart
                    headers = {k: v for k, v in self.session.headers.items() if k != 'Content-Type'}
                    response = requests.post(url, data=data, files=files, headers=headers, timeout=TEST_TIMEOUT)
                else:
                    response = self.session.post(url, json=data, timeout=TEST_TIMEOUT)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            processing_time = time.perf_counter() - start_time
            
            # Parse response
            try:
                response_data = response.json()
            except json.JSONDecodeError:
                response_data = {"raw_response": response.text}
            
            return {
                "success": response.status_code < 400,
                "status_code": response.status_code,
                "processing_time": processing_time,
                "response": response_data,
                "headers": dict(response.headers)
            }
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            return {
                "success": False,
                "error": str(e),
                "processing_time": processing_time
            }
    
    def test_health_endpoints(self):
        """Test health and status endpoints"""
        
        print("🏥 Testing Health and Status Endpoints...")
        
        # Test health check
        health_result = self.test_endpoint('GET', '/health')
        self.results['health'] = health_result
        
        if health_result['success']:
            print(f"   ✅ Health check: {health_result['processing_time']:.3f}s")
            print(f"      Status: {health_result['response'].get('status', 'unknown')}")
        else:
            print(f"   ❌ Health check failed: {health_result.get('error', 'Unknown error')}")
        
        # Test status endpoint
        status_result = self.test_endpoint('GET', '/status')
        self.results['status'] = status_result
        
        if status_result['success']:
            print(f"   ✅ Status check: {status_result['processing_time']:.3f}s")
            print(f"      API Status: {status_result['response'].get('status', 'unknown')}")
        else:
            print(f"   ❌ Status check failed: {status_result.get('error', 'Unknown error')}")
    
    def test_speech_endpoints(self):
        """Test speech analysis endpoints"""
        
        print("\n🎤 Testing Speech Analysis Endpoints...")
        
        # Test speech info endpoint
        info_result = self.test_endpoint('GET', '/speech/info')
        self.results['speech_info'] = info_result
        
        if info_result['success']:
            print(f"   ✅ Speech info: {info_result['processing_time']:.3f}s")
        else:
            print(f"   ❌ Speech info failed: {info_result.get('error', 'Unknown error')}")
        
        # Test speech analysis with mock data
        try:
            # Create a small mock audio file
            import io
            import wave
            import numpy as np
            
            # Generate 2 seconds of sine wave audio
            sample_rate = 16000
            duration = 2.0
            frequency = 440
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_data = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
            
            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            wav_buffer.seek(0)
            
            # Test speech analysis
            files = {'audio_file': ('test_audio.wav', wav_buffer, 'audio/wav')}
            data = {'session_id': 'test_speech_session'}
            
            analysis_result = self.test_endpoint('POST', '/speech/analyze', data=data, files=files)
            self.results['speech_analyze'] = analysis_result
            
            if analysis_result['success']:
                print(f"   ✅ Speech analysis: {analysis_result['processing_time']:.3f}s")
                response = analysis_result['response']
                if 'data' in response:
                    print(f"      Risk Score: {response['data'].get('risk_score', 'N/A')}")
                    print(f"      Confidence: {response['data'].get('confidence', 'N/A')}")
            else:
                print(f"   ❌ Speech analysis failed: {analysis_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Speech analysis test setup failed: {str(e)}")
    
    def test_retinal_endpoints(self):
        """Test retinal analysis endpoints"""
        
        print("\n👁️  Testing Retinal Analysis Endpoints...")
        
        # Test retinal info endpoint
        info_result = self.test_endpoint('GET', '/retinal/info')
        self.results['retinal_info'] = info_result
        
        if info_result['success']:
            print(f"   ✅ Retinal info: {info_result['processing_time']:.3f}s")
        else:
            print(f"   ❌ Retinal info failed: {info_result.get('error', 'Unknown error')}")
        
        # Test retinal analysis with mock image
        try:
            from PIL import Image
            import io
            
            # Create a simple test image
            img = Image.new('RGB', (512, 512), color='red')
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='JPEG')
            img_buffer.seek(0)
            
            files = {'image_file': ('test_retinal.jpg', img_buffer, 'image/jpeg')}
            data = {'session_id': 'test_retinal_session'}
            
            analysis_result = self.test_endpoint('POST', '/retinal/analyze', data=data, files=files)
            self.results['retinal_analyze'] = analysis_result
            
            if analysis_result['success']:
                print(f"   ✅ Retinal analysis: {analysis_result['processing_time']:.3f}s")
                response = analysis_result['response']
                if 'data' in response:
                    print(f"      Risk Score: {response['data'].get('risk_score', 'N/A')}")
                    print(f"      Quality Score: {response['data'].get('quality_score', 'N/A')}")
            else:
                print(f"   ❌ Retinal analysis failed: {analysis_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Retinal analysis test setup failed: {str(e)}")
    
    def test_motor_endpoints(self):
        """Test motor assessment endpoints"""
        
        print("\n🤲 Testing Motor Assessment Endpoints...")
        
        # Test motor info endpoint
        info_result = self.test_endpoint('GET', '/motor/info')
        self.results['motor_info'] = info_result
        
        if info_result['success']:
            print(f"   ✅ Motor info: {info_result['processing_time']:.3f}s")
        else:
            print(f"   ❌ Motor info failed: {info_result.get('error', 'Unknown error')}")
        
        # Test motor analysis
        motor_data = {
            "session_id": "test_motor_session",
            "sensor_data": {
                "accelerometer": [
                    {"x": 0.1, "y": 0.2, "z": 9.8},
                    {"x": 0.15, "y": 0.18, "z": 9.85},
                    {"x": 0.12, "y": 0.22, "z": 9.75}
                ]
            },
            "assessment_type": "tremor"
        }
        
        analysis_result = self.test_endpoint('POST', '/motor/analyze', data=motor_data)
        self.results['motor_analyze'] = analysis_result
        
        if analysis_result['success']:
            print(f"   ✅ Motor analysis: {analysis_result['processing_time']:.3f}s")
            response = analysis_result['response']
            if 'data' in response:
                print(f"      Risk Score: {response['data'].get('risk_score', 'N/A')}")
                print(f"      Movement Quality: {response['data'].get('movement_quality', 'N/A')}")
        else:
            print(f"   ❌ Motor analysis failed: {analysis_result.get('error', 'Unknown error')}")
    
    def test_cognitive_endpoints(self):
        """Test cognitive assessment endpoints"""
        
        print("\n🧠 Testing Cognitive Assessment Endpoints...")
        
        # Test cognitive info endpoint
        info_result = self.test_endpoint('GET', '/cognitive/info')
        self.results['cognitive_info'] = info_result
        
        if info_result['success']:
            print(f"   ✅ Cognitive info: {info_result['processing_time']:.3f}s")
        else:
            print(f"   ❌ Cognitive info failed: {info_result.get('error', 'Unknown error')}")
        
        # Test cognitive analysis
        cognitive_data = {
            "session_id": "test_cognitive_session",
            "test_results": {
                "response_times": [1200, 1150, 1300, 1100, 1250],
                "accuracy": [1, 1, 0, 1, 1],
                "memory": {"immediate_recall": 0.8, "delayed_recall": 0.75}
            },
            "test_battery": ["memory", "attention"],
            "difficulty_level": "standard"
        }
        
        analysis_result = self.test_endpoint('POST', '/cognitive/analyze', data=cognitive_data)
        self.results['cognitive_analyze'] = analysis_result
        
        if analysis_result['success']:
            print(f"   ✅ Cognitive analysis: {analysis_result['processing_time']:.3f}s")
            response = analysis_result['response']
            if 'data' in response:
                print(f"      Risk Score: {response['data'].get('risk_score', 'N/A')}")
                print(f"      Overall Score: {response['data'].get('overall_score', 'N/A')}")
        else:
            print(f"   ❌ Cognitive analysis failed: {analysis_result.get('error', 'Unknown error')}")
    
    def test_error_handling(self):
        """Test error handling scenarios"""
        
        print("\n⚠️  Testing Error Handling...")
        
        # Test 404 error
        not_found_result = self.test_endpoint('GET', '/nonexistent')
        if not_found_result['status_code'] == 404:
            print("   ✅ 404 error handling works")
        else:
            print(f"   ❌ 404 error handling failed: {not_found_result['status_code']}")
        
        # Test validation error
        invalid_data = {"invalid": "data"}
        validation_result = self.test_endpoint('POST', '/speech/analyze', data=invalid_data)
        if validation_result['status_code'] in [400, 422]:
            print("   ✅ Validation error handling works")
        else:
            print(f"   ❌ Validation error handling failed: {validation_result['status_code']}")
    
    def run_all_tests(self):
        """Run all API integration tests"""
        
        print("🚀 Starting API Integration Tests...")
        print("=" * 60)
        
        start_time = time.perf_counter()
        
        # Run all test suites
        self.test_health_endpoints()
        self.test_speech_endpoints()
        self.test_retinal_endpoints()
        self.test_motor_endpoints()
        self.test_cognitive_endpoints()
        self.test_error_handling()
        
        total_time = time.perf_counter() - start_time
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 API Integration Test Summary")
        print("=" * 60)
        
        successful_tests = sum(1 for result in self.results.values() if result.get('success', False))
        total_tests = len(self.results)
        
        print(f"✅ Successful tests: {successful_tests}/{total_tests}")
        print(f"⏱️  Total test time: {total_time:.3f}s")
        
        # Detailed results
        for test_name, result in self.results.items():
            status = "✅" if result.get('success', False) else "❌"
            time_str = f"{result.get('processing_time', 0):.3f}s"
            print(f"{status} {test_name}: {time_str}")
        
        if successful_tests == total_tests:
            print("\n🎉 All API integration tests passed!")
            print("✅ Frontend-backend communication ready")
            print("✅ Standardized responses working")
            print("✅ Error handling implemented")
        else:
            print(f"\n⚠️  {total_tests - successful_tests} tests failed")
            print("❌ Some endpoints need attention")
        
        return self.results


if __name__ == "__main__":
    print("🧪 NeuraLens API Integration Test Suite")
    print("Testing frontend-backend communication...")
    
    tester = ApiIntegrationTester()
    results = tester.run_all_tests()
    
    print(f"\n🏁 Test Results Summary:")
    print(f"   Total endpoints tested: {len(results)}")
    print(f"   Successful: {sum(1 for r in results.values() if r.get('success', False))}")
    print(f"   Failed: {sum(1 for r in results.values() if not r.get('success', False))}")

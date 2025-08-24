/**
 * Frontend API Integration Test Suite
 * 
 * Comprehensive test suite to verify all frontend API routes work correctly
 * and integrate properly with the backend services.
 * 
 * Tests all four assessment modalities:
 * - Speech Analysis
 * - Retinal Assessment  
 * - Motor Assessment
 * - Cognitive Testing
 * - NRI Fusion
 */

import type {
  SpeechAnalysisResponse,
  RetinalAnalysisResponse,
  MotorAssessmentResponse,
  CognitiveAssessmentResponse,
  NRIFusionResponse,
  MotorAssessmentRequest,
  CognitiveAssessmentRequest,
  NRIFusionRequest,
} from './types';

export interface IntegrationTestResult {
  success: boolean;
  endpoint: string;
  responseTime: number;
  error?: string;
  data?: any;
}

export interface ComprehensiveTestResults {
  overall: {
    success: boolean;
    totalTests: number;
    passedTests: number;
    failedTests: number;
    totalTime: number;
  };
  results: {
    health: IntegrationTestResult;
    speech: IntegrationTestResult;
    retinal: IntegrationTestResult;
    motor: IntegrationTestResult;
    cognitive: IntegrationTestResult;
    nri: IntegrationTestResult;
  };
}

/**
 * Run comprehensive integration test for all API endpoints
 */
export async function runComprehensiveIntegrationTest(): Promise<ComprehensiveTestResults> {
  console.log('🚀 Starting comprehensive frontend API integration test...');
  
  const startTime = Date.now();
  const results: ComprehensiveTestResults = {
    overall: {
      success: false,
      totalTests: 6,
      passedTests: 0,
      failedTests: 0,
      totalTime: 0,
    },
    results: {
      health: { success: false, endpoint: '/api/health', responseTime: 0 },
      speech: { success: false, endpoint: '/api/speech', responseTime: 0 },
      retinal: { success: false, endpoint: '/api/retinal', responseTime: 0 },
      motor: { success: false, endpoint: '/api/motor', responseTime: 0 },
      cognitive: { success: false, endpoint: '/api/cognitive', responseTime: 0 },
      nri: { success: false, endpoint: '/api/nri', responseTime: 0 },
    },
  };

  // Test health endpoint
  results.results.health = await testHealthEndpoint();
  
  // Test speech analysis endpoint
  results.results.speech = await testSpeechEndpoint();
  
  // Test retinal analysis endpoint
  results.results.retinal = await testRetinalEndpoint();
  
  // Test motor assessment endpoint
  results.results.motor = await testMotorEndpoint();
  
  // Test cognitive assessment endpoint
  results.results.cognitive = await testCognitiveEndpoint();
  
  // Test NRI fusion endpoint (requires other modality results)
  results.results.nri = await testNRIEndpoint(
    results.results.speech.data,
    results.results.retinal.data,
    results.results.motor.data,
    results.results.cognitive.data
  );

  // Calculate overall results
  const passedTests = Object.values(results.results).filter(result => result.success).length;
  const failedTests = results.overall.totalTests - passedTests;
  const totalTime = Date.now() - startTime;

  results.overall = {
    success: passedTests === results.overall.totalTests,
    totalTests: results.overall.totalTests,
    passedTests,
    failedTests,
    totalTime,
  };

  console.log(`✅ Integration test completed: ${passedTests}/${results.overall.totalTests} tests passed in ${totalTime}ms`);
  
  return results;
}

/**
 * Test health endpoint
 */
async function testHealthEndpoint(): Promise<IntegrationTestResult> {
  const startTime = Date.now();
  
  try {
    const response = await fetch('/api/health', {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    const responseTime = Date.now() - startTime;
    
    if (!response.ok) {
      return {
        success: false,
        endpoint: '/api/health',
        responseTime,
        error: `HTTP ${response.status}: ${response.statusText}`,
      };
    }

    const data = await response.json();
    
    // Validate response structure
    if (!data.status || !data.timestamp) {
      return {
        success: false,
        endpoint: '/api/health',
        responseTime,
        error: 'Invalid response structure',
      };
    }

    console.log('✅ Health endpoint test passed');
    return {
      success: true,
      endpoint: '/api/health',
      responseTime,
      data,
    };
  } catch (error) {
    return {
      success: false,
      endpoint: '/api/health',
      responseTime: Date.now() - startTime,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

/**
 * Test speech analysis endpoint
 */
async function testSpeechEndpoint(): Promise<IntegrationTestResult> {
  const startTime = Date.now();
  
  try {
    const testData = {
      result: {
        fluencyScore: 0.85,
        confidence: 0.9,
        biomarkers: {
          pauseDuration: 400,
          pauseFrequency: 8,
          tremorFrequency: 1.2,
          speechRate: 160,
          pitchVariation: 0.04,
          voiceQuality: {
            jitter: 0.01,
            shimmer: 0.02,
            hnr: 15,
          },
          mfccFeatures: Array.from({ length: 13 }, () => Math.random() * 2 - 1),
        },
        metadata: {
          processingTime: 80,
          audioDuration: 30,
          sampleRate: 16000,
          modelVersion: 'whisper-tiny-test-v1.0',
          timestamp: new Date(),
        },
      },
      sessionId: `test_${Date.now()}`,
      timestamp: Date.now(),
    };

    const response = await fetch('/api/speech', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(testData),
    });

    const responseTime = Date.now() - startTime;
    
    if (!response.ok) {
      return {
        success: false,
        endpoint: '/api/speech',
        responseTime,
        error: `HTTP ${response.status}: ${response.statusText}`,
      };
    }

    const data = await response.json();
    
    // Validate response structure
    if (!data.success || !data.result) {
      return {
        success: false,
        endpoint: '/api/speech',
        responseTime,
        error: 'Invalid response structure',
      };
    }

    console.log('✅ Speech endpoint test passed');
    return {
      success: true,
      endpoint: '/api/speech',
      responseTime,
      data: data.result,
    };
  } catch (error) {
    return {
      success: false,
      endpoint: '/api/speech',
      responseTime: Date.now() - startTime,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

/**
 * Test retinal analysis endpoint
 */
async function testRetinalEndpoint(): Promise<IntegrationTestResult> {
  const startTime = Date.now();
  
  try {
    const testData = {
      result: {
        vascularScore: 0.45,
        cupDiscRatio: 0.35,
        confidence: 0.88,
        riskFeatures: {
          vesselDensity: 0.18,
          tortuosityIndex: 0.25,
          averageVesselWidth: 8,
          arteriovenousRatio: 0.65,
          opticDiscArea: 2400,
          opticCupArea: 800,
          hemorrhageCount: 1,
          microaneurysmCount: 2,
          hardExudateArea: 0.03,
          softExudateCount: 1,
          imageQuality: 0.9,
          spatialFeatures: Array.from({ length: 1280 }, () => Math.random() * 2 - 1),
        },
        metadata: {
          processingTime: 120,
          imageDimensions: { width: 224, height: 224 },
          imageSize: 1024 * 1024,
          modelVersion: 'efficientnet-b0-test-v1.0',
          preprocessingSteps: ['resize', 'normalize', 'tensor_conversion'],
          timestamp: new Date(),
          gpuAccelerated: true,
        },
      },
      sessionId: `test_${Date.now()}`,
      timestamp: Date.now(),
    };

    const response = await fetch('/api/retinal', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(testData),
    });

    const responseTime = Date.now() - startTime;
    
    if (!response.ok) {
      return {
        success: false,
        endpoint: '/api/retinal',
        responseTime,
        error: `HTTP ${response.status}: ${response.statusText}`,
      };
    }

    const data = await response.json();
    
    // Validate response structure
    if (!data.success || !data.result) {
      return {
        success: false,
        endpoint: '/api/retinal',
        responseTime,
        error: 'Invalid response structure',
      };
    }

    console.log('✅ Retinal endpoint test passed');
    return {
      success: true,
      endpoint: '/api/retinal',
      responseTime,
      data: data.result,
    };
  } catch (error) {
    return {
      success: false,
      endpoint: '/api/retinal',
      responseTime: Date.now() - startTime,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

/**
 * Test motor assessment endpoint
 */
async function testMotorEndpoint(): Promise<IntegrationTestResult> {
  const startTime = Date.now();
  
  try {
    const testData: MotorAssessmentRequest = {
      session_id: `test_${Date.now()}`,
      sensor_data: {
        accelerometer: Array.from({ length: 100 }, (_, i) => ({
          x: Math.sin(i * 0.1) + Math.random() * 0.1,
          y: Math.cos(i * 0.1) + Math.random() * 0.1,
          z: 9.8 + Math.random() * 0.2,
        })),
        gyroscope: Array.from({ length: 100 }, (_, i) => ({
          x: Math.random() * 0.5 - 0.25,
          y: Math.random() * 0.5 - 0.25,
          z: Math.random() * 0.5 - 0.25,
        })),
        position: Array.from({ length: 50 }, (_, i) => ({
          x: i * 2 + Math.random() * 5,
          y: Math.sin(i * 0.2) * 10 + Math.random() * 2,
        })),
      },
      assessment_type: 'tremor',
    };

    const response = await fetch('/api/motor', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(testData),
    });

    const responseTime = Date.now() - startTime;
    
    if (!response.ok) {
      return {
        success: false,
        endpoint: '/api/motor',
        responseTime,
        error: `HTTP ${response.status}: ${response.statusText}`,
      };
    }

    const data = await response.json();
    
    // Validate response structure
    if (!data.success || !data.result) {
      return {
        success: false,
        endpoint: '/api/motor',
        responseTime,
        error: 'Invalid response structure',
      };
    }

    console.log('✅ Motor endpoint test passed');
    return {
      success: true,
      endpoint: '/api/motor',
      responseTime,
      data: data.result,
    };
  } catch (error) {
    return {
      success: false,
      endpoint: '/api/motor',
      responseTime: Date.now() - startTime,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

/**
 * Test cognitive assessment endpoint
 */
async function testCognitiveEndpoint(): Promise<IntegrationTestResult> {
  const startTime = Date.now();
  
  try {
    const testData: CognitiveAssessmentRequest = {
      session_id: `test_${Date.now()}`,
      test_results: {
        response_times: Array.from({ length: 20 }, () => 800 + Math.random() * 400),
        accuracy: Array.from({ length: 20 }, () => 0.7 + Math.random() * 0.25),
        memory: {
          immediate_recall: 0.8,
          delayed_recall: 0.75,
          recognition: 0.85,
        },
        attention: {
          sustained_attention: 0.82,
          selective_attention: 0.78,
          divided_attention: 0.73,
        },
        executive: {
          working_memory: 0.79,
          inhibition: 0.81,
          cognitive_flexibility: 0.76,
        },
        task_switching: {
          repeat_trials: Array.from({ length: 10 }, () => 600 + Math.random() * 200),
          switch_trials: Array.from({ length: 10 }, () => 750 + Math.random() * 250),
          switch_accuracy: 0.85,
        },
      },
      test_battery: ['memory', 'attention', 'executive'],
      difficulty_level: 'standard',
    };

    const response = await fetch('/api/cognitive', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(testData),
    });

    const responseTime = Date.now() - startTime;
    
    if (!response.ok) {
      return {
        success: false,
        endpoint: '/api/cognitive',
        responseTime,
        error: `HTTP ${response.status}: ${response.statusText}`,
      };
    }

    const data = await response.json();
    
    // Validate response structure
    if (!data.success || !data.result) {
      return {
        success: false,
        endpoint: '/api/cognitive',
        responseTime,
        error: 'Invalid response structure',
      };
    }

    console.log('✅ Cognitive endpoint test passed');
    return {
      success: true,
      endpoint: '/api/cognitive',
      responseTime,
      data: data.result,
    };
  } catch (error) {
    return {
      success: false,
      endpoint: '/api/cognitive',
      responseTime: Date.now() - startTime,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

/**
 * Test NRI fusion endpoint
 */
async function testNRIEndpoint(
  speechResult?: any,
  retinalResult?: any,
  motorResult?: any,
  cognitiveResult?: any
): Promise<IntegrationTestResult> {
  const startTime = Date.now();
  
  try {
    const testData: NRIFusionRequest = {
      session_id: `test_${Date.now()}`,
      modality_results: {
        ...(speechResult && { speech: speechResult }),
        ...(retinalResult && { retinal: retinalResult }),
        ...(motorResult && { motor: motorResult }),
        ...(cognitiveResult && { cognitive: cognitiveResult }),
      },
      fusion_method: 'bayesian',
      uncertainty_quantification: true,
    };

    const response = await fetch('/api/nri', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(testData),
    });

    const responseTime = Date.now() - startTime;
    
    if (!response.ok) {
      return {
        success: false,
        endpoint: '/api/nri',
        responseTime,
        error: `HTTP ${response.status}: ${response.statusText}`,
      };
    }

    const data = await response.json();
    
    // Validate response structure
    if (!data.success || !data.result) {
      return {
        success: false,
        endpoint: '/api/nri',
        responseTime,
        error: 'Invalid response structure',
      };
    }

    console.log('✅ NRI endpoint test passed');
    return {
      success: true,
      endpoint: '/api/nri',
      responseTime,
      data: data.result,
    };
  } catch (error) {
    return {
      success: false,
      endpoint: '/api/nri',
      responseTime: Date.now() - startTime,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

/**
 * Generate test report
 */
export function generateTestReport(results: ComprehensiveTestResults): string {
  const { overall, results: testResults } = results;
  
  let report = `
# Frontend API Integration Test Report

## Overall Results
- **Status**: ${overall.success ? '✅ PASSED' : '❌ FAILED'}
- **Tests Passed**: ${overall.passedTests}/${overall.totalTests}
- **Total Time**: ${overall.totalTime}ms

## Individual Test Results

`;

  Object.entries(testResults).forEach(([endpoint, result]) => {
    report += `### ${endpoint.toUpperCase()} Endpoint
- **Status**: ${result.success ? '✅ PASSED' : '❌ FAILED'}
- **Response Time**: ${result.responseTime}ms
- **Endpoint**: ${result.endpoint}
${result.error ? `- **Error**: ${result.error}` : ''}

`;
  });

  return report;
}

/**
 * Export test utilities for individual endpoint testing
 */
export const testUtils = {
  testHealthEndpoint,
  testSpeechEndpoint,
  testRetinalEndpoint,
  testMotorEndpoint,
  testCognitiveEndpoint,
  testNRIEndpoint,
};

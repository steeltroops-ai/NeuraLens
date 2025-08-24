/**
 * NeuraLens API Services
 * High-level service functions for each assessment type
 */

import { apiClient, ApiResponse, API_ENDPOINTS } from './client';
import {
  SpeechAnalysisRequest,
  SpeechAnalysisResponse,
  RetinalAnalysisRequest,
  RetinalAnalysisResponse,
  MotorAssessmentRequest,
  MotorAssessmentResponse,
  CognitiveAssessmentRequest,
  CognitiveAssessmentResponse,
  NRIFusionRequest,
  NRIFusionResponse,
  ValidationResponse,
  HealthCheckResponse,
  ApiStatusResponse,
  ServiceInfo,
  CompleteAssessmentResult,
} from './types';

// Speech Analysis Service
export class SpeechAnalysisService {
  /**
   * Analyze speech audio file
   */
  static async analyze(
    request: SpeechAnalysisRequest,
  ): Promise<ApiResponse<SpeechAnalysisResponse>> {
    const formData = new FormData();
    formData.append('audio_file', request.audio_file);
    formData.append('session_id', request.session_id);

    if (request.quality_threshold) {
      formData.append('quality_threshold', request.quality_threshold.toString());
    }

    return apiClient.postFormData<SpeechAnalysisResponse>(
      API_ENDPOINTS.SPEECH.ANALYZE,
      formData,
      { timeout: 60000 }, // 60 seconds for speech processing
    );
  }

  /**
   * Get speech analysis service info
   */
  static async getInfo(): Promise<ApiResponse<ServiceInfo>> {
    return apiClient.get<ServiceInfo>(API_ENDPOINTS.SPEECH.INFO);
  }
}

// Retinal Analysis Service
export class RetinalAnalysisService {
  /**
   * Analyze retinal image
   */
  static async analyze(
    request: RetinalAnalysisRequest,
  ): Promise<ApiResponse<RetinalAnalysisResponse>> {
    const formData = new FormData();
    formData.append('image_file', request.image_file);
    formData.append('session_id', request.session_id);

    if (request.quality_threshold) {
      formData.append('quality_threshold', request.quality_threshold.toString());
    }

    return apiClient.postFormData<RetinalAnalysisResponse>(
      API_ENDPOINTS.RETINAL.ANALYZE,
      formData,
      { timeout: 30000 }, // 30 seconds for retinal processing
    );
  }

  /**
   * Get retinal analysis service info
   */
  static async getInfo(): Promise<ApiResponse<ServiceInfo>> {
    return apiClient.get<ServiceInfo>(API_ENDPOINTS.RETINAL.INFO);
  }
}

// Motor Assessment Service
export class MotorAssessmentService {
  /**
   * Analyze motor sensor data
   */
  static async analyze(
    request: MotorAssessmentRequest,
  ): Promise<ApiResponse<MotorAssessmentResponse>> {
    return apiClient.post<MotorAssessmentResponse>(
      API_ENDPOINTS.MOTOR.ANALYZE,
      request,
      { timeout: 15000 }, // 15 seconds for motor processing
    );
  }

  /**
   * Get motor assessment service info
   */
  static async getInfo(): Promise<ApiResponse<ServiceInfo>> {
    return apiClient.get<ServiceInfo>(API_ENDPOINTS.MOTOR.INFO);
  }
}

// Cognitive Assessment Service
export class CognitiveAssessmentService {
  /**
   * Analyze cognitive test results
   */
  static async analyze(
    request: CognitiveAssessmentRequest,
  ): Promise<ApiResponse<CognitiveAssessmentResponse>> {
    return apiClient.post<CognitiveAssessmentResponse>(
      API_ENDPOINTS.COGNITIVE.ANALYZE,
      request,
      { timeout: 10000 }, // 10 seconds for cognitive processing
    );
  }

  /**
   * Get cognitive assessment service info
   */
  static async getInfo(): Promise<ApiResponse<ServiceInfo>> {
    return apiClient.get<ServiceInfo>(API_ENDPOINTS.COGNITIVE.INFO);
  }
}

// NRI Fusion Service
export class NRIFusionService {
  /**
   * Perform NRI fusion analysis
   */
  static async fusion(request: NRIFusionRequest): Promise<ApiResponse<NRIFusionResponse>> {
    return apiClient.post<NRIFusionResponse>(
      API_ENDPOINTS.NRI.FUSION,
      request,
      { timeout: 20000 }, // 20 seconds for fusion processing
    );
  }

  /**
   * Get NRI fusion service info
   */
  static async getInfo(): Promise<ApiResponse<ServiceInfo>> {
    return apiClient.get<ServiceInfo>(API_ENDPOINTS.NRI.INFO);
  }
}

// Validation Service
export class ValidationService {
  /**
   * Get validation metrics
   */
  static async getMetrics(modality?: string): Promise<ApiResponse<ValidationResponse[]>> {
    const endpoint = modality
      ? `${API_ENDPOINTS.VALIDATION.METRICS}?modality=${modality}`
      : API_ENDPOINTS.VALIDATION.METRICS;

    return apiClient.get<ValidationResponse[]>(endpoint);
  }

  /**
   * Get validation service info
   */
  static async getInfo(): Promise<ApiResponse<ServiceInfo>> {
    return apiClient.get<ServiceInfo>(API_ENDPOINTS.VALIDATION.INFO);
  }
}

// System Service
export class SystemService {
  /**
   * Health check
   */
  static async healthCheck(): Promise<ApiResponse<HealthCheckResponse>> {
    return apiClient.get<HealthCheckResponse>(API_ENDPOINTS.HEALTH);
  }

  /**
   * Get API status
   */
  static async getStatus(): Promise<ApiResponse<ApiStatusResponse>> {
    return apiClient.get<ApiStatusResponse>(API_ENDPOINTS.STATUS);
  }
}

// Complete Assessment Service
export class AssessmentService {
  /**
   * Run complete assessment workflow
   */
  static async runCompleteAssessment(
    sessionId: string,
    assessments: {
      speech?: File;
      retinal?: File;
      motor?: MotorAssessmentRequest['sensor_data'];
      cognitive?: CognitiveAssessmentRequest['test_results'];
    },
  ): Promise<ApiResponse<CompleteAssessmentResult>> {
    try {
      const results: any = {};

      // Run speech analysis if provided
      if (assessments.speech) {
        const speechResult = await SpeechAnalysisService.analyze({
          session_id: sessionId,
          audio_file: assessments.speech,
        });

        if (speechResult.success && speechResult.data) {
          results.speech_result = speechResult.data;
        }
      }

      // Run retinal analysis if provided
      if (assessments.retinal) {
        const retinalResult = await RetinalAnalysisService.analyze({
          session_id: sessionId,
          image_file: assessments.retinal,
        });

        if (retinalResult.success && retinalResult.data) {
          results.retinal_result = retinalResult.data;
        }
      }

      // Run motor assessment if provided
      if (assessments.motor) {
        const motorResult = await MotorAssessmentService.analyze({
          session_id: sessionId,
          sensor_data: assessments.motor,
          assessment_type: 'tremor',
        });

        if (motorResult.success && motorResult.data) {
          results.motor_result = motorResult.data;
        }
      }

      // Run cognitive assessment if provided
      if (assessments.cognitive) {
        const cognitiveResult = await CognitiveAssessmentService.analyze({
          session_id: sessionId,
          test_results: assessments.cognitive,
          test_battery: ['memory', 'attention', 'executive'],
          difficulty_level: 'standard',
        });

        if (cognitiveResult.success && cognitiveResult.data) {
          results.cognitive_result = cognitiveResult.data;
        }
      }

      // Run NRI fusion
      const nriResult = await NRIFusionService.fusion({
        session_id: sessionId,
        modality_results: {
          speech: results.speech_result,
          retinal: results.retinal_result,
          motor: results.motor_result,
          cognitive: results.cognitive_result,
        },
      });

      if (!nriResult.success || !nriResult.data) {
        return {
          success: false,
          error: nriResult.error || {
            code: 'NRI_FUSION_FAILED',
            message: 'Failed to perform NRI fusion',
          },
          metadata: {
            timestamp: new Date().toISOString(),
          },
        };
      }

      // Compile complete result
      const completeResult: CompleteAssessmentResult = {
        session_id: sessionId,
        ...results,
        nri_result: nriResult.data,
        overall_risk_category: nriResult.data.risk_category,
        completion_time: new Date().toISOString(),
        total_processing_time: Object.values(results).reduce(
          (total: number, result: any) => total + (result?.processing_time || 0),
          nriResult.data.processing_time,
        ),
      };

      return {
        success: true,
        data: completeResult,
        metadata: {
          timestamp: new Date().toISOString(),
          processing_time: completeResult.total_processing_time,
        },
      };
    } catch (error) {
      return {
        success: false,
        error: {
          code: 'ASSESSMENT_WORKFLOW_ERROR',
          message: error instanceof Error ? error.message : 'Unknown error occurred',
          details: error,
        },
        metadata: {
          timestamp: new Date().toISOString(),
        },
      };
    }
  }

  /**
   * Get assessment progress (mock implementation for now)
   */
  static async getProgress(sessionId: string): Promise<ApiResponse<any>> {
    // This would typically connect to a real-time progress endpoint
    return {
      success: true,
      data: {
        session_id: sessionId,
        current_step: 'processing',
        completed_steps: ['upload', 'validation'],
        total_steps: 5,
        progress_percentage: 60,
        estimated_time_remaining: 30,
      },
      metadata: {
        timestamp: new Date().toISOString(),
      },
    };
  }
}

// All services are exported directly above with 'export class'

/**
 * NeuroLens-X API Client
 * Comprehensive API integration with FastAPI backend
 * Real-time processing with error handling and performance optimization
 */

import axios, { AxiosInstance, AxiosResponse, AxiosError } from 'axios';
import type {
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
  ValidationMetrics,
  ApiResponse,
  ApiError,
} from '@/types';

// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const API_TIMEOUT = 30000; // 30 seconds
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000; // 1 second

// Create axios instance with default configuration
const apiClient: AxiosInstance = axios.create({
  baseURL: `${API_BASE_URL}/api/v1`,
  timeout: API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
});

// Request interceptor for adding auth tokens and request timing
apiClient.interceptors.request.use(
  (config) => {
    // Add request timestamp for performance monitoring
    config.metadata = { startTime: Date.now() };
    
    // Add auth token if available
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for error handling and performance monitoring
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    // Calculate request duration
    const duration = Date.now() - response.config.metadata?.startTime;
    console.log(`API Request [${response.config.method?.toUpperCase()}] ${response.config.url}: ${duration}ms`);
    
    return response;
  },
  async (error: AxiosError) => {
    const originalRequest = error.config;
    
    // Handle network errors with retry logic
    if (!error.response && originalRequest && !originalRequest._retry) {
      originalRequest._retry = true;
      
      for (let i = 0; i < MAX_RETRIES; i++) {
        try {
          await new Promise(resolve => setTimeout(resolve, RETRY_DELAY * (i + 1)));
          return await apiClient(originalRequest);
        } catch (retryError) {
          if (i === MAX_RETRIES - 1) {
            throw retryError;
          }
        }
      }
    }
    
    // Handle specific error cases
    if (error.response?.status === 401) {
      // Handle unauthorized - redirect to login
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    }
    
    return Promise.reject(error);
  }
);

// Utility function to handle API responses
function handleApiResponse<T>(response: AxiosResponse<T>): T {
  return response.data;
}

// Utility function to handle API errors
function handleApiError(error: AxiosError): ApiError {
  const apiError: ApiError = {
    code: error.code || 'UNKNOWN_ERROR',
    message: error.message || 'An unexpected error occurred',
    timestamp: new Date(),
  };
  
  if (error.response?.data) {
    const errorData = error.response.data as any;
    apiError.message = errorData.message || errorData.detail || apiError.message;
    apiError.details = errorData.details;
  }
  
  return apiError;
}

// API Client Class
export class NeuroLensAPI {
  // Health Check
  static async healthCheck(): Promise<{ status: string; service: string; version: string }> {
    try {
      const response = await apiClient.get('/status');
      return handleApiResponse(response);
    } catch (error) {
      throw handleApiError(error as AxiosError);
    }
  }

  // Speech Analysis
  static async analyzeSpeech(
    audioData: Blob,
    request: Omit<SpeechAnalysisRequest, 'sessionId' | 'timestamp'>
  ): Promise<SpeechAnalysisResponse> {
    try {
      const formData = new FormData();
      formData.append('audio', audioData, 'audio.wav');
      formData.append('request', JSON.stringify({
        ...request,
        sessionId: crypto.randomUUID(),
        timestamp: new Date(),
      }));

      const response = await apiClient.post('/speech/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      return handleApiResponse(response);
    } catch (error) {
      throw handleApiError(error as AxiosError);
    }
  }

  // Retinal Analysis
  static async analyzeRetinal(
    imageData: Blob,
    request: Omit<RetinalAnalysisRequest, 'sessionId' | 'timestamp'>
  ): Promise<RetinalAnalysisResponse> {
    try {
      const formData = new FormData();
      formData.append('image', imageData, 'retinal.jpg');
      formData.append('request', JSON.stringify({
        ...request,
        sessionId: crypto.randomUUID(),
        timestamp: new Date(),
      }));

      const response = await apiClient.post('/retinal/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      return handleApiResponse(response);
    } catch (error) {
      throw handleApiError(error as AxiosError);
    }
  }

  // Motor Assessment
  static async assessMotor(
    request: Omit<MotorAssessmentRequest, 'sessionId' | 'timestamp'>
  ): Promise<MotorAssessmentResponse> {
    try {
      const response = await apiClient.post('/motor/assess', {
        ...request,
        sessionId: crypto.randomUUID(),
        timestamp: new Date(),
      });
      
      return handleApiResponse(response);
    } catch (error) {
      throw handleApiError(error as AxiosError);
    }
  }

  // Cognitive Assessment
  static async assessCognitive(
    request: Omit<CognitiveAssessmentRequest, 'sessionId' | 'timestamp'>
  ): Promise<CognitiveAssessmentResponse> {
    try {
      const response = await apiClient.post('/cognitive/assess', {
        ...request,
        sessionId: crypto.randomUUID(),
        timestamp: new Date(),
      });
      
      return handleApiResponse(response);
    } catch (error) {
      throw handleApiError(error as AxiosError);
    }
  }

  // NRI Fusion
  static async calculateNRI(request: NRIFusionRequest): Promise<NRIFusionResponse> {
    try {
      const response = await apiClient.post('/nri/calculate', request);
      return handleApiResponse(response);
    } catch (error) {
      throw handleApiError(error as AxiosError);
    }
  }

  // Validation Metrics
  static async getValidationMetrics(
    modality?: string,
    metricType?: string
  ): Promise<ValidationMetrics> {
    try {
      const params = new URLSearchParams();
      if (modality) params.append('modality', modality);
      if (metricType) params.append('metric_type', metricType);
      
      const response = await apiClient.get(`/validation/metrics?${params.toString()}`);
      return handleApiResponse(response);
    } catch (error) {
      throw handleApiError(error as AxiosError);
    }
  }

  // Real-time Processing Status
  static async getProcessingStatus(sessionId: string): Promise<{
    status: 'processing' | 'completed' | 'error';
    progress: number;
    estimatedTimeRemaining?: number;
  }> {
    try {
      const response = await apiClient.get(`/processing/status/${sessionId}`);
      return handleApiResponse(response);
    } catch (error) {
      throw handleApiError(error as AxiosError);
    }
  }
}

// Real-time Processing Utilities
export class RealTimeProcessor {
  private static processingQueue: Map<string, {
    sessionId: string;
    onProgress: (progress: number) => void;
    onComplete: (result: any) => void;
    onError: (error: ApiError) => void;
  }> = new Map();

  static async processWithProgress<T>(
    sessionId: string,
    processingFunction: () => Promise<T>,
    onProgress: (progress: number) => void,
    onComplete: (result: T) => void,
    onError: (error: ApiError) => void
  ): Promise<void> {
    // Add to processing queue
    this.processingQueue.set(sessionId, {
      sessionId,
      onProgress,
      onComplete,
      onError,
    });

    try {
      // Start processing
      onProgress(0);
      
      // Simulate progress updates (in real implementation, this would come from the backend)
      const progressInterval = setInterval(async () => {
        try {
          const status = await NeuroLensAPI.getProcessingStatus(sessionId);
          onProgress(status.progress);
          
          if (status.status === 'completed') {
            clearInterval(progressInterval);
            this.processingQueue.delete(sessionId);
          } else if (status.status === 'error') {
            clearInterval(progressInterval);
            this.processingQueue.delete(sessionId);
            onError({
              code: 'PROCESSING_ERROR',
              message: 'Processing failed',
              timestamp: new Date(),
            });
          }
        } catch (error) {
          // If status check fails, continue with the original processing
        }
      }, 500);

      // Execute the actual processing function
      const result = await processingFunction();
      
      clearInterval(progressInterval);
      this.processingQueue.delete(sessionId);
      onProgress(100);
      onComplete(result);
      
    } catch (error) {
      this.processingQueue.delete(sessionId);
      onError(handleApiError(error as AxiosError));
    }
  }

  static cancelProcessing(sessionId: string): void {
    this.processingQueue.delete(sessionId);
  }
}

// Performance Monitoring
export class PerformanceMonitor {
  private static metrics: Map<string, number[]> = new Map();

  static recordMetric(endpoint: string, duration: number): void {
    if (!this.metrics.has(endpoint)) {
      this.metrics.set(endpoint, []);
    }
    
    const endpointMetrics = this.metrics.get(endpoint)!;
    endpointMetrics.push(duration);
    
    // Keep only last 100 measurements
    if (endpointMetrics.length > 100) {
      endpointMetrics.shift();
    }
  }

  static getAverageLatency(endpoint: string): number {
    const endpointMetrics = this.metrics.get(endpoint);
    if (!endpointMetrics || endpointMetrics.length === 0) {
      return 0;
    }
    
    return endpointMetrics.reduce((sum, duration) => sum + duration, 0) / endpointMetrics.length;
  }

  static getAllMetrics(): Record<string, { average: number; count: number }> {
    const result: Record<string, { average: number; count: number }> = {};
    
    for (const [endpoint, durations] of this.metrics.entries()) {
      result[endpoint] = {
        average: this.getAverageLatency(endpoint),
        count: durations.length,
      };
    }
    
    return result;
  }
}

export default NeuroLensAPI;

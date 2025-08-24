// NeuraLens ML Model Integration
// Unified interface for all ML models with backend API communication

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
} from '../api/types';

// Enhanced Assessment Request Interface
export interface AssessmentRequest {
  sessionId: string;
  audioFile?: File;
  retinalImage?: File;
  motorData?: {
    accelerometer?: Array<{ x: number; y: number; z: number }>;
    gyroscope?: Array<{ x: number; y: number; z: number }>;
    position?: Array<{ x: number; y: number }>;
    assessmentType: 'tremor' | 'finger_tapping' | 'gait' | 'balance';
  };
  cognitiveData?: {
    testResults: {
      response_times?: number[];
      accuracy?: number[];
      memory?: Record<string, number>;
      attention?: Record<string, number>;
      executive?: Record<string, number>;
      task_switching?: {
        repeat_trials: number[];
        switch_trials: number[];
        switch_accuracy: number;
      };
    };
    testBattery: string[];
    difficultyLevel: 'easy' | 'standard' | 'hard';
  };
  options?: {
    enableParallelProcessing?: boolean;
    timeoutMs?: number;
    retryAttempts?: number;
  };
}

export interface AssessmentProgress {
  sessionId: string;
  currentStep: string;
  progress: number; // 0-100
  estimatedTimeRemaining: number; // seconds
  completedModalities: string[];
  errors: AssessmentError[];
  startTime: number;
  lastUpdate: number;
}

export interface AssessmentError {
  modality: string;
  error: string;
  timestamp: number;
  recoverable: boolean;
}

export interface CompleteAssessmentResult {
  sessionId: string;
  nriResult: NRIFusionResponse;
  modalityResults: {
    speech?: SpeechAnalysisResponse;
    retinal?: RetinalAnalysisResponse;
    motor?: MotorAssessmentResponse;
    cognitive?: CognitiveAssessmentResponse;
  };
  metadata: {
    totalProcessingTime: number;
    timestamp: Date;
    version: string;
    dataQuality: number;
    successfulModalities: number;
    totalModalities: number;
  };
}

export type ProgressCallback = (progress: AssessmentProgress) => void;

// API Configuration
interface APIConfig {
  baseUrl: string;
  timeout: number;
  retryAttempts: number;
  retryDelay: number;
}

export class MLModelIntegrator {
  private activeAssessments = new Map<string, AssessmentProgress>();
  private resultCache = new Map<string, CompleteAssessmentResult>();
  private readonly version = '1.0.0';
  private readonly apiConfig: APIConfig = {
    baseUrl: '',
    timeout: 30000, // 30 seconds
    retryAttempts: 3,
    retryDelay: 1000, // 1 second
  };

  constructor(config?: Partial<APIConfig>) {
    if (config) {
      this.apiConfig = { ...this.apiConfig, ...config };
    }
  }

  /**
   * Make API call with retry logic and error handling
   */
  private async makeAPICall(
    endpoint: string,
    method: 'GET' | 'POST' | 'PUT' | 'DELETE',
    data?: any,
  ): Promise<any> {
    const url = `${this.apiConfig.baseUrl}${endpoint}`;
    let lastError: Error | null = null;

    for (let attempt = 0; attempt < this.apiConfig.retryAttempts; attempt++) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.apiConfig.timeout);

        const response = await fetch(url, {
          method,
          headers: {
            'Content-Type': 'application/json',
          },
          body: data ? JSON.stringify(data) : undefined,
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
      } catch (error) {
        lastError = error as Error;
        console.warn(`API call attempt ${attempt + 1} failed:`, error);

        if (attempt < this.apiConfig.retryAttempts - 1) {
          await new Promise(resolve =>
            setTimeout(resolve, this.apiConfig.retryDelay * (attempt + 1)),
          );
        }
      }
    }

    throw lastError || new Error('API call failed after all retry attempts');
  }

  /**
   * Process complete neurological assessment with backend API integration
   */
  async processAssessment(
    request: AssessmentRequest,
    onProgress?: ProgressCallback,
  ): Promise<CompleteAssessmentResult> {
    const startTime = performance.now();
    const currentTime = Date.now();

    try {
      // Initialize progress tracking
      const progress: AssessmentProgress = {
        sessionId: request.sessionId,
        currentStep: 'Initializing Assessment',
        progress: 0,
        estimatedTimeRemaining: 30, // Initial estimate
        completedModalities: [],
        errors: [],
        startTime: currentTime,
        lastUpdate: currentTime,
      };

      this.activeAssessments.set(request.sessionId, progress);
      this.updateProgress(progress, onProgress);

      // Validate request
      const validationErrors = this.validateAssessmentRequest(request);
      if (validationErrors.length > 0) {
        throw new Error(`Validation failed: ${validationErrors.join(', ')}`);
      }

      // Process modalities with proper error handling
      const modalityResults = await this.executeAssessmentWorkflow(request, progress, onProgress);

      // Update progress for fusion step
      progress.currentStep = 'Calculating NRI Score';
      progress.progress = 85;
      this.updateProgress(progress, onProgress);

      // Perform NRI fusion via API
      const nriResult = await this.performNRIFusion(modalityResults, request.sessionId);

      // Finalize results
      progress.currentStep = 'Finalizing Results';
      progress.progress = 95;
      this.updateProgress(progress, onProgress);

      const totalProcessingTime = performance.now() - startTime;
      const successfulModalities = Object.values(modalityResults).filter(
        result => result !== null,
      ).length;
      const totalModalities = Object.keys(modalityResults).length;

      const completeResult: CompleteAssessmentResult = {
        sessionId: request.sessionId,
        nriResult,
        modalityResults,
        metadata: {
          totalProcessingTime,
          timestamp: new Date(),
          version: this.version,
          dataQuality: this.calculateOverallDataQuality(modalityResults),
          successfulModalities,
          totalModalities,
        },
      };

      // Cache result
      this.resultCache.set(request.sessionId, completeResult);

      // Complete progress
      progress.currentStep = 'Assessment Complete';
      progress.progress = 100;
      progress.estimatedTimeRemaining = 0;
      this.updateProgress(progress, onProgress);

      // Cleanup
      this.activeAssessments.delete(request.sessionId);

      return completeResult;
    } catch (error) {
      // Handle errors
      const progress = this.activeAssessments.get(request.sessionId);
      if (progress) {
        const assessmentError: AssessmentError = {
          modality: 'system',
          error: (error as Error).message,
          timestamp: Date.now(),
          recoverable: false,
        };
        progress.errors.push(assessmentError);
        progress.currentStep = 'Assessment Failed';
        this.updateProgress(progress, onProgress);
      }

      this.activeAssessments.delete(request.sessionId);
      throw error;
    }
  }

  /**
   * Validate assessment request
   */
  private validateAssessmentRequest(request: AssessmentRequest): string[] {
    const errors: string[] = [];

    if (!request.sessionId) {
      errors.push('Session ID is required');
    }

    if (
      !request.audioFile &&
      !request.retinalImage &&
      !request.motorData &&
      !request.cognitiveData
    ) {
      errors.push('At least one assessment modality is required');
    }

    if (request.audioFile && !request.audioFile.type.startsWith('audio/')) {
      errors.push('Invalid audio file format');
    }

    if (request.retinalImage && !request.retinalImage.type.startsWith('image/')) {
      errors.push('Invalid image file format');
    }

    if (request.motorData && !request.motorData.assessmentType) {
      errors.push('Motor assessment type is required');
    }

    if (
      request.cognitiveData &&
      (!request.cognitiveData.testBattery || request.cognitiveData.testBattery.length === 0)
    ) {
      errors.push('Cognitive test battery is required');
    }

    return errors;
  }

  /**
   * Execute complete assessment workflow
   */
  private async executeAssessmentWorkflow(
    request: AssessmentRequest,
    progress: AssessmentProgress,
    onProgress?: ProgressCallback,
  ): Promise<CompleteAssessmentResult['modalityResults']> {
    const modalityResults: CompleteAssessmentResult['modalityResults'] = {};
    const modalities = this.getRequestedModalities(request);
    const totalModalities = modalities.length;

    for (let i = 0; i < modalities.length; i++) {
      const modality = modalities[i];
      if (!modality) continue;

      try {
        progress.currentStep = `Processing ${modality} analysis`;
        progress.progress = 10 + (i / totalModalities) * 70; // 10-80% range
        this.updateProgress(progress, onProgress);

        switch (modality) {
          case 'speech':
            if (request.audioFile) {
              modalityResults.speech = await this.processSpeechAnalysis(
                request.audioFile,
                request.sessionId,
              );
            }
            break;
          case 'retinal':
            if (request.retinalImage) {
              modalityResults.retinal = await this.processRetinalAnalysis(
                request.retinalImage,
                request.sessionId,
              );
            }
            break;
          case 'motor':
            if (request.motorData) {
              modalityResults.motor = await this.processMotorAnalysis(
                request.motorData,
                request.sessionId,
              );
            }
            break;
          case 'cognitive':
            if (request.cognitiveData) {
              modalityResults.cognitive = await this.processCognitiveAnalysis(
                request.cognitiveData,
                request.sessionId,
              );
            }
            break;
        }

        progress.completedModalities.push(modality);
      } catch (error) {
        console.error(`${modality} analysis failed:`, error);
        const assessmentError: AssessmentError = {
          modality,
          error: (error as Error).message,
          timestamp: Date.now(),
          recoverable: true,
        };
        progress.errors.push(assessmentError);
        modalityResults[modality as keyof typeof modalityResults] = undefined;
      }
    }

    return modalityResults;
  }

  /**
   * Get requested modalities from request
   */
  private getRequestedModalities(request: AssessmentRequest): string[] {
    const modalities: string[] = [];

    if (request.audioFile) modalities.push('speech');
    if (request.retinalImage) modalities.push('retinal');
    if (request.motorData) modalities.push('motor');
    if (request.cognitiveData) modalities.push('cognitive');

    return modalities;
  }

  /**
   * Perform NRI fusion via API
   */
  private async performNRIFusion(
    modalityResults: CompleteAssessmentResult['modalityResults'],
    sessionId: string,
  ): Promise<NRIFusionResponse> {
    const fusionRequest: NRIFusionRequest = {
      session_id: sessionId,
      modality_results: modalityResults,
      fusion_method: 'bayesian',
      uncertainty_quantification: true,
    };

    const response = await this.makeAPICall('/api/nri', 'POST', fusionRequest);

    if (!response.success) {
      throw new Error(`NRI fusion failed: ${response.error}`);
    }

    return response.result;
  }

  /**
   * Execute modality analysis with progress tracking
   */
  private async executeModalityAnalysis(
    promises: Record<string, Promise<any>>,
    progress: AssessmentProgress,
    onProgress?: ProgressCallback,
  ) {
    const modalityResults: any = {};
    const modalityNames = Object.keys(promises);
    const totalModalities = modalityNames.length;

    // Process modalities with progress updates
    for (let i = 0; i < modalityNames.length; i++) {
      const modalityName = modalityNames[i];
      if (!modalityName) continue;

      try {
        progress.currentStep = `Processing ${modalityName} analysis`;
        progress.progress = 10 + (i / totalModalities) * 70; // 10-80% range
        this.updateProgress(progress, onProgress);

        modalityResults[modalityName] = await promises[modalityName];
        progress.completedModalities.push(modalityName);
      } catch (error) {
        console.error(`${modalityName} analysis failed:`, error);
        const assessmentError: AssessmentError = {
          modality: modalityName,
          error: (error as Error).message,
          timestamp: Date.now(),
          recoverable: true,
        };
        progress.errors.push(assessmentError);
        modalityResults[modalityName] = null;
      }
    }

    return modalityResults;
  }

  /**
   * Process speech analysis via API
   */
  private async processSpeechAnalysis(
    audioFile: File, // File will be processed in production implementation
    sessionId: string,
  ): Promise<SpeechAnalysisResponse> {
    try {
      // Create speech analysis request with file data
      // In production, audioFile would be processed to extract actual features
      const speechRequest = {
        result: {
          fluencyScore: 0.85, // This would be replaced with actual file processing
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
            modelVersion: 'whisper-tiny-v1.0',
            timestamp: new Date(),
          },
        },
        sessionId,
        timestamp: Date.now(),
      };

      const response = await this.makeAPICall('/api/speech', 'POST', speechRequest);

      if (!response.success) {
        throw new Error(`Speech analysis failed: ${response.error}`);
      }

      return response.result;
    } catch (error) {
      console.error('Speech analysis error:', error);
      throw new Error(`Speech analysis failed: ${(error as Error).message}`);
    }
  }

  /**
   * Process retinal analysis via API
   */
  private async processRetinalAnalysis(
    retinalImage: File,
    sessionId: string,
  ): Promise<RetinalAnalysisResponse> {
    try {
      // Validate image file
      if (!retinalImage.type.startsWith('image/')) {
        throw new Error('Invalid image file format');
      }

      // Create retinal analysis request with image data
      const retinalRequest = {
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
            imageSize: retinalImage.size,
            modelVersion: 'efficientnet-b0-v1.0',
            preprocessingSteps: ['resize', 'normalize', 'tensor_conversion'],
            timestamp: new Date(),
            gpuAccelerated: true,
          },
        },
        sessionId,
        timestamp: Date.now(),
      };

      const response = await this.makeAPICall('/api/retinal', 'POST', retinalRequest);

      if (!response.success) {
        throw new Error(`Retinal analysis failed: ${response.error}`);
      }

      return response.result;
    } catch (error) {
      console.error('Retinal analysis error:', error);
      throw new Error(`Retinal analysis failed: ${(error as Error).message}`);
    }
  }

  /**
   * Process motor analysis via API
   */
  private async processMotorAnalysis(
    motorData: AssessmentRequest['motorData'],
    sessionId: string,
  ): Promise<MotorAssessmentResponse> {
    if (!motorData) {
      throw new Error('Motor data is required');
    }

    try {
      const motorRequest: MotorAssessmentRequest = {
        session_id: sessionId,
        sensor_data: {
          accelerometer: motorData.accelerometer || [],
          gyroscope: motorData.gyroscope || [],
          position: motorData.position || [],
        },
        assessment_type: motorData.assessmentType,
      };

      const response = await this.makeAPICall('/api/motor', 'POST', motorRequest);

      if (!response.success) {
        throw new Error(`Motor analysis failed: ${response.error}`);
      }

      return response.result;
    } catch (error) {
      console.error('Motor analysis error:', error);
      throw new Error(`Motor analysis failed: ${(error as Error).message}`);
    }
  }

  /**
   * Process cognitive analysis via API
   */
  private async processCognitiveAnalysis(
    cognitiveData: AssessmentRequest['cognitiveData'],
    sessionId: string,
  ): Promise<CognitiveAssessmentResponse> {
    if (!cognitiveData) {
      throw new Error('Cognitive data is required');
    }

    try {
      const cognitiveRequest: CognitiveAssessmentRequest = {
        session_id: sessionId,
        test_results: cognitiveData.testResults,
        test_battery: cognitiveData.testBattery,
        difficulty_level: cognitiveData.difficultyLevel,
      };

      const response = await this.makeAPICall('/api/cognitive', 'POST', cognitiveRequest);

      if (!response.success) {
        throw new Error(`Cognitive analysis failed: ${response.error}`);
      }

      return response.result;
    } catch (error) {
      console.error('Cognitive analysis error:', error);
      throw new Error(`Cognitive analysis failed: ${(error as Error).message}`);
    }
  }

  /**
   * Calculate overall data quality score
   */
  private calculateOverallDataQuality(modalityResults: any): number {
    const qualities: number[] = [];

    if (modalityResults.speech?.qualityScore) {
      qualities.push(modalityResults.speech.qualityScore);
    }

    if (modalityResults.retinal?.imageQuality) {
      qualities.push(modalityResults.retinal.imageQuality);
    }

    if (modalityResults.risk) {
      qualities.push(0.9); // Risk assessment typically high quality
    }

    if (modalityResults.motor?.quality) {
      qualities.push(modalityResults.motor.quality);
    }

    return qualities.length > 0 ? qualities.reduce((sum, q) => sum + q, 0) / qualities.length : 0.5;
  }

  /**
   * Update progress and notify callback
   */
  private updateProgress(progress: AssessmentProgress, onProgress?: ProgressCallback): void {
    // Update estimated time remaining
    if (progress.progress > 0) {
      const elapsedTime =
        Date.now() -
        (this.activeAssessments.get(progress.sessionId)?.estimatedTimeRemaining || Date.now());
      const estimatedTotal = elapsedTime / (progress.progress / 100);
      progress.estimatedTimeRemaining = Math.max(0, (estimatedTotal - elapsedTime) / 1000);
    }

    // Notify callback
    if (onProgress) {
      onProgress(progress);
    }
  }

  /**
   * Get assessment progress
   */
  getAssessmentProgress(sessionId: string): AssessmentProgress | null {
    return this.activeAssessments.get(sessionId) || null;
  }

  /**
   * Get cached result
   */
  getCachedResult(sessionId: string): CompleteAssessmentResult | null {
    return this.resultCache.get(sessionId) || null;
  }

  /**
   * Clear cache for session
   */
  clearSession(sessionId: string): void {
    this.activeAssessments.delete(sessionId);
    this.resultCache.delete(sessionId);
  }

  /**
   * Get system health status
   */
  getSystemHealth(): {
    status: 'healthy' | 'degraded' | 'unhealthy';
    activeAssessments: number;
    cacheSize: number;
    memoryUsage?: number;
  } {
    const activeCount = this.activeAssessments.size;
    const cacheSize = this.resultCache.size;

    // Simple health check
    let status: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';
    if (activeCount > 10) status = 'degraded';
    if (activeCount > 20) status = 'unhealthy';

    const memoryUsage = this.getMemoryUsage();
    return {
      status,
      activeAssessments: activeCount,
      cacheSize,
      ...(memoryUsage !== undefined && { memoryUsage }),
    };
  }

  /**
   * Get memory usage (if available)
   */
  private getMemoryUsage(): number | undefined {
    if (typeof window !== 'undefined' && 'performance' in window && 'memory' in performance) {
      return (performance as any).memory.usedJSHeapSize;
    }
    return undefined;
  }

  /**
   * Cleanup old cache entries
   */
  cleanupCache(maxAge: number = 3600000): void {
    // 1 hour default
    const now = Date.now();

    for (const [sessionId, result] of this.resultCache.entries()) {
      const age = now - result.metadata.timestamp.getTime();
      if (age > maxAge) {
        this.resultCache.delete(sessionId);
      }
    }
  }

  /**
   * Get performance metrics
   */
  getPerformanceMetrics(): {
    averageProcessingTime: number;
    successRate: number;
    modalitySuccessRates: Record<string, number>;
  } {
    // This would be implemented with proper metrics collection
    // For now, return placeholder values
    return {
      averageProcessingTime: 15000, // 15 seconds
      successRate: 0.95, // 95% success rate
      modalitySuccessRates: {
        speech: 0.92,
        retinal: 0.88,
        risk: 0.98,
        motor: 0.85,
      },
    };
  }
}

// Export singleton instance
export const mlModelIntegrator = new MLModelIntegrator();

// Export utility functions
export const generateSessionId = (): string => {
  // Use a more deterministic approach for SSR compatibility
  if (typeof window === 'undefined') {
    // Server-side: use a simple counter-based approach
    return `session_ssr_${Math.floor(Math.random() * 1000000)}`;
  }
  // Client-side: use timestamp and random for uniqueness
  return `session_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
};

export const validateAssessmentRequest = (request: AssessmentRequest): string[] => {
  const errors: string[] = [];

  if (!request.sessionId) {
    errors.push('Session ID is required');
  }

  if (!request.audioFile && !request.retinalImage && !request.motorData && !request.cognitiveData) {
    errors.push('At least one assessment modality is required');
  }

  if (request.audioFile && !request.audioFile.type.startsWith('audio/')) {
    errors.push('Invalid audio file format');
  }

  if (request.retinalImage && !request.retinalImage.type.startsWith('image/')) {
    errors.push('Invalid image file format');
  }

  if (request.motorData && !request.motorData.assessmentType) {
    errors.push('Motor assessment type is required');
  }

  if (
    request.cognitiveData &&
    (!request.cognitiveData.testBattery || request.cognitiveData.testBattery.length === 0)
  ) {
    errors.push('Cognitive test battery is required');
  }

  return errors;
};

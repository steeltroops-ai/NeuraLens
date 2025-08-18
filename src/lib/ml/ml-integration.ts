// NeuroLens-X ML Model Integration
// Unified interface for all ML models with performance optimization

import { speechAnalyzer, type SpeechAnalysisResult } from './speech-analysis';
import { retinalAnalyzer, type RetinalAnalysisResult } from './retinal-analysis';
import { riskAssessmentCalculator, type RiskAssessmentResult, type RiskAssessmentData } from './risk-assessment';
import { nriFusionCalculator, type NRIFusionResult } from './nri-fusion';

export interface AssessmentRequest {
  sessionId: string;
  audioFile?: File;
  retinalImage?: File;
  riskData?: RiskAssessmentData;
  motorData?: any; // Future implementation
}

export interface AssessmentProgress {
  sessionId: string;
  currentStep: string;
  progress: number; // 0-100
  estimatedTimeRemaining: number; // seconds
  completedModalities: string[];
  errors: string[];
}

export interface CompleteAssessmentResult {
  sessionId: string;
  nriResult: NRIFusionResult;
  modalityResults: {
    speech?: SpeechAnalysisResult;
    retinal?: RetinalAnalysisResult;
    risk?: RiskAssessmentResult;
    motor?: any;
  };
  metadata: {
    totalProcessingTime: number;
    timestamp: Date;
    version: string;
    dataQuality: number;
  };
}

export type ProgressCallback = (progress: AssessmentProgress) => void;

export class MLModelIntegrator {
  private activeAssessments = new Map<string, AssessmentProgress>();
  private resultCache = new Map<string, CompleteAssessmentResult>();
  private readonly version = '1.0.0';

  /**
   * Process complete neurological assessment
   */
  async processAssessment(
    request: AssessmentRequest,
    onProgress?: ProgressCallback
  ): Promise<CompleteAssessmentResult> {
    const startTime = performance.now();
    
    try {
      // Initialize progress tracking
      const progress: AssessmentProgress = {
        sessionId: request.sessionId,
        currentStep: 'Initializing',
        progress: 0,
        estimatedTimeRemaining: 30, // Initial estimate
        completedModalities: [],
        errors: [],
      };
      
      this.activeAssessments.set(request.sessionId, progress);
      this.updateProgress(progress, onProgress);
      
      // Process modalities in parallel for performance
      const modalityPromises = this.createModalityPromises(request, progress, onProgress);
      
      // Wait for all modalities to complete
      const modalityResults = await this.executeModalityAnalysis(modalityPromises, progress, onProgress);
      
      // Update progress for fusion step
      progress.currentStep = 'Calculating NRI Score';
      progress.progress = 85;
      this.updateProgress(progress, onProgress);
      
      // Perform NRI fusion
      const nriResult = await nriFusionCalculator.calculateNRI(
        modalityResults.speech,
        modalityResults.retinal,
        modalityResults.risk,
        modalityResults.motor
      );
      
      // Finalize results
      progress.currentStep = 'Finalizing Results';
      progress.progress = 95;
      this.updateProgress(progress, onProgress);
      
      const totalProcessingTime = performance.now() - startTime;
      
      const completeResult: CompleteAssessmentResult = {
        sessionId: request.sessionId,
        nriResult,
        modalityResults,
        metadata: {
          totalProcessingTime,
          timestamp: new Date(),
          version: this.version,
          dataQuality: this.calculateOverallDataQuality(modalityResults),
        },
      };
      
      // Cache result
      this.resultCache.set(request.sessionId, completeResult);
      
      // Complete progress
      progress.currentStep = 'Complete';
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
        progress.errors.push((error as Error).message);
        this.updateProgress(progress, onProgress);
      }
      
      this.activeAssessments.delete(request.sessionId);
      throw error;
    }
  }

  /**
   * Create promises for each modality analysis
   */
  private createModalityPromises(
    request: AssessmentRequest,
    progress: AssessmentProgress,
    onProgress?: ProgressCallback
  ) {
    const promises: Record<string, Promise<any>> = {};
    
    // Speech analysis
    if (request.audioFile) {
      promises.speech = this.processSpeechAnalysis(request.audioFile, progress, onProgress);
    }
    
    // Retinal analysis
    if (request.retinalImage) {
      promises.retinal = this.processRetinalAnalysis(request.retinalImage, progress, onProgress);
    }
    
    // Risk assessment
    if (request.riskData) {
      promises.risk = this.processRiskAssessment(request.riskData, progress, onProgress);
    }
    
    // Motor analysis (future implementation)
    if (request.motorData) {
      promises.motor = this.processMotorAnalysis(request.motorData, progress, onProgress);
    }
    
    return promises;
  }

  /**
   * Execute modality analysis with progress tracking
   */
  private async executeModalityAnalysis(
    promises: Record<string, Promise<any>>,
    progress: AssessmentProgress,
    onProgress?: ProgressCallback
  ) {
    const modalityResults: any = {};
    const modalityNames = Object.keys(promises);
    const totalModalities = modalityNames.length;
    
    // Process modalities with progress updates
    for (let i = 0; i < modalityNames.length; i++) {
      const modalityName = modalityNames[i];
      
      try {
        progress.currentStep = `Processing ${modalityName} analysis`;
        progress.progress = 10 + (i / totalModalities) * 70; // 10-80% range
        this.updateProgress(progress, onProgress);
        
        modalityResults[modalityName] = await promises[modalityName];
        progress.completedModalities.push(modalityName);
        
      } catch (error) {
        console.error(`${modalityName} analysis failed:`, error);
        progress.errors.push(`${modalityName} analysis failed: ${(error as Error).message}`);
        modalityResults[modalityName] = null;
      }
    }
    
    return modalityResults;
  }

  /**
   * Process speech analysis with error handling
   */
  private async processSpeechAnalysis(
    audioFile: File,
    progress: AssessmentProgress,
    onProgress?: ProgressCallback
  ): Promise<SpeechAnalysisResult | null> {
    try {
      // Convert file to ArrayBuffer
      const arrayBuffer = await audioFile.arrayBuffer();
      
      // Analyze speech
      const result = await speechAnalyzer.analyzeSpeech(arrayBuffer);
      
      // Validate result quality
      if (result.qualityScore < 0.3) {
        progress.errors.push('Audio quality too low for reliable analysis');
      }
      
      return result;
    } catch (error) {
      console.error('Speech analysis error:', error);
      throw new Error(`Speech analysis failed: ${(error as Error).message}`);
    }
  }

  /**
   * Process retinal analysis with error handling
   */
  private async processRetinalAnalysis(
    retinalImage: File,
    progress: AssessmentProgress,
    onProgress?: ProgressCallback
  ): Promise<RetinalAnalysisResult | null> {
    try {
      // Validate image file
      if (!retinalImage.type.startsWith('image/')) {
        throw new Error('Invalid image file format');
      }
      
      // Analyze retinal image
      const result = await retinalAnalyzer.analyzeRetinalImage(retinalImage);
      
      // Validate result quality
      if (result.imageQuality < 0.4) {
        progress.errors.push('Retinal image quality too low for reliable analysis');
      }
      
      return result;
    } catch (error) {
      console.error('Retinal analysis error:', error);
      throw new Error(`Retinal analysis failed: ${(error as Error).message}`);
    }
  }

  /**
   * Process risk assessment with validation
   */
  private async processRiskAssessment(
    riskData: RiskAssessmentData,
    progress: AssessmentProgress,
    onProgress?: ProgressCallback
  ): Promise<RiskAssessmentResult | null> {
    try {
      // Validate required fields
      this.validateRiskData(riskData);
      
      // Calculate risk assessment
      const result = await riskAssessmentCalculator.calculateRisk(riskData);
      
      return result;
    } catch (error) {
      console.error('Risk assessment error:', error);
      throw new Error(`Risk assessment failed: ${(error as Error).message}`);
    }
  }

  /**
   * Process motor analysis (placeholder for future implementation)
   */
  private async processMotorAnalysis(
    motorData: any,
    progress: AssessmentProgress,
    onProgress?: ProgressCallback
  ): Promise<any | null> {
    // Placeholder for future motor analysis implementation
    return null;
  }

  /**
   * Validate risk assessment data
   */
  private validateRiskData(riskData: RiskAssessmentData): void {
    if (!riskData.demographics?.age || riskData.demographics.age < 18 || riskData.demographics.age > 120) {
      throw new Error('Invalid age provided');
    }
    
    if (!riskData.demographics?.sex) {
      throw new Error('Sex information required');
    }
    
    // Add more validation as needed
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
    
    return qualities.length > 0 ? 
      qualities.reduce((sum, q) => sum + q, 0) / qualities.length : 0.5;
  }

  /**
   * Update progress and notify callback
   */
  private updateProgress(progress: AssessmentProgress, onProgress?: ProgressCallback): void {
    // Update estimated time remaining
    if (progress.progress > 0) {
      const elapsedTime = Date.now() - (this.activeAssessments.get(progress.sessionId)?.estimatedTimeRemaining || Date.now());
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
    
    return {
      status,
      activeAssessments: activeCount,
      cacheSize,
      memoryUsage: this.getMemoryUsage(),
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
  cleanupCache(maxAge: number = 3600000): void { // 1 hour default
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
  return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
};

export const validateAssessmentRequest = (request: AssessmentRequest): string[] => {
  const errors: string[] = [];
  
  if (!request.sessionId) {
    errors.push('Session ID is required');
  }
  
  if (!request.audioFile && !request.retinalImage && !request.riskData) {
    errors.push('At least one assessment modality is required');
  }
  
  if (request.audioFile && !request.audioFile.type.startsWith('audio/')) {
    errors.push('Invalid audio file format');
  }
  
  if (request.retinalImage && !request.retinalImage.type.startsWith('image/')) {
    errors.push('Invalid image file format');
  }
  
  return errors;
};

/**
 * Complete Assessment Workflow Engine
 * Orchestrates end-to-end assessment processing with real ML pipelines
 */

import {
  SpeechAnalysisService,
  RetinalAnalysisService,
  MotorAssessmentService,
  CognitiveAssessmentService,
  NRIFusionService,
} from '@/lib/api/services';
import {
  SpeechAnalysisResponse,
  RetinalAnalysisResponse,
  MotorAssessmentResponse,
  CognitiveAssessmentResponse,
  NRIFusionResponse,
  CompleteAssessmentResult,
} from '@/lib/api/types';
import { validateAudioFile, validateImageFile } from './validation';

// Assessment step definitions
export type AssessmentStep =
  | 'upload'
  | 'validation'
  | 'speech_processing'
  | 'retinal_processing'
  | 'motor_processing'
  | 'cognitive_processing'
  | 'nri_fusion'
  | 'results'
  | 'complete';

// Assessment input data
export interface AssessmentInput {
  sessionId: string;
  speechFile?: File;
  retinalImage?: File;
  motorData?: {
    accelerometer?: Array<{ x: number; y: number; z: number }>;
    gyroscope?: Array<{ x: number; y: number; z: number }>;
    position?: Array<{ x: number; y: number }>;
  };
  cognitiveData?: {
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
}

// Assessment progress state
export interface AssessmentProgress {
  currentStep: AssessmentStep;
  completedSteps: AssessmentStep[];
  totalSteps: number;
  progressPercentage: number;
  estimatedTimeRemaining?: number;
  stepProgress: Record<string, number>;
  errors: Record<string, string>;
}

// Assessment results
export interface AssessmentResults {
  sessionId: string;
  speechResult?: SpeechAnalysisResponse;
  retinalResult?: RetinalAnalysisResponse;
  motorResult?: MotorAssessmentResponse;
  cognitiveResult?: CognitiveAssessmentResponse;
  nriResult?: NRIFusionResponse;
  overallRiskCategory: 'low' | 'moderate' | 'high';
  completionTime: string;
  totalProcessingTime: number;
  metadata: {
    startTime: string;
    endTime: string;
    stepsCompleted: AssessmentStep[];
    errors: string[];
  };
}

// Progress callback type
export type ProgressCallback = (progress: AssessmentProgress) => void;

// Assessment workflow engine
export class AssessmentWorkflowEngine {
  private sessionId: string;
  private progress: AssessmentProgress;
  private results: Partial<AssessmentResults>;
  private onProgress?: ProgressCallback;
  private startTime: Date;

  constructor(sessionId: string, onProgress?: ProgressCallback) {
    this.sessionId = sessionId;
    this.onProgress = onProgress;
    this.startTime = new Date();

    this.progress = {
      currentStep: 'upload',
      completedSteps: [],
      totalSteps: 8,
      progressPercentage: 0,
      stepProgress: {},
      errors: {},
    };

    this.results = {
      sessionId,
      metadata: {
        startTime: this.startTime.toISOString(),
        endTime: '',
        stepsCompleted: [],
        errors: [],
      },
    };
  }

  /**
   * Execute complete assessment workflow
   */
  async executeAssessment(input: AssessmentInput): Promise<AssessmentResults> {
    try {
      // Step 1: Upload and Validation
      await this.executeStep('upload', async () => {
        await this.validateInput(input);
      });

      // Step 2: File Validation
      await this.executeStep('validation', async () => {
        await this.validateFiles(input);
      });

      // Step 3: Speech Processing
      if (input.speechFile) {
        await this.executeStep('speech_processing', async () => {
          this.results.speechResult = await this.processSpeech(input.speechFile!);
        });
      }

      // Step 4: Retinal Processing
      if (input.retinalImage) {
        await this.executeStep('retinal_processing', async () => {
          this.results.retinalResult = await this.processRetinal(input.retinalImage!);
        });
      }

      // Step 5: Motor Processing
      if (input.motorData) {
        await this.executeStep('motor_processing', async () => {
          this.results.motorResult = await this.processMotor(input.motorData!);
        });
      }

      // Step 6: Cognitive Processing
      if (input.cognitiveData) {
        await this.executeStep('cognitive_processing', async () => {
          this.results.cognitiveResult = await this.processCognitive(input.cognitiveData!);
        });
      }

      // Step 7: NRI Fusion
      await this.executeStep('nri_fusion', async () => {
        this.results.nriResult = await this.processNRIFusion();
      });

      // Step 8: Finalize Results
      await this.executeStep('results', async () => {
        await this.finalizeResults();
      });

      // Complete
      this.updateProgress('complete', 100);

      return this.results as AssessmentResults;
    } catch (error) {
      this.handleError(error);
      throw error;
    }
  }

  /**
   * Execute a single assessment step
   */
  private async executeStep(step: AssessmentStep, operation: () => Promise<void>): Promise<void> {
    try {
      this.updateProgress(step, 0);

      const startTime = Date.now();
      await operation();
      const processingTime = Date.now() - startTime;

      this.progress.completedSteps.push(step);
      this.progress.stepProgress[step] = 100;
      this.updateProgress(step, 100);

      // Update estimated time remaining
      this.updateTimeEstimate(processingTime);
    } catch (error) {
      this.progress.errors[step] = error instanceof Error ? error.message : 'Unknown error';
      throw error;
    }
  }

  /**
   * Validate assessment input
   */
  private async validateInput(input: AssessmentInput): Promise<void> {
    if (!input.sessionId) {
      throw new Error('Session ID is required');
    }

    const hasAnyInput =
      input.speechFile || input.retinalImage || input.motorData || input.cognitiveData;
    if (!hasAnyInput) {
      throw new Error('At least one assessment modality is required');
    }

    // Update progress
    this.updateStepProgress('upload', 100);
  }

  /**
   * Validate uploaded files
   */
  private async validateFiles(input: AssessmentInput): Promise<void> {
    let validationProgress = 0;
    const totalValidations = (input.speechFile ? 1 : 0) + (input.retinalImage ? 1 : 0);

    if (input.speechFile) {
      const audioValidation = await validateAudioFile(input.speechFile);
      if (!audioValidation.isValid) {
        throw new Error(`Audio file validation failed: ${audioValidation.errors.join(', ')}`);
      }
      validationProgress += 50;
      this.updateStepProgress('validation', validationProgress);
    }

    if (input.retinalImage) {
      const imageValidation = await validateImageFile(input.retinalImage);
      if (!imageValidation.isValid) {
        throw new Error(`Image file validation failed: ${imageValidation.errors.join(', ')}`);
      }
      validationProgress += 50;
      this.updateStepProgress('validation', validationProgress);
    }

    this.updateStepProgress('validation', 100);
  }

  /**
   * Process speech analysis
   */
  private async processSpeech(speechFile: File): Promise<SpeechAnalysisResponse> {
    this.updateStepProgress('speech_processing', 10);

    const response = await SpeechAnalysisService.analyze({
      session_id: this.sessionId,
      audio_file: speechFile,
    });

    if (!response.success || !response.data) {
      throw new Error(response.error?.message || 'Speech analysis failed');
    }

    this.updateStepProgress('speech_processing', 100);
    return response.data;
  }

  /**
   * Process retinal analysis
   */
  private async processRetinal(retinalImage: File): Promise<RetinalAnalysisResponse> {
    this.updateStepProgress('retinal_processing', 10);

    const response = await RetinalAnalysisService.analyze({
      session_id: this.sessionId,
      image_file: retinalImage,
    });

    if (!response.success || !response.data) {
      throw new Error(response.error?.message || 'Retinal analysis failed');
    }

    this.updateStepProgress('retinal_processing', 100);
    return response.data;
  }

  /**
   * Process motor assessment
   */
  private async processMotor(motorData: any): Promise<MotorAssessmentResponse> {
    this.updateStepProgress('motor_processing', 10);

    const response = await MotorAssessmentService.analyze({
      session_id: this.sessionId,
      sensor_data: motorData,
      assessment_type: 'tremor',
    });

    if (!response.success || !response.data) {
      throw new Error(response.error?.message || 'Motor assessment failed');
    }

    this.updateStepProgress('motor_processing', 100);
    return response.data;
  }

  /**
   * Process cognitive assessment
   */
  private async processCognitive(cognitiveData: any): Promise<CognitiveAssessmentResponse> {
    this.updateStepProgress('cognitive_processing', 10);

    const response = await CognitiveAssessmentService.analyze({
      session_id: this.sessionId,
      test_results: cognitiveData,
      test_battery: ['memory', 'attention', 'executive'],
      difficulty_level: 'standard',
    });

    if (!response.success || !response.data) {
      throw new Error(response.error?.message || 'Cognitive assessment failed');
    }

    this.updateStepProgress('cognitive_processing', 100);
    return response.data;
  }

  /**
   * Process NRI fusion
   */
  private async processNRIFusion(): Promise<NRIFusionResponse> {
    this.updateStepProgress('nri_fusion', 10);

    const response = await NRIFusionService.fusion({
      session_id: this.sessionId,
      modality_results: {
        speech: this.results.speechResult,
        retinal: this.results.retinalResult,
        motor: this.results.motorResult,
        cognitive: this.results.cognitiveResult,
      },
    });

    if (!response.success || !response.data) {
      throw new Error(response.error?.message || 'NRI fusion failed');
    }

    this.updateStepProgress('nri_fusion', 100);
    return response.data;
  }

  /**
   * Finalize assessment results
   */
  private async finalizeResults(): Promise<void> {
    const endTime = new Date();
    const totalProcessingTime = endTime.getTime() - this.startTime.getTime();

    this.results.overallRiskCategory = this.results.nriResult?.risk_category || 'moderate';
    this.results.completionTime = endTime.toISOString();
    this.results.totalProcessingTime = totalProcessingTime;

    if (this.results.metadata) {
      this.results.metadata.endTime = endTime.toISOString();
      this.results.metadata.stepsCompleted = [...this.progress.completedSteps];
    }

    this.updateStepProgress('results', 100);
  }

  /**
   * Update progress state
   */
  private updateProgress(step: AssessmentStep, stepProgress: number): void {
    this.progress.currentStep = step;
    this.progress.stepProgress[step] = stepProgress;

    // Calculate overall progress
    const completedSteps = this.progress.completedSteps.length;
    const currentStepProgress = stepProgress / 100;
    this.progress.progressPercentage =
      ((completedSteps + currentStepProgress) / this.progress.totalSteps) * 100;

    this.onProgress?.(this.progress);
  }

  /**
   * Update step-specific progress
   */
  private updateStepProgress(step: string, progress: number): void {
    this.progress.stepProgress[step] = progress;
    this.onProgress?.(this.progress);
  }

  /**
   * Update estimated time remaining
   */
  private updateTimeEstimate(lastStepTime: number): void {
    const remainingSteps = this.progress.totalSteps - this.progress.completedSteps.length;
    this.progress.estimatedTimeRemaining = Math.round((lastStepTime * remainingSteps) / 1000);
  }

  /**
   * Handle errors
   */
  private handleError(error: any): void {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    this.progress.errors[this.progress.currentStep] = errorMessage;

    if (this.results.metadata) {
      this.results.metadata.errors.push(errorMessage);
    }

    this.onProgress?.(this.progress);
  }

  /**
   * Get current progress
   */
  getProgress(): AssessmentProgress {
    return { ...this.progress };
  }

  /**
   * Get current results
   */
  getResults(): Partial<AssessmentResults> {
    return { ...this.results };
  }
}

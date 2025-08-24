/**
 * Assessment Workflow Hook
 * React hook for managing complete assessment workflow with progress tracking
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import {
  AssessmentWorkflowEngine,
  AssessmentInput,
  AssessmentProgress,
  AssessmentResults,
} from '@/lib/assessment/workflow';
import {
  ProgressTracker,
  ProgressEvent,
  ConnectionStatus,
} from '@/lib/assessment/progress-tracker';

// Workflow state
export interface WorkflowState {
  isRunning: boolean;
  isCompleted: boolean;
  hasError: boolean;
  error: string | null;
  progress: AssessmentProgress;
  results: Partial<AssessmentResults> | null;
  connectionStatus: ConnectionStatus;
}

// Workflow options
export interface WorkflowOptions {
  enableProgressTracking?: boolean;
  enablePersistence?: boolean;
  onStepCompleted?: (step: string) => void;
  onError?: (error: string) => void;
  onCompleted?: (results: AssessmentResults) => void;
}

// Assessment workflow hook
export function useAssessmentWorkflow(sessionId: string, options: WorkflowOptions = {}) {
  const [state, setState] = useState<WorkflowState>({
    isRunning: false,
    isCompleted: false,
    hasError: false,
    error: null,
    progress: {
      currentStep: 'upload',
      completedSteps: [],
      totalSteps: 8,
      progressPercentage: 0,
      stepProgress: {},
      errors: {},
    },
    results: null,
    connectionStatus: 'disconnected',
  });

  const workflowEngineRef = useRef<AssessmentWorkflowEngine | null>(null);
  const progressTrackerRef = useRef<ProgressTracker | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Initialize progress tracker
  useEffect(() => {
    if (options.enableProgressTracking && sessionId) {
      progressTrackerRef.current = new ProgressTracker({
        sessionId,
        enablePersistence: options.enablePersistence,
      });

      // Set up event listeners
      const handleProgressUpdate = (event: ProgressEvent) => {
        setState(prev => ({
          ...prev,
          progress: event.data,
        }));
      };

      const handleStepCompleted = (event: ProgressEvent) => {
        options.onStepCompleted?.(event.data.step);
      };

      const handleConnectionStatus = (event: ProgressEvent) => {
        setState(prev => ({
          ...prev,
          connectionStatus: event.data.status,
        }));
      };

      progressTrackerRef.current.addEventListener('progress_update', handleProgressUpdate);
      progressTrackerRef.current.addEventListener('step_completed', handleStepCompleted);
      progressTrackerRef.current.addEventListener('connection_status', handleConnectionStatus);

      // Start progress tracking
      progressTrackerRef.current.start();

      return () => {
        progressTrackerRef.current?.stop();
        progressTrackerRef.current = null;
      };
    }

    return undefined;
  }, [sessionId, options.enableProgressTracking, options.enablePersistence]);

  // Execute assessment workflow
  const executeAssessment = useCallback(
    async (input: AssessmentInput) => {
      if (state.isRunning) {
        throw new Error('Assessment is already running');
      }

      // Create abort controller
      abortControllerRef.current = new AbortController();

      // Reset state
      setState(prev => ({
        ...prev,
        isRunning: true,
        isCompleted: false,
        hasError: false,
        error: null,
        results: null,
      }));

      try {
        // Create workflow engine with progress callback
        const onProgress = (progress: AssessmentProgress) => {
          setState(prev => ({
            ...prev,
            progress,
          }));
        };

        workflowEngineRef.current = new AssessmentWorkflowEngine(sessionId, onProgress);

        // Execute assessment
        const results = await workflowEngineRef.current.executeAssessment(input);

        // Check if aborted
        if (abortControllerRef.current?.signal.aborted) {
          return;
        }

        // Update state with results
        setState(prev => ({
          ...prev,
          isRunning: false,
          isCompleted: true,
          results,
        }));

        // Clear persisted progress on completion
        if (options.enablePersistence) {
          progressTrackerRef.current?.clearPersistedProgress();
        }

        // Notify completion
        options.onCompleted?.(results);

        return results;
      } catch (error) {
        // Check if aborted
        if (abortControllerRef.current?.signal.aborted) {
          return;
        }

        const errorMessage = error instanceof Error ? error.message : 'Assessment failed';

        setState(prev => ({
          ...prev,
          isRunning: false,
          hasError: true,
          error: errorMessage,
        }));

        // Notify error
        options.onError?.(errorMessage);

        throw error;
      }
    },
    [sessionId, state.isRunning, options],
  );

  // Cancel assessment
  const cancelAssessment = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    setState(prev => ({
      ...prev,
      isRunning: false,
      hasError: false,
      error: null,
    }));
  }, []);

  // Reset workflow
  const resetWorkflow = useCallback(() => {
    cancelAssessment();

    setState({
      isRunning: false,
      isCompleted: false,
      hasError: false,
      error: null,
      progress: {
        currentStep: 'upload',
        completedSteps: [],
        totalSteps: 8,
        progressPercentage: 0,
        stepProgress: {},
        errors: {},
      },
      results: null,
      connectionStatus: 'disconnected',
    });

    // Clear persisted progress
    if (options.enablePersistence) {
      progressTrackerRef.current?.clearPersistedProgress();
    }
  }, [cancelAssessment, options.enablePersistence]);

  // Resume assessment from persisted state
  const resumeAssessment = useCallback(() => {
    if (progressTrackerRef.current) {
      const persistedProgress = progressTrackerRef.current.getProgress();
      setState(prev => ({
        ...prev,
        progress: persistedProgress,
      }));
    }
  }, []);

  // Get step status
  const getStepStatus = useCallback(
    (step: string) => {
      if (state.progress.completedSteps.includes(step as any)) {
        return 'completed';
      }
      if (state.progress.currentStep === step) {
        return 'active';
      }
      if (state.progress.errors[step]) {
        return 'error';
      }
      return 'pending';
    },
    [state.progress],
  );

  // Get step progress percentage
  const getStepProgress = useCallback(
    (step: string) => {
      return state.progress.stepProgress[step] || 0;
    },
    [state.progress.stepProgress],
  );

  // Check if step can be retried
  const canRetryStep = useCallback(
    (step: string) => {
      return state.progress.errors[step] && !state.isRunning;
    },
    [state.progress.errors, state.isRunning],
  );

  // Retry failed step
  const retryStep = useCallback(
    async (step: string, input: AssessmentInput) => {
      if (!canRetryStep(step)) {
        throw new Error('Step cannot be retried');
      }

      // Clear step error
      setState(prev => {
        const { [step]: removedError, ...remainingErrors } = prev.progress.errors;
        return {
          ...prev,
          progress: {
            ...prev.progress,
            errors: remainingErrors,
          },
        };
      });

      // Re-execute assessment from failed step
      return executeAssessment(input);
    },
    [canRetryStep, executeAssessment],
  );

  // Get assessment summary
  const getAssessmentSummary = useCallback(() => {
    const { progress, results } = state;

    return {
      totalSteps: progress.totalSteps,
      completedSteps: progress.completedSteps.length,
      currentStep: progress.currentStep,
      progressPercentage: progress.progressPercentage,
      hasErrors: Object.keys(progress.errors).length > 0,
      errorCount: Object.keys(progress.errors).length,
      isCompleted: state.isCompleted,
      estimatedTimeRemaining: progress.estimatedTimeRemaining,
      overallRiskCategory: results?.overallRiskCategory,
      totalProcessingTime: results?.totalProcessingTime,
    };
  }, [state]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cancelAssessment();
      progressTrackerRef.current?.stop();
    };
  }, [cancelAssessment]);

  return {
    // State
    ...state,

    // Actions
    executeAssessment,
    cancelAssessment,
    resetWorkflow,
    resumeAssessment,
    retryStep,

    // Utilities
    getStepStatus,
    getStepProgress,
    canRetryStep,
    getAssessmentSummary,

    // Progress tracking
    progressTracker: progressTrackerRef.current,
    workflowEngine: workflowEngineRef.current,
  };
}

// Hook for assessment step management
export function useAssessmentSteps() {
  const steps = [
    { id: 'upload', label: 'Upload Files', description: 'Upload audio and image files' },
    { id: 'validation', label: 'Validation', description: 'Validate file formats and quality' },
    {
      id: 'speech_processing',
      label: 'Speech Analysis',
      description: 'Analyze speech patterns and biomarkers',
    },
    {
      id: 'retinal_processing',
      label: 'Retinal Analysis',
      description: 'Process retinal image and extract features',
    },
    {
      id: 'motor_processing',
      label: 'Motor Assessment',
      description: 'Evaluate motor function and tremor',
    },
    {
      id: 'cognitive_processing',
      label: 'Cognitive Assessment',
      description: 'Assess cognitive performance',
    },
    { id: 'nri_fusion', label: 'NRI Calculation', description: 'Compute neurological risk index' },
    { id: 'results', label: 'Results', description: 'Generate final assessment report' },
  ];

  const getStepInfo = useCallback((stepId: string) => {
    return steps.find(step => step.id === stepId);
  }, []);

  const getStepIndex = useCallback((stepId: string) => {
    return steps.findIndex(step => step.id === stepId);
  }, []);

  const getNextStep = useCallback(
    (currentStepId: string) => {
      const currentIndex = getStepIndex(currentStepId);
      return currentIndex < steps.length - 1 ? steps[currentIndex + 1] : null;
    },
    [getStepIndex],
  );

  const getPreviousStep = useCallback(
    (currentStepId: string) => {
      const currentIndex = getStepIndex(currentStepId);
      return currentIndex > 0 ? steps[currentIndex - 1] : null;
    },
    [getStepIndex],
  );

  return {
    steps,
    getStepInfo,
    getStepIndex,
    getNextStep,
    getPreviousStep,
  };
}

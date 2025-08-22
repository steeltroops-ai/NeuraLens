/**
 * NeuroLens-X Assessment Hook
 * Custom React hook for managing assessment state and API interactions
 * Optimized for real-time processing and error handling
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { NeuroLensAPI, RealTimeProcessor } from '@/lib/api';
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
  ApiError,
} from '@/types';

// Assessment state interface
interface AssessmentState {
  isLoading: boolean;
  progress: number;
  error: ApiError | null;
  result: any | null;
  sessionId: string | null;
}

// Assessment hook return type
interface UseAssessmentReturn {
  // State
  state: AssessmentState;
  
  // Actions
  analyzeSpeech: (audioData: Blob, request: Omit<SpeechAnalysisRequest, 'sessionId' | 'timestamp'>) => Promise<void>;
  analyzeRetinal: (imageData: Blob, request: Omit<RetinalAnalysisRequest, 'sessionId' | 'timestamp'>) => Promise<void>;
  assessMotor: (request: Omit<MotorAssessmentRequest, 'sessionId' | 'timestamp'>) => Promise<void>;
  assessCognitive: (request: Omit<CognitiveAssessmentRequest, 'sessionId' | 'timestamp'>) => Promise<void>;
  calculateNRI: (request: NRIFusionRequest) => Promise<void>;
  
  // Utilities
  reset: () => void;
  cancel: () => void;
}

export function useAssessment(): UseAssessmentReturn {
  const [state, setState] = useState<AssessmentState>({
    isLoading: false,
    progress: 0,
    error: null,
    result: null,
    sessionId: null,
  });

  const currentSessionRef = useRef<string | null>(null);

  // Reset state
  const reset = useCallback(() => {
    if (currentSessionRef.current) {
      RealTimeProcessor.cancelProcessing(currentSessionRef.current);
    }
    
    setState({
      isLoading: false,
      progress: 0,
      error: null,
      result: null,
      sessionId: null,
    });
    
    currentSessionRef.current = null;
  }, []);

  // Cancel current processing
  const cancel = useCallback(() => {
    if (currentSessionRef.current) {
      RealTimeProcessor.cancelProcessing(currentSessionRef.current);
      setState(prev => ({
        ...prev,
        isLoading: false,
        progress: 0,
      }));
    }
  }, []);

  // Generic processing function
  const processAssessment = useCallback(async <T>(
    processingFunction: () => Promise<T>,
    sessionId: string
  ): Promise<void> => {
    currentSessionRef.current = sessionId;
    
    setState(prev => ({
      ...prev,
      isLoading: true,
      progress: 0,
      error: null,
      result: null,
      sessionId,
    }));

    await RealTimeProcessor.processWithProgress(
      sessionId,
      processingFunction,
      // onProgress
      (progress: number) => {
        setState(prev => ({
          ...prev,
          progress,
        }));
      },
      // onComplete
      (result: T) => {
        setState(prev => ({
          ...prev,
          isLoading: false,
          progress: 100,
          result,
        }));
        currentSessionRef.current = null;
      },
      // onError
      (error: ApiError) => {
        setState(prev => ({
          ...prev,
          isLoading: false,
          progress: 0,
          error,
        }));
        currentSessionRef.current = null;
      }
    );
  }, []);

  // Speech analysis
  const analyzeSpeech = useCallback(async (
    audioData: Blob,
    request: Omit<SpeechAnalysisRequest, 'sessionId' | 'timestamp'>
  ): Promise<void> => {
    const sessionId = crypto.randomUUID();
    
    await processAssessment(
      () => NeuroLensAPI.analyzeSpeech(audioData, request),
      sessionId
    );
  }, [processAssessment]);

  // Retinal analysis
  const analyzeRetinal = useCallback(async (
    imageData: Blob,
    request: Omit<RetinalAnalysisRequest, 'sessionId' | 'timestamp'>
  ): Promise<void> => {
    const sessionId = crypto.randomUUID();
    
    await processAssessment(
      () => NeuroLensAPI.analyzeRetinal(imageData, request),
      sessionId
    );
  }, [processAssessment]);

  // Motor assessment
  const assessMotor = useCallback(async (
    request: Omit<MotorAssessmentRequest, 'sessionId' | 'timestamp'>
  ): Promise<void> => {
    const sessionId = crypto.randomUUID();
    
    await processAssessment(
      () => NeuroLensAPI.assessMotor(request),
      sessionId
    );
  }, [processAssessment]);

  // Cognitive assessment
  const assessCognitive = useCallback(async (
    request: Omit<CognitiveAssessmentRequest, 'sessionId' | 'timestamp'>
  ): Promise<void> => {
    const sessionId = crypto.randomUUID();
    
    await processAssessment(
      () => NeuroLensAPI.assessCognitive(request),
      sessionId
    );
  }, [processAssessment]);

  // NRI calculation
  const calculateNRI = useCallback(async (
    request: NRIFusionRequest
  ): Promise<void> => {
    await processAssessment(
      () => NeuroLensAPI.calculateNRI(request),
      request.sessionId
    );
  }, [processAssessment]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (currentSessionRef.current) {
        RealTimeProcessor.cancelProcessing(currentSessionRef.current);
      }
    };
  }, []);

  return {
    state,
    analyzeSpeech,
    analyzeRetinal,
    assessMotor,
    assessCognitive,
    calculateNRI,
    reset,
    cancel,
  };
}

// Specialized hooks for individual assessment types
export function useSpeechAssessment() {
  const { state, analyzeSpeech, reset, cancel } = useAssessment();
  
  return {
    isLoading: state.isLoading,
    progress: state.progress,
    error: state.error,
    result: state.result as SpeechAnalysisResponse | null,
    sessionId: state.sessionId,
    analyze: analyzeSpeech,
    reset,
    cancel,
  };
}

export function useRetinalAssessment() {
  const { state, analyzeRetinal, reset, cancel } = useAssessment();
  
  return {
    isLoading: state.isLoading,
    progress: state.progress,
    error: state.error,
    result: state.result as RetinalAnalysisResponse | null,
    sessionId: state.sessionId,
    analyze: analyzeRetinal,
    reset,
    cancel,
  };
}

export function useMotorAssessment() {
  const { state, assessMotor, reset, cancel } = useAssessment();
  
  return {
    isLoading: state.isLoading,
    progress: state.progress,
    error: state.error,
    result: state.result as MotorAssessmentResponse | null,
    sessionId: state.sessionId,
    assess: assessMotor,
    reset,
    cancel,
  };
}

export function useCognitiveAssessment() {
  const { state, assessCognitive, reset, cancel } = useAssessment();
  
  return {
    isLoading: state.isLoading,
    progress: state.progress,
    error: state.error,
    result: state.result as CognitiveAssessmentResponse | null,
    sessionId: state.sessionId,
    assess: assessCognitive,
    reset,
    cancel,
  };
}

export function useNRIFusion() {
  const { state, calculateNRI, reset, cancel } = useAssessment();
  
  return {
    isLoading: state.isLoading,
    progress: state.progress,
    error: state.error,
    result: state.result as NRIFusionResponse | null,
    sessionId: state.sessionId,
    calculate: calculateNRI,
    reset,
    cancel,
  };
}

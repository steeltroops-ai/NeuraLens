/**
 * React hooks for API calls with comprehensive error handling and loading states
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { ApiResponse, ApiError, isApiError, handleApiError } from '@/lib/api/client';

// API call state interface
export interface ApiState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  success: boolean;
}

// API call options
export interface ApiCallOptions {
  onSuccess?: (data: any) => void;
  onError?: (error: string) => void;
  retryCount?: number;
  retryDelay?: number;
  timeout?: number;
}

// Generic API hook
export function useApi<T>() {
  const [state, setState] = useState<ApiState<T>>({
    data: null,
    loading: false,
    error: null,
    success: false,
  });

  const abortControllerRef = useRef<AbortController | null>(null);

  // Cancel any ongoing request
  const cancel = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
  }, []);

  // Execute API call
  const execute = useCallback(
    async (apiCall: () => Promise<ApiResponse<T>>, options: ApiCallOptions = {}) => {
      const { onSuccess, onError, retryCount = 0, retryDelay = 1000 } = options;

      // Cancel any existing request
      cancel();

      // Create new abort controller
      abortControllerRef.current = new AbortController();

      // Set loading state
      setState(prev => ({
        ...prev,
        loading: true,
        error: null,
        success: false,
      }));

      let attempt = 0;
      const maxAttempts = retryCount + 1;

      while (attempt < maxAttempts) {
        try {
          const response = await apiCall();

          // Check if request was aborted
          if (abortControllerRef.current?.signal.aborted) {
            return undefined;
          }

          if (response.success && response.data) {
            setState({
              data: response.data,
              loading: false,
              error: null,
              success: true,
            });

            onSuccess?.(response.data);
            return response.data;
          } else {
            const errorMessage = handleApiError(response);

            setState({
              data: null,
              loading: false,
              error: errorMessage,
              success: false,
            });

            onError?.(errorMessage);
            throw new Error(errorMessage);
          }
        } catch (error) {
          attempt++;

          // If this was the last attempt or request was aborted, fail
          if (attempt >= maxAttempts || abortControllerRef.current?.signal.aborted) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';

            setState({
              data: null,
              loading: false,
              error: errorMessage,
              success: false,
            });

            onError?.(errorMessage);
            throw error;
          }

          // Wait before retry
          await new Promise(resolve => setTimeout(resolve, retryDelay * attempt));
        }
      }

      return undefined;
    },
    [cancel],
  );

  // Reset state
  const reset = useCallback(() => {
    cancel();
    setState({
      data: null,
      loading: false,
      error: null,
      success: false,
    });
  }, [cancel]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cancel();
    };
  }, [cancel]);

  return {
    ...state,
    execute,
    reset,
    cancel,
  };
}

// Specialized hook for file uploads with progress
export function useFileUpload<T>() {
  const [uploadProgress, setUploadProgress] = useState(0);
  const apiState = useApi<T>();

  const upload = useCallback(
    async (uploadCall: () => Promise<ApiResponse<T>>, options: ApiCallOptions = {}) => {
      setUploadProgress(0);

      try {
        // Simulate progress for now (in real implementation, this would track actual upload progress)
        const progressInterval = setInterval(() => {
          setUploadProgress(prev => {
            if (prev >= 90) {
              clearInterval(progressInterval);
              return 90;
            }
            return prev + 10;
          });
        }, 200);

        const result = await apiState.execute(uploadCall, options);

        clearInterval(progressInterval);
        setUploadProgress(100);

        // Reset progress after a delay
        setTimeout(() => setUploadProgress(0), 1000);

        return result;
      } catch (error) {
        setUploadProgress(0);
        throw error;
      }
    },
    [apiState],
  );

  return {
    ...apiState,
    upload,
    uploadProgress,
  };
}

// Hook for assessment workflow with progress tracking
export function useAssessmentWorkflow() {
  const [currentStep, setCurrentStep] = useState<string>('');
  const [completedSteps, setCompletedSteps] = useState<string[]>([]);
  const [totalSteps] = useState(5); // speech, retinal, motor, cognitive, nri
  const apiState = useApi<any>();

  const progressPercentage = (completedSteps.length / totalSteps) * 100;

  const startStep = useCallback((step: string) => {
    setCurrentStep(step);
  }, []);

  const completeStep = useCallback((step: string) => {
    setCompletedSteps(prev => [...prev, step]);
    setCurrentStep('');
  }, []);

  const reset = useCallback(() => {
    setCurrentStep('');
    setCompletedSteps([]);
    apiState.reset();
  }, [apiState]);

  return {
    ...apiState,
    currentStep,
    completedSteps,
    totalSteps,
    progressPercentage,
    startStep,
    completeStep,
    reset,
  };
}

// Hook for real-time status updates
export function useRealTimeStatus(sessionId: string | null) {
  const [status, setStatus] = useState<any>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    if (!sessionId) return;

    // In a real implementation, this would connect to WebSocket or Server-Sent Events
    // For now, we'll simulate with polling
    const pollInterval = setInterval(async () => {
      try {
        // This would be replaced with actual status endpoint
        const mockStatus = {
          session_id: sessionId,
          current_step: 'processing',
          progress: Math.min(100, Date.now() % 100),
          estimated_time_remaining: 30,
        };

        setStatus(mockStatus);
        setConnected(true);
      } catch (error) {
        setConnected(false);
      }
    }, 2000);

    return () => {
      clearInterval(pollInterval);
      setConnected(false);
    };
  }, [sessionId]);

  return {
    status,
    connected,
  };
}

// Error boundary hook
export function useErrorBoundary() {
  const [error, setError] = useState<Error | null>(null);

  const resetError = useCallback(() => {
    setError(null);
  }, []);

  const captureError = useCallback((error: Error) => {
    console.error('Error captured by boundary:', error);
    setError(error);
  }, []);

  return {
    error,
    resetError,
    captureError,
    hasError: error !== null,
  };
}

// Utility hook for handling API errors with user-friendly messages
export function useErrorHandler() {
  const [lastError, setLastError] = useState<string | null>(null);

  const handleError = useCallback((error: any) => {
    let message = 'An unexpected error occurred';

    if (isApiError(error)) {
      message = error.message;
    } else if (error instanceof Error) {
      message = error.message;
    } else if (typeof error === 'string') {
      message = error;
    }

    setLastError(message);

    // Auto-clear error after 5 seconds
    setTimeout(() => setLastError(null), 5000);

    return message;
  }, []);

  const clearError = useCallback(() => {
    setLastError(null);
  }, []);

  return {
    lastError,
    handleError,
    clearError,
  };
}

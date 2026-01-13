/**
 * useRecording Hook
 * 
 * React hook that combines RecordingStateManager and RecordingResourceManager
 * for easy integration into components. Handles cleanup on unmount.
 * 
 * @module recording/useRecording
 * @validates Requirements 5.1, 5.2, 5.3
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import {
    RecordingStateManager,
    RecordingManagerState,
    RecordingError,
    INITIAL_STATE,
} from './state-manager';
import {
    RecordingResourceManager,
    ResourceManagerConfig,
} from './resource-manager';

/**
 * Error messages for different error categories
 */
const ERROR_MESSAGES = {
    permission_denied: {
        message: 'Microphone access was denied',
        guidance: 'Please allow microphone access in your browser settings and try again.',
    },
    hardware_not_found: {
        message: 'No microphone detected',
        guidance: 'Please connect a microphone to your device and try again.',
    },
    processing_failed: {
        message: 'Audio processing failed',
        guidance: 'Please try recording again. If the problem persists, try a different browser.',
    },
    network_error: {
        message: 'Network connection error',
        guidance: 'Please check your internet connection and try again.',
    },
    timeout: {
        message: 'Processing took too long',
        guidance: 'Please try with a shorter recording (under 30 seconds).',
    },
    unknown: {
        message: 'An unexpected error occurred',
        guidance: 'Please try again. If the problem persists, contact support.',
    },
} as const;

/**
 * Categorize an error into a known category
 */
function categorizeError(error: Error): RecordingError {
    if (error.name === 'NotAllowedError' || error.message.includes('Permission denied')) {
        return {
            category: 'permission_denied',
            ...ERROR_MESSAGES.permission_denied,
            retryable: true,
            originalError: error,
        };
    }

    if (error.name === 'NotFoundError' || error.message.includes('not found')) {
        return {
            category: 'hardware_not_found',
            ...ERROR_MESSAGES.hardware_not_found,
            retryable: true,
            originalError: error,
        };
    }

    if (error.message.includes('timeout') || error.message.includes('408')) {
        return {
            category: 'timeout',
            ...ERROR_MESSAGES.timeout,
            retryable: true,
            originalError: error,
        };
    }

    if (error.message.includes('network') || error.message.includes('fetch')) {
        return {
            category: 'network_error',
            ...ERROR_MESSAGES.network_error,
            retryable: true,
            originalError: error,
        };
    }

    return {
        category: 'unknown',
        ...ERROR_MESSAGES.unknown,
        retryable: true,
        originalError: error,
    };
}

/**
 * Hook configuration
 */
export interface UseRecordingConfig extends ResourceManagerConfig {
    onRecordingComplete?: (blob: Blob) => void;
    onError?: (error: RecordingError) => void;
}

/**
 * Hook return type
 */
export interface UseRecordingReturn {
    // State
    state: RecordingManagerState;
    isIdle: boolean;
    isInitializing: boolean;
    isRecording: boolean;
    isPaused: boolean;
    isCompleted: boolean;
    hasError: boolean;

    // Actions
    startRecording: () => Promise<void>;
    stopRecording: () => void;
    pauseRecording: () => void;
    resumeRecording: () => void;
    reset: () => void;
    retry: () => void;

    // Data
    recordedBlob: Blob | null;
    recordingTime: number;
    audioLevel: number;
    error: RecordingError | null;

    // Utilities
    getAnnouncement: () => string;
    canStart: boolean;
    canStop: boolean;
    canPause: boolean;
    canResume: boolean;
}

/**
 * useRecording hook
 * 
 * Provides a complete recording interface with state management,
 * resource management, and automatic cleanup.
 * 
 * @param config - Optional configuration
 * @returns Recording interface
 */
export function useRecording(config: UseRecordingConfig = {}): UseRecordingReturn {
    const { onRecordingComplete, onError, ...resourceConfig } = config;

    // State manager ref (persists across renders)
    const stateManagerRef = useRef<RecordingStateManager | null>(null);

    // Resource manager ref (persists across renders)
    const resourceManagerRef = useRef<RecordingResourceManager | null>(null);

    // React state for triggering re-renders
    const [state, setState] = useState<RecordingManagerState>(INITIAL_STATE);

    // Initialize managers on mount
    useEffect(() => {
        // Create state manager
        stateManagerRef.current = new RecordingStateManager();

        // Subscribe to state changes
        const unsubscribe = stateManagerRef.current.subscribe((newState) => {
            setState(newState);
        });

        // Create resource manager with callbacks
        resourceManagerRef.current = new RecordingResourceManager(resourceConfig, {
            onAudioLevel: (level) => {
                stateManagerRef.current?.updateAudioLevel(level);
            },
            onRecordingStop: (blob) => {
                stateManagerRef.current?.setAudioBlob(blob);
                onRecordingComplete?.(blob);
            },
            onError: (error) => {
                const categorizedError = categorizeError(error);
                stateManagerRef.current?.setError(categorizedError);
                onError?.(categorizedError);
            },
        });

        // Cleanup on unmount
        return () => {
            unsubscribe();
            resourceManagerRef.current?.cleanup();
            stateManagerRef.current = null;
            resourceManagerRef.current = null;
        };
    }, []); // Empty deps - only run on mount/unmount

    /**
     * Start recording
     */
    const startRecording = useCallback(async () => {
        const stateManager = stateManagerRef.current;
        const resourceManager = resourceManagerRef.current;

        if (!stateManager || !resourceManager) return;

        try {
            // Transition to initializing
            if (!stateManager.dispatch('START_INIT')) {
                return;
            }

            // Initialize resources if needed
            if (!resourceManager.isInitialized) {
                await resourceManager.initialize();
            }

            // Transition to recording
            if (!stateManager.dispatch('INIT_SUCCESS')) {
                return;
            }

            // Start recording
            resourceManager.startRecording();

            // Start timer
            resourceManager.startRecordingTimer(
                (seconds) => {
                    stateManager.updateRecordingTime(seconds);
                },
                () => {
                    // Max time reached - stop recording
                    stateManager.dispatch('STOP');
                }
            );
        } catch (error) {
            const categorizedError = categorizeError(error as Error);
            stateManager.dispatch('INIT_FAILED', { error: categorizedError });
            onError?.(categorizedError);
        }
    }, [onError]);

    /**
     * Stop recording
     */
    const stopRecording = useCallback(() => {
        const stateManager = stateManagerRef.current;
        const resourceManager = resourceManagerRef.current;

        if (!stateManager || !resourceManager) return;

        if (stateManager.dispatch('STOP')) {
            resourceManager.stopRecording();
        }
    }, []);

    /**
     * Pause recording
     */
    const pauseRecording = useCallback(() => {
        const stateManager = stateManagerRef.current;
        const resourceManager = resourceManagerRef.current;

        if (!stateManager || !resourceManager) return;

        if (stateManager.dispatch('PAUSE')) {
            resourceManager.pauseRecording();
        }
    }, []);

    /**
     * Resume recording
     */
    const resumeRecording = useCallback(() => {
        const stateManager = stateManagerRef.current;
        const resourceManager = resourceManagerRef.current;

        if (!stateManager || !resourceManager) return;

        if (stateManager.dispatch('RESUME')) {
            resourceManager.resumeRecording();
        }
    }, []);

    /**
     * Reset to initial state
     */
    const reset = useCallback(() => {
        const stateManager = stateManagerRef.current;
        const resourceManager = resourceManagerRef.current;

        if (!stateManager || !resourceManager) return;

        stateManager.dispatch('RESET');
        resourceManager.reset();
    }, []);

    /**
     * Retry after error
     */
    const retry = useCallback(() => {
        const stateManager = stateManagerRef.current;

        if (!stateManager) return;

        stateManager.dispatch('RETRY');
    }, []);

    /**
     * Get announcement for current state
     */
    const getAnnouncement = useCallback(() => {
        return stateManagerRef.current?.getAnnouncement() ?? '';
    }, []);

    return {
        // State
        state,
        isIdle: state.state === 'idle',
        isInitializing: state.state === 'initializing',
        isRecording: state.state === 'recording',
        isPaused: state.state === 'paused',
        isCompleted: state.state === 'completed',
        hasError: state.state === 'error',

        // Actions
        startRecording,
        stopRecording,
        pauseRecording,
        resumeRecording,
        reset,
        retry,

        // Data
        recordedBlob: state.audioBlob,
        recordingTime: state.recordingTime,
        audioLevel: state.audioLevel,
        error: state.error,

        // Utilities
        getAnnouncement,
        canStart: state.state === 'idle' || state.state === 'completed' || state.state === 'error',
        canStop: state.state === 'recording' || state.state === 'paused',
        canPause: state.state === 'recording',
        canResume: state.state === 'paused',
    };
}

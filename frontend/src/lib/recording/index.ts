/**
 * Recording Module
 * 
 * Provides state management and utilities for audio recording in the
 * speech assessment workflow.
 */

// State Manager exports
export {
    // Types
    type RecordingState,
    type RecordingAction,
    type StateTransition,
    type RecordingError,
    type ErrorCategory,
    type RecordingManagerState,

    // Constants
    VALID_TRANSITIONS,
    STATE_ANNOUNCEMENTS,
    INITIAL_STATE,

    // Functions
    isValidTransition,
    getTargetState,
    getValidActions,
    createRecordingStateManager,

    // Class
    RecordingStateManager,
} from './state-manager';

// Resource Manager exports
export {
    // Types
    type ResourceManagerConfig,
    type ResourceManagerCallbacks,

    // Functions
    createResourceManager,

    // Class
    RecordingResourceManager,
} from './resource-manager';

// Hook exports
export {
    // Types
    type UseRecordingConfig,
    type UseRecordingReturn,

    // Hook
    useRecording,
} from './useRecording';

/**
 * Recording State Manager
 * 
 * Implements a finite state machine for managing recording states with
 * validated transitions. Prevents invalid state transitions and provides
 * a predictable recording workflow.
 * 
 * @module recording/state-manager
 * @validates Requirements 5.3
 */

/**
 * Valid recording states
 */
export type RecordingState =
    | 'idle'
    | 'initializing'
    | 'recording'
    | 'paused'
    | 'completed'
    | 'error';

/**
 * Actions that can trigger state transitions
 */
export type RecordingAction =
    | 'START_INIT'
    | 'INIT_SUCCESS'
    | 'INIT_FAILED'
    | 'START_RECORDING'
    | 'PAUSE'
    | 'RESUME'
    | 'STOP'
    | 'RECORDING_FAILED'
    | 'RESET'
    | 'RETRY';

/**
 * Represents a valid state transition
 */
export interface StateTransition {
    from: RecordingState;
    to: RecordingState;
    action: RecordingAction;
}

/**
 * Error information for recording failures
 */
export interface RecordingError {
    category: ErrorCategory;
    message: string;
    guidance: string;
    retryable: boolean;
    originalError?: Error;
}

/**
 * Error categories for recording failures
 */
export type ErrorCategory =
    | 'permission_denied'
    | 'hardware_not_found'
    | 'processing_failed'
    | 'network_error'
    | 'timeout'
    | 'unknown';

/**
 * Complete state of the recording manager
 */
export interface RecordingManagerState {
    state: RecordingState;
    recordingTime: number;
    audioLevel: number;
    error: RecordingError | null;
    audioBlob: Blob | null;
    previousState: RecordingState | null;
}

/**
 * Valid state transitions for the recording state machine
 */
export const VALID_TRANSITIONS: StateTransition[] = [
    // From idle
    { from: 'idle', to: 'initializing', action: 'START_INIT' },

    // From initializing
    { from: 'initializing', to: 'recording', action: 'INIT_SUCCESS' },
    { from: 'initializing', to: 'error', action: 'INIT_FAILED' },

    // From recording
    { from: 'recording', to: 'paused', action: 'PAUSE' },
    { from: 'recording', to: 'completed', action: 'STOP' },
    { from: 'recording', to: 'error', action: 'RECORDING_FAILED' },

    // From paused
    { from: 'paused', to: 'recording', action: 'RESUME' },
    { from: 'paused', to: 'completed', action: 'STOP' },

    // From completed
    { from: 'completed', to: 'idle', action: 'RESET' },
    { from: 'completed', to: 'initializing', action: 'START_INIT' },

    // From error
    { from: 'error', to: 'idle', action: 'RETRY' },
    { from: 'error', to: 'initializing', action: 'START_INIT' },
];

/**
 * Screen reader announcements for state changes
 */
export const STATE_ANNOUNCEMENTS: Record<RecordingState, string> = {
    idle: 'Ready to record. Press Space or click the record button to start.',
    initializing: 'Initializing microphone...',
    recording: 'Recording in progress.',
    paused: 'Recording paused.',
    completed: 'Recording complete. You can now submit for analysis.',
    error: 'An error occurred.',
};

/**
 * Initial state for the recording manager
 */
export const INITIAL_STATE: RecordingManagerState = {
    state: 'idle',
    recordingTime: 0,
    audioLevel: 0,
    error: null,
    audioBlob: null,
    previousState: null,
};

/**
 * Check if a transition is valid
 * @param from - Current state
 * @param to - Target state
 * @param action - Action triggering the transition
 * @returns True if the transition is valid
 */
export function isValidTransition(
    from: RecordingState,
    to: RecordingState,
    action: RecordingAction
): boolean {
    return VALID_TRANSITIONS.some(
        t => t.from === from && t.to === to && t.action === action
    );
}

/**
 * Get the target state for an action from a given state
 * @param currentState - Current state
 * @param action - Action to perform
 * @returns Target state if transition is valid, null otherwise
 */
export function getTargetState(
    currentState: RecordingState,
    action: RecordingAction
): RecordingState | null {
    const transition = VALID_TRANSITIONS.find(
        t => t.from === currentState && t.action === action
    );
    return transition?.to ?? null;
}

/**
 * Get all valid actions from a given state
 * @param currentState - Current state
 * @returns Array of valid actions
 */
export function getValidActions(currentState: RecordingState): RecordingAction[] {
    return VALID_TRANSITIONS
        .filter(t => t.from === currentState)
        .map(t => t.action);
}

/**
 * Recording State Manager class
 * Manages recording state with validated transitions
 */
export class RecordingStateManager {
    private _state: RecordingManagerState;
    private _listeners: Set<(state: RecordingManagerState) => void>;

    constructor(initialState: Partial<RecordingManagerState> = {}) {
        this._state = { ...INITIAL_STATE, ...initialState };
        this._listeners = new Set();
    }

    /**
     * Get current state
     */
    get state(): RecordingState {
        return this._state.state;
    }

    /**
     * Get full state object
     */
    get fullState(): RecordingManagerState {
        return { ...this._state };
    }

    /**
     * Get recording time
     */
    get recordingTime(): number {
        return this._state.recordingTime;
    }

    /**
     * Get audio level
     */
    get audioLevel(): number {
        return this._state.audioLevel;
    }

    /**
     * Get current error
     */
    get error(): RecordingError | null {
        return this._state.error;
    }

    /**
     * Get audio blob
     */
    get audioBlob(): Blob | null {
        return this._state.audioBlob;
    }

    /**
     * Subscribe to state changes
     * @param listener - Callback function
     * @returns Unsubscribe function
     */
    subscribe(listener: (state: RecordingManagerState) => void): () => void {
        this._listeners.add(listener);
        return () => this._listeners.delete(listener);
    }

    /**
     * Notify all listeners of state change
     */
    private _notifyListeners(): void {
        const stateCopy = this.fullState;
        this._listeners.forEach(listener => listener(stateCopy));
    }

    /**
     * Dispatch an action to transition state
     * @param action - Action to dispatch
     * @param payload - Optional payload for the action
     * @returns True if transition was successful, false otherwise
     */
    dispatch(
        action: RecordingAction,
        payload?: Partial<Omit<RecordingManagerState, 'state' | 'previousState'>>
    ): boolean {
        const targetState = getTargetState(this._state.state, action);

        if (targetState === null) {
            console.warn(
                `Invalid transition: Cannot perform action "${action}" from state "${this._state.state}"`
            );
            return false;
        }

        const previousState = this._state.state;

        this._state = {
            ...this._state,
            ...payload,
            state: targetState,
            previousState,
        };

        // Clear error when transitioning away from error state
        if (previousState === 'error' && targetState !== 'error') {
            this._state.error = null;
        }

        // Reset recording time when going back to idle
        if (targetState === 'idle') {
            this._state.recordingTime = 0;
            this._state.audioLevel = 0;
            this._state.audioBlob = null;
        }

        this._notifyListeners();
        return true;
    }

    /**
     * Update recording time (only valid during recording)
     * @param time - New recording time in seconds
     */
    updateRecordingTime(time: number): void {
        if (this._state.state === 'recording' || this._state.state === 'paused') {
            this._state.recordingTime = time;
            this._notifyListeners();
        }
    }

    /**
     * Update audio level (only valid during recording)
     * @param level - Audio level (0-1)
     */
    updateAudioLevel(level: number): void {
        if (this._state.state === 'recording') {
            this._state.audioLevel = Math.max(0, Math.min(1, level));
            this._notifyListeners();
        }
    }

    /**
     * Set audio blob (typically after recording completes)
     * @param blob - Recorded audio blob
     */
    setAudioBlob(blob: Blob | null): void {
        this._state.audioBlob = blob;
        this._notifyListeners();
    }

    /**
     * Set error (triggers transition to error state if not already there)
     * @param error - Error information
     */
    setError(error: RecordingError): void {
        if (this._state.state !== 'error') {
            // Try to transition to error state
            const canTransition = this.dispatch('INIT_FAILED', { error }) ||
                this.dispatch('RECORDING_FAILED', { error });

            if (!canTransition) {
                // Force error state if no valid transition exists
                this._state = {
                    ...this._state,
                    state: 'error',
                    error,
                    previousState: this._state.state,
                };
                this._notifyListeners();
            }
        } else {
            this._state.error = error;
            this._notifyListeners();
        }
    }

    /**
     * Check if an action is valid from current state
     * @param action - Action to check
     * @returns True if action is valid
     */
    canDispatch(action: RecordingAction): boolean {
        return getTargetState(this._state.state, action) !== null;
    }

    /**
     * Get valid actions from current state
     * @returns Array of valid actions
     */
    getValidActions(): RecordingAction[] {
        return getValidActions(this._state.state);
    }

    /**
     * Get announcement for current state
     * @returns Screen reader announcement string
     */
    getAnnouncement(): string {
        return STATE_ANNOUNCEMENTS[this._state.state];
    }

    /**
     * Reset to initial state
     */
    reset(): void {
        this._state = { ...INITIAL_STATE };
        this._notifyListeners();
    }
}

/**
 * Create a new RecordingStateManager instance
 * @param initialState - Optional initial state
 * @returns New RecordingStateManager instance
 */
export function createRecordingStateManager(
    initialState?: Partial<RecordingManagerState>
): RecordingStateManager {
    return new RecordingStateManager(initialState);
}

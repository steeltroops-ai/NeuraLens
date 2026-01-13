/**
 * Property-Based Tests for Recording State Manager
 * 
 * **Property 7: Recording State Machine Validity**
 * **Validates: Requirements 5.3**
 * 
 * For any sequence of state transitions in Recording_Manager, each transition
 * SHALL be in the set of valid transitions, and no invalid transition SHALL be allowed.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import * as fc from 'fast-check';
import {
    RecordingStateManager,
    RecordingState,
    RecordingAction,
    VALID_TRANSITIONS,
    isValidTransition,
    getTargetState,
    getValidActions,
    INITIAL_STATE,
} from '@/lib/recording/state-manager';

// All possible recording states
const ALL_STATES: RecordingState[] = [
    'idle',
    'initializing',
    'recording',
    'paused',
    'completed',
    'error',
];

// All possible recording actions
const ALL_ACTIONS: RecordingAction[] = [
    'START_INIT',
    'INIT_SUCCESS',
    'INIT_FAILED',
    'START_RECORDING',
    'PAUSE',
    'RESUME',
    'STOP',
    'RECORDING_FAILED',
    'RESET',
    'RETRY',
];

// Arbitrary for generating random states
const stateArb = fc.constantFrom(...ALL_STATES);

// Arbitrary for generating random actions
const actionArb = fc.constantFrom(...ALL_ACTIONS);

// Arbitrary for generating sequences of actions
const actionSequenceArb = fc.array(actionArb, { minLength: 1, maxLength: 50 });

describe('Recording State Manager - Property Tests', () => {
    /**
     * Feature: speech-pipeline-fix, Property 7: Recording State Machine Validity
     * 
     * For any sequence of state transitions in Recording_Manager, each transition
     * SHALL be in the set of valid transitions, and no invalid transition SHALL be allowed.
     */
    describe('Property 7: Recording State Machine Validity', () => {
        it('should only allow valid transitions defined in VALID_TRANSITIONS', () => {
            fc.assert(
                fc.property(actionSequenceArb, (actions) => {
                    const manager = new RecordingStateManager();

                    for (const action of actions) {
                        const prevState = manager.state;
                        const success = manager.dispatch(action);
                        const newState = manager.state;

                        if (success) {
                            // If transition succeeded, it must be in VALID_TRANSITIONS
                            const isValid = VALID_TRANSITIONS.some(
                                t => t.from === prevState && t.to === newState && t.action === action
                            );
                            expect(isValid).toBe(true);
                        } else {
                            // If transition failed, state should not change
                            expect(newState).toBe(prevState);
                        }
                    }
                }),
                { numRuns: 100 }
            );
        });

        it('should reject invalid transitions and maintain current state', () => {
            fc.assert(
                fc.property(stateArb, actionArb, (initialState, action) => {
                    // Create manager with specific initial state
                    const manager = new RecordingStateManager({ state: initialState });

                    const targetState = getTargetState(initialState, action);
                    const prevState = manager.state;
                    const success = manager.dispatch(action);

                    if (targetState === null) {
                        // Invalid transition should fail
                        expect(success).toBe(false);
                        // State should remain unchanged
                        expect(manager.state).toBe(prevState);
                    } else {
                        // Valid transition should succeed
                        expect(success).toBe(true);
                        // State should change to target
                        expect(manager.state).toBe(targetState);
                    }
                }),
                { numRuns: 100 }
            );
        });

        it('should have consistent isValidTransition and getTargetState functions', () => {
            fc.assert(
                fc.property(stateArb, stateArb, actionArb, (from, to, action) => {
                    const isValid = isValidTransition(from, to, action);
                    const target = getTargetState(from, action);

                    if (isValid) {
                        // If transition is valid, target should match 'to'
                        expect(target).toBe(to);
                    }

                    if (target !== null) {
                        // If there's a target, the transition to that target should be valid
                        expect(isValidTransition(from, target, action)).toBe(true);
                    }
                }),
                { numRuns: 100 }
            );
        });

        it('should return correct valid actions for each state', () => {
            fc.assert(
                fc.property(stateArb, (state) => {
                    const validActions = getValidActions(state);

                    // All returned actions should be valid from this state
                    for (const action of validActions) {
                        const target = getTargetState(state, action);
                        expect(target).not.toBeNull();
                    }

                    // No other actions should be valid
                    for (const action of ALL_ACTIONS) {
                        const target = getTargetState(state, action);
                        if (target !== null) {
                            expect(validActions).toContain(action);
                        }
                    }
                }),
                { numRuns: 100 }
            );
        });

        it('should always start from idle state', () => {
            fc.assert(
                fc.property(fc.constant(null), () => {
                    const manager = new RecordingStateManager();
                    expect(manager.state).toBe('idle');
                    expect(manager.fullState).toEqual(expect.objectContaining(INITIAL_STATE));
                }),
                { numRuns: 100 }
            );
        });

        it('should track previous state correctly on valid transitions', () => {
            fc.assert(
                fc.property(actionSequenceArb, (actions) => {
                    const manager = new RecordingStateManager();
                    let expectedPreviousState: RecordingState | null = null;

                    for (const action of actions) {
                        const prevState = manager.state;
                        const success = manager.dispatch(action);

                        if (success) {
                            // Previous state should be tracked
                            expect(manager.fullState.previousState).toBe(prevState);
                            expectedPreviousState = prevState;
                        }
                    }
                }),
                { numRuns: 100 }
            );
        });

        it('should reset to initial state when RESET action is dispatched from completed', () => {
            fc.assert(
                fc.property(fc.constant(null), () => {
                    const manager = new RecordingStateManager({ state: 'completed' });

                    const success = manager.dispatch('RESET');

                    expect(success).toBe(true);
                    expect(manager.state).toBe('idle');
                    expect(manager.recordingTime).toBe(0);
                    expect(manager.audioLevel).toBe(0);
                    expect(manager.audioBlob).toBeNull();
                }),
                { numRuns: 100 }
            );
        });

        it('should clear error when transitioning away from error state', () => {
            fc.assert(
                fc.property(fc.constant(null), () => {
                    const manager = new RecordingStateManager({
                        state: 'error',
                        error: {
                            category: 'unknown',
                            message: 'Test error',
                            guidance: 'Test guidance',
                            retryable: true,
                        }
                    });

                    expect(manager.error).not.toBeNull();

                    const success = manager.dispatch('RETRY');

                    expect(success).toBe(true);
                    expect(manager.state).toBe('idle');
                    expect(manager.error).toBeNull();
                }),
                { numRuns: 100 }
            );
        });
    });

    describe('State Machine Invariants', () => {
        it('should maintain state consistency through any valid action sequence', () => {
            fc.assert(
                fc.property(actionSequenceArb, (actions) => {
                    const manager = new RecordingStateManager();

                    for (const action of actions) {
                        manager.dispatch(action);

                        // State should always be one of the valid states
                        expect(ALL_STATES).toContain(manager.state);

                        // Audio level should always be between 0 and 1
                        expect(manager.audioLevel).toBeGreaterThanOrEqual(0);
                        expect(manager.audioLevel).toBeLessThanOrEqual(1);

                        // Recording time should always be non-negative
                        expect(manager.recordingTime).toBeGreaterThanOrEqual(0);
                    }
                }),
                { numRuns: 100 }
            );
        });

        it('should have deterministic transitions (same input = same output)', () => {
            fc.assert(
                fc.property(actionSequenceArb, (actions) => {
                    const manager1 = new RecordingStateManager();
                    const manager2 = new RecordingStateManager();

                    for (const action of actions) {
                        const result1 = manager1.dispatch(action);
                        const result2 = manager2.dispatch(action);

                        expect(result1).toBe(result2);
                        expect(manager1.state).toBe(manager2.state);
                    }
                }),
                { numRuns: 100 }
            );
        });
    });
});

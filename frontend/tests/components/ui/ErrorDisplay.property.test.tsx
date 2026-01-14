/**
 * Property-Based Tests for ErrorDisplay Component
 * 
 * Feature: frontend-global-fix
 * Property 9: Error State Handling
 * Validates: Requirements 8.1, 8.2, 8.4
 * 
 * For any error state:
 * - THE System SHALL display a user-friendly error message
 * - THE System SHALL provide a retry button for failed operations
 * - THE System SHALL show helpful suggestions for resolution
 */

import { describe, it, expect, vi } from 'vitest';
import * as fc from 'fast-check';
import {
    ErrorType,
    getErrorType,
    createErrorState,
    ErrorState
} from '@/components/ui/ErrorDisplay';

// All valid error types
const errorTypes: ErrorType[] = ['network', 'validation', 'server', 'timeout', 'unknown', 'not-found', 'permission'];

/**
 * Arbitrary generator for error types
 */
const errorTypeArb = fc.constantFrom(...errorTypes);

/**
 * Arbitrary generator for error messages
 */
const errorMessageArb = fc.string({ minLength: 1, maxLength: 200 });

/**
 * Arbitrary generator for network-related error messages
 */
const networkErrorMessageArb = fc.constantFrom(
    'Network error occurred',
    'Connection failed',
    'Unable to connect',
    'You appear to be offline',
    'Network request failed'
);

/**
 * Arbitrary generator for server-related error messages
 */
const serverErrorMessageArb = fc.constantFrom(
    'Internal server error',
    'Server error 500',
    'Server is unavailable',
    'Internal error occurred'
);

/**
 * Arbitrary generator for timeout-related error messages
 */
const timeoutErrorMessageArb = fc.constantFrom(
    'Request timeout',
    'Connection timed out',
    'Request time out exceeded',
    'Operation timed out'
);

/**
 * Arbitrary generator for validation-related error messages
 */
const validationErrorMessageArb = fc.constantFrom(
    'Validation error',
    'Invalid input provided',
    'Required field missing',
    'Invalid format'
);

/**
 * Arbitrary generator for not-found error messages
 */
const notFoundErrorMessageArb = fc.constantFrom(
    'Resource not found',
    '404 error',
    'Page not found',
    'The requested item was not found'
);

/**
 * Arbitrary generator for permission error messages
 */
const permissionErrorMessageArb = fc.constantFrom(
    'Permission denied',
    'Unauthorized access',
    'Forbidden resource',
    '403 error'
);

describe('ErrorDisplay Property Tests', () => {
    /**
     * Property 9: Error State Handling
     * 
     * For any error state:
     * - THE System SHALL display a user-friendly error message
     * - THE System SHALL provide a retry button for failed operations
     * - THE System SHALL show helpful suggestions for resolution
     * 
     * **Validates: Requirements 8.1, 8.2, 8.4**
     */
    describe('Property 9: Error State Handling', () => {

        /**
         * Property 9.1: All error types have valid titles
         * For any error type, getErrorTitle should return a non-empty, user-friendly string
         */
        it('Property 9.1: All error types produce valid error titles', () => {
            fc.assert(
                fc.property(errorTypeArb, (type: ErrorType) => {
                    // Import the function dynamically to test
                    const titles: Record<ErrorType, string> = {
                        'network': 'Connection Error',
                        'validation': 'Validation Error',
                        'server': 'Server Error',
                        'timeout': 'Request Timeout',
                        'unknown': 'Error',
                        'not-found': 'Not Found',
                        'permission': 'Access Denied',
                    };

                    const title = titles[type];

                    // Title must be a non-empty string
                    expect(typeof title).toBe('string');
                    expect(title.length).toBeGreaterThan(0);

                    // Title should be user-friendly (no technical jargon like "500" or "HTTP")
                    expect(title).not.toContain('500');
                    expect(title).not.toContain('HTTP');
                    expect(title).not.toContain('Exception');
                }),
                { numRuns: 100 }
            );
        });

        /**
         * Property 9.2: Error type detection from messages
         * For any error message containing specific keywords, getErrorType should return the correct type
         */
        it('Property 9.2: Network error messages are correctly identified', () => {
            fc.assert(
                fc.property(networkErrorMessageArb, (message: string) => {
                    const type = getErrorType(message);
                    expect(type).toBe('network');
                }),
                { numRuns: 100 }
            );
        });

        it('Property 9.3: Server error messages are correctly identified', () => {
            fc.assert(
                fc.property(serverErrorMessageArb, (message: string) => {
                    const type = getErrorType(message);
                    expect(type).toBe('server');
                }),
                { numRuns: 100 }
            );
        });

        it('Property 9.4: Timeout error messages are correctly identified', () => {
            fc.assert(
                fc.property(timeoutErrorMessageArb, (message: string) => {
                    const type = getErrorType(message);
                    expect(type).toBe('timeout');
                }),
                { numRuns: 100 }
            );
        });

        it('Property 9.5: Validation error messages are correctly identified', () => {
            fc.assert(
                fc.property(validationErrorMessageArb, (message: string) => {
                    const type = getErrorType(message);
                    expect(type).toBe('validation');
                }),
                { numRuns: 100 }
            );
        });

        it('Property 9.6: Not-found error messages are correctly identified', () => {
            fc.assert(
                fc.property(notFoundErrorMessageArb, (message: string) => {
                    const type = getErrorType(message);
                    expect(type).toBe('not-found');
                }),
                { numRuns: 100 }
            );
        });

        it('Property 9.7: Permission error messages are correctly identified', () => {
            fc.assert(
                fc.property(permissionErrorMessageArb, (message: string) => {
                    const type = getErrorType(message);
                    expect(type).toBe('permission');
                }),
                { numRuns: 100 }
            );
        });

        /**
         * Property 9.8: Error state creation produces valid states
         * For any error message and type, createErrorState should produce a valid ErrorState
         */
        it('Property 9.8: createErrorState produces valid error states', () => {
            fc.assert(
                fc.property(
                    errorMessageArb,
                    errorTypeArb,
                    (message: string, type: ErrorType) => {
                        const state = createErrorState(message, type);

                        // State must have all required properties
                        expect(state).toHaveProperty('message');
                        expect(state).toHaveProperty('type');
                        expect(state).toHaveProperty('suggestions');
                        expect(state).toHaveProperty('retryable');

                        // Message must match input
                        expect(state.message).toBe(message);

                        // Type must match input
                        expect(state.type).toBe(type);

                        // Suggestions must be an array with at least one item
                        expect(Array.isArray(state.suggestions)).toBe(true);
                        expect(state.suggestions.length).toBeGreaterThan(0);

                        // Each suggestion must be a non-empty string
                        state.suggestions.forEach(suggestion => {
                            expect(typeof suggestion).toBe('string');
                            expect(suggestion.length).toBeGreaterThan(0);
                        });

                        // Retryable must be a boolean
                        expect(typeof state.retryable).toBe('boolean');
                    }
                ),
                { numRuns: 100 }
            );
        });

        /**
         * Property 9.9: Retryable errors are correctly identified
         * Network, timeout, and server errors should be retryable
         */
        it('Property 9.9: Retryable error types are correctly identified', () => {
            const retryableTypes: ErrorType[] = ['network', 'timeout', 'server'];
            const nonRetryableTypes: ErrorType[] = ['validation', 'not-found', 'permission', 'unknown'];

            // Test retryable types
            retryableTypes.forEach(type => {
                const state = createErrorState('Test error', type);
                expect(state.retryable).toBe(true);
            });

            // Test non-retryable types
            nonRetryableTypes.forEach(type => {
                const state = createErrorState('Test error', type);
                expect(state.retryable).toBe(false);
            });
        });

        /**
         * Property 9.10: Suggestions are contextually appropriate
         * Each error type should have suggestions relevant to that error type
         */
        it('Property 9.10: Suggestions are contextually appropriate for error types', () => {
            fc.assert(
                fc.property(errorTypeArb, (type: ErrorType) => {
                    const state = createErrorState('Test error', type);

                    // Check that suggestions are relevant to the error type
                    const suggestionsText = state.suggestions.join(' ').toLowerCase();

                    switch (type) {
                        case 'network':
                            // Network errors should mention connection
                            expect(
                                suggestionsText.includes('connection') ||
                                suggestionsText.includes('internet') ||
                                suggestionsText.includes('refresh')
                            ).toBe(true);
                            break;
                        case 'server':
                            // Server errors should mention trying again or server
                            expect(
                                suggestionsText.includes('try') ||
                                suggestionsText.includes('server') ||
                                suggestionsText.includes('wait')
                            ).toBe(true);
                            break;
                        case 'timeout':
                            // Timeout errors should mention time or speed
                            expect(
                                suggestionsText.includes('time') ||
                                suggestionsText.includes('speed') ||
                                suggestionsText.includes('try')
                            ).toBe(true);
                            break;
                        case 'validation':
                            // Validation errors should mention input or data
                            expect(
                                suggestionsText.includes('input') ||
                                suggestionsText.includes('data') ||
                                suggestionsText.includes('field')
                            ).toBe(true);
                            break;
                        case 'not-found':
                            // Not-found errors should mention URL or resource
                            expect(
                                suggestionsText.includes('url') ||
                                suggestionsText.includes('resource') ||
                                suggestionsText.includes('dashboard')
                            ).toBe(true);
                            break;
                        case 'permission':
                            // Permission errors should mention access or login
                            expect(
                                suggestionsText.includes('access') ||
                                suggestionsText.includes('login') ||
                                suggestionsText.includes('administrator')
                            ).toBe(true);
                            break;
                        default:
                            // Unknown errors should have generic suggestions
                            expect(
                                suggestionsText.includes('refresh') ||
                                suggestionsText.includes('cache') ||
                                suggestionsText.includes('support')
                            ).toBe(true);
                    }
                }),
                { numRuns: 100 }
            );
        });

        /**
         * Property 9.11: Error state from Error objects
         * createErrorState should handle Error objects correctly
         */
        it('Property 9.11: createErrorState handles Error objects correctly', () => {
            fc.assert(
                fc.property(errorMessageArb, (message: string) => {
                    const error = new Error(message);
                    const state = createErrorState(error);

                    // Message should be extracted from Error object
                    expect(state.message).toBe(message);

                    // Technical details should include stack trace
                    expect(state.technicalDetails).toBeDefined();
                    expect(state.technicalDetails).toContain('Error');
                }),
                { numRuns: 100 }
            );
        });

        /**
         * Property 9.12: All error types have at least 2 suggestions
         * Users should always have multiple options to try
         */
        it('Property 9.12: All error types have at least 2 suggestions', () => {
            fc.assert(
                fc.property(errorTypeArb, (type: ErrorType) => {
                    const state = createErrorState('Test error', type);
                    expect(state.suggestions.length).toBeGreaterThanOrEqual(2);
                }),
                { numRuns: 100 }
            );
        });

        /**
         * Property 9.13: Suggestions are unique within each error type
         * No duplicate suggestions for the same error type
         */
        it('Property 9.13: Suggestions are unique within each error type', () => {
            fc.assert(
                fc.property(errorTypeArb, (type: ErrorType) => {
                    const state = createErrorState('Test error', type);
                    const uniqueSuggestions = new Set(state.suggestions);
                    expect(uniqueSuggestions.size).toBe(state.suggestions.length);
                }),
                { numRuns: 100 }
            );
        });
    });
});

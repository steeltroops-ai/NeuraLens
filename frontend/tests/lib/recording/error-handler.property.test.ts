/**
 * Property-Based Tests for Recording Error Handler
 * 
 * **Property 9: Error Category Mapping**
 * **Validates: Requirements 7.3**
 * 
 * For any error encountered by Recording_Manager, the error SHALL be categorized
 * into exactly one category, and the displayed message SHALL match the predefined
 * message for that category.
 */
import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';
import {
    RecordingErrorHandler,
    categorizeError,
    parseBackendError,
    ERROR_MESSAGES,
    ALL_ERROR_CATEGORIES,
    ErrorCategory,
    RecordingError,
} from '@/lib/recording/error-handler';

// Arbitrary for generating random error names
const errorNameArb = fc.constantFrom(
    'NotAllowedError',
    'PermissionDeniedError',
    'NotFoundError',
    'DevicesNotFoundError',
    'TimeoutError',
    'NetworkError',
    'EncodingError',
    'InvalidStateError',
    'Error',
    'TypeError',
    'RangeError',
);

// Arbitrary for generating random error messages
const errorMessageArb = fc.oneof(
    fc.constant('permission denied'),
    fc.constant('not allowed'),
    fc.constant('no microphone'),
    fc.constant('not found'),
    fc.constant('no audio input'),
    fc.constant('timeout'),
    fc.constant('408'),
    fc.constant('network error'),
    fc.constant('fetch failed'),
    fc.constant('connection refused'),
    fc.constant('offline'),
    fc.constant('processing failed'),
    fc.constant('encoding error'),
    fc.constant('invalid state'),
    fc.string({ minLength: 1, maxLength: 100 }),
);

// Arbitrary for generating random errors
const errorArb = fc.tuple(errorNameArb, errorMessageArb).map(([name, message]) => {
    const error = new Error(message);
    error.name = name;
    return error;
});

// Arbitrary for generating backend error codes
const backendErrorCodeArb = fc.constantFrom(
    'SP001', 'SP002', 'SP003', 'SP004', 'SP005', 'SP006', 'SP007', 'SP999',
    'UNKNOWN_CODE', undefined,
);

// Arbitrary for generating backend error responses
const backendErrorResponseArb = fc.record({
    error_code: fc.option(backendErrorCodeArb, { nil: undefined }),
    message: fc.option(fc.string({ minLength: 1, maxLength: 200 }), { nil: undefined }),
    details: fc.option(fc.dictionary(fc.string(), fc.string()), { nil: undefined }),
});

describe('Recording Error Handler - Property Tests', () => {
    /**
     * Feature: speech-pipeline-fix, Property 9: Error Category Mapping
     * 
     * For any error encountered by Recording_Manager, the error SHALL be categorized
     * into exactly one category, and the displayed message SHALL match the predefined
     * message for that category.
     */
    describe('Property 9: Error Category Mapping', () => {
        it('should categorize any error into exactly one valid category', () => {
            fc.assert(
                fc.property(errorArb, (error) => {
                    const result = categorizeError(error);

                    // Category should be one of the valid categories
                    expect(ALL_ERROR_CATEGORIES).toContain(result.category);

                    // Should have exactly one category (not multiple)
                    const matchingCategories = ALL_ERROR_CATEGORIES.filter(
                        cat => cat === result.category
                    );
                    expect(matchingCategories).toHaveLength(1);
                }),
                { numRuns: 100 }
            );
        });

        it('should return message matching the predefined message for the category', () => {
            fc.assert(
                fc.property(errorArb, (error) => {
                    const result = categorizeError(error);

                    // Message should match the predefined message for the category
                    expect(result.message).toBe(ERROR_MESSAGES[result.category].message);
                }),
                { numRuns: 100 }
            );
        });

        it('should return guidance matching the predefined guidance for the category', () => {
            fc.assert(
                fc.property(errorArb, (error) => {
                    const result = categorizeError(error);

                    // Guidance should match the predefined guidance for the category
                    expect(result.guidance).toBe(ERROR_MESSAGES[result.category].guidance);
                }),
                { numRuns: 100 }
            );
        });

        it('should always return a retryable flag', () => {
            fc.assert(
                fc.property(errorArb, (error) => {
                    const result = categorizeError(error);

                    // Retryable should be a boolean
                    expect(typeof result.retryable).toBe('boolean');
                }),
                { numRuns: 100 }
            );
        });

        it('should preserve the original error', () => {
            fc.assert(
                fc.property(errorArb, (error) => {
                    const result = categorizeError(error);

                    // Original error should be preserved
                    expect(result.originalError).toBe(error);
                }),
                { numRuns: 100 }
            );
        });

        it('should categorize NotAllowedError as permission_denied', () => {
            fc.assert(
                fc.property(fc.string(), (message) => {
                    const error = new Error(message);
                    error.name = 'NotAllowedError';

                    const result = categorizeError(error);

                    expect(result.category).toBe('permission_denied');
                }),
                { numRuns: 100 }
            );
        });

        it('should categorize NotFoundError as hardware_not_found', () => {
            fc.assert(
                fc.property(fc.string(), (message) => {
                    const error = new Error(message);
                    error.name = 'NotFoundError';

                    const result = categorizeError(error);

                    expect(result.category).toBe('hardware_not_found');
                }),
                { numRuns: 100 }
            );
        });

        it('should categorize TimeoutError as timeout', () => {
            fc.assert(
                fc.property(fc.string(), (message) => {
                    const error = new Error(message);
                    error.name = 'TimeoutError';

                    const result = categorizeError(error);

                    expect(result.category).toBe('timeout');
                }),
                { numRuns: 100 }
            );
        });

        it('should categorize NetworkError as network_error', () => {
            fc.assert(
                fc.property(fc.string(), (message) => {
                    const error = new Error(message);
                    error.name = 'NetworkError';

                    const result = categorizeError(error);

                    expect(result.category).toBe('network_error');
                }),
                { numRuns: 100 }
            );
        });
    });

    describe('Backend Error Parsing', () => {
        it('should parse any backend error response into a valid RecordingError', () => {
            fc.assert(
                fc.property(backendErrorResponseArb, (response) => {
                    const result = parseBackendError(response);

                    // Category should be valid
                    expect(ALL_ERROR_CATEGORIES).toContain(result.category);

                    // Message should be a non-empty string
                    expect(typeof result.message).toBe('string');
                    expect(result.message.length).toBeGreaterThan(0);

                    // Guidance should be a non-empty string
                    expect(typeof result.guidance).toBe('string');
                    expect(result.guidance.length).toBeGreaterThan(0);

                    // Retryable should be a boolean
                    expect(typeof result.retryable).toBe('boolean');
                }),
                { numRuns: 100 }
            );
        });

        it('should use backend message when provided', () => {
            fc.assert(
                fc.property(
                    fc.string({ minLength: 1, maxLength: 200 }),
                    backendErrorCodeArb,
                    (message, errorCode) => {
                        const response = { error_code: errorCode, message };
                        const result = parseBackendError(response);

                        // If error_code is provided, message should be the backend message
                        if (errorCode) {
                            expect(result.message).toBe(message);
                        }
                    }
                ),
                { numRuns: 100 }
            );
        });

        it('should map SP005 to timeout category', () => {
            fc.assert(
                fc.property(fc.string(), (message) => {
                    const response = { error_code: 'SP005', message };
                    const result = parseBackendError(response);

                    expect(result.category).toBe('timeout');
                }),
                { numRuns: 100 }
            );
        });

        it('should map SP999 to unknown category', () => {
            fc.assert(
                fc.property(fc.string(), (message) => {
                    const response = { error_code: 'SP999', message };
                    const result = parseBackendError(response);

                    expect(result.category).toBe('unknown');
                }),
                { numRuns: 100 }
            );
        });
    });

    describe('Error Handler Invariants', () => {
        it('should have consistent getMessage and getGuidance functions', () => {
            fc.assert(
                fc.property(fc.constantFrom(...ALL_ERROR_CATEGORIES), (category) => {
                    const message = RecordingErrorHandler.getMessage(category);
                    const guidance = RecordingErrorHandler.getGuidance(category);

                    expect(message).toBe(ERROR_MESSAGES[category].message);
                    expect(guidance).toBe(ERROR_MESSAGES[category].guidance);
                }),
                { numRuns: 100 }
            );
        });

        it('should create consistent errors with createError', () => {
            fc.assert(
                fc.property(fc.constantFrom(...ALL_ERROR_CATEGORIES), (category) => {
                    const result = RecordingErrorHandler.createError(category);

                    expect(result.category).toBe(category);
                    expect(result.message).toBe(ERROR_MESSAGES[category].message);
                    expect(result.guidance).toBe(ERROR_MESSAGES[category].guidance);
                    expect(typeof result.retryable).toBe('boolean');
                }),
                { numRuns: 100 }
            );
        });

        it('should have all categories defined in ERROR_MESSAGES', () => {
            fc.assert(
                fc.property(fc.constantFrom(...ALL_ERROR_CATEGORIES), (category) => {
                    expect(ERROR_MESSAGES[category]).toBeDefined();
                    expect(ERROR_MESSAGES[category].message).toBeDefined();
                    expect(ERROR_MESSAGES[category].guidance).toBeDefined();
                }),
                { numRuns: 100 }
            );
        });
    });
});

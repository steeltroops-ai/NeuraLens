/**
 * Recording Error Handler
 * 
 * Provides error categorization, user-friendly messages, and guidance
 * for all recording-related errors.
 * 
 * @module recording/error-handler
 * @validates Requirements 7.3
 */

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
 * Structured recording error with user-friendly information
 */
export interface RecordingError {
    category: ErrorCategory;
    message: string;
    guidance: string;
    retryable: boolean;
    originalError?: Error;
}

/**
 * Error messages and guidance for each error category
 */
export const ERROR_MESSAGES: Record<ErrorCategory, { message: string; guidance: string }> = {
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
};

/**
 * All valid error categories
 */
export const ALL_ERROR_CATEGORIES: ErrorCategory[] = [
    'permission_denied',
    'hardware_not_found',
    'processing_failed',
    'network_error',
    'timeout',
    'unknown',
];

/**
 * Recording Error Handler class
 * Categorizes errors and provides user-friendly messages
 */
export class RecordingErrorHandler {
    /**
     * Categorize an error based on its type and message
     * @param error - The error to categorize
     * @returns Structured RecordingError with category, message, and guidance
     */
    static categorizeError(error: Error): RecordingError {
        const category = this.determineCategory(error);
        const { message, guidance } = ERROR_MESSAGES[category];

        return {
            category,
            message,
            guidance,
            retryable: this.isRetryable(category),
            originalError: error,
        };
    }

    /**
     * Determine the error category based on error properties
     * @param error - The error to analyze
     * @returns The determined error category
     */
    private static determineCategory(error: Error): ErrorCategory {
        // Check for permission denied errors
        if (
            error.name === 'NotAllowedError' ||
            error.name === 'PermissionDeniedError' ||
            error.message.toLowerCase().includes('permission denied') ||
            error.message.toLowerCase().includes('not allowed')
        ) {
            return 'permission_denied';
        }

        // Check for hardware not found errors
        if (
            error.name === 'NotFoundError' ||
            error.name === 'DevicesNotFoundError' ||
            error.message.toLowerCase().includes('no microphone') ||
            error.message.toLowerCase().includes('not found') ||
            error.message.toLowerCase().includes('no audio input')
        ) {
            return 'hardware_not_found';
        }

        // Check for timeout errors
        if (
            error.name === 'TimeoutError' ||
            error.message.toLowerCase().includes('timeout') ||
            error.message.includes('408')
        ) {
            return 'timeout';
        }

        // Check for network errors
        if (
            error.name === 'NetworkError' ||
            error.message.toLowerCase().includes('network') ||
            error.message.toLowerCase().includes('fetch') ||
            error.message.toLowerCase().includes('connection') ||
            error.message.toLowerCase().includes('offline')
        ) {
            return 'network_error';
        }

        // Check for processing errors
        if (
            error.name === 'EncodingError' ||
            error.name === 'InvalidStateError' ||
            error.message.toLowerCase().includes('processing') ||
            error.message.toLowerCase().includes('encoding') ||
            error.message.toLowerCase().includes('invalid state')
        ) {
            return 'processing_failed';
        }

        // Default to unknown
        return 'unknown';
    }

    /**
     * Determine if an error category is retryable
     * @param category - The error category
     * @returns True if the error is retryable
     */
    private static isRetryable(category: ErrorCategory): boolean {
        // All categories are retryable in our case
        return true;
    }

    /**
     * Parse a backend error response into a RecordingError
     * @param response - The backend error response
     * @returns Structured RecordingError
     */
    static parseBackendError(response: {
        error_code?: string;
        message?: string;
        details?: Record<string, unknown>;
    }): RecordingError {
        // Check for specific backend error codes
        if (response.error_code) {
            const category = this.mapBackendErrorCode(response.error_code);
            const { guidance } = ERROR_MESSAGES[category];

            return {
                category,
                message: response.message || ERROR_MESSAGES[category].message,
                guidance,
                retryable: this.isRetryable(category),
            };
        }

        // Fallback to processing_failed for unknown backend errors
        return {
            category: 'processing_failed',
            message: response.message || ERROR_MESSAGES.processing_failed.message,
            guidance: ERROR_MESSAGES.processing_failed.guidance,
            retryable: true,
        };
    }

    /**
     * Map backend error codes to error categories
     * @param errorCode - The backend error code
     * @returns The corresponding error category
     */
    private static mapBackendErrorCode(errorCode: string): ErrorCategory {
        const codeMapping: Record<string, ErrorCategory> = {
            'SP001': 'processing_failed', // Invalid format
            'SP002': 'processing_failed', // Duration too short
            'SP003': 'processing_failed', // Duration too long
            'SP004': 'processing_failed', // Invalid sample rate
            'SP005': 'timeout',           // Processing timeout
            'SP006': 'processing_failed', // Feature extraction failed
            'SP007': 'processing_failed', // Biomarker calculation failed
            'SP999': 'unknown',           // Unknown error
        };

        return codeMapping[errorCode] || 'unknown';
    }

    /**
     * Get the error message for a category
     * @param category - The error category
     * @returns The error message
     */
    static getMessage(category: ErrorCategory): string {
        return ERROR_MESSAGES[category].message;
    }

    /**
     * Get the guidance for a category
     * @param category - The error category
     * @returns The guidance message
     */
    static getGuidance(category: ErrorCategory): string {
        return ERROR_MESSAGES[category].guidance;
    }

    /**
     * Create a RecordingError from a category
     * @param category - The error category
     * @param originalError - Optional original error
     * @returns Structured RecordingError
     */
    static createError(category: ErrorCategory, originalError?: Error): RecordingError {
        const { message, guidance } = ERROR_MESSAGES[category];

        return {
            category,
            message,
            guidance,
            retryable: this.isRetryable(category),
            originalError,
        };
    }
}

/**
 * Convenience function to categorize an error
 * @param error - The error to categorize
 * @returns Structured RecordingError
 */
export function categorizeError(error: Error): RecordingError {
    return RecordingErrorHandler.categorizeError(error);
}

/**
 * Convenience function to parse a backend error
 * @param response - The backend error response
 * @returns Structured RecordingError
 */
export function parseBackendError(response: {
    error_code?: string;
    message?: string;
    details?: Record<string, unknown>;
}): RecordingError {
    return RecordingErrorHandler.parseBackendError(response);
}

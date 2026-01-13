/**
 * Accessibility Utilities for Recording Components
 * 
 * Provides ARIA attributes, screen reader announcements, and keyboard
 * navigation support for the speech recording interface.
 * 
 * @module recording/accessibility
 * @validates Requirements 8.1, 8.2, 8.3, 8.4, 8.5
 */

import { RecordingState, STATE_ANNOUNCEMENTS } from './state-manager';

/**
 * Accessibility props interface for components
 */
export interface AccessibilityProps {
    ariaLabel: string;
    ariaDescribedBy?: string;
    ariaLive?: 'polite' | 'assertive' | 'off';
    role?: string;
}

/**
 * Recording state for accessibility context
 */
export interface RecordingAccessibilityState {
    state: RecordingState;
    recordingTime: number;
    audioLevel: number;
    hasError: boolean;
    errorMessage?: string;
}

/**
 * ARIA labels for recording control buttons
 */
export const BUTTON_ARIA_LABELS = {
    startRecording: 'Start voice recording',
    stopRecording: 'Stop voice recording',
    pauseRecording: 'Pause voice recording',
    resumeRecording: 'Resume voice recording',
    tryAgain: 'Try recording again',
    recordAgain: 'Record a new voice sample',
    continueAnalysis: 'Continue with speech analysis',
    skipStep: 'Skip voice evaluation step',
    goBack: 'Go back to previous step',
} as const;

/**
 * ARIA descriptions for recording states
 */
export const STATE_ARIA_DESCRIPTIONS: Record<RecordingState, string> = {
    idle: 'Voice recording is ready. Press the record button or Space key to begin.',
    initializing: 'Setting up microphone. Please wait.',
    recording: 'Recording in progress. Press the stop button or Space key to finish.',
    paused: 'Recording is paused. Press resume to continue or stop to finish.',
    completed: 'Recording complete. You can submit for analysis or record again.',
    error: 'An error occurred. Please read the error message and try again.',
};

/**
 * Get the announcement text for a state change
 * @param state - The new recording state
 * @param context - Additional context for the announcement
 * @returns The announcement text for screen readers
 */
export function getStateAnnouncement(
    state: RecordingState,
    context?: {
        recordingTime?: number;
        errorMessage?: string;
    }
): string {
    let announcement = STATE_ANNOUNCEMENTS[state];

    if (state === 'recording' && context?.recordingTime !== undefined) {
        const minutes = Math.floor(context.recordingTime / 60);
        const seconds = context.recordingTime % 60;
        const timeStr = minutes > 0
            ? `${minutes} minute${minutes !== 1 ? 's' : ''} and ${seconds} second${seconds !== 1 ? 's' : ''}`
            : `${seconds} second${seconds !== 1 ? 's' : ''}`;
        announcement = `Recording in progress. ${timeStr} recorded.`;
    }

    if (state === 'error' && context?.errorMessage) {
        announcement = `Error: ${context.errorMessage}`;
    }

    return announcement;
}

/**
 * Get ARIA label for the audio level indicator
 * @param level - Audio level as percentage (0-100)
 * @returns ARIA label describing the audio level
 */
export function getAudioLevelAriaLabel(level: number): string {
    const roundedLevel = Math.round(level);

    if (roundedLevel < 10) {
        return `Audio level very low at ${roundedLevel} percent. Please speak louder.`;
    } else if (roundedLevel < 30) {
        return `Audio level low at ${roundedLevel} percent.`;
    } else if (roundedLevel < 70) {
        return `Audio level good at ${roundedLevel} percent.`;
    } else {
        return `Audio level high at ${roundedLevel} percent.`;
    }
}

/**
 * Get ARIA label for the recording timer
 * @param seconds - Recording time in seconds
 * @param maxSeconds - Maximum recording time in seconds
 * @returns ARIA label describing the recording time
 */
export function getRecordingTimeAriaLabel(seconds: number, maxSeconds: number): string {
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    const remaining = maxSeconds - seconds;
    const remainingMinutes = Math.floor(remaining / 60);
    const remainingSecs = remaining % 60;

    const currentTime = minutes > 0
        ? `${minutes} minute${minutes !== 1 ? 's' : ''} ${secs} second${secs !== 1 ? 's' : ''}`
        : `${secs} second${secs !== 1 ? 's' : ''}`;

    const remainingTime = remainingMinutes > 0
        ? `${remainingMinutes} minute${remainingMinutes !== 1 ? 's' : ''} ${remainingSecs} second${remainingSecs !== 1 ? 's' : ''}`
        : `${remainingSecs} second${remainingSecs !== 1 ? 's' : ''}`;

    return `Recording time: ${currentTime}. ${remainingTime} remaining.`;
}

/**
 * Keyboard event handler for recording controls
 * @param event - Keyboard event
 * @param handlers - Object containing handler functions
 * @returns True if the event was handled
 */
export function handleRecordingKeyboard(
    event: React.KeyboardEvent,
    handlers: {
        onStartStop?: () => void;
        onPauseResume?: () => void;
        onCancel?: () => void;
    }
): boolean {
    switch (event.key) {
        case ' ':
        case 'Enter':
            // Only handle if not on a button (buttons handle their own events)
            if ((event.target as HTMLElement).tagName !== 'BUTTON') {
                event.preventDefault();
                handlers.onStartStop?.();
                return true;
            }
            return false;
        case 'p':
        case 'P':
            if (event.ctrlKey || event.metaKey) {
                event.preventDefault();
                handlers.onPauseResume?.();
                return true;
            }
            return false;
        case 'Escape':
            event.preventDefault();
            handlers.onCancel?.();
            return true;
        default:
            return false;
    }
}

/**
 * Hook-friendly keyboard handler creator
 * Creates a keyboard event handler with the specified callbacks
 */
export function createKeyboardHandler(handlers: {
    onStartStop?: () => void;
    onPauseResume?: () => void;
    onCancel?: () => void;
}): (event: React.KeyboardEvent) => void {
    return (event: React.KeyboardEvent) => {
        handleRecordingKeyboard(event, handlers);
    };
}

/**
 * Generate unique IDs for ARIA relationships
 * @param prefix - Prefix for the ID
 * @returns Unique ID string
 */
export function generateAriaId(prefix: string): string {
    return `${prefix}-${Math.random().toString(36).substring(2, 9)}`;
}

/**
 * Props for the live region component
 */
export interface LiveRegionProps {
    message: string;
    politeness: 'polite' | 'assertive';
    atomic?: boolean;
}

/**
 * Create props for an aria-live region
 * @param message - The message to announce
 * @param isUrgent - Whether the message is urgent (uses assertive)
 * @returns Props object for the live region
 */
export function createLiveRegionProps(
    message: string,
    isUrgent: boolean = false
): LiveRegionProps {
    return {
        message,
        politeness: isUrgent ? 'assertive' : 'polite',
        atomic: true,
    };
}

/**
 * Focus management utilities
 */
export const FocusManager = {
    /**
     * Focus an element by ID
     * @param elementId - The ID of the element to focus
     */
    focusById(elementId: string): void {
        const element = document.getElementById(elementId);
        if (element) {
            element.focus();
        }
    },

    /**
     * Trap focus within a container
     * @param container - The container element
     * @param event - The keyboard event
     */
    trapFocus(container: HTMLElement, event: KeyboardEvent): void {
        if (event.key !== 'Tab') return;

        const focusableElements = container.querySelectorAll<HTMLElement>(
            'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
        );

        const firstElement = focusableElements[0];
        const lastElement = focusableElements[focusableElements.length - 1];

        if (event.shiftKey && document.activeElement === firstElement) {
            event.preventDefault();
            lastElement?.focus();
        } else if (!event.shiftKey && document.activeElement === lastElement) {
            event.preventDefault();
            firstElement?.focus();
        }
    },
};

/**
 * Visible focus indicator styles (CSS class names)
 */
export const FOCUS_STYLES = {
    default: 'focus:outline-none focus:ring-2 focus:ring-medical-500 focus:ring-offset-2',
    button: 'focus:outline-none focus:ring-2 focus:ring-medical-500 focus:ring-offset-2 focus-visible:ring-2',
    input: 'focus:outline-none focus:ring-2 focus:ring-medical-500 focus:border-medical-500',
    card: 'focus-within:ring-2 focus-within:ring-medical-500 focus-within:ring-offset-2',
    /** High contrast focus style for better visibility */
    highContrast: 'focus:outline-none focus:ring-4 focus:ring-medical-600 focus:ring-offset-2 focus:ring-offset-white',
    /** Focus style for recording button (larger, more prominent) */
    recordButton: 'focus:outline-none focus:ring-4 focus:ring-medical-500 focus:ring-offset-4 focus-visible:ring-4 transition-shadow',
} as const;

/**
 * Keyboard shortcut descriptions for help text
 */
export const KEYBOARD_SHORTCUTS = {
    startStop: 'Space or Enter',
    pauseResume: 'Ctrl+P or Cmd+P',
    cancel: 'Escape',
} as const;

/**
 * Get keyboard shortcut help text
 * @returns Formatted help text for keyboard shortcuts
 */
export function getKeyboardShortcutsHelp(): string {
    return `Keyboard shortcuts: ${KEYBOARD_SHORTCUTS.startStop} to start/stop recording, ${KEYBOARD_SHORTCUTS.pauseResume} to pause/resume, ${KEYBOARD_SHORTCUTS.cancel} to cancel.`;
}

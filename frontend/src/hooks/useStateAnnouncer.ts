/**
 * State Announcer Hook
 * 
 * Provides a React hook for announcing state changes to screen readers
 * using aria-live regions.
 * 
 * @module hooks/useStateAnnouncer
 * @validates Requirements 8.2
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { RecordingState, STATE_ANNOUNCEMENTS } from '@/lib/recording/state-manager';

/**
 * Options for the state announcer hook
 */
export interface UseStateAnnouncerOptions {
    /** Politeness level for announcements */
    politeness?: 'polite' | 'assertive';
    /** Delay before announcing (ms) */
    delay?: number;
    /** Whether to clear announcement after a timeout */
    clearAfter?: number;
}

/**
 * Return type for the state announcer hook
 */
export interface UseStateAnnouncerReturn {
    /** Current announcement text */
    announcement: string;
    /** Function to manually announce a message */
    announce: (message: string, isUrgent?: boolean) => void;
    /** Function to clear the current announcement */
    clearAnnouncement: () => void;
    /** Props to spread on the live region element */
    liveRegionProps: {
        role: 'status' | 'alert';
        'aria-live': 'polite' | 'assertive';
        'aria-atomic': boolean;
        className: string;
    };
}

/**
 * Hook for managing screen reader announcements
 * 
 * @param currentState - Current recording state
 * @param options - Configuration options
 * @returns Announcement state and control functions
 * 
 * @example
 * ```tsx
 * const { announcement, announce, liveRegionProps } = useStateAnnouncer(recordingState);
 * 
 * return (
 *   <div {...liveRegionProps}>
 *     {announcement}
 *   </div>
 * );
 * ```
 */
export function useStateAnnouncer(
    currentState: RecordingState,
    options: UseStateAnnouncerOptions = {}
): UseStateAnnouncerReturn {
    const {
        politeness = 'polite',
        delay = 100,
        clearAfter = 5000,
    } = options;

    const [announcement, setAnnouncement] = useState<string>('');
    const [currentPoliteness, setCurrentPoliteness] = useState<'polite' | 'assertive'>(politeness);
    const previousStateRef = useRef<RecordingState | null>(null);
    const clearTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    const announceTimeoutRef = useRef<NodeJS.Timeout | null>(null);

    /**
     * Clear any pending timeouts
     */
    const clearTimeouts = useCallback(() => {
        if (clearTimeoutRef.current) {
            clearTimeout(clearTimeoutRef.current);
            clearTimeoutRef.current = null;
        }
        if (announceTimeoutRef.current) {
            clearTimeout(announceTimeoutRef.current);
            announceTimeoutRef.current = null;
        }
    }, []);

    /**
     * Clear the current announcement
     */
    const clearAnnouncement = useCallback(() => {
        setAnnouncement('');
    }, []);

    /**
     * Announce a message to screen readers
     */
    const announce = useCallback((message: string, isUrgent: boolean = false) => {
        clearTimeouts();

        // Set politeness based on urgency
        setCurrentPoliteness(isUrgent ? 'assertive' : politeness);

        // Delay announcement slightly to ensure screen readers pick it up
        announceTimeoutRef.current = setTimeout(() => {
            setAnnouncement(message);

            // Clear announcement after timeout if configured
            if (clearAfter > 0) {
                clearTimeoutRef.current = setTimeout(() => {
                    setAnnouncement('');
                }, clearAfter);
            }
        }, delay);
    }, [clearTimeouts, politeness, delay, clearAfter]);

    /**
     * Announce state changes automatically
     */
    useEffect(() => {
        if (previousStateRef.current !== currentState) {
            const stateAnnouncement = STATE_ANNOUNCEMENTS[currentState];
            const isUrgent = currentState === 'error';

            announce(stateAnnouncement, isUrgent);
            previousStateRef.current = currentState;
        }
    }, [currentState, announce]);

    /**
     * Cleanup on unmount
     */
    useEffect(() => {
        return () => {
            clearTimeouts();
        };
    }, [clearTimeouts]);

    /**
     * Props for the live region element
     */
    const liveRegionProps = {
        role: currentPoliteness === 'assertive' ? 'alert' as const : 'status' as const,
        'aria-live': currentPoliteness,
        'aria-atomic': true,
        className: 'sr-only',
    };

    return {
        announcement,
        announce,
        clearAnnouncement,
        liveRegionProps,
    };
}

/**
 * Component props for the LiveRegion component
 */
export interface LiveRegionProps {
    /** The message to announce */
    message: string;
    /** Politeness level */
    politeness?: 'polite' | 'assertive';
    /** Additional CSS classes */
    className?: string;
    /** ID for the element */
    id?: string;
}

/**
 * Standalone LiveRegion component for announcements
 * 
 * @example
 * ```tsx
 * <LiveRegion message={announcement} politeness="polite" />
 * ```
 */
export function LiveRegion({
    message,
    politeness = 'polite',
    className = 'sr-only',
    id,
}: LiveRegionProps): JSX.Element {
    return (
        <div
            id= { id }
    role = { politeness === 'assertive' ? 'alert' : 'status'
}
aria - live={ politeness }
aria - atomic={ true }
className = { className }
    >
    { message }
    </div>
    );
}

export default useStateAnnouncer;

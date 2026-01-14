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

    const clearAnnouncement = useCallback(() => {
        setAnnouncement('');
    }, []);

    const announce = useCallback((message: string, isUrgent: boolean = false) => {
        clearTimeouts();
        setCurrentPoliteness(isUrgent ? 'assertive' : politeness);

        announceTimeoutRef.current = setTimeout(() => {
            setAnnouncement(message);

            if (clearAfter > 0) {
                clearTimeoutRef.current = setTimeout(() => {
                    setAnnouncement('');
                }, clearAfter);
            }
        }, delay);
    }, [clearTimeouts, politeness, delay, clearAfter]);

    useEffect(() => {
        if (previousStateRef.current !== currentState) {
            const stateAnnouncement = STATE_ANNOUNCEMENTS[currentState];
            const isUrgent = currentState === 'error';

            announce(stateAnnouncement, isUrgent);
            previousStateRef.current = currentState;
        }
    }, [currentState, announce]);

    useEffect(() => {
        return () => {
            clearTimeouts();
        };
    }, [clearTimeouts]);

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
    message: string;
    politeness?: 'polite' | 'assertive';
    className?: string;
    id?: string;
}

/**
 * Standalone LiveRegion component for announcements
 */
export function LiveRegion({
    message,
    politeness = 'polite',
    className = 'sr-only',
    id,
}: LiveRegionProps): JSX.Element {
    return (
        <div
            id={id}
            role={politeness === 'assertive' ? 'alert' : 'status'}
            aria-live={politeness}
            aria-atomic={true}
            className={className}
        >
            {message}
        </div>
    );
}

export default useStateAnnouncer;

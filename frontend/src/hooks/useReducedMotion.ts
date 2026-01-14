/**
 * useReducedMotion Hook
 * 
 * React hook to detect and respond to user's reduced motion preference.
 * Implements prefers-reduced-motion media query support.
 * 
 * Requirements: 16.5
 * 
 * @example
 * ```tsx
 * const prefersReducedMotion = useReducedMotion();
 * 
 * return (
 *   <motion.div
 *     animate={prefersReducedMotion ? {} : { y: 0, opacity: 1 }}
 *     transition={prefersReducedMotion ? { duration: 0 } : { duration: 0.3 }}
 *   >
 *     Content
 *   </motion.div>
 * );
 * ```
 */

'use client';

import { useState, useEffect } from 'react';

/**
 * Media query string for reduced motion preference
 */
const REDUCED_MOTION_QUERY = '(prefers-reduced-motion: reduce)';

/**
 * Hook to detect if user prefers reduced motion
 * 
 * @returns boolean - true if user prefers reduced motion, false otherwise
 */
export function useReducedMotion(): boolean {
    // Default to false on server-side rendering
    const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);

    useEffect(() => {
        // Check if window is available (client-side)
        if (typeof window === 'undefined') return;

        // Create media query
        const mediaQuery = window.matchMedia(REDUCED_MOTION_QUERY);

        // Set initial value
        setPrefersReducedMotion(mediaQuery.matches);

        // Handler for media query changes
        const handleChange = (event: MediaQueryListEvent) => {
            setPrefersReducedMotion(event.matches);
        };

        // Add listener for changes
        // Use addEventListener for modern browsers, addListener for older ones
        if (mediaQuery.addEventListener) {
            mediaQuery.addEventListener('change', handleChange);
        } else {
            // Fallback for older browsers
            mediaQuery.addListener(handleChange);
        }

        // Cleanup
        return () => {
            if (mediaQuery.removeEventListener) {
                mediaQuery.removeEventListener('change', handleChange);
            } else {
                // Fallback for older browsers
                mediaQuery.removeListener(handleChange);
            }
        };
    }, []);

    return prefersReducedMotion;
}

/**
 * Hook that returns animation props based on reduced motion preference
 * 
 * @param enabledProps - Props to use when animations are enabled
 * @param disabledProps - Props to use when animations are disabled (optional)
 * @returns The appropriate props based on user preference
 */
export function useAnimationProps<T>(
    enabledProps: T,
    disabledProps?: Partial<T>
): T | Partial<T> {
    const prefersReducedMotion = useReducedMotion();

    if (prefersReducedMotion && disabledProps) {
        return disabledProps;
    }

    if (prefersReducedMotion) {
        // Return empty object to disable animations
        return {} as Partial<T>;
    }

    return enabledProps;
}

/**
 * Hook that returns transition duration based on reduced motion preference
 * 
 * @param normalDuration - Duration in seconds when animations are enabled
 * @returns 0 if reduced motion is preferred, otherwise the normal duration
 */
export function useAnimationDuration(normalDuration: number): number {
    const prefersReducedMotion = useReducedMotion();
    return prefersReducedMotion ? 0 : normalDuration;
}

/**
 * Hook that returns Framer Motion variants based on reduced motion preference
 * 
 * @param variants - The animation variants to use when enabled
 * @returns Static variants if reduced motion is preferred, otherwise the provided variants
 */
export function useMotionVariants<T extends Record<string, unknown>>(variants: T): T {
    const prefersReducedMotion = useReducedMotion();

    if (prefersReducedMotion) {
        // Return static variants that don't animate
        return {
            initial: { opacity: 1 },
            animate: { opacity: 1 },
            exit: { opacity: 1 },
        } as unknown as T;
    }

    return variants;
}

export default useReducedMotion;

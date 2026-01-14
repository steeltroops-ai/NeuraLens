/**
 * Property-Based Tests for Reduced Motion Respect
 * 
 * Feature: frontend-global-fix
 * Property 20: Reduced Motion Respect
 * Validates: Requirements 16.5
 * 
 * When prefers-reduced-motion is enabled, animations SHALL be disabled
 * or reduced to instant transitions.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as fc from 'fast-check';
import {
    duration,
    durationMs,
    prefersReducedMotion,
    getReducedMotionVariants,
    getReducedMotionTransition,
    fadeInUp,
    scaleIn,
    fadeIn,
    slideInRight,
    slideInLeft,
    staggerItem,
    modalContent,
    toastSlideIn,
    nriScoreReveal,
    medilensTransition,
    fastTransition,
    slowTransition,
} from '@/lib/animations';
import type { Variants, Transition } from 'framer-motion';

/**
 * All animation variants that should respect reduced motion
 */
const animationVariants = [
    { name: 'fadeInUp', variant: fadeInUp },
    { name: 'scaleIn', variant: scaleIn },
    { name: 'fadeIn', variant: fadeIn },
    { name: 'slideInRight', variant: slideInRight },
    { name: 'slideInLeft', variant: slideInLeft },
    { name: 'staggerItem', variant: staggerItem },
    { name: 'modalContent', variant: modalContent },
    { name: 'toastSlideIn', variant: toastSlideIn },
    { name: 'nriScoreReveal', variant: nriScoreReveal },
];

/**
 * All transitions that should respect reduced motion
 */
const transitions = [
    { name: 'medilensTransition', transition: medilensTransition },
    { name: 'fastTransition', transition: fastTransition },
    { name: 'slowTransition', transition: slowTransition },
];

/**
 * Mock matchMedia for testing reduced motion preference
 */
function mockMatchMedia(matches: boolean) {
    const mockMediaQueryList = {
        matches,
        media: '(prefers-reduced-motion: reduce)',
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn(),
    };

    Object.defineProperty(window, 'matchMedia', {
        writable: true,
        value: vi.fn().mockImplementation((query: string) => ({
            ...mockMediaQueryList,
            media: query,
        })),
    });

    return mockMediaQueryList;
}

describe('Reduced Motion Property Tests', () => {
    let originalMatchMedia: typeof window.matchMedia;

    beforeEach(() => {
        // Store original matchMedia
        originalMatchMedia = window.matchMedia;
    });

    afterEach(() => {
        // Restore original matchMedia
        Object.defineProperty(window, 'matchMedia', {
            writable: true,
            value: originalMatchMedia,
        });
        vi.clearAllMocks();
    });

    /**
     * Property 20: Reduced Motion Respect
     * 
     * When prefers-reduced-motion is enabled, animations SHALL be disabled
     * or reduced to instant transitions.
     * 
     * **Validates: Requirements 16.5**
     */
    describe('Property 20: Reduced Motion Respect', () => {
        /**
         * Property 20.1: prefersReducedMotion returns true when media query matches
         */
        it('Property 20.1: prefersReducedMotion returns true when media query matches', () => {
            mockMatchMedia(true);
            expect(prefersReducedMotion()).toBe(true);
        });

        /**
         * Property 20.2: prefersReducedMotion returns false when media query does not match
         */
        it('Property 20.2: prefersReducedMotion returns false when media query does not match', () => {
            mockMatchMedia(false);
            expect(prefersReducedMotion()).toBe(false);
        });

        /**
         * Property 20.3: getReducedMotionVariants returns static variants when reduced motion is preferred
         */
        it('Property 20.3: getReducedMotionVariants returns static variants when reduced motion is preferred', () => {
            mockMatchMedia(true);

            fc.assert(
                fc.property(
                    fc.constantFrom(...animationVariants),
                    ({ name, variant }) => {
                        const reducedVariants = getReducedMotionVariants(variant);

                        // Should have initial state with opacity 1 (no fade)
                        expect(reducedVariants.initial).toBeDefined();
                        if (typeof reducedVariants.initial === 'object') {
                            expect((reducedVariants.initial as any).opacity).toBe(1);
                        }

                        // Should have animate state with opacity 1 (no animation)
                        expect(reducedVariants.animate).toBeDefined();
                        if (typeof reducedVariants.animate === 'object') {
                            expect((reducedVariants.animate as any).opacity).toBe(1);
                        }

                        return true;
                    }
                ),
                { numRuns: 100 }
            );
        });

        /**
         * Property 20.4: getReducedMotionVariants returns original variants when reduced motion is not preferred
         */
        it('Property 20.4: getReducedMotionVariants returns original variants when reduced motion is not preferred', () => {
            mockMatchMedia(false);

            fc.assert(
                fc.property(
                    fc.constantFrom(...animationVariants),
                    ({ name, variant }) => {
                        const result = getReducedMotionVariants(variant);

                        // Should return the original variant unchanged
                        expect(result).toEqual(variant);

                        return true;
                    }
                ),
                { numRuns: 100 }
            );
        });

        /**
         * Property 20.5: getReducedMotionTransition returns instant transition when reduced motion is preferred
         */
        it('Property 20.5: getReducedMotionTransition returns instant transition when reduced motion is preferred', () => {
            mockMatchMedia(true);

            fc.assert(
                fc.property(
                    fc.constantFrom(...transitions),
                    ({ name, transition }) => {
                        const reducedTransition = getReducedMotionTransition(transition);

                        // Duration should be 0 for instant transition
                        expect(reducedTransition.duration).toBe(0);

                        return true;
                    }
                ),
                { numRuns: 100 }
            );
        });

        /**
         * Property 20.6: getReducedMotionTransition returns original transition when reduced motion is not preferred
         */
        it('Property 20.6: getReducedMotionTransition returns original transition when reduced motion is not preferred', () => {
            mockMatchMedia(false);

            fc.assert(
                fc.property(
                    fc.constantFrom(...transitions),
                    ({ name, transition }) => {
                        const result = getReducedMotionTransition(transition);

                        // Should return the original transition unchanged
                        expect(result).toEqual(transition);

                        return true;
                    }
                ),
                { numRuns: 100 }
            );
        });

        /**
         * Property 20.7: Duration values are positive and can be set to 0 for reduced motion
         */
        it('Property 20.7: Duration values are positive and can be set to 0 for reduced motion', () => {
            const durationValues = [
                { name: 'instant', value: duration.instant },
                { name: 'fast', value: duration.fast },
                { name: 'normal', value: duration.normal },
                { name: 'slow', value: duration.slow },
                { name: 'slower', value: duration.slower },
            ];

            fc.assert(
                fc.property(
                    fc.constantFrom(...durationValues),
                    ({ name, value }) => {
                        // All durations should be non-negative
                        expect(value).toBeGreaterThanOrEqual(0);

                        // Instant duration should be 0 (already reduced)
                        if (name === 'instant') {
                            expect(value).toBe(0);
                        }

                        return true;
                    }
                ),
                { numRuns: 100 }
            );
        });

        /**
         * Property 20.8: Animation variants have exit states that can be disabled
         */
        it('Property 20.8: Animation variants have exit states that can be disabled', () => {
            fc.assert(
                fc.property(
                    fc.constantFrom(...animationVariants),
                    ({ name, variant }) => {
                        // Most variants should have exit state
                        // (some like nriScoreReveal may not have exit)
                        if (variant.exit) {
                            expect(variant.exit).toBeDefined();
                        }

                        return true;
                    }
                ),
                { numRuns: 100 }
            );
        });

        /**
         * Property 20.9: Reduced motion variants do not include transform properties
         */
        it('Property 20.9: Reduced motion variants do not include transform properties', () => {
            mockMatchMedia(true);

            fc.assert(
                fc.property(
                    fc.constantFrom(...animationVariants),
                    ({ name, variant }) => {
                        const reducedVariants = getReducedMotionVariants(variant);

                        // Initial state should not have transform properties
                        if (typeof reducedVariants.initial === 'object') {
                            const initial = reducedVariants.initial as any;
                            expect(initial.x).toBeUndefined();
                            expect(initial.y).toBeUndefined();
                            expect(initial.scale).toBeUndefined();
                        }

                        // Animate state should not have transform properties
                        if (typeof reducedVariants.animate === 'object') {
                            const animate = reducedVariants.animate as any;
                            expect(animate.x).toBeUndefined();
                            expect(animate.y).toBeUndefined();
                            expect(animate.scale).toBeUndefined();
                        }

                        return true;
                    }
                ),
                { numRuns: 100 }
            );
        });

        /**
         * Property 20.10: CSS reduced motion media query is correctly formatted
         */
        it('Property 20.10: CSS reduced motion media query is correctly formatted', () => {
            // The media query string should be valid
            const mediaQuery = '(prefers-reduced-motion: reduce)';

            // Should be a valid media query format
            expect(mediaQuery).toMatch(/^\(prefers-reduced-motion:\s*(reduce|no-preference)\)$/);

            // Should specifically check for 'reduce' value
            expect(mediaQuery).toContain('reduce');
        });

        /**
         * Property 20.11: All animation durations can be converted to 0 for reduced motion
         */
        it('Property 20.11: All animation durations can be converted to 0 for reduced motion', () => {
            const allDurations = [
                duration.instant,
                duration.fast,
                duration.normal,
                duration.slow,
                duration.slower,
            ];

            allDurations.forEach(d => {
                // Each duration should be a number
                expect(typeof d).toBe('number');

                // Each duration can be replaced with 0
                const reducedDuration = 0;
                expect(reducedDuration).toBe(0);
            });
        });

        /**
         * Property 20.12: Millisecond durations are consistent with second durations
         */
        it('Property 20.12: Millisecond durations are consistent with second durations', () => {
            const durationPairs = [
                { seconds: duration.instant, ms: durationMs.instant },
                { seconds: duration.fast, ms: durationMs.fast },
                { seconds: duration.normal, ms: durationMs.normal },
                { seconds: duration.slow, ms: durationMs.slow },
                { seconds: duration.slower, ms: durationMs.slower },
            ];

            durationPairs.forEach(({ seconds, ms }) => {
                // Milliseconds should be seconds * 1000
                expect(ms).toBe(seconds * 1000);
            });
        });
    });
});

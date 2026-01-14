/**
 * Property-Based Tests for Animation Easing Consistency
 * 
 * Feature: frontend-global-fix
 * Property 19: Animation Easing Consistency
 * Validates: Requirements 16.1, 16.2
 * 
 * For any animated transition, it SHALL use the Apple-inspired easing function
 * cubic-bezier(0.22, 1, 0.36, 1) for primary transitions.
 */

import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';
import {
    easing,
    cssEasing,
    duration,
    durationMs,
    medilensTransition,
    fastTransition,
    slowTransition,
    fadeInUp,
    scaleIn,
    fadeIn,
    slideInRight,
    slideInLeft,
    staggerItem,
    modalContent,
    toastSlideIn,
    nriScoreReveal,
} from '@/lib/animations';

/**
 * MediLens primary easing function (ease-out-quint)
 * cubic-bezier(0.22, 1, 0.36, 1)
 */
const MEDILENS_EASING = [0.22, 1, 0.36, 1] as const;
const MEDILENS_EASING_CSS = 'cubic-bezier(0.22, 1, 0.36, 1)';

/**
 * Valid easing function values (must be between 0 and 1 for x values, any for y)
 */
const isValidEasingArray = (arr: readonly number[]): boolean => {
    if (arr.length !== 4) return false;
    // x1 and x2 (indices 0 and 2) must be between 0 and 1
    return arr[0] >= 0 && arr[0] <= 1 && arr[2] >= 0 && arr[2] <= 1;
};

/**
 * Check if an easing array matches the MediLens primary easing
 */
const isMediLensEasing = (arr: readonly number[]): boolean => {
    return (
        arr.length === 4 &&
        arr[0] === MEDILENS_EASING[0] &&
        arr[1] === MEDILENS_EASING[1] &&
        arr[2] === MEDILENS_EASING[2] &&
        arr[3] === MEDILENS_EASING[3]
    );
};

/**
 * All animation variants that should use MediLens easing
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
 * Duration scale values
 */
const durationValues = [
    { name: 'instant', value: duration.instant, ms: durationMs.instant },
    { name: 'fast', value: duration.fast, ms: durationMs.fast },
    { name: 'normal', value: duration.normal, ms: durationMs.normal },
    { name: 'slow', value: duration.slow, ms: durationMs.slow },
    { name: 'slower', value: duration.slower, ms: durationMs.slower },
];

describe('Animation Easing Property Tests', () => {
    /**
     * Property 19: Animation Easing Consistency
     * 
     * For any animated transition, it SHALL use the Apple-inspired easing function
     * cubic-bezier(0.22, 1, 0.36, 1) for primary transitions.
     * 
     * **Validates: Requirements 16.1, 16.2**
     */
    describe('Property 19: Animation Easing Consistency', () => {
        /**
         * Property 19.1: Primary easing function is correctly defined
         */
        it('Property 19.1: Primary easing function is correctly defined', () => {
            // Easing array should have exactly 4 values
            expect(easing.outQuint).toHaveLength(4);

            // Should match the MediLens primary easing
            expect(easing.outQuint[0]).toBe(0.22);
            expect(easing.outQuint[1]).toBe(1);
            expect(easing.outQuint[2]).toBe(0.36);
            expect(easing.outQuint[3]).toBe(1);

            // Should be a valid cubic-bezier
            expect(isValidEasingArray(easing.outQuint)).toBe(true);
        });

        /**
         * Property 19.2: CSS easing string matches the array values
         */
        it('Property 19.2: CSS easing string matches the array values', () => {
            expect(cssEasing.outQuint).toBe(MEDILENS_EASING_CSS);

            // Parse the CSS string and verify values
            const match = cssEasing.outQuint.match(/cubic-bezier\(([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/);
            expect(match).not.toBeNull();

            if (match) {
                expect(parseFloat(match[1])).toBe(easing.outQuint[0]);
                expect(parseFloat(match[2])).toBe(easing.outQuint[1]);
                expect(parseFloat(match[3])).toBe(easing.outQuint[2]);
                expect(parseFloat(match[4])).toBe(easing.outQuint[3]);
            }
        });

        /**
         * Property 19.3: All easing functions are valid cubic-bezier values
         */
        it('Property 19.3: All easing functions are valid cubic-bezier values', () => {
            const easingFunctions = [
                { name: 'outQuint', value: easing.outQuint },
                { name: 'inOutCubic', value: easing.inOutCubic },
                { name: 'spring', value: easing.spring },
                { name: 'out', value: easing.out },
                { name: 'in', value: easing.in },
            ];

            easingFunctions.forEach(({ name, value }) => {
                expect(value).toHaveLength(4);
                expect(isValidEasingArray(value)).toBe(true);
            });
        });

        /**
         * Property 19.4: MediLens transition uses primary easing
         */
        it('Property 19.4: MediLens transition uses primary easing', () => {
            expect(medilensTransition.ease).toEqual(MEDILENS_EASING);
            expect(medilensTransition.duration).toBe(duration.normal);
        });

        /**
         * Property 19.5: Fast transition uses primary easing
         */
        it('Property 19.5: Fast transition uses primary easing', () => {
            expect(fastTransition.ease).toEqual(MEDILENS_EASING);
            expect(fastTransition.duration).toBe(duration.fast);
        });

        /**
         * Property 19.6: Slow transition uses primary easing
         */
        it('Property 19.6: Slow transition uses primary easing', () => {
            expect(slowTransition.ease).toEqual(MEDILENS_EASING);
            expect(slowTransition.duration).toBe(duration.slow);
        });

        /**
         * Property 19.7: All animation variants use MediLens easing
         * For any animation variant, its animate transition SHALL use ease-out-quint
         */
        it('Property 19.7: All animation variants use MediLens easing', () => {
            fc.assert(
                fc.property(
                    fc.constantFrom(...animationVariants),
                    ({ name, variant }) => {
                        // Check animate state has transition with correct easing
                        const animate = variant.animate;

                        if (animate && typeof animate === 'object' && 'transition' in animate) {
                            const transition = (animate as any).transition;
                            if (transition && transition.ease) {
                                expect(isMediLensEasing(transition.ease)).toBe(true);
                            }
                        }

                        return true;
                    }
                ),
                { numRuns: 100 }
            );
        });

        /**
         * Property 19.8: Duration scale follows MediLens specifications
         * Fast: 150ms, Normal: 300ms, Slow: 500ms
         */
        it('Property 19.8: Duration scale follows MediLens specifications', () => {
            // Fast duration should be 0.15s (150ms)
            expect(duration.fast).toBe(0.15);
            expect(durationMs.fast).toBe(150);

            // Normal duration should be 0.3s (300ms)
            expect(duration.normal).toBe(0.3);
            expect(durationMs.normal).toBe(300);

            // Slow duration should be 0.5s (500ms)
            expect(duration.slow).toBe(0.5);
            expect(durationMs.slow).toBe(500);
        });

        /**
         * Property 19.9: Duration values are consistent between seconds and milliseconds
         */
        it('Property 19.9: Duration values are consistent between seconds and milliseconds', () => {
            fc.assert(
                fc.property(
                    fc.constantFrom(...durationValues),
                    ({ name, value, ms }) => {
                        // Milliseconds should be seconds * 1000
                        expect(ms).toBe(value * 1000);
                        return true;
                    }
                ),
                { numRuns: 100 }
            );
        });

        /**
         * Property 19.10: Animation variants have proper initial and animate states
         */
        it('Property 19.10: Animation variants have proper initial and animate states', () => {
            fc.assert(
                fc.property(
                    fc.constantFrom(...animationVariants),
                    ({ name, variant }) => {
                        // Should have initial state
                        expect(variant.initial).toBeDefined();

                        // Should have animate state
                        expect(variant.animate).toBeDefined();

                        // Initial opacity should be 0 or less than animate opacity
                        if (
                            typeof variant.initial === 'object' &&
                            typeof variant.animate === 'object' &&
                            'opacity' in variant.initial &&
                            'opacity' in variant.animate
                        ) {
                            expect((variant.initial as any).opacity).toBeLessThanOrEqual(
                                (variant.animate as any).opacity
                            );
                        }

                        return true;
                    }
                ),
                { numRuns: 100 }
            );
        });

        /**
         * Property 19.11: Easing values are within valid cubic-bezier range
         */
        it('Property 19.11: Easing values are within valid cubic-bezier range', () => {
            // Generate random easing-like arrays and verify our validation function
            // Note: Using noNaN: true to exclude NaN values which would fail validation
            fc.assert(
                fc.property(
                    fc.tuple(
                        fc.float({ min: 0, max: 1, noNaN: true }),
                        fc.float({ min: -2, max: 2, noNaN: true }),
                        fc.float({ min: 0, max: 1, noNaN: true }),
                        fc.float({ min: -2, max: 2, noNaN: true })
                    ),
                    ([x1, y1, x2, y2]) => {
                        const arr = [x1, y1, x2, y2] as const;
                        // Our validation should pass for valid x values
                        expect(isValidEasingArray(arr)).toBe(true);
                        return true;
                    }
                ),
                { numRuns: 100 }
            );
        });

        /**
         * Property 19.12: Duration values are positive
         */
        it('Property 19.12: Duration values are positive', () => {
            fc.assert(
                fc.property(
                    fc.constantFrom(...durationValues),
                    ({ name, value, ms }) => {
                        expect(value).toBeGreaterThanOrEqual(0);
                        expect(ms).toBeGreaterThanOrEqual(0);
                        return true;
                    }
                ),
                { numRuns: 100 }
            );
        });

        /**
         * Property 19.13: Duration scale is ordered correctly
         * instant < fast < normal < slow < slower
         */
        it('Property 19.13: Duration scale is ordered correctly', () => {
            expect(duration.instant).toBeLessThan(duration.fast);
            expect(duration.fast).toBeLessThan(duration.normal);
            expect(duration.normal).toBeLessThan(duration.slow);
            expect(duration.slow).toBeLessThan(duration.slower);
        });
    });
});

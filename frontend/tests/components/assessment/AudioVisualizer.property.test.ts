/**
 * Property-Based Tests for AudioVisualizer Component
 * 
 * **Property 8: Volume Level Normalization**
 * **Validates: Requirements 6.2**
 * 
 * For any audio level reading from Audio_Visualizer, the displayed value
 * SHALL be in the range [0, 100] percent.
 */
import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';
import { normalizeAudioLevel } from '@/components/assessment/AudioVisualizer';

describe('AudioVisualizer - Property Tests', () => {
    /**
     * Feature: speech-pipeline-fix, Property 8: Volume Level Normalization
     * 
     * For any audio level reading from Audio_Visualizer, the displayed value
     * SHALL be in the range [0, 100] percent.
     */
    describe('Property 8: Volume Level Normalization', () => {
        it('should always return a value between 0 and 100 for any input', () => {
            fc.assert(
                fc.property(
                    fc.float({ min: -1000, max: 1000, noNaN: true }),
                    (rawLevel) => {
                        const normalizedLevel = normalizeAudioLevel(rawLevel);

                        // The normalized level must always be >= 0
                        expect(normalizedLevel).toBeGreaterThanOrEqual(0);
                        // The normalized level must always be <= 100
                        expect(normalizedLevel).toBeLessThanOrEqual(100);
                    }
                ),
                { numRuns: 100 }
            );
        });

        it('should return 0 for zero or negative input values', () => {
            fc.assert(
                fc.property(
                    fc.float({ min: -1000, max: 0, noNaN: true }),
                    (rawLevel) => {
                        const normalizedLevel = normalizeAudioLevel(rawLevel);

                        // Zero or negative values should normalize to 0
                        expect(normalizedLevel).toBe(0);
                    }
                ),
                { numRuns: 100 }
            );
        });

        it('should return 100 for values at or above the max value', () => {
            fc.assert(
                fc.property(
                    fc.float({ min: 255, max: 1000, noNaN: true }),
                    (rawLevel) => {
                        const normalizedLevel = normalizeAudioLevel(rawLevel);

                        // Values at or above max should normalize to 100
                        expect(normalizedLevel).toBe(100);
                    }
                ),
                { numRuns: 100 }
            );
        });

        it('should scale linearly within the valid range', () => {
            fc.assert(
                fc.property(
                    fc.float({ min: 0, max: 255, noNaN: true }),
                    (rawLevel) => {
                        const normalizedLevel = normalizeAudioLevel(rawLevel);
                        const expectedLevel = (rawLevel / 255) * 100;

                        // Should be approximately equal (accounting for floating point)
                        expect(normalizedLevel).toBeCloseTo(expectedLevel, 5);
                    }
                ),
                { numRuns: 100 }
            );
        });

        it('should handle custom max values correctly', () => {
            fc.assert(
                fc.property(
                    fc.float({ min: 0, max: 1000, noNaN: true }),
                    fc.float({ min: 1, max: 1000, noNaN: true }),
                    (rawLevel, maxValue) => {
                        const normalizedLevel = normalizeAudioLevel(rawLevel, maxValue);

                        // Result should always be in [0, 100] range
                        expect(normalizedLevel).toBeGreaterThanOrEqual(0);
                        expect(normalizedLevel).toBeLessThanOrEqual(100);
                    }
                ),
                { numRuns: 100 }
            );
        });

        it('should be monotonically increasing within valid range', () => {
            fc.assert(
                fc.property(
                    fc.integer({ min: 0, max: 254 }),
                    fc.integer({ min: 1, max: 10 }),
                    (rawLevel, delta) => {
                        const level1 = normalizeAudioLevel(rawLevel);
                        const level2 = normalizeAudioLevel(rawLevel + delta);

                        // Higher input should produce higher or equal output
                        expect(level2).toBeGreaterThanOrEqual(level1);
                    }
                ),
                { numRuns: 100 }
            );
        });

        it('should produce consistent results for the same input', () => {
            fc.assert(
                fc.property(
                    fc.float({ min: 0, max: 255, noNaN: true }),
                    (rawLevel) => {
                        const result1 = normalizeAudioLevel(rawLevel);
                        const result2 = normalizeAudioLevel(rawLevel);

                        // Same input should always produce same output
                        expect(result1).toBe(result2);
                    }
                ),
                { numRuns: 100 }
            );
        });

        it('should handle typical Web Audio API frequency data values (0-255)', () => {
            fc.assert(
                fc.property(
                    fc.integer({ min: 0, max: 255 }),
                    (rawLevel) => {
                        const normalizedLevel = normalizeAudioLevel(rawLevel);

                        // Result should be in valid percentage range
                        expect(normalizedLevel).toBeGreaterThanOrEqual(0);
                        expect(normalizedLevel).toBeLessThanOrEqual(100);

                        // Should be proportional to input
                        const expectedPercentage = (rawLevel / 255) * 100;
                        expect(normalizedLevel).toBeCloseTo(expectedPercentage, 5);
                    }
                ),
                { numRuns: 100 }
            );
        });
    });

    describe('Edge Cases', () => {
        it('should handle boundary values correctly', () => {
            // Test exact boundary values
            expect(normalizeAudioLevel(0)).toBe(0);
            expect(normalizeAudioLevel(255)).toBe(100);
            expect(normalizeAudioLevel(127.5)).toBeCloseTo(50, 1);
        });

        it('should handle very small positive values', () => {
            fc.assert(
                fc.property(
                    fc.integer({ min: 1, max: 10 }),
                    (rawLevel) => {
                        const normalizedLevel = normalizeAudioLevel(rawLevel);

                        // Small positive values should produce small positive results
                        expect(normalizedLevel).toBeGreaterThan(0);
                        expect(normalizedLevel).toBeLessThan(5);
                    }
                ),
                { numRuns: 100 }
            );
        });
    });
});

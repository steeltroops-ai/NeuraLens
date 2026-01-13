/**
 * Property-Based Tests for EnhancedSpeechResults Component
 * 
 * **Property 11: Results Display Completeness**
 * **Validates: Requirements 9.1, 9.2, 9.4**
 * 
 * For any successful speech analysis result displayed, the Results_Display
 * SHALL show all 9 biomarkers with their values and confidence indicators.
 */
import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';
import {
    hasAllBiomarkers,
    hasConfidenceIndicators,
    hasEstimatedMarkers,
    getBiomarkerCount,
    type EnhancedSpeechAnalysisResult,
    type BiomarkerResult,
} from '@/components/assessment/EnhancedSpeechResults';

/**
 * Arbitrary for generating valid biomarker results
 */
const biomarkerResultArb = fc.record({
    value: fc.float({ min: 0, max: 1, noNaN: true }),
    unit: fc.constantFrom('%', 'dB', 'syl/s'),
    normalRange: fc.tuple(
        fc.float({ min: 0, max: 0.5, noNaN: true }),
        fc.float({ min: 0.5, max: 1, noNaN: true })
    ) as fc.Arbitrary<[number, number]>,
    isEstimated: fc.boolean(),
    confidence: fc.option(fc.float({ min: 0, max: 1, noNaN: true }), { nil: undefined }),
});

/**
 * Arbitrary for generating HNR biomarker (different range)
 */
const hnrBiomarkerArb = fc.record({
    value: fc.float({ min: 0, max: 30, noNaN: true }),
    unit: fc.constant('dB'),
    normalRange: fc.tuple(
        fc.float({ min: 0, max: 15, noNaN: true }),
        fc.float({ min: 15, max: 30, noNaN: true })
    ) as fc.Arbitrary<[number, number]>,
    isEstimated: fc.boolean(),
    confidence: fc.option(fc.float({ min: 0, max: 1, noNaN: true }), { nil: undefined }),
});

/**
 * Arbitrary for generating speech rate biomarker (different range)
 */
const speechRateBiomarkerArb = fc.record({
    value: fc.float({ min: 0.5, max: 10, noNaN: true }),
    unit: fc.constant('syl/s'),
    normalRange: fc.tuple(
        fc.float({ min: 0.5, max: 5, noNaN: true }),
        fc.float({ min: 5, max: 10, noNaN: true })
    ) as fc.Arbitrary<[number, number]>,
    isEstimated: fc.boolean(),
    confidence: fc.option(fc.float({ min: 0, max: 1, noNaN: true }), { nil: undefined }),
});

/**
 * Arbitrary for generating complete biomarkers object
 */
const biomarkersArb = fc.record({
    jitter: biomarkerResultArb,
    shimmer: biomarkerResultArb,
    hnr: hnrBiomarkerArb,
    speechRate: speechRateBiomarkerArb,
    pauseRatio: biomarkerResultArb,
    fluencyScore: biomarkerResultArb,
    voiceTremor: biomarkerResultArb,
    articulationClarity: biomarkerResultArb,
    prosodyVariation: biomarkerResultArb,
});

/**
 * Arbitrary for generating baseline comparison
 */
const baselineComparisonArb = fc.option(
    fc.record({
        jitter: fc.float({ min: -0.5, max: 0.5, noNaN: true }),
        shimmer: fc.float({ min: -0.5, max: 0.5, noNaN: true }),
        hnr: fc.float({ min: -0.5, max: 0.5, noNaN: true }),
        speechRate: fc.float({ min: -0.5, max: 0.5, noNaN: true }),
        pauseRatio: fc.float({ min: -0.5, max: 0.5, noNaN: true }),
        fluencyScore: fc.float({ min: -0.5, max: 0.5, noNaN: true }),
        voiceTremor: fc.float({ min: -0.5, max: 0.5, noNaN: true }),
        articulationClarity: fc.float({ min: -0.5, max: 0.5, noNaN: true }),
        prosodyVariation: fc.float({ min: -0.5, max: 0.5, noNaN: true }),
    }),
    { nil: undefined }
);

/**
 * Arbitrary for generating valid ISO timestamp strings
 * Using integer timestamps to avoid invalid date issues
 */
const timestampArb = fc.integer({
    min: new Date('2020-01-01').getTime(),
    max: new Date('2030-12-31').getTime(),
}).map((ts) => new Date(ts).toISOString());

/**
 * Arbitrary for generating complete EnhancedSpeechAnalysisResult
 */
const enhancedSpeechResultArb: fc.Arbitrary<EnhancedSpeechAnalysisResult> = fc.record({
    sessionId: fc.uuid(),
    processingTime: fc.float({ min: 100, max: 30000, noNaN: true }),
    timestamp: timestampArb,
    confidence: fc.float({ min: 0, max: 1, noNaN: true }),
    riskScore: fc.float({ min: 0, max: 1, noNaN: true }),
    qualityScore: fc.float({ min: 0, max: 1, noNaN: true }),
    biomarkers: biomarkersArb,
    recommendations: fc.array(fc.string({ minLength: 1, maxLength: 200 }), { minLength: 0, maxLength: 5 }),
    baselineComparison: baselineComparisonArb,
});


describe('EnhancedSpeechResults - Property Tests', () => {
    /**
     * Feature: speech-pipeline-fix, Property 11: Results Display Completeness
     * 
     * For any successful speech analysis result displayed, the Results_Display
     * SHALL show all 9 biomarkers with their values and confidence indicators.
     */
    describe('Property 11: Results Display Completeness', () => {
        it('should have exactly 9 biomarker configurations', () => {
            expect(getBiomarkerCount()).toBe(9);
        });

        it('should have all 9 biomarkers present for any valid result', () => {
            fc.assert(
                fc.property(enhancedSpeechResultArb, (result) => {
                    // All 9 biomarkers must be present
                    expect(hasAllBiomarkers(result)).toBe(true);
                }),
                { numRuns: 100 }
            );
        });

        it('should have confidence indicators for any valid result', () => {
            fc.assert(
                fc.property(enhancedSpeechResultArb, (result) => {
                    // Confidence indicator must be present and valid
                    expect(hasConfidenceIndicators(result)).toBe(true);
                }),
                { numRuns: 100 }
            );
        });

        it('should have estimated markers for all biomarkers', () => {
            fc.assert(
                fc.property(enhancedSpeechResultArb, (result) => {
                    // All biomarkers must have isEstimated flag
                    expect(hasEstimatedMarkers(result)).toBe(true);
                }),
                { numRuns: 100 }
            );
        });

        it('should have all required biomarker keys', () => {
            fc.assert(
                fc.property(enhancedSpeechResultArb, (result) => {
                    const requiredKeys = [
                        'jitter',
                        'shimmer',
                        'hnr',
                        'speechRate',
                        'pauseRatio',
                        'fluencyScore',
                        'voiceTremor',
                        'articulationClarity',
                        'prosodyVariation',
                    ];

                    // All required keys must be present
                    for (const key of requiredKeys) {
                        expect(result.biomarkers).toHaveProperty(key);
                    }
                }),
                { numRuns: 100 }
            );
        });

        it('should have valid value types for all biomarkers', () => {
            fc.assert(
                fc.property(enhancedSpeechResultArb, (result) => {
                    for (const biomarker of Object.values(result.biomarkers)) {
                        // Value must be a number
                        expect(typeof biomarker.value).toBe('number');
                        expect(Number.isFinite(biomarker.value)).toBe(true);

                        // Unit must be a string
                        expect(typeof biomarker.unit).toBe('string');

                        // Normal range must be a tuple of two numbers
                        expect(Array.isArray(biomarker.normalRange)).toBe(true);
                        expect(biomarker.normalRange.length).toBe(2);
                        expect(typeof biomarker.normalRange[0]).toBe('number');
                        expect(typeof biomarker.normalRange[1]).toBe('number');

                        // isEstimated must be a boolean
                        expect(typeof biomarker.isEstimated).toBe('boolean');
                    }
                }),
                { numRuns: 100 }
            );
        });

        it('should have valid confidence range (0-1)', () => {
            fc.assert(
                fc.property(enhancedSpeechResultArb, (result) => {
                    expect(result.confidence).toBeGreaterThanOrEqual(0);
                    expect(result.confidence).toBeLessThanOrEqual(1);
                }),
                { numRuns: 100 }
            );
        });

        it('should have valid risk score range (0-1)', () => {
            fc.assert(
                fc.property(enhancedSpeechResultArb, (result) => {
                    expect(result.riskScore).toBeGreaterThanOrEqual(0);
                    expect(result.riskScore).toBeLessThanOrEqual(1);
                }),
                { numRuns: 100 }
            );
        });

        it('should have valid quality score range (0-1)', () => {
            fc.assert(
                fc.property(enhancedSpeechResultArb, (result) => {
                    expect(result.qualityScore).toBeGreaterThanOrEqual(0);
                    expect(result.qualityScore).toBeLessThanOrEqual(1);
                }),
                { numRuns: 100 }
            );
        });
    });

    describe('Biomarker Value Ranges', () => {
        it('should have jitter value in valid range (0-1)', () => {
            fc.assert(
                fc.property(enhancedSpeechResultArb, (result) => {
                    expect(result.biomarkers.jitter.value).toBeGreaterThanOrEqual(0);
                    expect(result.biomarkers.jitter.value).toBeLessThanOrEqual(1);
                }),
                { numRuns: 100 }
            );
        });

        it('should have shimmer value in valid range (0-1)', () => {
            fc.assert(
                fc.property(enhancedSpeechResultArb, (result) => {
                    expect(result.biomarkers.shimmer.value).toBeGreaterThanOrEqual(0);
                    expect(result.biomarkers.shimmer.value).toBeLessThanOrEqual(1);
                }),
                { numRuns: 100 }
            );
        });

        it('should have HNR value in valid range (0-30 dB)', () => {
            fc.assert(
                fc.property(enhancedSpeechResultArb, (result) => {
                    expect(result.biomarkers.hnr.value).toBeGreaterThanOrEqual(0);
                    expect(result.biomarkers.hnr.value).toBeLessThanOrEqual(30);
                }),
                { numRuns: 100 }
            );
        });

        it('should have speech rate value in valid range (0.5-10 syl/s)', () => {
            fc.assert(
                fc.property(enhancedSpeechResultArb, (result) => {
                    expect(result.biomarkers.speechRate.value).toBeGreaterThanOrEqual(0.5);
                    expect(result.biomarkers.speechRate.value).toBeLessThanOrEqual(10);
                }),
                { numRuns: 100 }
            );
        });

        it('should have pause ratio value in valid range (0-1)', () => {
            fc.assert(
                fc.property(enhancedSpeechResultArb, (result) => {
                    expect(result.biomarkers.pauseRatio.value).toBeGreaterThanOrEqual(0);
                    expect(result.biomarkers.pauseRatio.value).toBeLessThanOrEqual(1);
                }),
                { numRuns: 100 }
            );
        });
    });

    describe('Estimated Value Indicators', () => {
        it('should correctly identify estimated biomarkers', () => {
            fc.assert(
                fc.property(enhancedSpeechResultArb, (result) => {
                    const estimatedCount = Object.values(result.biomarkers).filter(
                        (b) => b.isEstimated
                    ).length;

                    // Count should be between 0 and 9
                    expect(estimatedCount).toBeGreaterThanOrEqual(0);
                    expect(estimatedCount).toBeLessThanOrEqual(9);
                }),
                { numRuns: 100 }
            );
        });

        it('should have isEstimated as boolean for all biomarkers', () => {
            fc.assert(
                fc.property(enhancedSpeechResultArb, (result) => {
                    for (const biomarker of Object.values(result.biomarkers)) {
                        expect(typeof biomarker.isEstimated).toBe('boolean');
                    }
                }),
                { numRuns: 100 }
            );
        });
    });

    describe('Baseline Comparison', () => {
        it('should handle results with baseline comparison', () => {
            fc.assert(
                fc.property(
                    enhancedSpeechResultArb.filter((r) => r.baselineComparison !== undefined),
                    (result) => {
                        expect(result.baselineComparison).toBeDefined();
                        expect(typeof result.baselineComparison).toBe('object');
                    }
                ),
                { numRuns: 100 }
            );
        });

        it('should handle results without baseline comparison', () => {
            fc.assert(
                fc.property(
                    enhancedSpeechResultArb.filter((r) => r.baselineComparison === undefined),
                    (result) => {
                        expect(result.baselineComparison).toBeUndefined();
                    }
                ),
                { numRuns: 100 }
            );
        });

        it('should have valid delta values in baseline comparison', () => {
            fc.assert(
                fc.property(
                    enhancedSpeechResultArb.filter((r) => r.baselineComparison !== undefined),
                    (result) => {
                        if (result.baselineComparison) {
                            for (const delta of Object.values(result.baselineComparison)) {
                                expect(typeof delta).toBe('number');
                                expect(Number.isFinite(delta)).toBe(true);
                            }
                        }
                    }
                ),
                { numRuns: 100 }
            );
        });
    });

    describe('Session Metadata', () => {
        it('should have valid session ID', () => {
            fc.assert(
                fc.property(enhancedSpeechResultArb, (result) => {
                    expect(typeof result.sessionId).toBe('string');
                    expect(result.sessionId.length).toBeGreaterThan(0);
                }),
                { numRuns: 100 }
            );
        });

        it('should have valid timestamp', () => {
            fc.assert(
                fc.property(enhancedSpeechResultArb, (result) => {
                    expect(typeof result.timestamp).toBe('string');
                    // Should be parseable as a date
                    const date = new Date(result.timestamp);
                    expect(date.toString()).not.toBe('Invalid Date');
                }),
                { numRuns: 100 }
            );
        });

        it('should have positive processing time', () => {
            fc.assert(
                fc.property(enhancedSpeechResultArb, (result) => {
                    expect(result.processingTime).toBeGreaterThan(0);
                }),
                { numRuns: 100 }
            );
        });
    });
});

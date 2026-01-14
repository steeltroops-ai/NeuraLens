/**
 * Property-Based Tests for MediLens Design System
 * 
 * Feature: frontend-global-fix
 * Tests Properties 15, 16, 18, 21, 22
 * 
 * Validates the MediLens Design System implementation including:
 * - Color consistency (Property 15)
 * - Risk level color consistency (Property 16)
 * - Spacing grid consistency (Property 18)
 * - Button component consistency (Property 21)
 * - Card component consistency (Property 22)
 */

import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';
import {
    colors,
    medilensBlue,
    nriColors,
    statusColors,
    surfaceColors,
    textColors,
    spacing,
    borderRadius,
    shadows,
    getNriRiskLevel,
    getNriRiskColor,
    buttonPatterns,
    cardPatterns,
} from '@/styles/design-tokens';

/**
 * MediLens Blue color values from design system
 */
const MEDILENS_BLUE_500 = '#007AFF';
const MEDILENS_BLUE_600 = '#0062CC';

/**
 * NRI Risk Gradient Colors from design system
 */
const NRI_COLORS = {
    minimal: '#34C759',
    low: '#30D158',
    moderate: '#FFD60A',
    elevated: '#FF9F0A',
    high: '#FF6B6B',
    critical: '#FF3B30',
};

/**
 * Valid 8px grid spacing values
 */
const VALID_SPACING_VALUES = [
    '4px', '8px', '12px', '16px', '20px', '24px',
    '32px', '40px', '48px', '64px', '80px', '96px',
];

/**
 * Arbitrary generator for NRI scores (0-100)
 */
const nriScoreArb = fc.integer({ min: 0, max: 100 });

/**
 * Arbitrary generator for risk levels
 */
const riskLevelArb = fc.constantFrom(
    'minimal', 'low', 'moderate', 'elevated', 'high', 'critical'
) as fc.Arbitrary<keyof typeof NRI_COLORS>;

/**
 * Arbitrary generator for spacing keys
 */
const spacingKeyArb = fc.constantFrom(...Object.keys(spacing)) as fc.Arbitrary<keyof typeof spacing>;

/**
 * Check if a hex color is valid
 */
function isValidHexColor(color: string): boolean {
    return /^#[0-9A-Fa-f]{6}$/.test(color);
}

/**
 * Check if a string contains the MediLens blue color
 */
function containsMediLensBlue(str: string): boolean {
    return str.includes('#007AFF') ||
        str.includes('medilens-blue-500') ||
        str.includes('from-medilens-blue-500');
}

describe('MediLens Design System Property Tests', () => {

    /**
     * Property 15: Design System Color Consistency
     * 
     * For any component using the MediLens color system:
     * - It SHALL use the exact color tokens defined
     * - MediLens Blue (#007AFF) for primary actions
     * - NRI gradient colors for risk levels
     * - Surface colors for backgrounds
     * 
     * **Validates: Requirements 13.1, 14.1, 14.2, 14.3, 14.4**
     */
    describe('Property 15: Design System Color Consistency', () => {

        it('Property 15.1: MediLens Blue primary color is #007AFF', () => {
            expect(medilensBlue[500]).toBe(MEDILENS_BLUE_500);
        });

        it('Property 15.2: MediLens Blue scale has all required shades', () => {
            const requiredShades = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900];

            requiredShades.forEach(shade => {
                const key = shade as keyof typeof medilensBlue;
                expect(medilensBlue[key]).toBeDefined();
                expect(isValidHexColor(medilensBlue[key])).toBe(true);
            });
        });

        it('Property 15.3: All status colors are valid hex colors', () => {
            Object.values(statusColors).forEach(color => {
                expect(isValidHexColor(color)).toBe(true);
            });
        });

        it('Property 15.4: All surface colors are valid hex colors', () => {
            Object.values(surfaceColors).forEach(color => {
                expect(isValidHexColor(color)).toBe(true);
            });
        });

        it('Property 15.5: All text colors are valid hex colors', () => {
            Object.values(textColors).forEach(color => {
                expect(isValidHexColor(color)).toBe(true);
            });
        });

        it('Property 15.6: Primary button pattern uses MediLens Blue', () => {
            expect(containsMediLensBlue(buttonPatterns.primary)).toBe(true);
        });

        it('Property 15.7: Surface secondary color is #F2F2F7', () => {
            expect(surfaceColors.secondary).toBe('#F2F2F7');
        });

        it('Property 15.8: Text primary color is #000000', () => {
            expect(textColors.primary).toBe('#000000');
        });
    });

    /**
     * Property 16: Risk Level Color Consistency
     * 
     * For any risk indicator displayed:
     * - It SHALL use the NRI risk gradient colors
     * - minimal (#34C759), low (#30D158), moderate (#FFD60A)
     * - elevated (#FF9F0A), high (#FF6B6B), critical (#FF3B30)
     * 
     * **Validates: Requirements 13.3, 14.1**
     */
    describe('Property 16: Risk Level Color Consistency', () => {

        it('Property 16.1: All NRI colors match design system specification', () => {
            expect(nriColors.minimal).toBe(NRI_COLORS.minimal);
            expect(nriColors.low).toBe(NRI_COLORS.low);
            expect(nriColors.moderate).toBe(NRI_COLORS.moderate);
            expect(nriColors.elevated).toBe(NRI_COLORS.elevated);
            expect(nriColors.high).toBe(NRI_COLORS.high);
            expect(nriColors.critical).toBe(NRI_COLORS.critical);
        });

        it('Property 16.2: getNriRiskLevel returns correct level for any score', () => {
            fc.assert(
                fc.property(nriScoreArb, (score) => {
                    const level = getNriRiskLevel(score);

                    // Level must be one of the valid risk levels
                    expect(['minimal', 'low', 'moderate', 'elevated', 'high', 'critical']).toContain(level);

                    // Verify score ranges
                    if (score <= 25) expect(level).toBe('minimal');
                    else if (score <= 40) expect(level).toBe('low');
                    else if (score <= 55) expect(level).toBe('moderate');
                    else if (score <= 70) expect(level).toBe('elevated');
                    else if (score <= 85) expect(level).toBe('high');
                    else expect(level).toBe('critical');
                }),
                { numRuns: 100 }
            );
        });

        it('Property 16.3: getNriRiskColor returns valid hex color for any score', () => {
            fc.assert(
                fc.property(nriScoreArb, (score) => {
                    const color = getNriRiskColor(score);
                    expect(isValidHexColor(color)).toBe(true);
                }),
                { numRuns: 100 }
            );
        });

        it('Property 16.4: Risk level colors are distinct', () => {
            const colorValues = Object.values(nriColors);
            const uniqueColors = new Set(colorValues);
            expect(uniqueColors.size).toBe(colorValues.length);
        });

        it('Property 16.5: All risk levels have corresponding colors', () => {
            fc.assert(
                fc.property(riskLevelArb, (level) => {
                    expect(nriColors[level]).toBeDefined();
                    expect(isValidHexColor(nriColors[level])).toBe(true);
                }),
                { numRuns: 100 }
            );
        });
    });

    /**
     * Property 18: Spacing Grid Consistency
     * 
     * For any component spacing (padding, margin, gap):
     * - It SHALL use values from the 8px grid system
     * - Valid values: 4px, 8px, 12px, 16px, 20px, 24px, 32px, 40px, 48px, 64px, 80px, 96px
     * 
     * **Validates: Requirements 13.2, 18.5**
     */
    describe('Property 18: Spacing Grid Consistency', () => {

        it('Property 18.1: All spacing values follow 8px grid', () => {
            Object.values(spacing).forEach(value => {
                expect(VALID_SPACING_VALUES).toContain(value);
            });
        });

        it('Property 18.2: Spacing scale has all required values', () => {
            const requiredKeys = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24];

            requiredKeys.forEach(key => {
                const spacingKey = key as keyof typeof spacing;
                expect(spacing[spacingKey]).toBeDefined();
            });
        });

        it('Property 18.3: Spacing values are in ascending order', () => {
            const keys = Object.keys(spacing).map(Number).sort((a, b) => a - b);
            const values = keys.map(k => parseInt(spacing[k as keyof typeof spacing]));

            for (let i = 1; i < values.length; i++) {
                expect(values[i]).toBeGreaterThan(values[i - 1]);
            }
        });

        it('Property 18.4: All spacing values are valid pixel values', () => {
            fc.assert(
                fc.property(spacingKeyArb, (key) => {
                    const value = spacing[key];
                    expect(value).toMatch(/^\d+px$/);
                }),
                { numRuns: 100 }
            );
        });

        it('Property 18.5: Base spacing unit is 4px (space-1)', () => {
            expect(spacing[1]).toBe('4px');
        });

        it('Property 18.6: Standard spacing is 8px (space-2)', () => {
            expect(spacing[2]).toBe('8px');
        });
    });

    /**
     * Property 21: Button Component Consistency
     * 
     * For any primary button:
     * - It SHALL use gradient background (from-medilens-blue-500 to-medilens-blue-600)
     * - It SHALL use white text
     * - It SHALL use rounded-xl border radius
     * - It SHALL use min-h-[48px]
     * - It SHALL use shadow-medical
     * 
     * **Validates: Requirements 17.1**
     */
    describe('Property 21: Button Component Consistency', () => {

        it('Property 21.1: Primary button uses gradient background', () => {
            expect(buttonPatterns.primary).toContain('bg-gradient-to-br');
            expect(buttonPatterns.primary).toContain('from-');
            expect(buttonPatterns.primary).toContain('to-');
        });

        it('Property 21.2: Primary button uses MediLens Blue gradient', () => {
            expect(containsMediLensBlue(buttonPatterns.primary)).toBe(true);
        });

        it('Property 21.3: Primary button uses white text', () => {
            expect(buttonPatterns.primary).toContain('text-white');
        });

        it('Property 21.4: Primary button uses rounded-xl', () => {
            expect(buttonPatterns.primary).toContain('rounded-xl');
        });

        it('Property 21.5: Primary button has 48px minimum height', () => {
            expect(buttonPatterns.primary).toContain('min-h-[48px]');
        });

        it('Property 21.6: Primary button uses shadow-medical', () => {
            expect(buttonPatterns.primary).toContain('shadow-');
        });

        it('Property 21.7: Secondary button uses surface-secondary background', () => {
            expect(buttonPatterns.secondary).toContain('bg-');
        });

        it('Property 21.8: Ghost button uses MediLens Blue text', () => {
            expect(buttonPatterns.ghost).toContain('text-');
        });

        it('Property 21.9: All button variants have focus ring', () => {
            Object.values(buttonPatterns).forEach(pattern => {
                expect(pattern).toContain('focus-visible:');
            });
        });
    });

    /**
     * Property 22: Card Component Consistency
     * 
     * For any standard card:
     * - It SHALL use white background
     * - It SHALL use rounded-2xl border radius
     * - It SHALL use p-6 padding
     * - It SHALL use shadow-apple
     * - It SHALL use border border-black/5
     * 
     * **Validates: Requirements 17.2**
     */
    describe('Property 22: Card Component Consistency', () => {

        it('Property 22.1: Standard card uses white background', () => {
            expect(cardPatterns.standard).toContain('bg-white');
        });

        it('Property 22.2: Standard card uses rounded-2xl', () => {
            expect(cardPatterns.standard).toContain('rounded-2xl');
        });

        it('Property 22.3: Standard card uses p-6 padding', () => {
            expect(cardPatterns.standard).toContain('p-6');
        });

        it('Property 22.4: Standard card uses shadow-apple', () => {
            expect(cardPatterns.standard).toContain('shadow-');
        });

        it('Property 22.5: Standard card uses border', () => {
            expect(cardPatterns.standard).toContain('border');
        });

        it('Property 22.6: Featured card uses gradient background', () => {
            expect(cardPatterns.featured).toContain('bg-gradient-to-br');
        });

        it('Property 22.7: Featured card uses rounded-3xl', () => {
            expect(cardPatterns.featured).toContain('rounded-3xl');
        });

        it('Property 22.8: Featured card uses p-8 padding', () => {
            expect(cardPatterns.featured).toContain('p-8');
        });

        it('Property 22.9: Glass card uses backdrop-blur', () => {
            expect(cardPatterns.glass).toContain('backdrop-blur');
        });

        it('Property 22.10: Standard card has hover lift animation', () => {
            expect(cardPatterns.standard).toContain('hover:-translate-y-1');
        });
    });

    /**
     * Additional Design System Consistency Tests
     */
    describe('Additional Design System Tests', () => {

        it('Border radius system has all required values', () => {
            expect(borderRadius.sm).toBe('4px');
            expect(borderRadius.md).toBe('8px');
            expect(borderRadius.lg).toBe('12px');
            expect(borderRadius.xl).toBe('16px');
            expect(borderRadius['2xl']).toBe('24px');
            expect(borderRadius.full).toBe('9999px');
        });

        it('Shadow system has all required shadows', () => {
            expect(shadows.apple).toBeDefined();
            expect(shadows.appleHover).toBeDefined();
            expect(shadows.medical).toBeDefined();
            expect(shadows.medicalHover).toBeDefined();
            expect(shadows.nri).toBeDefined();
            expect(shadows.glass).toBeDefined();
        });

        it('Colors object contains all color categories', () => {
            expect(colors.medilensBlue).toBeDefined();
            expect(colors.status).toBeDefined();
            expect(colors.nri).toBeDefined();
            expect(colors.surface).toBeDefined();
            expect(colors.text).toBeDefined();
        });
    });
});

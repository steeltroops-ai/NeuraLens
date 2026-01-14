/**
 * Property-Based Tests for Typography System Consistency
 * 
 * Feature: frontend-global-fix
 * Property 17: Typography System Consistency
 * Validates: Requirements 15.1, 15.2, 15.3, 15.4
 * 
 * For any text element, it SHALL use the Apple system font stack and follow
 * the typography scale: Display (48px), Title 1 (34px), Title 2 (28px),
 * Title 3 (22px), Headline (17px), Body (17px).
 */

import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';
import {
    fontFamily,
    typographyScale,
    letterSpacing,
    maxTextWidth,
    fluidTypography,
    typography,
} from '@/styles/design-tokens';

/**
 * MediLens Typography Specifications
 */
const APPLE_SYSTEM_FONT_STACK = "-apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', 'Inter', system-ui, sans-serif";
const MONOSPACE_FONT_STACK = "'SF Mono', 'Monaco', 'Menlo', 'JetBrains Mono', monospace";
const MAX_TEXT_WIDTH = '65ch';

/**
 * Typography scale specifications from MediLens Design System
 */
const TYPOGRAPHY_SCALE_SPECS = {
    display: { size: '48px', weight: 700, lineHeight: 1.1 },
    title1: { size: '34px', weight: 700, lineHeight: 1.2 },
    title2: { size: '28px', weight: 600, lineHeight: 1.25 },
    title3: { size: '22px', weight: 600, lineHeight: 1.3 },
    headline: { size: '17px', weight: 600, lineHeight: 1.4 },
    body: { size: '17px', weight: 400, lineHeight: 1.5 },
    callout: { size: '16px', weight: 400, lineHeight: 1.5 },
    subhead: { size: '15px', weight: 400, lineHeight: 1.4 },
    footnote: { size: '13px', weight: 400, lineHeight: 1.35 },
    caption: { size: '12px', weight: 400, lineHeight: 1.3 },
} as const;

/**
 * Fluid typography specifications
 */
const FLUID_TYPOGRAPHY_SPECS = {
    display: 'clamp(32px, 5vw + 1rem, 48px)',
    title1: 'clamp(28px, 4vw + 0.5rem, 34px)',
    title2: 'clamp(22px, 3vw + 0.5rem, 28px)',
    title3: 'clamp(18px, 2.5vw + 0.25rem, 22px)',
} as const;

/**
 * Typography scale entries for property testing
 */
const typographyScaleEntries = Object.entries(TYPOGRAPHY_SCALE_SPECS).map(([name, spec]) => ({
    name,
    spec,
}));

/**
 * Fluid typography entries for property testing
 */
const fluidTypographyEntries = Object.entries(FLUID_TYPOGRAPHY_SPECS).map(([name, value]) => ({
    name,
    value,
}));

/**
 * Parse pixel value from string (e.g., '48px' -> 48)
 */
const parsePixelValue = (value: string): number => {
    const match = value.match(/^(\d+)px$/);
    return match ? parseInt(match[1], 10) : 0;
};

/**
 * Check if a font stack contains Apple system fonts
 */
const containsAppleSystemFonts = (fontStack: string): boolean => {
    return fontStack.includes('-apple-system') && fontStack.includes('BlinkMacSystemFont');
};

/**
 * Check if a font stack contains SF Pro fonts
 */
const containsSFProFonts = (fontStack: string): boolean => {
    return fontStack.includes('SF Pro Display') || fontStack.includes('SF Pro Text');
};

/**
 * Check if a font stack contains monospace fonts
 */
const containsMonospaceFonts = (fontStack: string): boolean => {
    return fontStack.includes('SF Mono') || fontStack.includes('Monaco') || fontStack.includes('Menlo');
};

describe('Typography System Property Tests', () => {
    /**
     * Property 17: Typography System Consistency
     * 
     * For any text element, it SHALL use the Apple system font stack and follow
     * the typography scale: Display (48px), Title 1 (34px), Title 2 (28px),
     * Title 3 (22px), Headline (17px), Body (17px).
     * 
     * **Validates: Requirements 15.1, 15.2, 15.3, 15.4**
     */
    describe('Property 17: Typography System Consistency', () => {
        /**
         * Property 17.1: Apple system font stack is correctly defined
         */
        it('Property 17.1: Apple system font stack is correctly defined', () => {
            expect(fontFamily.system).toBe(APPLE_SYSTEM_FONT_STACK);
            expect(containsAppleSystemFonts(fontFamily.system)).toBe(true);
            expect(containsSFProFonts(fontFamily.system)).toBe(true);
        });

        /**
         * Property 17.2: Monospace font stack is correctly defined for clinical data
         */
        it('Property 17.2: Monospace font stack is correctly defined for clinical data', () => {
            expect(fontFamily.mono).toBe(MONOSPACE_FONT_STACK);
            expect(containsMonospaceFonts(fontFamily.mono)).toBe(true);
        });

        /**
         * Property 17.3: Typography scale matches MediLens specifications
         */
        it('Property 17.3: Typography scale matches MediLens specifications', () => {
            fc.assert(
                fc.property(
                    fc.constantFrom(...typographyScaleEntries),
                    ({ name, spec }) => {
                        const scaleEntry = typographyScale[name as keyof typeof typographyScale];

                        expect(scaleEntry.size).toBe(spec.size);
                        expect(scaleEntry.weight).toBe(spec.weight);
                        expect(scaleEntry.lineHeight).toBe(spec.lineHeight);

                        return true;
                    }
                ),
                { numRuns: 100 }
            );
        });

        /**
         * Property 17.4: Display typography is 48px with weight 700
         */
        it('Property 17.4: Display typography is 48px with weight 700', () => {
            expect(typographyScale.display.size).toBe('48px');
            expect(typographyScale.display.weight).toBe(700);
            expect(typographyScale.display.lineHeight).toBe(1.1);
        });

        /**
         * Property 17.5: Title 1 typography is 34px with weight 700
         */
        it('Property 17.5: Title 1 typography is 34px with weight 700', () => {
            expect(typographyScale.title1.size).toBe('34px');
            expect(typographyScale.title1.weight).toBe(700);
            expect(typographyScale.title1.lineHeight).toBe(1.2);
        });

        /**
         * Property 17.6: Title 2 typography is 28px with weight 600
         */
        it('Property 17.6: Title 2 typography is 28px with weight 600', () => {
            expect(typographyScale.title2.size).toBe('28px');
            expect(typographyScale.title2.weight).toBe(600);
            expect(typographyScale.title2.lineHeight).toBe(1.25);
        });

        /**
         * Property 17.7: Title 3 typography is 22px with weight 600
         */
        it('Property 17.7: Title 3 typography is 22px with weight 600', () => {
            expect(typographyScale.title3.size).toBe('22px');
            expect(typographyScale.title3.weight).toBe(600);
            expect(typographyScale.title3.lineHeight).toBe(1.3);
        });

        /**
         * Property 17.8: Body typography is 17px with weight 400
         */
        it('Property 17.8: Body typography is 17px with weight 400', () => {
            expect(typographyScale.body.size).toBe('17px');
            expect(typographyScale.body.weight).toBe(400);
            expect(typographyScale.body.lineHeight).toBe(1.5);
        });

        /**
         * Property 17.9: Maximum text width is 65ch for readability
         */
        it('Property 17.9: Maximum text width is 65ch for readability', () => {
            expect(maxTextWidth).toBe(MAX_TEXT_WIDTH);
            expect(typography.maxWidth).toBe(MAX_TEXT_WIDTH);
        });

        /**
         * Property 17.10: Letter spacing is defined for headlines and body
         */
        it('Property 17.10: Letter spacing is defined for headlines and body', () => {
            expect(letterSpacing.tight).toBe('-0.02em');
            expect(letterSpacing.normal).toBe('0');
        });

        /**
         * Property 17.11: Fluid typography uses clamp() for responsive scaling
         */
        it('Property 17.11: Fluid typography uses clamp() for responsive scaling', () => {
            fc.assert(
                fc.property(
                    fc.constantFrom(...fluidTypographyEntries),
                    ({ name, value }) => {
                        const fluidValue = fluidTypography[name as keyof typeof fluidTypography];

                        // Should match the expected clamp value
                        expect(fluidValue).toBe(value);

                        // Should start with 'clamp('
                        expect(fluidValue.startsWith('clamp(')).toBe(true);

                        // Should end with ')'
                        expect(fluidValue.endsWith(')')).toBe(true);

                        return true;
                    }
                ),
                { numRuns: 100 }
            );
        });

        /**
         * Property 17.12: Typography scale sizes are in descending order
         */
        it('Property 17.12: Typography scale sizes are in descending order', () => {
            const sizes = [
                parsePixelValue(typographyScale.display.size),
                parsePixelValue(typographyScale.title1.size),
                parsePixelValue(typographyScale.title2.size),
                parsePixelValue(typographyScale.title3.size),
                parsePixelValue(typographyScale.headline.size),
                parsePixelValue(typographyScale.body.size),
                parsePixelValue(typographyScale.callout.size),
                parsePixelValue(typographyScale.subhead.size),
                parsePixelValue(typographyScale.footnote.size),
                parsePixelValue(typographyScale.caption.size),
            ];

            // Each size should be >= the next size (descending order)
            for (let i = 0; i < sizes.length - 1; i++) {
                expect(sizes[i]).toBeGreaterThanOrEqual(sizes[i + 1]);
            }
        });

        /**
         * Property 17.13: All typography weights are valid CSS font-weight values
         */
        it('Property 17.13: All typography weights are valid CSS font-weight values', () => {
            const validWeights = [100, 200, 300, 400, 500, 600, 700, 800, 900];

            fc.assert(
                fc.property(
                    fc.constantFrom(...typographyScaleEntries),
                    ({ name, spec }) => {
                        const scaleEntry = typographyScale[name as keyof typeof typographyScale];
                        expect(validWeights).toContain(scaleEntry.weight);
                        return true;
                    }
                ),
                { numRuns: 100 }
            );
        });

        /**
         * Property 17.14: All line heights are positive numbers
         */
        it('Property 17.14: All line heights are positive numbers', () => {
            fc.assert(
                fc.property(
                    fc.constantFrom(...typographyScaleEntries),
                    ({ name }) => {
                        const scaleEntry = typographyScale[name as keyof typeof typographyScale];
                        expect(scaleEntry.lineHeight).toBeGreaterThan(0);
                        expect(typeof scaleEntry.lineHeight).toBe('number');
                        return true;
                    }
                ),
                { numRuns: 100 }
            );
        });

        /**
         * Property 17.15: Typography object exports all required properties
         */
        it('Property 17.15: Typography object exports all required properties', () => {
            expect(typography.fontFamily).toBeDefined();
            expect(typography.scale).toBeDefined();
            expect(typography.letterSpacing).toBeDefined();
            expect(typography.maxWidth).toBeDefined();
            expect(typography.fluid).toBeDefined();

            // Verify nested properties
            expect(typography.fontFamily.system).toBe(APPLE_SYSTEM_FONT_STACK);
            expect(typography.fontFamily.mono).toBe(MONOSPACE_FONT_STACK);
            expect(typography.maxWidth).toBe(MAX_TEXT_WIDTH);
        });

        /**
         * Property 17.16: Headline and body have same size but different weights
         */
        it('Property 17.16: Headline and body have same size but different weights', () => {
            expect(typographyScale.headline.size).toBe(typographyScale.body.size);
            expect(typographyScale.headline.weight).toBeGreaterThan(typographyScale.body.weight);
        });
    });
});
